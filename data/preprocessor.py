"""
Functions for preprocessing seismic data.
"""
import numpy as np
from obspy import Trace, Stream, UTCDateTime
from config import DEFAULT_SAMPLING_RATE, DEFAULT_WAVEFORM_LENGTH

def to_velocity(wf_np, row, inventory, *, mode="full"):
    """
    Convert a (3, n) numpy array in SEED counts to ground velocity [m/s].

    Args:
        wf_np: Numpy array of shape (3, n) containing raw counts
        row: Row from metadata containing station information
        inventory: ObsPy inventory object with instrument response information
        mode: "full" → ObsPy remove_response (amplitude + phase)
              "scalar"→ divide by InstrumentSensitivity only
              
    Returns:
        Numpy array of shape (3, n) containing velocity in m/s
    """
    net, sta = row['station_network_code'], row['station_code']
    sr = row.get('trace_sampling_rate_hz', DEFAULT_SAMPLING_RATE)
    t0 = UTCDateTime(row['trace_start_time'])

    print(f"DEBUG - Net: {net}, Sta: {sta})")

    comps = ["Z", "N", "E"]
    ch_codes = [f"HH{c}" for c in comps]

    if mode == "scalar":
        # --- quick-and-dirty: divide by InstrumentSensitivity
        out = np.empty_like(wf_np, dtype=float)
        for i, cha in enumerate(ch_codes):
            resp = inventory.select(network=net, station=sta, channel=cha)
            sens = resp[0][0][0].response.instrument_sensitivity.value
            out[i] = wf_np[i] / sens
        return out

    else:
        # --- full deconvolution
        stream = Stream()
        for i, cha in enumerate(ch_codes):
            tr = Trace(wf_np[i].astype("float64"))
            tr.stats.network = net
            tr.stats.station = sta
            tr.stats.channel = cha
            tr.stats.starttime = t0
            tr.stats.sampling_rate = sr
            stream.append(tr)

        # 0.005–0.01 / 40–50 Hz taper keeps numerical noise down at the ends
        stream.remove_response(inventory=inventory, output="VEL",
                              pre_filt=[0.005, 0.01, 40.0, 50.0])

        return np.vstack([tr.data for tr in stream])

def standardize_waveforms(data_dict, data_format, target_length=DEFAULT_WAVEFORM_LENGTH):
    """
    Standardize all waveforms to the same length by padding or trimming

    Args:
        data_dict: Dictionary of event data
        data_format: Format of the data ('multi_station' or 'single_station')
        target_length: Target length for all waveforms (samples)
        
    Returns:
        modified_count: Number of waveforms that were modified
    """
    print("\n" + "=" * 50)
    print(f"STANDARDIZING WAVEFORMS TO {target_length} SAMPLES")
    print("=" * 50)

    modified_count = 0

    if data_format == "multi_station":
        for event_key, stations_data in data_dict.items():
            for station_key, station_data in stations_data.items():
                waveform = station_data["waveform"]

                if waveform.shape[1] == target_length:
                    continue

                if waveform.shape[1] > target_length:
                    data_dict[event_key][station_key]["waveform"] = waveform[:, :target_length]
                else:
                    padded = np.zeros((3, target_length))
                    padded[:, : waveform.shape[1]] = waveform
                    data_dict[event_key][station_key]["waveform"] = padded

                modified_count += 1
    else:
        for event_key, event_data in data_dict.items():
            waveform = event_data["waveform"]

            if waveform.shape[1] == target_length:
                continue

            if waveform.shape[1] > target_length:
                data_dict[event_key]["waveform"] = waveform[:, :target_length]
            else:
                padded = np.zeros((3, target_length))
                padded[:, : waveform.shape[1]] = waveform
                data_dict[event_key]["waveform"] = padded

            modified_count += 1

    print(f"Standardized {modified_count} waveforms to length {target_length}")
    return modified_count