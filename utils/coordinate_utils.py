"""
Utilities for coordinate conversion between geographic and cartesian systems.
"""

import numpy as np
from config import EARTH_RADIUS


def geographic_to_cartesian(lat, lon, depth, ref_lat, ref_lon, ref_depth):
    """
    Convert geographic coordinates to cartesian coordinates
    with the reference point (mainshock) as the origin

    Returns (x, y, z) where:
    x: East-West distance (positive = east)
    y: North-South distance (positive = north)
    z: Depth difference (positive = deeper than mainshock)
    """
    # Convert to radians
    lat1, lon1 = np.radians(ref_lat), np.radians(ref_lon)
    lat2, lon2 = np.radians(lat), np.radians(lon)

    # Calculate the differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # North-South distance (y)
    y = EARTH_RADIUS * dlat

    # East-West distance (x)
    x = EARTH_RADIUS * dlon * np.cos(lat1)

    # Depth difference (z) - positive means deeper than mainshock
    z = depth - ref_depth

    return x, y, z


def cartesian_to_geographic(x, y, z, ref_lat, ref_lon, ref_depth):
    """
    Convert cartesian coordinates back to geographic coordinates
    """
    # Convert reference point to radians
    ref_lat_rad = np.radians(ref_lat)

    # Calculate latitude difference in radians
    dlat = y / EARTH_RADIUS

    # Calculate longitude difference in radians
    dlon = x / (EARTH_RADIUS * np.cos(ref_lat_rad))

    # Convert to absolute latitude and longitude in radians
    lat_rad = ref_lat_rad + dlat
    lon_rad = np.radians(ref_lon) + dlon

    # Convert back to degrees
    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)

    # Calculate absolute depth
    depth = ref_depth + z

    return lat, lon, depth


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points in km.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * EARTH_RADIUS
