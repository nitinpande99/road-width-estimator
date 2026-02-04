"""
GPS utilities for Road Width Estimator.

Handles reading GPS data, interpolation by timestamp,
and aggregation by distance.
"""

from typing import List, Tuple, Optional
import csv
import math
import xml.etree.ElementTree as ET


# -----------------------------
# Helpers
# -----------------------------

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate distance (meters) between two lat/lon points.
    """
    R = 6371000.0  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# -----------------------------
# Readers
# -----------------------------

def read_gps_csv(path: str) -> List[Tuple[float, float, float]]:
    """
    Read GPS CSV file.

    Expected columns: timestamp, lat, lon
    """
    data = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp = float(row["timestamp"])
                lat = float(row["lat"])
                lon = float(row["lon"])
                data.append((timestamp, lat, lon))
            except Exception:
                continue

    return data


def read_gps_gpx(path: str) -> List[Tuple[float, float, float]]:
    """
    Read GPS GPX file.

    Returns list of (timestamp, lat, lon)
    """
    data = []

    tree = ET.parse(path)
    root = tree.getroot()

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}

    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])

        time_el = trkpt.find("gpx:time", ns)
        if time_el is None:
            continue

        timestamp = (
            time_el.text.replace("Z", "")
        )

        try:
            ts = (
                float(
                    (
                        __import__("datetime")
                        .datetime.fromisoformat(timestamp)
                        .timestamp()
                    )
                )
            )
        except Exception:
            continue

        data.append((ts, lat, lon))

    return data


# -----------------------------
# Interpolation
# -----------------------------

def interpolate_gps(
    gps_data: List[Tuple[float, float, float]],
    timestamps: List[float],
) -> List[Tuple[float, float]]:
    """
    Interpolate GPS coordinates for given timestamps.
    """
    if not gps_data:
        return [(0.0, 0.0) for _ in timestamps]

    gps_data = sorted(gps_data, key=lambda x: x[0])

    result = []

    idx = 0
    for ts in timestamps:
        while idx < len(gps_data) - 1 and gps_data[idx + 1][0] <= ts:
            idx += 1

        t1, lat1, lon1 = gps_data[idx]

        if idx == len(gps_data) - 1:
            result.append((lat1, lon1))
        else:
            t2, lat2, lon2 = gps_data[idx + 1]

            if t2 == t1:
                result.append((lat1, lon1))
            else:
                ratio = (ts - t1) / (t2 - t1)
                lat = lat1 + ratio * (lat2 - lat1)
                lon = lon1 + ratio * (lon2 - lon1)
                result.append((lat, lon))

    return result


# -----------------------------
# Aggregation
# -----------------------------

def aggregate_by_distance(
    results: List[dict],
    distance_threshold_m: float = 10.0,
) -> List[dict]:
    """
    Aggregate results when distance exceeds threshold.
    """
    if not results:
        return []

    aggregated = [results[0]]
    last = results[0]

    for r in results[1:]:
        d = haversine_distance(
            last["lat"], last["lon"],
            r["lat"], r["lon"]
        )

        if d >= distance_threshold_m:
            aggregated.append(r)
            last = r

    return aggregated
