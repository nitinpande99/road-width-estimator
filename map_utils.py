"""
Map utilities for Road Width Estimator.

Streamlit-safe: no shapely, no native dependencies.
"""

from typing import List, Tuple
import math


# --------------------------------------------------
# Color helpers
# --------------------------------------------------

def get_width_color(width: float, min_w: float, max_w: float) -> str:
    if max_w <= min_w:
        return "#00ff00"

    ratio = (width - min_w) / (max_w - min_w)
    ratio = max(0.0, min(1.0, ratio))

    r = int(255 * ratio)
    g = int(255 * (1 - abs(ratio - 0.5) * 2))
    b = int(255 * (1 - ratio))

    return f"#{r:02x}{g:02x}{b:02x}"


def get_confidence_color(conf: float) -> str:
    conf = max(0.0, min(1.0, conf))
    r = int(255 * (1 - conf))
    g = int(255 * conf)
    return f"#{r:02x}{g:02x}00"


def get_confidence_style(conf: float) -> dict:
    if conf < 0.5:
        return {
            "fillOpacity": 0.3,
            "opacity": 0.5,
            "dashArray": "5,5",
        }
    return {
        "fillOpacity": 0.6,
        "opacity": 0.9,
        "dashArray": None,
    }


# --------------------------------------------------
# Geometry helpers
# --------------------------------------------------

def _offset_point(
    lat: float, lon: float, dx_m: float, dy_m: float
) -> Tuple[float, float]:
    """
    Offset lat/lon by meters (approx).
    """
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def create_buffered_corridor(
    points: List[Tuple[float, float]],
    widths: List[float],
) -> List[List[Tuple[float, float]]]:
    """
    Create corridor polygons as simple rectangles between points.
    """
    polygons = []

    for i in range(len(points) - 1):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]

        w = widths[i] / 2.0

        dx = lon2 -
