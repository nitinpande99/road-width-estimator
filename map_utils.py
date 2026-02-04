"""
Map utilities for Road Width Estimator.

Streamlit-safe:
- No shapely
- No native dependencies
- Pure Python + math
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
    return {
        "weight": 4,
        "opacity": max(0.3, min(1.0, conf)),
    }


# --------------------------------------------------
# Corridor utilities (pure math fallback)
# --------------------------------------------------

def create_buffered_corridor(
    points: List[Tuple[float, float]],
    half_width_m: float
) -> List[Tuple[float, float]]:
    """
    Create a simple rectangular corridor approximation
    around a polyline (lat, lon).
    """
    if len(points) < 2:
        return points

    buffered = []
    scale = half_width_m / 111_000.0  # meters → degrees (approx)

    for lat, lon in points:
