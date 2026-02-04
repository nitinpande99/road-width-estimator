"""
Projection utilities for Road Width Estimator.

All OpenCV (cv2) imports are done INSIDE functions
to remain compatible with Streamlit Cloud.
"""

import numpy as np
from typing import Tuple


def extract_front_view(
    frame: np.ndarray,
    fov_deg: float = 90.0
) -> np.ndarray:
    """
    Extract a front-facing view from an equirectangular frame.

    Args:
        frame: Input frame (H x W x 3)
        fov_deg: Horizontal field of view in degrees

    Returns:
        Cropped front-view image
    """
    # 🔴 Streamlit-safe import
    import cv2

    if frame is None or frame.size == 0:
        raise ValueError("Empty frame provided to extract_front_view")

    h, w, _ = frame.shape

    # Simple & SAFE default behavior:
    # Take central vertical slice as "front view"
    center_x = w // 2
    half_width = int((fov_deg / 360.0) * w / 2)

    x1 = max(0, center_x - half_width)
    x2 = min(w, center_x + half_width)

    front_view = frame[:, x1:x2]

    # Ensure output is valid
    if front_view.size == 0:
        raise ValueError("Front view extraction failed")

    return front_view
