"""
Projection utilities for Road Width Estimator.

Converts equirectangular frames to a front-facing perspective view.
"""

import numpy as np
import cv2


def extract_front_view(
    frame: np.ndarray,
    fov_deg: float = 90.0,
    out_width: int = 640,
    out_height: int = 480
) -> np.ndarray:
    """
    Extract a front-facing perspective view from an equirectangular frame.

    Args:
        frame: Input equirectangular image (H x W x 3)
        fov_deg: Horizontal field of view in degrees
        out_width: Output width
        out_height: Output height

    Returns:
        Perspective (front-view) image
    """

    h, w, _ = frame.shape
    fov = np.deg2rad(fov_deg)

    # Output image grid
    x = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), out_width)
    y = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), out_height)
    xv, yv = np.meshgrid(x, -y)
    zv = np.ones_like(xv)

    # Normalize direction vectors
    norm = np.sqrt(xv**2 + yv**2 + zv**2)
    xv /= norm
    yv /= norm
    zv /= norm

    # Spherical coordinates
    lon = np.arctan2(xv, zv)
    lat = np.arcsin(yv)

    # Map to equirectangular image
    map_x = (lon / (2 * np.pi) + 0.5) * w
    map_y = (0.5 - lat / np.pi) * h

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Remap
    front_view = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )

    return front_view
