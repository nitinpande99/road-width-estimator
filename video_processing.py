"""
Video Processing Module
Handles video frame extraction from equirectangular MP4 files.

This module safely uses OpenCV in Streamlit Cloud by importing cv2
inside functions (not at module import time).
"""
import os
import numpy as np
from typing import Generator, Tuple, Optional


def extract_frames(
    video_path: str,
    fps: float = 2.0,
    max_frames: Optional[int] = None
) -> Generator[Tuple[int, np.ndarray, float], None, None]:
    """
    Extract frames from a video at a specified FPS.

    Args:
        video_path: Path to input video file
        fps: Target frames per second to extract
        max_frames: Maximum number of frames to extract (None for all)

    Yields:
        (frame_index, frame_array, timestamp_seconds)
    """
    # 🔴 IMPORTANT: Import cv2 inside function (Streamlit-safe)
    import cv2

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # fallback safety

    frame_interval = max(1, int(video_fps / fps))

    frame_count = 0
    extracted_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                yield extracted_count, frame, timestamp
                extracted_count += 1

                if max_frames is not None and extracted_count >= max_frames:
                    break

            frame_count += 1
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """
    Get basic metadata from a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties
    """
    # 🔴 IMPORTANT: Import cv2 inside function (Streamlit-safe)
    import cv2

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "frame_count": frame_count,
    }
