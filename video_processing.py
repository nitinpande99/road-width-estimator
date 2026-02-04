"""
Video Processing Module
Handles video frame extraction from equirectangular MP4 files.
"""

import os
import numpy as np
from typing import Generator, Tuple, Optional
import os


def extract_frames(
    video_path: str,
    fps: float = 2.0,
    max_frames: Optional[int] = None
) -> Generator[Tuple[int, np.ndarray, float], None, None]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to input video file
        fps: Target frames per second to extract
        max_frames: Maximum number of frames to extract (None for all)
    
    Yields:
        Tuple of (frame_number, frame_array, timestamp_seconds)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))  # Skip frames to achieve target FPS
    
    frame_count = 0
    extracted_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                yield (extracted_count, frame, timestamp)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video properties (fps, width, height, duration, frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration,
        'frame_count': frame_count
    }
