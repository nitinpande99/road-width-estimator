"""
Road segmentation module for Road Width Estimator.

All heavy libraries (cv2, torch, ultralytics) are imported
INSIDE methods to keep Streamlit Cloud stable.
"""

import numpy as np
from typing import Optional


class RoadSegmenter:
    """
    Handles road segmentation using a deep learning model (YOLO / segmentation).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the segmenter.

        Args:
            model_path: Optional path to a trained segmentation model
        """
        self.model_path = model_path
        self.model = None

        if model_path:
            self._load_model()

    def _load_model(self):
        """
        Load segmentation model lazily.
        """
        try:
            # 🔴 Streamlit-safe imports
            from ultralytics import YOLO
