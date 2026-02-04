"""
Road segmentation module (Streamlit-safe).
"""

import numpy as np
from typing import Optional


class RoadSegmenter:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None

        if model_path:
            self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.model = None
            print(f"[WARN] Model load failed: {e}")

    def segment(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")

        if self.model is None:
            return self._fallback_mask(frame)

        try:
            results = self.model(frame, verbose=False)
            masks = results[0].masks

            if masks is None:
                return self._fallback_mask(frame)

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for m in masks.data:
                mask |= (m.cpu().numpy() > 0.5).astype(np.uint8)

            return mask
        except Exception:
            return self._fallback_mask(frame)

    def _fallback_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.5):, int(w * 0.25):int(w * 0.75)] = 1
        return mask
