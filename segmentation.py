"""
Segmentation Module
Road segmentation using YOLOv8-seg for detecting road surface and boundaries.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from ultralytics import YOLO
import os


class RoadSegmenter:
    """
    Road segmentation using YOLOv8-segmentation model.
    Detects road surface and extracts left/right boundaries.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize road segmenter.
        
        Args:
            model_path: Path to custom YOLOv8-seg model (None for default)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.use_yolo = False
        
        # Try to load YOLOv8-seg model (optional, falls back to heuristic)
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.use_yolo = True
            else:
                # Use pre-trained YOLOv8-seg model
                # Note: This will download the model on first use
                # For prototype, we'll skip YOLO and use heuristic only
                # Uncomment below to enable YOLO (requires internet for first download)
                # self.model = YOLO('yolov8n-seg.pt')
                # self.use_yolo = True
                pass
        except Exception as e:
            # If YOLO fails to load, use heuristic only
            print(f"Warning: Could not load YOLO model, using heuristic segmentation only: {e}")
            self.model = None
            self.use_yolo = False
        
        # COCO class IDs that might represent roads
        # Note: Standard COCO doesn't have "road" class, so we'll use:
        # - Class 0: person (not useful)
        # - We'll need to segment based on lower image region and geometric assumptions
        # For a proper solution, you'd need a custom trained model on road data
        # This is a prototype, so we'll use a heuristic approach
        
    def segment_road_heuristic(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[int], Optional[int], Dict]:
        """
        Heuristic road segmentation for prototype.
        Assumes road is in lower portion of image with distinct edges.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            Tuple of (road_mask, left_edge_x, right_edge_x, edge_quality_dict)
            left_edge_x and right_edge_x are pixel coordinates at scanline
            edge_quality_dict contains metrics about edge detection quality
        """
        h, w = image.shape[:2]
        
        # Focus on lower 60% of image (road is typically below horizon)
        lower_region = image[int(h * 0.4):, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge detection quality metrics
        edge_pixel_count = np.sum(edges > 0)
        edge_density = edge_pixel_count / (lower_region.shape[0] * lower_region.shape[1])
        
        # Use HoughLinesP to detect road boundaries
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        # Select scanline in lower half (e.g., 75% down from top of lower region)
        scanline_y = int(lower_region.shape[0] * 0.75)
        scanline_absolute_y = int(h * 0.4) + scanline_y
        
        # Find left and right edges at scanline
        left_edge = None
        right_edge = None
        left_edge_strength = 0.0
        right_edge_strength = 0.0
        left_lines = []
        right_lines = []
        
        if lines is not None:
            # Filter lines that intersect scanline
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line intersects scanline
                y_min, y_max = min(y1, y2), max(y1, y2)
                if y_min <= scanline_y <= y_max:
                    # Calculate x at scanline
                    if y2 != y1:
                        x_at_scanline = int(x1 + (x2 - x1) * (scanline_y - y1) / (y2 - y1))
                        
                        # Classify as left or right edge based on position and slope
                        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        # Left edge: negative slope, left side of image
                        if slope < -0.1 and x_at_scanline < w * 0.6:
                            if left_edge is None or x_at_scanline < left_edge:
                                left_edge = x_at_scanline
                            left_lines.append((x_at_scanline, line_length, abs(slope)))
                            left_edge_strength += line_length * abs(slope)
                        
                        # Right edge: positive slope, right side of image
                        elif slope > 0.1 and x_at_scanline > w * 0.4:
                            if right_edge is None or x_at_scanline > right_edge:
                                right_edge = x_at_scanline
                            right_lines.append((x_at_scanline, line_length, abs(slope)))
                            right_edge_strength += line_length * abs(slope)
        
        # Normalize edge strengths
        max_possible_strength = lower_region.shape[0] * lower_region.shape[1] * 0.5
        left_edge_strength = min(1.0, left_edge_strength / max_possible_strength) if max_possible_strength > 0 else 0.0
        right_edge_strength = min(1.0, right_edge_strength / max_possible_strength) if max_possible_strength > 0 else 0.0
        
        # Fallback: if no edges found, use image boundaries or center-based estimate
        if left_edge is None:
            left_edge = int(w * 0.1)  # Default to 10% from left
        if right_edge is None:
            right_edge = int(w * 0.9)  # Default to 10% from right
        
        # Create road mask (simple rectangular region between edges)
        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_mask[scanline_absolute_y:, left_edge:right_edge] = 255
        
        # Calculate edge quality metrics
        edge_quality = {
            'edge_density': edge_density,
            'left_edge_strength': left_edge_strength,
            'right_edge_strength': right_edge_strength,
            'num_lines_detected': len(lines) if lines is not None else 0,
            'left_lines_count': len(left_lines),
            'right_lines_count': len(right_lines),
            'edges_detected': left_edge is not None and right_edge is not None
        }
        
        return road_mask, left_edge, right_edge, edge_quality
    
    def segment_road_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[int], Optional[int], Dict]:
        """
        Road segmentation using YOLOv8-seg.
        Note: Standard YOLOv8 doesn't have road class, so this is a placeholder.
        For production, train custom model on road segmentation dataset.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            Tuple of (road_mask, left_edge_x, right_edge_x, edge_quality_dict)
        """
        if self.model is None:
            # Fall back to heuristic if model not available
            return self.segment_road_heuristic(image)
        
        try:
            h, w = image.shape[:2]
            
            # Run YOLOv8 inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            # For prototype, fall back to heuristic if no road-specific detections
            # In production, you'd filter for road class IDs
            road_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Use heuristic as primary method for prototype
            return self.segment_road_heuristic(image)
        except Exception as e:
            # If YOLO fails, fall back to heuristic
            print(f"Warning: YOLO inference failed, using heuristic: {e}")
            return self.segment_road_heuristic(image)
    
    def segment(self, image: np.ndarray, use_yolo: bool = False) -> Tuple[np.ndarray, Optional[int], Optional[int], Dict]:
        """
        Main segmentation method.
        
        Args:
            image: Input image
            use_yolo: Whether to use YOLOv8 (currently falls back to heuristic)
        
        Returns:
            Tuple of (road_mask, left_edge_x, right_edge_x, edge_quality_dict)
        """
        if use_yolo:
            return self.segment_road_yolo(image)
        else:
            return self.segment_road_heuristic(image)
