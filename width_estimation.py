"""
Width Estimation Module
Estimates road width using monocular vision and geometric assumptions.
"""
from typing import Optional, Dict, Tuple, List



def pixel_to_angular_width(
    pixel_width: float,
    image_width: int,
    fov_degrees: float
) -> float:
    """
    Convert pixel width to angular width in radians.
    
    Args:
        pixel_width: Width in pixels
        image_width: Total image width in pixels
        fov_degrees: Horizontal field of view in degrees
    
    Returns:
        Angular width in radians
    """
    fov_rad = np.deg2rad(fov_degrees)
    angular_width_rad = (pixel_width / image_width) * fov_rad
    return angular_width_rad


def angular_to_real_width(
    angular_width_rad: float,
    camera_height: float
) -> float:
    """
    Convert angular width to real-world width using monocular geometry.
    
    Assumes flat road surface and camera looking forward horizontally.
    Formula: width_m = 2 * camera_height * tan(angle_rad / 2)
    
    Args:
        angular_width_rad: Angular width in radians
        camera_height: Camera height above road in meters
    
    Returns:
        Estimated width in meters
    """
    width_m = 2 * camera_height * np.tan(angular_width_rad / 2)
    return width_m


def estimate_road_width(
    left_edge_x: int,
    right_edge_x: int,
    image_width: int,
    fov_degrees: float,
    camera_height: float
) -> float:
    """
    Estimate road width from pixel coordinates.
    
    Args:
        left_edge_x: X coordinate of left road edge (pixels)
        right_edge_x: X coordinate of right road edge (pixels)
        image_width: Total image width (pixels)
        fov_degrees: Horizontal field of view (degrees)
        camera_height: Camera height above road (meters)
    
    Returns:
        Estimated road width in meters
    """
    pixel_width = abs(right_edge_x - left_edge_x)
    
    if pixel_width <= 0:
        return 0.0
    
    # Convert to angular width
    angular_width = pixel_to_angular_width(pixel_width, image_width, fov_degrees)
    
    # Convert to real-world width
    width_m = angular_to_real_width(angular_width, camera_height)
    
    return width_m


def smooth_width_estimates(
    width_estimates: List[float],
    window_size: int = 5
) -> List[float]:
    """
    Smooth width estimates using moving average.
    
    Args:
        width_estimates: List of width estimates in meters
        window_size: Size of smoothing window
    
    Returns:
        Smoothed width estimates
    """
    if len(width_estimates) < window_size:
        return width_estimates
    
    # Use simple moving average
    smoothed = []
    for i in range(len(width_estimates)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(width_estimates), i + window_size // 2 + 1)
        window = width_estimates[start_idx:end_idx]
        smoothed.append(np.mean(window))
    
    return smoothed


def calculate_confidence_score(
    left_edge_x: Optional[int],
    right_edge_x: Optional[int],
    image_width: int,
    width_estimate: float,
    edge_quality: Optional[Dict] = None,
    previous_width: Optional[float] = None
) -> Tuple[float, Dict]:
    """
    Calculate enhanced confidence score for width estimate.
    Considers edge detection quality, left-right symmetry, and temporal consistency.
    
    Args:
        left_edge_x: X coordinate of left edge (None if not detected)
        right_edge_x: X coordinate of right edge (None if not detected)
        image_width: Total image width
        width_estimate: Estimated width in meters
        edge_quality: Dictionary with edge detection quality metrics
        previous_width: Previous frame's width estimate for temporal consistency
    
    Returns:
        Tuple of (confidence_score, confidence_breakdown_dict)
        confidence_score is between 0.0 and 1.0
        confidence_breakdown contains individual component scores
    """
    breakdown = {
        'edge_detection': 1.0,
        'edge_quality': 1.0,
        'symmetry': 1.0,
        'boundary_proximity': 1.0,
        'width_realism': 1.0,
        'temporal_consistency': 1.0
    }
    
    # 1. Edge Detection Quality (0.0-1.0)
    if edge_quality is not None:
        # Edge density component
        edge_density = edge_quality.get('edge_density', 0.0)
        edge_density_score = min(1.0, edge_density * 10)  # Normalize
        
        # Edge strength component
        left_strength = edge_quality.get('left_edge_strength', 0.0)
        right_strength = edge_quality.get('right_edge_strength', 0.0)
        avg_strength = (left_strength + right_strength) / 2
        strength_score = avg_strength
        
        # Number of lines detected
        num_lines = edge_quality.get('num_lines_detected', 0)
        lines_score = min(1.0, num_lines / 10)  # More lines = better
        
        # Combined edge quality score
        breakdown['edge_quality'] = (edge_density_score * 0.3 + 
                                      strength_score * 0.5 + 
                                      lines_score * 0.2)
    else:
        breakdown['edge_quality'] = 0.5  # Default if no quality data
    
    # 2. Edge Detection (basic check)
    if left_edge_x is None or right_edge_x is None:
        breakdown['edge_detection'] = 0.3
    elif not edge_quality.get('edges_detected', True):
        breakdown['edge_detection'] = 0.5
    else:
        breakdown['edge_detection'] = 1.0
    
    # 3. Left-Right Symmetry (0.0-1.0)
    if left_edge_x is not None and right_edge_x is not None and edge_quality is not None:
        # Distance from center symmetry
        center_x = image_width / 2
        left_dist_from_center = abs(left_edge_x - center_x)
        right_dist_from_center = abs(right_edge_x - center_x)
        
        # Symmetry score based on how balanced the edges are
        max_dist = max(left_dist_from_center, right_dist_from_center)
        if max_dist > 0:
            symmetry_ratio = min(left_dist_from_center, right_dist_from_center) / max_dist
            breakdown['symmetry'] = symmetry_ratio
        
        # Edge strength symmetry
        left_strength = edge_quality.get('left_edge_strength', 0.0)
        right_strength = edge_quality.get('right_edge_strength', 0.0)
        if left_strength > 0 or right_strength > 0:
            strength_symmetry = min(left_strength, right_strength) / max(left_strength, right_strength) if max(left_strength, right_strength) > 0 else 0.0
            breakdown['symmetry'] = (breakdown['symmetry'] * 0.6 + strength_symmetry * 0.4)
    else:
        breakdown['symmetry'] = 0.5  # Default if edges not detected
    
    # 4. Boundary Proximity (penalize if edges too close to image edges)
    margin = image_width * 0.1
    boundary_penalty = 1.0
    if left_edge_x is not None and left_edge_x < margin:
        boundary_penalty *= 0.8
    if right_edge_x is not None and right_edge_x > (image_width - margin):
        boundary_penalty *= 0.8
    breakdown['boundary_proximity'] = boundary_penalty
    
    # 5. Width Realism (penalize unrealistic widths)
    width_score = 1.0
    if width_estimate < 2.0:  # Very narrow road
        width_score = 0.6
    elif width_estimate < 3.0:  # Narrow road
        width_score = 0.8
    elif width_estimate > 20.0:  # Very wide road
        width_score = 0.6
    elif width_estimate > 15.0:  # Wide road
        width_score = 0.8
    breakdown['width_realism'] = width_score
    
    # 6. Temporal Consistency (compare with previous frame)
    if previous_width is not None and previous_width > 0:
        width_change_ratio = abs(width_estimate - previous_width) / previous_width
        # Penalize large changes (>30% change)
        if width_change_ratio > 0.3:
            breakdown['temporal_consistency'] = 0.5
        elif width_change_ratio > 0.2:
            breakdown['temporal_consistency'] = 0.7
        elif width_change_ratio > 0.1:
            breakdown['temporal_consistency'] = 0.9
        else:
            breakdown['temporal_consistency'] = 1.0
    else:
        breakdown['temporal_consistency'] = 1.0  # No previous frame, no penalty
    
    # Weighted combination of all factors
    confidence = (
        breakdown['edge_detection'] * 0.20 +
        breakdown['edge_quality'] * 0.25 +
        breakdown['symmetry'] * 0.20 +
        breakdown['boundary_proximity'] * 0.15 +
        breakdown['width_realism'] * 0.10 +
        breakdown['temporal_consistency'] * 0.10
    )
    
    return max(0.0, min(1.0, confidence)), breakdown
