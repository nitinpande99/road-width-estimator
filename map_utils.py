"""
Map Utilities Module
Functions for creating buffered corridor polygons from GPS points and road widths.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
import pyproj
from functools import partial


def create_buffered_corridor(
    points: List[Tuple[float, float]],
    widths: List[float],
    buffer_ratio: float = 0.5
) -> List[Polygon]:
    """
    Create buffered corridor polygons from GPS points and road widths.
    
    Args:
        points: List of (latitude, longitude) tuples
        widths: List of road widths in meters (corresponding to points)
        buffer_ratio: Ratio to buffer (0.5 = half width on each side)
    
    Returns:
        List of Polygon objects representing buffered corridors
    """
    if len(points) < 2:
        return []
    
    polygons = []
    
    # Create line segments between consecutive points
    for i in range(len(points) - 1):
        p1 = points[i]  # (lat, lon)
        p2 = points[i + 1]  # (lat, lon)
        
        # Use average width for the segment
        avg_width = (widths[i] + widths[i + 1]) / 2
        
        # Create line segment - Shapely expects (lon, lat) for geographic coordinates
        line = LineString([(p1[1], p1[0]), (p2[1], p2[0])])  # Convert to (lon, lat)
        
        # Buffer in meters (convert to approximate degrees)
        # Note: This is approximate - for better accuracy, use UTM projection
        buffer_meters = avg_width * buffer_ratio
        
        # Approximate conversion: 1 degree latitude ≈ 111,000 meters
        # Longitude varies by latitude, but we'll use a simple approximation
        buffer_degrees = buffer_meters / 111000.0
        
        # Simple buffering (not perfect but works for small areas)
        buffered = line.buffer(buffer_degrees)
        
        # For better accuracy, use UTM projection
        try:
            # Determine UTM zone from longitude
            lon = (p1[1] + p2[1]) / 2
            lat = (p1[0] + p2[0]) / 2
            utm_zone = int((lon + 180) / 6) + 1
            
            # Create UTM projection
            utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
            wgs84_crs = "+proj=longlat +datum=WGS84 +no_defs"
            
            # Project to UTM
            project_to_utm = pyproj.Transformer.from_crs(
                wgs84_crs, utm_crs, always_xy=True
            )
            project_to_wgs84 = pyproj.Transformer.from_crs(
                utm_crs, wgs84_crs, always_xy=True
            )
            
            # Transform line to UTM
            line_utm = transform(project_to_utm.transform, line)
            
            # Buffer in meters
            buffered_utm = line_utm.buffer(buffer_meters)
            
            # Transform back to WGS84
            buffered = transform(project_to_wgs84.transform, buffered_utm)
        except Exception:
            # Fallback to simple buffering if projection fails
            buffered = line.buffer(buffer_degrees)
        
        polygons.append(buffered)
    
    return polygons


def get_width_color(width: float, min_width: float, max_width: float) -> str:
    """
    Get color for width value using a color scale.
    Uses a gradient from blue (narrow) to green (medium) to red (wide).
    
    Args:
        width: Road width in meters
        min_width: Minimum width in dataset
        max_width: Maximum width in dataset
    
    Returns:
        Hex color code
    """
    if max_width == min_width:
        return '#808080'  # Gray if no variation
    
    # Normalize width to 0-1
    normalized = (width - min_width) / (max_width - min_width)
    normalized = max(0.0, min(1.0, normalized))
    
    # Color gradient: blue -> cyan -> green -> yellow -> orange -> red
    if normalized < 0.2:
        # Blue to cyan
        r = 0
        g = int(255 * (normalized / 0.2))
        b = 255
    elif normalized < 0.4:
        # Cyan to green
        r = 0
        g = 255
        b = int(255 * (1 - (normalized - 0.2) / 0.2))
    elif normalized < 0.6:
        # Green to yellow
        r = int(255 * ((normalized - 0.4) / 0.2))
        g = 255
        b = 0
    elif normalized < 0.8:
        # Yellow to orange
        r = 255
        g = int(255 * (1 - (normalized - 0.6) / 0.2))
        b = 0
    else:
        # Orange to red
        r = 255
        g = int(255 * (1 - (normalized - 0.8) / 0.2) * 0.5)
        b = 0
    
    return f'#{r:02x}{g:02x}{b:02x}'


def get_confidence_color(confidence: float) -> str:
    """
    Get color for confidence score using a color scale.
    Uses a gradient from red (low confidence) to yellow (medium) to green (high confidence).
    
    Args:
        confidence: Confidence score between 0.0 and 1.0
    
    Returns:
        Hex color code
    """
    # Clamp confidence to 0-1
    normalized = max(0.0, min(1.0, confidence))
    
    # Color gradient: red -> orange -> yellow -> green
    if normalized < 0.25:
        # Red to orange-red
        r = 255
        g = int(255 * (normalized / 0.25) * 0.5)
        b = 0
    elif normalized < 0.5:
        # Orange-red to orange
        r = 255
        g = int(255 * (0.5 + (normalized - 0.25) / 0.25 * 0.5))
        b = 0
    elif normalized < 0.75:
        # Orange to yellow
        r = 255
        g = 255
        b = int(255 * (1 - (normalized - 0.5) / 0.25))
    else:
        # Yellow to green
        r = int(255 * (1 - (normalized - 0.75) / 0.25))
        g = 255
        b = 0
    
    return f'#{r:02x}{g:02x}{b:02x}'


def get_confidence_style(confidence: float, low_confidence_threshold: float = 0.5) -> Dict:
    """
    Get styling parameters for confidence-based visual distinction.
    Low confidence polygons get reduced opacity and dashed borders.
    
    Args:
        confidence: Confidence score between 0.0 and 1.0
        low_confidence_threshold: Threshold below which polygons are considered low confidence
    
    Returns:
        Dictionary with fillOpacity, opacity, and dashArray settings
    """
    is_low_confidence = confidence < low_confidence_threshold
    
    if is_low_confidence:
        # Fade low confidence polygons
        # Opacity scales from 0.2 (very low) to 0.4 (at threshold)
        normalized_low = confidence / low_confidence_threshold
        fill_opacity = 0.2 + (normalized_low * 0.2)  # 0.2 to 0.4
        border_opacity = 0.4 + (normalized_low * 0.3)  # 0.4 to 0.7
        dash_array = '10, 5'  # Dashed border pattern
    else:
        # Normal opacity for high confidence
        fill_opacity = 0.6
        border_opacity = 0.8
        dash_array = None  # Solid border
    
    return {
        'fillOpacity': fill_opacity,
        'opacity': border_opacity,
        'dashArray': dash_array
    }


def create_corridor_geojson(
    results: List[Dict]
) -> Dict:
    """
    Create GeoJSON FeatureCollection from results with buffered corridors.
    
    Args:
        results: List of result dictionaries with lat, lon, width
    
    Returns:
        GeoJSON dictionary
    """
    valid_results = [r for r in results if r.get('lat', 0.0) != 0.0 and r.get('lon', 0.0) != 0.0]
    
    if len(valid_results) < 2:
        return {"type": "FeatureCollection", "features": []}
    
    # Extract points and widths
    points = [(r['lat'], r['lon']) for r in valid_results]
    widths = [r.get('width', 0.0) for r in valid_results]
    
    # Create buffered corridors
    polygons = create_buffered_corridor(points, widths)
    
    # Get width range for coloring
    min_width = min(widths)
    max_width = max(widths)
    
    # Create features
    features = []
    for i, polygon in enumerate(polygons):
        # Get average width for this segment
        avg_width = (widths[i] + widths[i + 1]) / 2 if i + 1 < len(widths) else widths[i]
        avg_confidence = (valid_results[i].get('confidence', 0.5) + 
                         valid_results[i + 1].get('confidence', 0.5)) / 2 if i + 1 < len(valid_results) else valid_results[i].get('confidence', 0.5)
        
        # Convert polygon to GeoJSON coordinates (lon, lat order)
        # Shapely polygon coordinates are already in (lon, lat) format
        coords = [list(polygon.exterior.coords)]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            },
            "properties": {
                "width": avg_width,
                "confidence": avg_confidence,
                "color": get_width_color(avg_width, min_width, max_width),
                "segment_index": i
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }
