"""
GPS Utilities Module
Handles GPS metadata extraction from video files and CSV/GPX files.
"""

import os
import csv
import json
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np


def extract_gps_from_video(video_path: str) -> Optional[List[Dict]]:
    """
    Extract GPS metadata from video file using exiftool or similar.
    For prototype, returns None (requires external tool).
    
    Args:
        video_path: Path to video file
    
    Returns:
        List of GPS data dictionaries or None if not available
    """
    # In production, you would use exiftool or ffprobe to extract GPS data
    # For prototype, we'll assume GPS data is provided separately
    return None


def read_gps_csv(csv_path: str) -> List[Dict]:
    """
    Read GPS data from CSV file.
    Expected format: timestamp, latitude, longitude, (optional: altitude)
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of GPS data dictionaries with keys: timestamp, lat, lon, alt
    """
    gps_data = []
    
    if not os.path.exists(csv_path):
        return gps_data
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Try to parse different column name variations
                lat = float(row.get('latitude', row.get('lat', row.get('y', 0))))
                lon = float(row.get('longitude', row.get('lon', row.get('x', 0)))
                timestamp = row.get('timestamp', row.get('time', row.get('t', '')))
                alt = float(row.get('altitude', row.get('alt', row.get('z', 0))))
                
                gps_data.append({
                    'timestamp': timestamp,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt
                })
            except (ValueError, KeyError) as e:
                continue
    
    return gps_data


def read_gps_gpx(gpx_path: str) -> List[Dict]:
    """
    Read GPS data from GPX file.
    
    Args:
        gpx_path: Path to GPX file
    
    Returns:
        List of GPS data dictionaries with keys: timestamp, lat, lon, alt
    """
    gps_data = []
    
    if not os.path.exists(gpx_path):
        return gps_data
    
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()
        
        # Handle GPX namespace - try with namespace first
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        
        # Try to find trackpoints with namespace
        trkpts = root.findall('.//gpx:trkpt', ns)
        if not trkpts:
            # Fallback: try without namespace
            trkpts = root.findall('.//trkpt')
            ns = {}
        
        for trkpt in trkpts:
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            
            # Try to find elevation element
            if ns:
                ele_elem = trkpt.find('gpx:ele', ns)
            else:
                ele_elem = trkpt.find('ele')
            alt = float(ele_elem.text) if ele_elem is not None and ele_elem.text is not None else 0.0
            
            # Try to find time element
            if ns:
                time_elem = trkpt.find('gpx:time', ns)
            else:
                time_elem = trkpt.find('time')
            timestamp = time_elem.text if time_elem is not None and time_elem.text is not None else ''
            
            gps_data.append({
                'timestamp': timestamp,
                'lat': lat,
                'lon': lon,
                'alt': alt
            })
    except (ET.ParseError, ValueError, AttributeError) as e:
        # Silently fail if GPX parsing fails
        pass
    
    return gps_data


def interpolate_gps(
    gps_data: List[Dict],
    video_timestamps: List[float]
) -> List[Tuple[float, float]]:
    """
    Interpolate GPS coordinates for video frame timestamps.
    
    Args:
        gps_data: List of GPS data points with timestamps
        video_timestamps: List of video frame timestamps in seconds
    
    Returns:
        List of (latitude, longitude) tuples for each video timestamp
    """
    if not gps_data:
        return [(0.0, 0.0)] * len(video_timestamps)
    
    # Convert timestamps to numeric if needed
    gps_times = []
    gps_coords = []
    
    for point in gps_data:
        try:
            # Try to parse timestamp
            if isinstance(point['timestamp'], str):
                # Simple numeric timestamp or ISO format
                try:
                    t = float(point['timestamp'])
                except ValueError:
                    # Try ISO format
                    try:
                        dt = datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00'))
                        t = dt.timestamp()
                    except:
                        continue
            else:
                t = float(point['timestamp'])
            
            gps_times.append(t)
            gps_coords.append((point['lat'], point['lon']))
        except (ValueError, KeyError):
            continue
    
    if not gps_times:
        return [(0.0, 0.0)] * len(video_timestamps)
    
    # Interpolate for each video timestamp
    interpolated = []
    for vid_time in video_timestamps:
        # Find closest GPS point
        closest_idx = min(range(len(gps_times)), key=lambda i: abs(gps_times[i] - vid_time))
        
        # Simple nearest neighbor (can be improved with linear interpolation)
        interpolated.append(gps_coords[closest_idx])
    
    return interpolated


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in meters."""
    R = 6371000  # Earth radius in meters
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 + 
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def _aggregate_group(current_group: List[Dict], aggregated: List[Dict]):
    """Helper function to aggregate a group of results."""
    if not current_group:
        return
    
    avg_lat = np.mean([r.get('lat', 0.0) for r in current_group])
    avg_lon = np.mean([r.get('lon', 0.0) for r in current_group])
    avg_width = np.mean([r.get('width', 0.0) for r in current_group])
    avg_confidence = np.mean([r.get('confidence', 0.0) for r in current_group])
    
    # Preserve other keys from first item
    result_item = {
        'lat': avg_lat,
        'lon': avg_lon,
        'width': avg_width,
        'confidence': avg_confidence,
        'count': len(current_group)
    }
    
    # Add frame and timestamp if they exist
    if 'frame' in current_group[0]:
        result_item['frame'] = current_group[0]['frame']
    if 'timestamp' in current_group[0]:
        result_item['timestamp'] = current_group[0]['timestamp']
    
    aggregated.append(result_item)


def aggregate_by_distance(
    results: List[Dict],
    distance_threshold_m: float = 10.0
) -> List[Dict]:
    """
    Aggregate width estimates by distance traveled.
    Groups consecutive points within distance threshold.
    
    Args:
        results: List of result dictionaries with lat, lon, width
        distance_threshold_m: Distance threshold in meters
    
    Returns:
        Aggregated results
    """
    if not results:
        return []
    
    # Check if we have valid GPS data
    has_valid_gps = any(r.get('lat', 0.0) != 0.0 or r.get('lon', 0.0) != 0.0 for r in results)
    
    if not has_valid_gps:
        # No GPS data, return results as-is
        return results
    
    aggregated = []
    current_group = [results[0]]
    
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        
        # Check if both have valid GPS
        prev_has_gps = prev.get('lat', 0.0) != 0.0 or prev.get('lon', 0.0) != 0.0
        curr_has_gps = curr.get('lat', 0.0) != 0.0 or curr.get('lon', 0.0) != 0.0
        
        if prev_has_gps and curr_has_gps:
            try:
                distance = haversine_distance(
                    prev.get('lat', 0.0), prev.get('lon', 0.0),
                    curr.get('lat', 0.0), curr.get('lon', 0.0)
                )
                
                if distance < distance_threshold_m:
                    current_group.append(curr)
                else:
                    # Aggregate current group
                    _aggregate_group(current_group, aggregated)
                    current_group = [curr]
            except Exception:
                # If distance calculation fails, add individually
                _aggregate_group(current_group, aggregated)
                current_group = [curr]
        else:
            # One or both missing GPS, aggregate current group and start new
            _aggregate_group(current_group, aggregated)
            current_group = [curr]
    
    # Add last group
    _aggregate_group(current_group, aggregated)
    
    return aggregated
