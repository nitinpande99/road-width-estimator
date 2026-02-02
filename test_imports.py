"""Test script to check for import errors"""
import sys

try:
    print("Testing imports...")
    import video_processing
    print("✓ video_processing")
    
    import projection
    print("✓ projection")
    
    import segmentation
    print("✓ segmentation")
    
    import width_estimation
    print("✓ width_estimation")
    
    import gps_utils
    print("✓ gps_utils")
    
    import streamlit
    print("✓ streamlit")
    
    import cv2
    print("✓ opencv")
    
    import numpy
    print("✓ numpy")
    
    from ultralytics import YOLO
    print("✓ ultralytics")
    
    import pandas
    print("✓ pandas")
    
    import folium
    print("✓ folium")
    
    from streamlit_folium import st_folium
    print("✓ streamlit_folium")
    
    print("\n✅ All imports successful!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
