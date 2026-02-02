# Road Width Estimator

A prototype Python application that estimates road width (in meters) from geo-tagged 360° equirectangular video captured using an Insta360 camera.

## ⚠️ Disclaimer

**This prototype estimates road width using monocular vision and geometric assumptions. Results are indicative only and not suitable for engineering or legal surveys.**

## Features

- **Video Processing**: Extract frames from equirectangular MP4 videos at configurable FPS
- **360° Projection**: Convert equirectangular frames to front-facing perspective views
- **Road Segmentation**: Detect road surface and boundaries using computer vision
- **Width Estimation**: Estimate road width using monocular geometry (no LiDAR, no SfM)
- **Geo-Referencing**: Associate width estimates with GPS coordinates
- **Interactive UI**: Streamlit-based web interface
- **Export Options**: CSV and GeoJSON output formats

## Architecture

The application is modular with the following components:

- `video_processing.py`: Frame extraction from video files
- `projection.py`: Equirectangular to perspective projection conversion
- `segmentation.py`: Road segmentation using YOLOv8-seg (heuristic fallback)
- `width_estimation.py`: Monocular width estimation using geometric formulas
- `gps_utils.py`: GPS data handling (CSV/GPX)
- `app.py`: Streamlit user interface

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Note**: The application will automatically download YOLOv8-seg model on first use (~6MB)

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

### Using the Application

1. **Upload Video**: Upload your equirectangular 360° MP4 video file
2. **Configure Parameters**:
   - **Camera Height**: Height of camera above road surface (default: 1.5m)
   - **Horizontal FOV**: Field of view for perspective projection (default: 90°)
   - **Frame Extraction Rate**: Frames per second to extract (default: 2 FPS)
   - **Maximum Frames**: Limit processing (0 = all frames)
3. **Upload GPS Data** (Optional): Provide CSV or GPX file with GPS coordinates
4. **Start Processing**: Click "Start Processing" to begin analysis

### GPS Data Format

**CSV Format**:
```csv
timestamp,latitude,longitude,altitude
0.0,37.7749,-122.4194,10.0
1.0,37.7750,-122.4195,10.0
```

**GPX Format**: Standard GPX track format

## Outputs

### CSV File
Contains columns:
- `latitude`: GPS latitude
- `longitude`: GPS longitude
- `width`: Estimated road width in meters
- `confidence`: Confidence score (0.0-1.0)
- `frame`: Frame number
- `timestamp`: Video timestamp

### GeoJSON File
QGIS-compatible format with point features containing width and confidence properties.

### Statistics
- Minimum width
- Maximum width
- Average width
- Average confidence

### Interactive Map
Visualization of road width estimates on a map (requires GPS data).

## Technical Details

### Width Estimation Formula

The application uses monocular geometry to estimate road width:

```
width_m = 2 * camera_height * tan(angular_width_rad / 2)
```

Where:
- `angular_width_rad` = (pixel_width / image_width) * FOV_rad
- Assumes flat road surface and horizontal camera orientation

### Road Segmentation

The prototype uses a heuristic approach:
- Focuses on lower 60% of image (road region)
- Uses Canny edge detection and Hough line transform
- Detects left and right road boundaries
- Falls back to image boundaries if edges not detected

**Note**: For production use, train a custom YOLOv8-seg model on road segmentation datasets.

### Performance

- Frame sampling reduces processing time
- Caching of processed frames
- CPU-based processing (no GPU required, but GPU will speed up YOLOv8)

## Limitations

1. **Accuracy**: Target accuracy is ±2-3 meters (prototype-grade)
2. **Road Segmentation**: Uses heuristic method (not trained on road data)
3. **Geometric Assumptions**: Assumes flat road and horizontal camera
4. **No 3D Reconstruction**: Pure monocular vision approach
5. **GPS Required**: For geo-referencing, GPS data must be provided separately

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- Streamlit
- Pandas
- SciPy
- Folium (for mapping)

## File Structure

```
road_width_estimator/
├── app.py                 # Streamlit application
├── video_processing.py    # Video frame extraction
├── projection.py          # Equirectangular projection
├── segmentation.py        # Road segmentation
├── width_estimation.py    # Width calculation
├── gps_utils.py           # GPS data handling
├── map_utils.py           # Map visualization utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Future Improvements

- Custom trained road segmentation model
- Better edge detection algorithms
- Support for camera pitch/roll compensation
- Real-time processing mode
- Batch processing multiple videos
- Integration with video GPS metadata extraction

## License

This is a prototype/demo application. Use at your own risk.

## Support

For issues or questions, please refer to the code comments and documentation within each module.
