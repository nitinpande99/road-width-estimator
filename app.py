"""
Streamlit Application
Main UI for road width estimation from 360° video.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
from pathlib import Path
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
from video_processing import extract_frames, get_video_info
from projection import extract_front_view
from segmentation import RoadSegmenter
from width_estimation import (
    estimate_road_width,
    smooth_width_estimates,
    calculate_confidence_score
)
from gps_utils import (
    read_gps_csv,
    read_gps_gpx,
    interpolate_gps,
    aggregate_by_distance
)
from map_utils import (
    create_buffered_corridor,
    get_width_color,
    get_confidence_color,
    get_confidence_style,
    create_corridor_geojson
)

# Page configuration
st.set_page_config(
    page_title="Road Width Estimator",
    page_icon="🛣️",
    layout="wide"
)

# Disclaimer
st.sidebar.markdown("""
### ⚠️ Disclaimer
**This prototype estimates road width using monocular vision and geometric assumptions. 
Results are indicative only and not suitable for engineering or legal surveys.**
""")

# Title
st.title("🛣️ Road Width Estimator")
st.markdown("Estimate road width from 360° equirectangular video using computer vision")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'segmenter' not in st.session_state:
    st.session_state.segmenter = None

# Sidebar inputs
st.sidebar.header("Configuration")

# Video upload
uploaded_video = st.sidebar.file_uploader(
    "Upload Video (MP4)",
    type=['mp4'],
    help="Upload equirectangular 360° video file"
)

# GPS data upload
uploaded_gps = st.sidebar.file_uploader(
    "Upload GPS Data (Optional)",
    type=['csv', 'gpx'],
    help="Upload GPS track file (CSV or GPX format)"
)

# Parameters
camera_height = st.sidebar.number_input(
    "Camera Height (meters)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="Height of camera above road surface"
)

fov_degrees = st.sidebar.number_input(
    "Horizontal FOV (degrees)",
    min_value=30,
    max_value=180,
    value=90,
    step=5,
    help="Field of view for perspective projection"
)

fps = st.sidebar.number_input(
    "Frame Extraction Rate (FPS)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="Frames per second to extract from video"
)

max_frames = st.sidebar.number_input(
    "Maximum Frames (0 = all)",
    min_value=0,
    value=0,
    help="Limit number of frames to process (0 for all)"
)

# Processing button
process_button = st.sidebar.button("🚀 Start Processing", type="primary")

# Main content area
if uploaded_video is None:
    st.info("👆 Please upload a video file to get started")
    st.markdown("""
    ### How to Use
    
    1. **Upload Video**: Upload your equirectangular 360° MP4 video file
    2. **Configure Parameters**: 
       - Set camera height (default: 1.5m)
       - Set horizontal FOV (default: 90°)
       - Set frame extraction rate (default: 2 FPS)
    3. **Upload GPS Data** (Optional): Provide CSV or GPX file with GPS coordinates
    4. **Start Processing**: Click the button to begin analysis
    
    ### Outputs
    
    - **CSV File**: Road width estimates with GPS coordinates
    - **GeoJSON File**: QGIS-compatible format for mapping
    - **Interactive Map**: Visualize results on a map
    - **Statistics**: Min/max/average road width
    """)
else:
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name
    
    # Store temp file path for cleanup
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    st.session_state.temp_files.append(video_path)
    
    # Get video info
    try:
        video_info = get_video_info(video_path)
        st.sidebar.success(f"✅ Video loaded: {video_info['width']}x{video_info['height']}, {video_info['duration']:.1f}s")
    except Exception as e:
        st.error(f"Error loading video: {e}")
        video_path = None
    
    # Load GPS data if provided
    gps_data = []
    gps_path = None
    if uploaded_gps is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_gps.name).suffix) as tmp_gps:
            tmp_gps.write(uploaded_gps.read())
            gps_path = tmp_gps.name
            if 'temp_files' not in st.session_state:
                st.session_state.temp_files = []
            st.session_state.temp_files.append(gps_path)
        
        try:
            if uploaded_gps.name.endswith('.csv'):
                gps_data = read_gps_csv(gps_path)
            elif uploaded_gps.name.endswith('.gpx'):
                gps_data = read_gps_gpx(gps_path)
            
            if gps_data:
                st.sidebar.success(f"✅ GPS data loaded: {len(gps_data)} points")
        except Exception as e:
            st.warning(f"Could not load GPS data: {e}")
    
    # Processing
    if process_button and video_path:
        with st.spinner("Processing video... This may take a while."):
            try:
                # Initialize segmenter (lazy loading)
                if st.session_state.segmenter is None:
                    st.session_state.segmenter = RoadSegmenter()
                
                segmenter = st.session_state.segmenter
                
                # Process frames
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_gen = extract_frames(
                    video_path,
                    fps=fps,
                    max_frames=max_frames if max_frames > 0 else None
                )
                
                frames_list = list(frame_gen)
                total_frames = len(frames_list)
                
                timestamps = []
                width_estimates = []
                previous_width = None
                
                for idx, (frame_num, frame, timestamp) in enumerate(frames_list):
                    # Update progress
                    progress = (idx + 1) / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {idx + 1}/{total_frames}")
                    
                    # Convert equirectangular to perspective
                    perspective = extract_front_view(
                        frame,
                        fov_degrees=fov_degrees,
                        output_width=640,
                        output_height=480
                    )
                    
                    # Segment road (now returns edge_quality dict)
                    road_mask, left_edge, right_edge, edge_quality = segmenter.segment(perspective)
                    
                    # Estimate width
                    width_m = estimate_road_width(
                        left_edge or 0,
                        right_edge or 640,
                        perspective.shape[1],
                        fov_degrees,
                        camera_height
                    )
                    
                    # Calculate enhanced confidence with temporal consistency
                    confidence, confidence_breakdown = calculate_confidence_score(
                        left_edge,
                        right_edge,
                        perspective.shape[1],
                        width_m,
                        edge_quality=edge_quality,
                        previous_width=previous_width
                    )
                    
                    timestamps.append(timestamp)
                    width_estimates.append(width_m)
                    previous_width = width_m  # Update for next frame
                    
                    # Store result with confidence breakdown
                    results.append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'width': width_m,
                        'confidence': confidence,
                        'confidence_breakdown': confidence_breakdown,
                        'edge_quality': edge_quality,
                        'lat': 0.0,
                        'lon': 0.0
                    })
                
                # Smooth width estimates
                smoothed_widths = smooth_width_estimates(width_estimates)
                for i, width in enumerate(smoothed_widths):
                    results[i]['width'] = width
                
                # Add GPS data if available
                if gps_data:
                    gps_coords = interpolate_gps(gps_data, timestamps)
                    for i, (lat, lon) in enumerate(gps_coords):
                        results[i]['lat'] = lat
                        results[i]['lon'] = lon
                
                # Aggregate by distance if GPS available and valid
                if gps_data and any(r['lat'] != 0.0 or r['lon'] != 0.0 for r in results):
                    results = aggregate_by_distance(results, distance_threshold_m=10.0)
                
                st.session_state.results = results
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Processing complete! Processed {len(results)} frames.")
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.results is not None and len(st.session_state.results) > 0:
        results = st.session_state.results
        
        # Statistics
        st.header("📊 Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        widths = [r['width'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        with col1:
            st.metric("Min Width", f"{min(widths):.2f} m")
        with col2:
            st.metric("Max Width", f"{max(widths):.2f} m")
        with col3:
            st.metric("Avg Width", f"{np.mean(widths):.2f} m")
        with col4:
            st.metric("Avg Confidence", f"{np.mean(confidences):.3f}")
        with col5:
            st.metric("Min Confidence", f"{min(confidences):.3f}")
        
        # Confidence Visualization
        st.header("📈 Confidence Analysis")
        
        # Confidence over time chart
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Confidence Score Over Time', 'Confidence Breakdown Components'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Plot 1: Confidence over time
        frames_list = [r.get('frame', i) for i, r in enumerate(results)]
        fig.add_trace(
            go.Scatter(
                x=frames_list,
                y=confidences,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Plot 2: Confidence breakdown components
        if results and 'confidence_breakdown' in results[0]:
            breakdown_keys = ['edge_detection', 'edge_quality', 'symmetry', 
                             'boundary_proximity', 'width_realism', 'temporal_consistency']
            
            # Calculate average for each component
            avg_breakdown = {}
            for key in breakdown_keys:
                values = [r.get('confidence_breakdown', {}).get(key, 0.0) for r in results if 'confidence_breakdown' in r]
                avg_breakdown[key] = np.mean(values) if values else 0.0
            
            # Create bar chart
            component_names = ['Edge Detection', 'Edge Quality', 'Symmetry', 
                              'Boundary', 'Width Realism', 'Temporal']
            component_values = [avg_breakdown.get(k, 0.0) for k in breakdown_keys]
            
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            fig.add_trace(
                go.Bar(
                    x=component_names,
                    y=component_values,
                    name='Component Scores',
                    marker_color=colors,
                    text=[f'{v:.3f}' for v in component_values],
                    textposition='outside'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Frame Number", row=1, col=1)
        fig.update_yaxes(title_text="Confidence Score", range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="Component", row=2, col=1)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=2, col=1)
        fig.update_layout(height=700, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confidence Distribution")
            confidence_hist = go.Figure(data=[
                go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    marker_color='#1f77b4',
                    opacity=0.7
                )
            ])
            confidence_hist.update_layout(
                title="Distribution of Confidence Scores",
                xaxis_title="Confidence Score",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(confidence_hist, use_container_width=True)
        
        with col2:
            st.subheader("Confidence vs Width")
            scatter_fig = go.Figure(data=[
                go.Scatter(
                    x=widths,
                    y=confidences,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=confidences,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Confidence")
                    ),
                    text=[f"Frame {r.get('frame', i)}" for i, r in enumerate(results)],
                    hovertemplate='Width: %{x:.2f}m<br>Confidence: %{y:.3f}<br>%{text}<extra></extra>'
                )
            ])
            scatter_fig.update_layout(
                title="Width vs Confidence",
                xaxis_title="Width (m)",
                yaxis_title="Confidence Score",
                height=300
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Results table
        st.header("📋 Results")
        
        # Prepare dataframe (flatten confidence breakdown for display)
        display_results = []
        for r in results:
            display_row = {
                'frame': r.get('frame', 0),
                'timestamp': r.get('timestamp', 0),
                'width (m)': r.get('width', 0),
                'confidence': r.get('confidence', 0),
                'lat': r.get('lat', 0),
                'lon': r.get('lon', 0)
            }
            # Add breakdown components if available
            if 'confidence_breakdown' in r:
                breakdown = r['confidence_breakdown']
                display_row['edge_quality'] = breakdown.get('edge_quality', 0)
                display_row['symmetry'] = breakdown.get('symmetry', 0)
                display_row['temporal'] = breakdown.get('temporal_consistency', 0)
            display_results.append(display_row)
        
        df = pd.DataFrame(display_results)
        
        # Format confidence columns
        if not df.empty:
            for col in ['confidence', 'edge_quality', 'symmetry', 'temporal']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(df, use_container_width=True)
        
        # Map visualization
        if any(r['lat'] != 0.0 or r['lon'] != 0.0 for r in results):
            st.header("🗺️ Map Visualization")
            
            # Color mode toggle
            col1, col2 = st.columns([1, 3])
            with col1:
                color_mode = st.radio(
                    "Color by:",
                    ["Road Width", "Confidence Score"],
                    horizontal=True,
                    help="Choose whether to color corridors by road width or confidence score"
                )
            
            # Create map
            valid_results = [r for r in results if r['lat'] != 0.0 and r['lon'] != 0.0]
            if valid_results and len(valid_results) >= 2:
                # Calculate center
                center_lat = np.mean([r['lat'] for r in valid_results])
                center_lon = np.mean([r['lon'] for r in valid_results])
                
                # Create Folium map
                m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
                
                # Extract points and widths for corridor creation
                points = [(r['lat'], r['lon']) for r in valid_results]
                widths = [r.get('width', 0.0) for r in valid_results]
                confidences = [r.get('confidence', 0.5) for r in valid_results]
                min_width = min(widths)
                max_width = max(widths)
                min_confidence = min(confidences)
                max_confidence = max(confidences)
                
                # Create buffered corridor polygons
                polygons = create_buffered_corridor(points, widths)
                
                # Add corridor polygons to map
                for i, polygon in enumerate(polygons):
                    if i < len(valid_results) - 1:
                        # Get average width and confidence for this segment
                        avg_width = (widths[i] + widths[i + 1]) / 2
                        avg_confidence = (valid_results[i].get('confidence', 0.5) + 
                                         valid_results[i + 1].get('confidence', 0.5)) / 2
                        
                        # Get color based on selected mode
                        if color_mode == "Road Width":
                            color = get_width_color(avg_width, min_width, max_width)
                        else:  # Confidence Score
                            color = get_confidence_color(avg_confidence)
                        
                        # Get confidence-based styling (fading/hatching for low confidence)
                        confidence_style = get_confidence_style(avg_confidence)
                        
                        # Convert polygon to list of (lat, lon) tuples for Folium
                        # Shapely polygon coordinates are (lon, lat), Folium expects (lat, lon)
                        polygon_coords = [[coord[1], coord[0]] for coord in polygon.exterior.coords]
                        
                        # Create popup text
                        popup_text = f"Width: {avg_width:.2f}m<br>Confidence: {avg_confidence:.3f}"
                        if avg_confidence < 0.5:
                            popup_text += " <span style='color:red;'>(Low Confidence)</span>"
                        if 'confidence_breakdown' in valid_results[i]:
                            breakdown = valid_results[i]['confidence_breakdown']
                            popup_text += f"<br><br>Breakdown:<br>"
                            popup_text += f"Edge Quality: {breakdown.get('edge_quality', 0):.2f}<br>"
                            popup_text += f"Symmetry: {breakdown.get('symmetry', 0):.2f}<br>"
                            popup_text += f"Temporal: {breakdown.get('temporal_consistency', 0):.2f}"
                        
                        # Add polygon to map with confidence-based opacity
                        folium.Polygon(
                            locations=polygon_coords,
                            popup=folium.Popup(popup_text, max_width=200),
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=confidence_style['fillOpacity'],
                            weight=2,
                            opacity=confidence_style['opacity']
                        ).add_to(m)
                        
                        # For low confidence, add a dashed border overlay using PolyLine
                        if confidence_style['dashArray']:
                            folium.PolyLine(
                                locations=polygon_coords,
                                color=color,
                                weight=2,
                                opacity=confidence_style['opacity'],
                                dashArray=confidence_style['dashArray'],
                                fill=False
                            ).add_to(m)
                
                # Add legend based on color mode
                if color_mode == "Road Width":
                    legend_html = f'''
                    <div style="position: fixed; 
                                bottom: 50px; right: 50px; width: 220px; height: 160px; 
                                background-color: white; border:2px solid grey; z-index:9999; 
                                font-size:14px; padding: 10px">
                    <h4 style="margin-top:0">Road Width (m)</h4>
                    <p><span style="color:#0000ff">●</span> Narrow ({min_width:.1f}m)</p>
                    <p><span style="color:#00ff00">●</span> Medium</p>
                    <p><span style="color:#ff0000">●</span> Wide ({max_width:.1f}m)</p>
                    <hr style="margin: 8px 0;">
                    <p style="font-size:12px; margin:0;"><span style="opacity:0.3;">●</span> Faded/Dashed = Low Confidence</p>
                    </div>
                    '''
                else:  # Confidence Score
                    legend_html = f'''
                    <div style="position: fixed; 
                                bottom: 50px; right: 50px; width: 220px; height: 160px; 
                                background-color: white; border:2px solid grey; z-index:9999; 
                                font-size:14px; padding: 10px">
                    <h4 style="margin-top:0">Confidence Score</h4>
                    <p><span style="color:#ff0000">●</span> Low ({min_confidence:.2f})</p>
                    <p><span style="color:#ffaa00">●</span> Medium</p>
                    <p><span style="color:#00ff00">●</span> High ({max_confidence:.2f})</p>
                    <hr style="margin: 8px 0;">
                    <p style="font-size:12px; margin:0;"><span style="opacity:0.3;">●</span> Faded/Dashed = Low Confidence</p>
                    </div>
                    '''
                m.get_root().html.add_child(folium.Element(legend_html))
                
                st_folium(m, width=700, height=500)
                
                # Add color scale explanation
                if color_mode == "Road Width":
                    st.info(f"💡 **Color Scale**: Road corridors are color-coded by width. "
                           f"Blue indicates narrow roads ({min_width:.1f}m), "
                           f"green indicates medium-width roads, "
                           f"and red indicates wide roads ({max_width:.1f}m). "
                           f"**Visual Cues**: Faded polygons with dashed borders indicate low confidence estimates (<0.5). "
                           f"Each polygon represents a buffered corridor segment.")
                else:  # Confidence Score
                    st.info(f"💡 **Color Scale**: Road corridors are color-coded by confidence score. "
                           f"Red indicates low confidence ({min_confidence:.2f}), "
                           f"yellow indicates medium confidence, "
                           f"and green indicates high confidence ({max_confidence:.2f}). "
                           f"**Visual Cues**: Faded polygons with dashed borders indicate low confidence estimates (<0.5). "
                           f"Each polygon represents a buffered corridor segment.")
            elif valid_results:
                # Fallback to point markers if only one point
                center_lat = valid_results[0]['lat']
                center_lon = valid_results[0]['lon']
                m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
                
                r = valid_results[0]
                color = get_width_color(r.get('width', 0.0), r.get('width', 0.0), r.get('width', 1.0))
                
                folium.CircleMarker(
                    location=[r['lat'], r['lon']],
                    radius=10,
                    popup=f"Width: {r['width']:.2f}m<br>Confidence: {r['confidence']:.3f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
                
                st_folium(m, width=700, height=500)
        
        # Export options
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="road_width_estimates.csv",
                mime="text/csv"
            )
        
        with col2:
            # GeoJSON export (with corridor polygons)
            if any(r['lat'] != 0.0 or r['lon'] != 0.0 for r in results):
                geojson = create_corridor_geojson(results)
                geojson_str = json.dumps(geojson, indent=2)
                st.download_button(
                    label="📥 Download GeoJSON (Corridors)",
                    data=geojson_str,
                    file_name="road_width_corridors.geojson",
                    mime="application/json"
                )
            else:
                st.info("GPS data required for GeoJSON export")
