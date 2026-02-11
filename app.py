import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import yt_dlp
from ultralytics import YOLO
import torch
from collections import defaultdict

# Import tracking algorithms
from trackers.sort import SORTTracker
from trackers.deepsort import DeepSORTTracker
from trackers.bytetrack import ByteTracker

# Import depth estimation
from depth_estimator import DepthEstimator

# Page configuration
st.set_page_config(
    page_title="Multi-Object Tracking System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .upload-section {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

def download_youtube_video(url):
    """Download YouTube video and return local path"""
    try:
        with st.spinner("Downloading YouTube video..."):
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': '/tmp/youtube_video.mp4',
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return '/tmp/youtube_video.mp4'
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

def get_tracker(tracker_name, frame_size):
    """Initialize the selected tracker"""
    if tracker_name == "SORT":
        return SORTTracker()
    elif tracker_name == "DeepSORT":
        return DeepSORTTracker(frame_size)
    elif tracker_name == "BYTETrack":
        return ByteTracker()
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")

def generate_colors(num_colors):
    """Generate distinct colors for different track IDs"""
    np.random.seed(42)
    colors = {}
    for i in range(num_colors):
        colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
    return colors

def process_video(video_path, weights_path, tracker_name, confidence_threshold, use_uav=False, use_lite=False):
    """Process video with detection, tracking, and depth estimation"""
    
    # Load YOLOv8 model
    model = YOLO(weights_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize tracker
    tracker = get_tracker(tracker_name, (width, height))
    
    # Initialize depth estimator based on mode
    if use_uav:
        from depth_estimator import UAVDepthEstimator
        depth_estimator = UAVDepthEstimator()
    elif use_lite:
        from depth_estimator import LiteDepthEstimator
        depth_estimator = LiteDepthEstimator()
    else:
        from depth_estimator import DepthEstimator
        depth_estimator = DepthEstimator()
    
    # Initialize video writers
    output_dir = '/tmp/mot_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Use H264 codec for better Streamlit compatibility
    # Try different codecs in order of preference
    codecs_to_try = [
        ('avc1', '.mp4'),  # H264 - best for web
        ('mp4v', '.mp4'),  # MPEG-4
        ('X264', '.mp4'),  # Alternative H264
    ]
    
    codec_found = False
    for codec, ext in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(
                f'{output_dir}/test{ext}', fourcc, fps, (width, height))
            if test_writer.isOpened():
                test_writer.release()
                os.remove(f'{output_dir}/test{ext}')
                codec_found = True
                break
        except:
            continue
    
    if not codec_found:
        codec = 'mp4v'  # Fallback
        ext = '.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    original_writer = cv2.VideoWriter(
        f'{output_dir}/original{ext}', fourcc, fps, (width, height))
    tracking_writer = cv2.VideoWriter(
        f'{output_dir}/tracking{ext}', fourcc, fps, (width, height))
    depth_writer = cv2.VideoWriter(
        f'{output_dir}/depth{ext}', fourcc, fps, (width, height))
    
    # Track history for polylines
    track_history = defaultdict(lambda: [])
    colors = generate_colors(1000)  # Pre-generate colors for up to 1000 objects
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Original frame
        original_frame = frame.copy()
        
        # YOLOv8 Detection
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                detections.append([x1, y1, x2, y2, conf, cls])
        
        # Update tracker
        if tracker_name == "DeepSORT":
            tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 6)), frame)
        else:
            tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 6)))
        
        # Create tracking visualization
        tracking_frame = frame.copy()
        
        for track in tracked_objects:
            if len(track) >= 5:
                x1, y1, x2, y2, track_id = track[:5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)
                
                # Get color for this track ID
                color = colors.get(track_id % 1000, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID label
                label = f"ID: {track_id}"
                cv2.putText(tracking_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Update track history
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                track_history[track_id].append(center)
                
                # Keep only last 30 points
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                
                # Draw polyline
                if len(track_history[track_id]) > 1:
                    points = np.array(track_history[track_id], dtype=np.int32)
                    cv2.polylines(tracking_frame, [points], False, color, 2)
        
        # Add tracker info
        cv2.putText(tracking_frame, f"Tracker: {tracker_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(tracking_frame, f"Objects: {len(tracked_objects)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Depth estimation
        depth_frame = depth_estimator.estimate(frame)
        
        # Write frames
        original_writer.write(original_frame)
        tracking_writer.write(tracking_frame)
        depth_writer.write(depth_frame)
        
        # Update progress
        frame_idx += 1
        progress = int((frame_idx / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx}/{total_frames}")
    
    # Release resources
    cap.release()
    original_writer.release()
    tracking_writer.release()
    depth_writer.release()
    
    progress_bar.empty()
    status_text.empty()
    
    return {
        'original': f'{output_dir}/original{ext}',
        'tracking': f'{output_dir}/tracking{ext}',
        'depth': f'{output_dir}/depth{ext}'
    }

def main():
    # Header
    st.title("Multi-Object Tracking System")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Tracker selection
        st.subheader("Tracking Algorithm")
        tracker_name = st.selectbox(
            "Select Tracker",
            ["SORT", "DeepSORT", "BYTETrack"],
            help="Choose the tracking algorithm to use"
        )
        
        st.markdown("---")
        
        # Model weights
        st.subheader("YOLOv8 Weights")
        
        # Get available models from models directory
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            st.info(f" Models directory created at: {models_dir}")
        
        # List all .pt files in models directory
        available_models = []
        if os.path.exists(models_dir):
            available_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        
        if available_models:
            weights_file = st.selectbox(
                "Select Model",
                options=available_models,
                help="Choose YOLOv8 weights from models directory"
            )
            weights_path = os.path.join(models_dir, weights_file)
            
            # Show model info
            model_size = os.path.getsize(weights_path) / (1024 * 1024)
            st.caption(f" Model size: {model_size:.1f} MB")
        else:
            st.warning(" No models found in 'models' directory")
            st.info("""
            **Please add YOLOv8 weights to the 'models' directory:**
            
            1. Download models from Ultralytics:
               - yolov8n.pt (6 MB) - Fastest
               - yolov8s.pt (22 MB) - Small
               - yolov8m.pt (50 MB) - Medium
               - yolov8l.pt (84 MB) - Large
               - yolov8x.pt (131 MB) - Extra Large
            
            2. Place them in the 'models' folder
            
            3. Refresh the page
            """)
            weights_path = None
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        st.markdown("---")
        
        # Depth Estimation Settings
        st.subheader("Depth Estimation")
        
        depth_mode = st.selectbox(
            "Depth Mode",
            ["MiDaS (Deep Learning)", "UAV/Aerial Optimized", "Lite (Fast)"],
            index=1,  # Default to UAV mode
            help="Choose depth estimation method"
        )
        
        if depth_mode == "MiDaS (Deep Learning)":
            use_midas = True
            use_uav = False
            use_lite = False
        elif depth_mode == "UAV/Aerial Optimized":
            use_midas = False
            use_uav = True
            use_lite = False
        else:  # Lite
            use_midas = False
            use_uav = False
            use_lite = True
        
        st.markdown("---")
        
        # Information
        st.subheader("ℹAbout")
        st.info("""
        **Features:**
        - YOLOv8 Object Detection
        - SORT/DeepSORT/BYTETrack
        - Real-time Depth Estimation
        - Track Visualization
        - Multi-view Output
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a local video file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.video_path = tfile.name
            st.success(f"Video uploaded: {uploaded_file.name}")
    
    with col2:
        st.subheader("YouTube Link")
        youtube_url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube video URL"
        )
        
        if st.button("Download YouTube Video"):
            if youtube_url:
                video_path = download_youtube_video(youtube_url)
                if video_path:
                    st.session_state.video_path = video_path
                    st.success("YouTube video downloaded successfully!")
            else:
                st.warning("Please enter a YouTube URL")
    
    st.markdown("---")
    
    # Process button
    if st.session_state.video_path:
        st.info(f"Current video: {st.session_state.video_path}")
        
        if st.button("🚀 Start Processing", type="primary"):
            if weights_path is None:
                st.error(" No model selected. Please add YOLOv8 weights to the 'models' directory.")
            elif not os.path.exists(weights_path):
                st.error(f" Weights file not found: {weights_path}")
            else:
                try:
                    # Process video
                    output_paths = process_video(
                        st.session_state.video_path,
                        weights_path,
                        tracker_name,
                        confidence_threshold,
                        use_uav=use_uav,
                        use_lite=use_lite
                    )
                    
                    st.success("Processing complete!")
                    st.session_state.processed = True
                    st.session_state.output_paths = output_paths
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
    
    # Display results
    if st.session_state.get('processed', False) and st.session_state.get('output_paths'):
        st.markdown("---")
        st.header("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Video")
            st.video(st.session_state.output_paths['original'])
        
        with col2:
            st.subheader(f"{tracker_name} Tracking")
            st.video(st.session_state.output_paths['tracking'])
        
        with col3:
            st.subheader("Depth Estimation")
            st.video(st.session_state.output_paths['depth'])
        
        # Download buttons
        st.markdown("---")
        st.subheader("Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with open(st.session_state.output_paths['original'], 'rb') as f:
                st.download_button(
                    "Download Original",
                    f,
                    file_name="original.mp4",
                    mime="video/mp4"
                )
        
        with col2:
            with open(st.session_state.output_paths['tracking'], 'rb') as f:
                st.download_button(
                    "Download Tracking",
                    f,
                    file_name=f"tracking_{tracker_name.lower()}.mp4",
                    mime="video/mp4"
                )
        
        with col3:
            with open(st.session_state.output_paths['depth'], 'rb') as f:
                st.download_button(
                    "Download Depth",
                    f,
                    file_name="depth_estimation.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()