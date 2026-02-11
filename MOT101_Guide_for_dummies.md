# Multi-Object Tracking System - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding the Interface](#understanding-the-interface)
3. [Tracking Algorithms Explained](#tracking-algorithms-explained)
4. [Best Practices](#best-practices)
5. [Common Use Cases](#common-use-cases)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Advanced Tips](#advanced-tips)

---

## Getting Started

### First Time Setup

1. **Install Python 3.8+** from [python.org](https://python.org)

2. **Quick Start (Recommended)**
   ```bash
   # Linux/Mac
   chmod +x start.sh
   ./start.sh
   
   # Windows
   start.bat
   ```

3. **Manual Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run application
   streamlit run app.py
   ```

4. **Verify Installation**
   ```bash
   python test_installation.py
   ```

---

## Understanding the Interface

### Sidebar Controls

#### 1. Tracking Algorithm Selection
- **SORT**: Best for speed, simple scenarios
- **DeepSORT**: Best for handling occlusions
- **BYTETrack**: Best overall performance

#### 2. YOLOv8 Weights
- Path to your YOLOv8 model file
- Default: `yolov8n.pt` (automatically downloaded)
- Larger models = better accuracy but slower

#### 3. Detection Confidence
- Slider: 0.0 to 1.0
- Higher = fewer but more accurate detections
- Lower = more detections but possible false positives
- Recommended: 0.4-0.6

### Main Interface

#### Video Upload Section
Two options:
1. **Local Upload**: Click to browse and select video file
2. **YouTube**: Paste URL and click download

#### Processing
- Click "Start Processing" button
- Progress bar shows completion status
- Processing time depends on video length and settings

#### Results Display
Three synchronized videos:
1. **Original**: Unmodified input
2. **Tracking**: Colored boxes with IDs and trails
3. **Depth**: Real-time depth estimation

---

## Tracking Algorithms Explained

### SORT (Simple Online and Realtime Tracking)

**How it works:**
- Uses Kalman filter to predict object positions
- Matches detections using IoU (Intersection over Union)
- Very fast, minimal computational overhead

**Best for:**
- Simple surveillance scenarios
- When speed is critical
- Objects move predictably
- Minimal occlusions

**Parameters:**
```python
max_age = 1         # Frames to keep track without detection
min_hits = 3        # Hits needed to confirm track
iou_threshold = 0.3 # Matching threshold
```

**Pros:**
- ⚡ Very fast
- 💾 Low memory usage
- 🎯 Good for simple cases

**Cons:**
- ❌ Poor with occlusions
- ❌ Can't re-identify objects
- ❌ Struggles with crowded scenes

---

### DeepSORT (Deep SORT)

**How it works:**
- Extends SORT with deep learning features
- Extracts appearance features from crops
- Uses both motion and appearance for matching
- Better object re-identification

**Best for:**
- Scenarios with occlusions
- Need for re-identification
- Multiple similar-looking objects
- Medium-complexity scenes

**Parameters:**
```python
max_age = 30               # More forgiving with lost tracks
min_hits = 3
iou_threshold = 0.3
max_cosine_distance = 0.2  # Appearance matching threshold
```

**Pros:**
- ✓ Handles occlusions well
- ✓ Can re-identify objects
- ✓ Better for complex scenes

**Cons:**
- ⏱️ Slower than SORT
- 💾 More memory usage
- 🔧 More complex

---

### BYTETrack

**How it works:**
- Associates EVERY detection box, including low-confidence ones
- Two-stage association process
- Robust to detection failures
- State-of-the-art performance

**Best for:**
- Crowded scenes
- Fast-moving objects
- When you need best accuracy
- Complex tracking scenarios

**Parameters:**
```python
track_thresh = 0.5    # High confidence first association
track_buffer = 30     # Buffer for lost tracks
match_thresh = 0.8    # Matching threshold
```

**Pros:**
- 🏆 Best overall accuracy
- ✓ Excellent with occlusions
- ✓ Handles crowded scenes
- ✓ Robust to detection errors

**Cons:**
- ⏱️ Slower than SORT
- 💾 Higher memory usage

---

## Best Practices

### Choosing the Right Model

#### For Real-time Applications:
```
YOLOv8n + SORT + LiteDepth
- Speed: ⚡⚡⚡⚡⚡
- Accuracy: ⭐⭐⭐
- Memory: 2-3 GB
```

#### For Balanced Performance:
```
YOLOv8s + DeepSORT + MiDaS (Hybrid)
- Speed: ⚡⚡⚡
- Accuracy: ⭐⭐⭐⭐
- Memory: 4-5 GB
```

#### For Maximum Accuracy:
```
YOLOv8l + BYTETrack + MiDaS (Large)
- Speed: ⚡⚡
- Accuracy: ⭐⭐⭐⭐⭐
- Memory: 8-10 GB
```

### Optimizing Detection Confidence

| Scenario | Confidence | Reasoning |
|----------|-----------|-----------|
| Crowded scenes | 0.3-0.4 | Catch more objects |
| Simple scenes | 0.5-0.7 | Reduce false positives |
| High accuracy needed | 0.6-0.8 | Only confident detections |
| Distant objects | 0.3-0.5 | Objects are harder to detect |

### Video Quality Tips

1. **Resolution**: 1080p recommended
2. **Frame rate**: 30 FPS ideal
3. **Lighting**: Good lighting improves detection
4. **Camera angle**: Front-facing better than top-down
5. **Stability**: Stable camera helps tracking

---

## Common Use Cases

### 1. People Counting
```
Model: YOLOv8m
Tracker: BYTETrack
Confidence: 0.5
Classes: [0]  # Person only

Tips:
- Use entrance/exit zones
- Count track IDs that cross line
- Handle bi-directional flow
```

### 2. Traffic Monitoring
```
Model: YOLOv8s
Tracker: SORT
Confidence: 0.4
Classes: [2, 3, 5, 7]  # car, motorcycle, bus, truck

Tips:
- Set up virtual lanes
- Track speed using distance/time
- Classify by vehicle type
```

### 3. Sports Analysis
```
Model: YOLOv8l
Tracker: DeepSORT
Confidence: 0.6
Classes: [0, 32]  # person, sports ball

Tips:
- High confidence for accuracy
- DeepSORT handles occlusions
- Track player movements
```

### 4. Retail Analytics
```
Model: YOLOv8m
Tracker: BYTETrack
Confidence: 0.5
Classes: [0]  # person

Tips:
- Track customer paths
- Dwell time analysis
- Heat map generation
```

### 5. Wildlife Monitoring
```
Model: YOLOv8x (fine-tuned)
Tracker: DeepSORT
Confidence: 0.4
Classes: Custom trained

Tips:
- Lower confidence for distant animals
- DeepSORT for re-identification
- Handle partial occlusions
```

---

## Troubleshooting Guide

### Problem: Slow Processing

**Solutions:**
1. Use smaller YOLOv8 model (yolov8n.pt)
2. Lower video resolution
3. Use SORT tracker
4. Use LiteDepthEstimator
5. Enable GPU if available
6. Process fewer frames (frame skip)

**Code:**
```python
# In config.ini
[video]
frame_skip = 2  # Process every other frame
output_width = 640  # Resize to 640px width
```

### Problem: Poor Tracking

**Solutions:**
1. Lower confidence threshold
2. Try different tracker
3. Use larger YOLOv8 model
4. Check video quality
5. Adjust lighting

**Recommended settings:**
```ini
[detection]
confidence_threshold = 0.4  # Lower for more detections

[bytetrack]
track_thresh = 0.4  # Lower for more associations
```

### Problem: Too Many False Positives

**Solutions:**
1. Increase confidence threshold
2. Use larger model
3. Filter by classes
4. Post-process tracks

**Code:**
```python
# In app.py, filter detections
detections = [d for d in detections if d[5] in [0, 2, 3]]  # person, car, motorcycle only
```

### Problem: Lost Tracks

**Solutions:**
1. Increase max_age
2. Use DeepSORT or BYTETrack
3. Lower track_thresh (BYTETrack)

**Settings:**
```ini
[deepsort]
max_age = 50  # Keep tracks longer

[bytetrack]
track_buffer = 50
track_thresh = 0.4
```

### Problem: GPU Out of Memory

**Solutions:**
1. Use smaller model
2. Reduce batch size
3. Lower resolution
4. Close other applications

**Code:**
```python
# Process in smaller batches
results = model(frame, conf=conf, imgsz=640)  # Smaller image size
```

---

## Advanced Tips

### Custom Object Detection

Train custom YOLOv8 model:
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
model.train(data='custom.yaml', epochs=100)

# Use in app
# Set weights_path to 'runs/detect/train/weights/best.pt'
```

### Extract Track Data

Save track information:
```python
# Add to app.py
track_data = []
for track in tracked_objects:
    track_data.append({
        'frame': frame_idx,
        'id': int(track[4]),
        'bbox': track[:4].tolist(),
        'center': [(track[0] + track[2])/2, (track[1] + track[3])/2]
    })

# Save to CSV
import pandas as pd
df = pd.DataFrame(track_data)
df.to_csv('tracks.csv', index=False)
```

### Speed Estimation

Calculate object speed:
```python
# Track positions over time
speeds = {}
for track_id, positions in track_history.items():
    if len(positions) >= 2:
        # Distance in pixels
        dist = np.linalg.norm(
            np.array(positions[-1]) - np.array(positions[-2])
        )
        # Speed (pixels per frame)
        speeds[track_id] = dist
        
# Convert to real units if camera calibration known
# speed_mph = pixels_per_frame * fps * scale_factor
```

### Region of Interest (ROI)

Process only specific areas:
```python
# Define ROI polygon
roi_points = np.array([[100, 100], [500, 100], [500, 400], [100, 400]])

# Check if detection center is in ROI
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Filter detections
filtered_dets = []
for det in detections:
    center = [(det[0] + det[2])/2, (det[1] + det[3])/2]
    if point_in_polygon(center, roi_points):
        filtered_dets.append(det)
```

### Multi-Camera Setup

Track across multiple cameras:
```python
# Initialize tracker for each camera
trackers = {
    'cam1': ByteTracker(),
    'cam2': ByteTracker(),
    'cam3': ByteTracker()
}

# Process each camera
for cam_id, tracker in trackers.items():
    frame = get_frame(cam_id)
    results = tracker.update(detections)
    
# Cross-camera re-identification
# Use DeepSORT features to match tracks across cameras
```

---

## Performance Optimization Checklist

- [ ] Use appropriate model size for your hardware
- [ ] Enable GPU acceleration
- [ ] Choose tracker based on requirements
- [ ] Adjust confidence threshold
- [ ] Set appropriate frame skip
- [ ] Use ROI if applicable
- [ ] Batch process when possible
- [ ] Monitor memory usage
- [ ] Profile bottlenecks
- [ ] Consider video preprocessing

---

## Tips for Best Results

1. **Good lighting** - Improves detection significantly
2. **Stable camera** - Reduces false tracks
3. **Appropriate angle** - 30-60° from horizontal
4. **Consistent conditions** - Same lighting throughout
5. **Clear subjects** - Avoid heavy occlusions
6. **Quality video** - Higher resolution = better detection
7. **Proper settings** - Tune confidence and thresholds
8. **Test different trackers** - Each excels in different scenarios

---

## Getting Help

- Check `README.md` for installation issues
- Run `python test_installation.py` to verify setup
- Review `config.ini` for available settings
- Check logs in terminal for error messages

---

**Happy Tracking! 🎯**