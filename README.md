# Hybrid Railway Track Detection

A Python-based computer vision and deep learning system designed to enhance railway safety by automatically detecting tracks, anomalies, and obstacles.

---

### Features

* **Hybrid Detection System**: Combines traditional computer vision techniques (Edge Detection, Hough Transform) with modern Deep Learning models for robust real-time tracking.
* **Weather & Lighting Robustness**: Accurately detects tracks under challenging environments such as rain, fog, low light, and shadows.
* **Obstacle & Fault Detection**: Real-time scanning to identify track anomalies, cracks, or unauthorized obstacles ahead.
* **Instant Alerts**: Provides visual overlays and high-priority logging for immediate automated decision-making.

---

### How It Works (System Workflow)

The system processes video feeds or image frames from front-facing train cameras through a multi-stage pipeline:

1. **Preprocessing**: Frames are resized, denoised (using Gaussian Blur), and converted to grayscale/HSV color spaces to handle varying lighting conditions.
2. **Track Segmentation (U-Net)**: A Deep Learning U-Net architecture segments the exact region of the railway tracks from the background.
3. **Line Fitting (Hough Transform)**: Traditional computer vision (Canny Edge Detection + Probabilistic Hough Line Transform) fits precise straight/curved lines on the segmented tracks for geometry alignment.
4. **Obstacle & Defect Detection (YOLOv8)**: A concurrent pipeline runs YOLOv8 to detect foreign objects (humans, animals, vehicles) and structural defects (cracks, missing clips) on the path.
5. **Decision & Alert Logic**: If an obstacle or track break is detected within the safe braking distance zone, the system triggers a high-priority visual/acoustic warning overlay.

---

### Tech Stack

* **Language**: Python
* **Core Libraries**: OpenCV, PyTorch, TensorFlow, NumPy, SciPy, Matplotlib
* **Models Used**: YOLOv8 (for obstacle detection), Custom CNN / U-Net (for track segmentation)

---

### Project Structure

```text
hybrid-railway-track-detection/
├── datasets/             # Training and validation images
├── models/               # Saved weights (.pt / .h5 files)
├── src/
│   ├── preprocessing.py  # Image cleaning & filtering
│   ├── track_detect.py   # Segmentation & Hough transform
│   └── obstacle_det.py   # YOLOv8 inference pipeline
├── main.py               # Main application entry point
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
