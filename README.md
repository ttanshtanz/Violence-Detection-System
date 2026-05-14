# Violence Detection System

A real-time violence detection system using **YOLOv8**, **CNN-LSTM**, and **OpenCV** for intelligent video surveillance.

The system detects violent activities from video streams, draws bounding boxes around detected persons, and triggers an alarm during violence detection.

---

# Features

- Real-time violence detection
- YOLOv8 person detection
- CNN + LSTM temporal action recognition
- Bounding box visualization
- Alarm triggering system
- Webcam and video file support
- Lightweight implementation

---

# Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Object Detection | YOLOv8 |
| Video Processing | OpenCV |
| Numerical Computing | NumPy |
| Alarm System | winsound |
| Dataset | Real Life Violence Dataset |

---

# Project Architecture

```text
Video Input
     ↓
YOLOv8 Person Detection
     ↓
Frame Sequence Generation
     ↓
CNN Feature Extraction
     ↓
LSTM Temporal Learning
     ↓
Violence Classification
     ↓
Bounding Boxes + Alarm
```

---

# Dataset

Dataset Used:

- Real Life Violence Dataset

Dataset Structure:

```text
Real Life Violence Dataset/
├── NonViolence/
└── Violence/
```

Dataset Link:

https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/violence-detection-system.git

cd violence-detection-system
```

---

## Create Virtual Environment

```bash
python -m venv vio
```

### Activate Environment

#### Windows

```bash
vio\Scripts\activate
```

---

## Install Dependencies

```bash
pip install tensorflow ultralytics opencv-python numpy scikit-learn tqdm
```

---

# Training the Model

## Configure Dataset Path

Inside `train.py`:

```python
DATASET_PATH = r"E:\Violence Detection\real-life-violence-situations-dataset\Real Life Violence Dataset"
```

---

## Train Model

```bash
python train.py
```

---

## Output Model

```text
violence_model.h5
```

---

# Running Inference

## Using Video File 

```python
VIDEO_PATH = r"E:\Violence Detection\test_videos\violence.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
```

---

## Run Detection

```bash
python infer.py
```

---

# Model Architecture

## CNN Backbone

- MobileNetV2 (Pretrained on ImageNet)

Used for:
- spatial feature extraction
- transfer learning

---

## Temporal Learning

- LSTM Layer

Used for:
- motion understanding
- temporal sequence learning
- violence recognition across frames

---

# Parameters

| Parameter | Value |
|---|---|
| Image Size | 64x64 |
| Sequence Length | 8 |
| Batch Size | 2 |
| Threshold | 0.90 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

---

# Real-Time Detection Workflow

1. Video frames are captured
2. YOLOv8 detects persons
3. Frames are converted into sequences
4. CNN extracts spatial features
5. LSTM learns temporal motion patterns
6. Violence probability is predicted
7. Bounding boxes are drawn
8. Alarm triggers during violence detection

---

# Output

## Normal Activity

- Green bounding boxes
- NORMAL label displayed

## Violence Detected

- Red bounding boxes
- VIOLENCE DETECTED label displayed
- Alarm sound triggered

---

# Challenges Faced

- Handling temporal video sequences
- False positives during prediction
- Video corruption issues
- Memory optimization
- Real-time inference optimization

---

# Future Improvements

- Person tracking
- Pose estimation
- Optical flow analysis
- MoViNet / SlowFast architectures
- FastAPI integration
- GPU optimization
- Cloud deployment

---

# Interview Concepts Covered

- Computer Vision
- Deep Learning
- Action Recognition
- Transfer Learning
- Temporal Sequence Modeling
- Real-Time Video Analytics
- Object Detection
- CNN-LSTM Architecture
