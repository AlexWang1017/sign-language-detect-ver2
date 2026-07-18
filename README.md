# AI Static Hand Gesture Recognition System

A real-time hand gesture recognition system based on **Computer Vision and Deep Learning**.

This project uses **MediaPipe Hands** for hand detection and tracking, and applies a **CNN-based image classification model** to recognize 20 different static hand gestures, including left and right hand gestures from numbers 0 to 9.

The system supports real-time webcam inference and displays the predicted gesture with confidence score.

---

# System Overview

```
Webcam
   |
   v
MediaPipe Hands Detection
   |
   v
Hand Region Extraction
   |
   v
Image Preprocessing
   |
   v
CNN Classification Model
   |
   v
Gesture Prediction
```

---

# Features

- Real-time hand detection using MediaPipe Hands
- CNN-based static gesture classification
- Supports 20 gesture categories
  - Left hand: 0~9
  - Right hand: 0~9
- Automatic hand region extraction
- Webcam real-time recognition
- Confidence score display
- FPS monitoring

---

# Technologies

## Programming Language

- Python

## Computer Vision

- OpenCV
- MediaPipe Hands

## Deep Learning

- TensorFlow
- Keras
- Convolutional Neural Network (CNN)

## Data Processing

- NumPy
- Scikit-learn

---

# Project Structure

```
AI-Hand-Gesture-Recognition/

│
├── data/
│   └── dataset/
│       ├── left_0/
│       ├── left_1/
│       ├── ...
│       ├── right_9/
│       │
│       ├── x_data.npy
│       ├── y_data.npy
│       └── static_gesture_model_20_classes.h5
│
├── collect_data.py
├── preprocess_data.py
├── train_model.py
├── realtime_prediction.py
│
└── README.md
```

---

# Dataset Collection

The dataset is collected using a webcam.

For each gesture class:

- 350 images are collected
- MediaPipe Hands detects hand landmarks
- Bounding boxes are generated automatically
- Hand regions are cropped and resized

Image size:

```
256 × 256 × 3
```

Dataset:

```
20 Classes

10 Left-hand gestures
+
10 Right-hand gestures


Total:

20 × 350 = 7000 images
```

Dataset example:

```
dataset/

├── left_0/
│   ├── 001.jpg
│   ├── 002.jpg
│
├── left_1/
│
├── ...
│
└── right_9/
```

---

# Data Preprocessing

The collected images are processed by:

1. Loading image data
2. Converting BGR images to RGB
3. Resizing images to 256×256
4. Normalizing pixel values

Normalization:

```
pixel value / 255.0
```

Processed datasets are saved as:

```
x_data.npy
y_data.npy
```

---

# CNN Model Architecture

The model uses a Convolutional Neural Network for static gesture classification.

Architecture:

```
Input
(256,256,3)

        |
        v

Conv2D
32 Filters

        |
        v

MaxPooling

        |
        v

Conv2D
64 Filters

        |
        v

MaxPooling

        |
        v

Conv2D
128 Filters

        |
        v

MaxPooling

        |
        v

Flatten

        |
        v

Dense
128 Neurons

        |
        v

Dropout

        |
        v

Softmax Output

20 Classes
```

---

# Model Training

Training configuration:

| Parameter | Value |
|---|---|
| Input Size | 256 × 256 |
| Classes | 20 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Crossentropy |
| Epochs | 5~30 |

Training example:

```python
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=30,
    batch_size=32
)
```

---

# Real-Time Recognition

The inference pipeline:

1. Capture webcam frame
2. Detect hand using MediaPipe
3. Extract hand bounding box
4. Resize image to 256×256
5. Normalize image data
6. Predict gesture using CNN model

Example output:

```
Gesture: left_5

Confidence: 0.94

FPS: 30
```

---

# Installation

Install required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:

```
tensorflow
opencv-python
mediapipe
numpy
scikit-learn
```

---

# Usage

## 1. Collect Dataset

Run:

```bash
python collect_data.py
```

The program will:

- Open webcam
- Detect hand landmarks
- Crop hand images
- Save images into gesture folders


---

## 2. Preprocess Dataset

Run:

```bash
python preprocess_data.py
```

Generated files:

```
x_data.npy
y_data.npy
```

---

## 3. Train Model

Run:

```bash
python train_model.py
```

The trained model will be saved:

```
data/dataset/static_gesture_model_20_classes.h5
```

---

## 4. Real-Time Recognition

Run:

```bash
python realtime_prediction.py
```

Press:

```
q
```

to exit.

---

# Performance

Evaluation metrics:

```
Test Accuracy:
XX.XX%
```

Real-time performance:

```
FPS:
XX
```

Performance depends on hardware configuration.

---

# Future Improvements

- [ ] Replace CNN image classification with MediaPipe landmark classifier
- [ ] Add dynamic gesture recognition using CNN-LSTM
- [ ] Increase gesture categories
- [ ] Apply data augmentation
- [ ] Deploy on Raspberry Pi / Edge AI devices
- [ ] Convert model to TensorFlow Lite

---

# Learning Outcomes

Through this project, I gained practical experience in:

- Computer Vision system development
- Real-time image processing
- MediaPipe hand tracking
- CNN model training and optimization
- Dataset collection and preprocessing
- Deep learning model deployment

---

# Author

**Alex Wang**

Computer Science Student

Skills:

- Python
- TensorFlow / Keras
- OpenCV
- Computer Vision
- Deep Learning
