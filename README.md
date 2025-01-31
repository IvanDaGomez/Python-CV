# 🖥️ Computer Vision Projects  

This repository contains multiple computer vision projects utilizing OpenCV, Mediapipe, and YOLOv8 for various tasks, including hand tracking, face recognition, face mesh analysis, pose detection, and object detection.  

## 📌 Table of Contents  
- [🔧 Setup](#-setup)  
- [📂 Module Descriptions](#-module-descriptions)  
- [🚀 Running the Projects](#-running-the-projects)  
- [📜 Dependencies](#-dependencies)  
- [📸 Examples](#-examples)  

## 🔧 Setup  
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/computer-vision-projects.git
   cd computer-vision-projects
   ```
2. Install dependencies:  
   ```bash
   pip install opencv-python mediapipe ultralytics numpy
   ```

## 📂 Module Descriptions  
### 1️⃣ OpenCV Basic Operations  
- Image processing techniques such as edge detection, thresholding, and filtering.  
- Video processing with OpenCV.  

### 2️⃣ Mediapipe Projects (cvModules)
#### ✋ Hand Tracking  
- Tracks and identifies hand landmarks in real-time.  
- Used for gesture recognition and virtual controls.  

#### 😀 Face Detection  
- Detects and tracks faces using Mediapipe’s built-in face detection model.  

#### 🕸️ Face Mesh  
- Extracts 468 facial landmarks for applications like AR filters and expression analysis.  

#### 🏃 Pose Estimation  
- Identifies body keypoints for fitness tracking, gesture-based interactions, and movement analysis.  

### 3️⃣ YOLOv8 Object Detection  
- Utilizes Ultralytics YOLOv8 for real-time object detection.  
- There is a basic object detector pretrained and a car plate registrator.

### 4️⃣ Face recognition (faceRecognition)
- Utilizes face_recognition library to analyze specific faces and name them based on images loaded.
- Done a mark attendance implementation for a class that could use this method

## 🚀 Running the Projects  


- Mediapipe:
  Just import the library
  ```bash
  from cvModules.handDetector import HandDetector
  ```
  and in your project initialize the class and read the image.
  ```bash
  detector = HandDetector()
  img = detector.findHands(img)
  ```
- YOLOv8 Object Detection:  
  ```bash
  cd YOLO
  python yoloFirst.py
  ```
- Face recognition
  ```bash
  cd faceRecognition
  python faceRecognition.py
  ```
  You can upload images to the 'Images' folder or take a picture while the program runs pressing 's'

## 📸 Examples  
Here are some examples of the projects in action:  
🖼️ Add example images or GIFs of your models detecting faces, hands, poses, or objects.  
