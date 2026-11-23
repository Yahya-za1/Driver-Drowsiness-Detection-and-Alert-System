# Driver-Drowsiness-Detection-and-Alert-System
Camera-Based Driver Drowsiness Detection and Alert System Using Edge Machine Learning.

### 1. Project Overview

Driver drowsiness is a significant road safety concern that can reduce driver attention, reaction time, and overall driving performance.

This project focuses on developing a camera-based Driver Drowsiness Detection and Alert System using edge machine learning. The proposed system monitors driver's eye activity through a camera, processes visual data.

The system uses MediaPipe Face Landmarker to identify facial landmarks around the eyes and calculates the Eye Aspect Ratio (EAR) to estimate whether the driver's eyes are open or closed. It also uses continuous eye-closure duration and Percentage of Eyelid Closure (PERCLOS) to identify patterns associated with drowsiness.

When the detected eye-closure conditions exceed predefined thresholds, the system generates a warning. The intended hardware implementation uses a Raspberry Pi connected to a camera, buzzer, and LED to provide real-time alerts.

A Convolutional Neural Network (CNN) based on MobileNetV2 was also developed and evaluated as an alternative eye-state classification approach. However, initial webcam testing showed limitations in its ability to generalize to live camera images. Consequently, the current prototype uses MediaPipe and EAR for real-time eye-state detection.

The project aims to develop a system that captures and analyzes eye activity in real time and activates an alert when signs of drowsiness are detected

### 2. Project Objectives
The main objective of this project is to develop a camera-based driver drowsiness detection and alert system using a Raspberry Pi. The system aims to monitor the driver's eye movements and facial features in real time to identify signs of drowsiness and provide immediate warnings through a buzzer and LED.

The specific objectives are to:
1. Develop a real-time camera-based system to monitor the driver's face and eyes.
2. Apply computer vision techniques to detect prolonged eye closure and yawning.
3. Use eye-related measurements, such as the Eye Aspect Ratio (EAR), to identify potential signs of drowsiness.
4. Implement an alert mechanism using a GPIO-connected buzzer and LED.
5. Evaluate the system under different conditions, including lighting changes, head movements, normal blinking, and prolonged eye closure.
6. Assess system performance in terms of detection accuracy, response time, and false alerts.

The overall goal is to create a low-cost, non-intrusive prototype that demonstrates how edge computing and computer vision can be integrated into a real-time driver monitoring system.
### 3. System Architecture

The proposed system consists of four main components: image acquisition, facial landmark detection, drowsiness analysis, and alert generation.

## 3.1 Image Acquisition

A camera connected to the Raspberry Pi captures real-time video frames of the driver's face. The frames are processed using OpenCV and prepared for facial landmark detection.

## 3.2 Facial Landmark Detection

MediaPipe Face Landmarker processes the captured images to identify facial landmarks, including points around both eyes.

These landmarks are used to estimate eyelid opening through the Eye Aspect Ratio (EAR).

## 3.3 Drowsiness Detection

The system analyzes eye activity using two measurements:

Eye Aspect Ratio (EAR): Estimates whether the driver's eyes are open or closed based on the distances between eye landmarks.

Percentage of Eyelid Closure (PERCLOS): Estimates the proportion of observed time during which the eyes are classified as closed within a specified time window.

The system also monitors continuous eye-closure duration. If either the sustained closure duration or PERCLOS exceeds its configured threshold, a drowsiness warning is generated.

### 4. Hardware and Software Requirements

Components

#### Hardware

Raspberry Pi 4 Model B  <br>
Camera  <br>
Buzzer  <br>
LED and current-limiting resistor  <br>
Breadboard and jumper wires  <br>
microSD card and power supply  <br>

#### Software

Raspberry Pi OS <br>
Python 3  <br>
OpenCV - Video  <br>
GPIO library  <br>


### 5. Implementation

### 6. Testing and Evaluation

### 7. Team Members
Yahya Al-Khateeb<br>
Osama Abdulghani<br>
Oluwakorede Oseni<br>
Oloto Arni <br>

### 8. Project Status
In Progress
