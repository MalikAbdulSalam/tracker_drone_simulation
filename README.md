# Drone Object Tracking Simulation

This project demonstrates an intelligent drone system that combines computer vision with autonomous control to track an object in real-time using **YOLOv8** and **BoT-SORT**. The drone's movement is adjusted based on the tracked object's position to keep it centered in the camera frame.

## Features

- **Object Detection**: YOLOv8 identifies and tracks objects in real-time.
- **Autonomous Flight**: The drone adjusts its position to keep the tracked object centered.
- **User Input**: Allows users to enter waypoints for the drone and initiate landing.
- **Offboard Control**: The drone operates in offboard mode for precise movement control.
- **Camera Feed**: Displays the camera feed with annotations for tracked objects.
- **Tracking Direction**: Based on the object’s position, the drone moves left, right, forward, or backward to keep the object centered.
![Chatbot Flow Screenshot](resources/rag.png)
## Requirements

Before running the simulation, ensure you have the following dependencies installed:

- **PX4 SITL** (Software-In-The-Loop) for simulation.
- **ROS** (Robot Operating System) for communication and control.
- **QGroundControl** for mission planning and drone telemetry.
- **OpenCV** for handling video feed and tracking.
- **Ultralytics YOLOv8** for object detection and tracking.
- **MAVSDK** for controlling the drone via Python.
- Python 3.7+ and required libraries.

### Setup
- **PX4 SITL** (Software-In-The-Loop) for simulation.
- **ROS** (Robot Operating System) for communication and control.
- **QGroundControl** for mission planning and drone telemetry.
### Clone the repository and setup environment:
```bash
git clone https://github.com/MalikAbdulSalam/RAG.git
cd tracker_drone_simulation
conda env create -f environment.yaml
conda activate droncuda
python test2.py

