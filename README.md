# Robotic Arm 3D Vision with YOLOv11-OBB

This repository implements a robotic vision system for automatic part assembly using a 5-DOF robotic arm and Intel RealSense D435 camera. The system uses YOLOv11-OBB for object detection and 3D point cloud data for pose estimation and control.

## 📦 Project Files

- `me5110q.py`: Main logic for robot vision and control
- `untitled.py`: UI or auxiliary functions
- `untitled.ui`: Qt Designer UI layout
- `best_obb_1.pt`, `best_obb_2.pt`: YOLOv11-OBB models
- `note.txt`: Notes or configuration details

## 🛠 Requirements

- Python ≥ 3.8
- USB 3.1+ port for RealSense camera
- IPv4 address set to `192.168.3.1` for robot communication

### Suggested Python Libraries

- `ultralytics` (YOLOv11-OBB)
- `pyrealsense2`
- `open3d`, `numpy`, `opencv-python`, `torch`, `matplotlib`

### CUDA (optional)

Install CUDA Toolkit if using GPU acceleration with PyTorch/YOLO.

## 🚀 Getting Started

```bash
python me5110q.py
```

Make sure:
- RealSense D435 is connected via USB 3.1+
- Robot is reachable at IP `192.168.3.1`

## 📄 License

This project is intended for educational and research purposes.
