# YOLOv8 Drone Detection with Java & Spring Boot

A production-ready Java implementation of a YOLOv8-based drone detection system that processes live video streams in real-time.


<div align="center">
  <img src="drone_detection_webcam.gif" alt="Drone Detection" />
  <br>
  <em>Streaming App Displaying Live Drone Detection from a 1920×1080 Webcam Feed</em>
</div>

## Overview

This project demonstrates how to deploy a Python-trained YOLOv8 model into a Java Spring Boot application for scalable, real-time drone detection. The system captures live video from webcam feeds, performs object detection using ONNX runtime, and streams annotated video directly to web browsers.

## Key Features

- **Real-time Processing**: Handles HD video feeds at ~30 FPS
- **Web Streaming**: MJPEG streaming to web browsers
- **Production Ready**: Built with Java and Spring Boot for enterprise deployment
- **Framework Agnostic**: Uses ONNX format for model interoperability
- **Accurate Detection**: Maintains Python model accuracy in Java environment


## Quick Start

**Export Model**: First export your trained YOLOv8 model to ONNX format:

   ```python
   model = YOLO("runs/detect/train7/weights/best.pt")
   model.export(format="onnx", imgsz=120, dynamic=True)
   ```

**Configure Source**: Update `application.yml` to use webcam (source: "0") or video file  
**Run Application**:
```bash
mvn clean spring-boot:run
```

**Access Stream**: Open http://localhost:8080/stream in your browser
**Model Details**:

- Input Size: 128×128 pixels (automatically adjusted from training size)
- Output Shape: (1, 5, 336) - 1 image, 5 features per detection, 336 anchor points
- Features: center x, center y, width, height, and confidence
- Post-processing: Non-Maximum Suppression (NMS) for duplicate removal

## Why Java for Production?

- Performance: Compiled speed and efficient memory management
- Scalability: Strong concurrency support for multiple streams
- Stability: Robust enterprise-grade deployment capabilities
- Maintenance: Single JAR deployment avoids dependency issues

## License

MIT

## Acknowledgments

Built with Ultralytics YOLOv8, ONNX Runtime, and Spring Boot framework.