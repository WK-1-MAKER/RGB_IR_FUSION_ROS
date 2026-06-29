# two_stream

`two_stream` is a ROS1 (`catkin`) C++ package for deploying the current repository's dual-stream YOLOv8 ONNX model with ONNX Runtime, SuperPoint, and LightGlue.

## Build

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make -DCMAKE_BUILD_TYPE=Release
```

## Status

The package now includes:

- Dual-input YOLOv8 ONNX Runtime inference for RGB and IR images
- C++-side YOLOv8 detection decode and NMS after the current rank-3 ONNX output
- SuperPoint and LightGlue runtime/resources copied into this repository
- ROS launch/config/model assets under the `two_stream` package

## Scope

This package is intentionally narrowed to the current repository's YOLOv8 dual-input ONNX export and does not keep compatibility branches for legacy YOLOv5-style outputs.

## onnxruntime版本和tensorrt版本
- onnxruntime-gpu-1.18.0
- tensorrt-8.6.1.6
- CUDA-11.8

