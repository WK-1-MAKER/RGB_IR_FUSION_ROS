# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# From the ROS workspace root (not the package directory)
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make -DCMAKE_BUILD_TYPE=Release
```

## Run

```bash
roslaunch two_stream yolo_topic.launch
```

## Tests

```bash
# Run all tests
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make run_tests

# Run individual test binaries after building
./devel/lib/two_stream/test_yolov8_decode
./devel/lib/two_stream/test_detector_postprocess
./devel/lib/two_stream/test_read_config
```

## Architecture

This is a ROS1 (catkin) C++14 package that fuses RGB and thermal (IR) image streams for object detection and 3D feature matching.

**Pipeline:**
1. `yolo_node.cpp` — ROS node that subscribes to synchronized RGB/IR image pairs (via `message_filters` ApproximateTime policy), orchestrates the pipeline, and publishes annotated output
2. `detector.cpp` + `yolov8_decode.cpp` — YOLOv8 inference via ONNX Runtime GPU; decodes rank-3 output tensor, applies confidence thresholding and NMS
3. `super_point.cpp` — TensorRT-accelerated SuperPoint feature extractor (keypoints + 258-dim descriptors); builds/deserializes `.engine` from `.onnx` on first run
4. `light_glue.cpp` — TensorRT-accelerated LightGlue matcher; matches descriptors from RGB and IR views
5. `yolo_node.cpp` (stereo geometry) — Loads stereo calibration from `config/config.yaml`, computes homography (H) and fundamental matrix (F), rectifies matched points, triangulates 3D positions

**Key data structures:**
- `Detection` (yolo_utils.h): `cv::Rect box`, `float conf`, `int classId`
- `Configs` / `CameraCalibrationConfig` (read_config.h): parsed from `config/config.yaml` via yaml-cpp

**External dependencies (hardcoded paths in CMakeLists.txt):**
- ONNX Runtime: `/home/SENSETIME/wenkai/3rd/onnxruntime-linux-x64-gpu-1.18.0`
- TensorRT: `/home/SENSETIME/wenkai/3rd/TensorRT-8.6.1.6`
- Versions: ONNX Runtime 1.18.0, TensorRT 8.6.1.6, CUDA 11.8

**Model/weight files (git-ignored, must be present manually):**
- `models/best.onnx` — dual-input YOLOv8 ONNX model (189 MB)
- `weights/superpoint_v1.{onnx,engine}` — SuperPoint
- `weights/superpoint_lightglue.{onnx,engine}` — LightGlue

TensorRT `.engine` files are built from `.onnx` on first run and cached; if hardware changes, delete `.engine` files to force rebuild.

**Launch parameters** (configurable in `launch/yolo_topic.launch`):
- `rgb_topic` / `ir_topic`: input image topics
- `conf_threshold` / `iou_threshold`: detection thresholds
- `model_path`, `config_path`, `model_dir`: file paths
- `input_width` / `input_height`: model input resolution (default 640×640)

- **开发过程中禁止自动提交代码**:修改代码过程中禁止通过git提交代码，禁止git add，git commit, git push等操作，代码的git管理统一由人工完成。
- **未经允许不可修改代码**:每次修改代码前都要询问一次，未经允许不可直接修改代码
