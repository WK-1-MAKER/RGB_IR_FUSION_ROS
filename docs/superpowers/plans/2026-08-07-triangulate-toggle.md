# `traingulate` Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a launch-controlled switch that disables all feature matching and 3D localization work while preserving YOLO detection output.

**Architecture:** Read private parameter `~traingulate` with default `true`. Construct SuperPoint and LightGlue only when enabled, and guard the existing post-detection geometry pipeline with the same flag. Keep annotated image publication active in both modes.

**Tech Stack:** ROS1 roscpp, C++14, catkin, GoogleTest.

---

### Task 1: Add configuration contract test

**Files:**
- Create: `include/ros_yolo/triangulation_options.h`
- Create: `tests/test_triangulation_options.cpp`
- Modify: `CMakeLists.txt`

- [x] Write tests for the exact parameter name `traingulate`, default-enabled state, and explicit disabled state.
- [x] Run the new test before implementation and confirm it fails because the header is absent.
- [x] Add the small options type and build the test target.
- [x] Run the test and confirm all cases pass.

### Task 2: Add the ROS parameter and conditional model initialization

**Files:**
- Modify: `include/ros_yolo/yolo_node.h`
- Modify: `src/yolo_node.cpp`
- Modify: `launch/yolo_topic.launch`

- [x] Add `triangulate_` with default `true` and read private parameter `traingulate`.
- [x] Wrap SuperPoint and LightGlue construction/build calls in `if (triangulate_)`.
- [x] Add a default-true `traingulate` launch argument and pass it to the node parameter.

### Task 3: Guard per-frame 3D post-processing

**Files:**
- Modify: `src/yolo_node.cpp`

- [x] Put SuperPoint inference, LightGlue matching, triangulation, body-frame aggregation and 3D position logs under `if (triangulate_)`.
- [x] Keep detection visualization and annotated image publication outside the branch.
- [x] Preserve the existing true-path code and output semantics.

### Task 4: Verify

- [x] Build `ros_yolo_node` and the new test target.
- [x] Run the new test plus latest-frame, camera-geometry, YOLO decode and detector post-processing tests.
- [x] Run `git diff --check` and inspect that the launch parameter and conditional branches are present.
