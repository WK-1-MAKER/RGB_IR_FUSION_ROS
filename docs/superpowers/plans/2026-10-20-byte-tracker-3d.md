# BYTETracker3D Implementation Plan

**Goal:** Port the existing image/world XY tracker into an independent C++14 module.

**Architecture:** Public input/config/result structs and a stateful Pimpl tracker
in `include/ros_yolo/byte_tracker_3d.h`. Private image/world filters, Hungarian
assignment with unmatched choices, and lifecycle pools live in
`src/byte_tracker_3d.cpp`. Eigen supplies matrix algebra; OpenCV supplies value
types only. No changes to the ROS execution pipeline or existing user edits.

**Execution:** Inline in the authorized current checkout; no git mutations,
commits, extra approval loops or subagents for implementation.

- [x] Read Python tracker, base tracker, filter, matching, pool helpers and tests.
- [x] Review design against the user's standalone-module requirement.
- [x] Write public interface and initial GTests in `tests/test_byte_tracker_3d.cpp`.
- [x] Compile tests and verify missing implementation produces undefined symbols.
- [x] Add regression cases for lifecycle, confidence split, missing measurements,
  world gates, validation and global assignment before implementing behavior.
- [x] Implement XYAH and world filters with Eigen double matrices, per-frame
  world prediction/expiry, thresholded global assignment and ByteTrack pools.
- [x] Build with `g++ -std=c++14 -Iinclude -I/usr/include/opencv4
  -I/usr/include/eigen3 tests/test_byte_tracker_3d.cpp src/byte_tracker_3d.cpp
  -lgtest_main -lgtest -lopencv_core -pthread -o /tmp/test_byte_tracker_3d`;
  execute `/tmp/test_byte_tracker_3d` and require every test to pass.
- [x] Export/build a separate `byte_tracker_3d` CMake library and catkin GTest.
- [x] Compare deterministic C++ outputs to Python on shared synthetic inputs;
  require identical IDs/indices/lifecycle and close filtered numeric outputs.
- [x] Run fresh build/tests, inspect diff and report created files and validation.
