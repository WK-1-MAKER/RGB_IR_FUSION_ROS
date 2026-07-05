# Body Camera Extrinsics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `two_stream` camera extrinsics to the `T_body_camera` convention while preserving current RGB-IR alignment geometry and emitting final 3D estimates in body/IMU coordinates.

**Architecture:** Add a small header-only transform helper for camera pose inversion, composition, and point transformation. Parse the same config fields as `T_body_rgb` and `T_body_ir`, derive inverse camera-to-body transforms, then derive `T_ir_rgb` for the existing OpenCV homography, fundamental matrix, rectification, and triangulation logic.

**Tech Stack:** ROS1 catkin, C++14, OpenCV `cv::Matx33d` and `cv::Vec3d`, yaml-cpp, gtest.

---

## File Structure

- Create `ROS/src/two_stream/include/ros_yolo/camera_geometry.h`
  - Owns small, testable rigid transform helpers for camera/body geometry.
- Create `ROS/src/two_stream/tests/test_camera_geometry.cpp`
  - Verifies inversion, composition, and the temporary config values derive the legacy `T_ir_rgb`.
- Modify `ROS/src/two_stream/CMakeLists.txt`
  - Registers the new geometry test.
- Modify `ROS/src/two_stream/config/config.yaml`
  - Updates the extrinsic convention comment and temporary IR translation.
- Modify `ROS/src/two_stream/tests/test_read_config.cpp`
  - Updates expected parsed camera-to-body temporary values.
- Modify `ROS/src/two_stream/include/ros_yolo/yolo_node.h`
  - Renames camera geometry members to explicit body/camera and RGB/IR-relative names.
- Modify `ROS/src/two_stream/src/yolo_node.cpp`
  - Derives `T_ir_rgb` from `T_body_rgb` and `T_body_ir`, keeps `H_` and `F_` direction unchanged, and converts final RGB-frame 3D points to body.

Do not run `git add`, `git commit`, or `git push` in this package. The package instructions reserve git management for the human operator.

---

### Task 1: Add Transform Helper and Unit Test

**Files:**
- Create: `ROS/src/two_stream/include/ros_yolo/camera_geometry.h`
- Create: `ROS/src/two_stream/tests/test_camera_geometry.cpp`
- Modify: `ROS/src/two_stream/CMakeLists.txt`

- [ ] **Step 1: Write the failing geometry test**

Create `ROS/src/two_stream/tests/test_camera_geometry.cpp`:

```cpp
#include "ros_yolo/camera_geometry.h"

#include <gtest/gtest.h>

namespace
{
void expectVecNear(const cv::Vec3d& actual, const cv::Vec3d& expected)
{
    constexpr double kTol = 1e-12;
    EXPECT_NEAR(actual[0], expected[0], kTol);
    EXPECT_NEAR(actual[1], expected[1], kTol);
    EXPECT_NEAR(actual[2], expected[2], kTol);
}

void expectMatNear(const cv::Matx33d& actual, const cv::Matx33d& expected)
{
    constexpr double kTol = 1e-12;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            EXPECT_NEAR(actual(r, c), expected(r, c), kTol);
        }
    }
}
}  // namespace

TEST(CameraGeometryTest, InvertsBodyCameraTransform)
{
    const cv::Matx33d rotation(0.0, -1.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0);
    const cv::Vec3d translation(1.0, 2.0, 3.0);
    const ros_yolo::CameraTransform bodyCamera(rotation, translation);

    const ros_yolo::CameraTransform cameraBody = ros_yolo::inverseTransform(bodyCamera);

    expectMatNear(cameraBody.rotation, rotation.t());
    expectVecNear(cameraBody.translation, cv::Vec3d(-2.0, 1.0, -3.0));
}

TEST(CameraGeometryTest, DerivesLegacyRgbToIrTransformFromTemporaryBodyCameraPoses)
{
    const ros_yolo::CameraTransform bodyRgb(cv::Matx33d::eye(), cv::Vec3d(0.0, 0.0, 0.0));
    const ros_yolo::CameraTransform bodyIr(cv::Matx33d::eye(), cv::Vec3d(0.036, -0.03, 0.009));

    const ros_yolo::CameraTransform irBody = ros_yolo::inverseTransform(bodyIr);
    const ros_yolo::CameraTransform irRgb = ros_yolo::composeTransforms(irBody, bodyRgb);

    expectMatNear(irRgb.rotation, cv::Matx33d::eye());
    expectVecNear(irRgb.translation, cv::Vec3d(-0.036, 0.03, -0.009));
}

TEST(CameraGeometryTest, TransformsPointIntoBodyFrame)
{
    const ros_yolo::CameraTransform bodyRgb(cv::Matx33d::eye(), cv::Vec3d(0.1, -0.2, 0.3));
    const cv::Point3f pointRgb(1.0F, 2.0F, 3.0F);

    const cv::Point3f pointBody = ros_yolo::transformPoint(bodyRgb, pointRgb);

    EXPECT_NEAR(pointBody.x, 1.1F, 1e-6F);
    EXPECT_NEAR(pointBody.y, 1.8F, 1e-6F);
    EXPECT_NEAR(pointBody.z, 3.3F, 1e-6F);
}
```

- [ ] **Step 2: Register the geometry test in CMake**

In `ROS/src/two_stream/CMakeLists.txt`, add this block inside `if(CATKIN_ENABLE_TESTING)` after the `test_read_config` block:

```cmake
  catkin_add_gtest(test_camera_geometry
    tests/test_camera_geometry.cpp
  )
  if(TARGET test_camera_geometry)
    target_link_libraries(test_camera_geometry
      ${OpenCV_LIBS}
      gtest_main
    )
  endif()
```

- [ ] **Step 3: Run the test to verify it fails before the helper exists**

Run from the ROS workspace root:

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make run_tests_two_stream_gtest_test_camera_geometry
```

Expected result: build fails because `ros_yolo/camera_geometry.h` does not exist.

- [ ] **Step 4: Add the transform helper header**

Create `ROS/src/two_stream/include/ros_yolo/camera_geometry.h`:

```cpp
#pragma once

#include <opencv2/opencv.hpp>

namespace ros_yolo
{
struct CameraTransform
{
    CameraTransform()
        : rotation(cv::Matx33d::eye()), translation(0.0, 0.0, 0.0)
    {
    }

    CameraTransform(const cv::Matx33d& rotationIn, const cv::Vec3d& translationIn)
        : rotation(rotationIn), translation(translationIn)
    {
    }

    cv::Matx33d rotation;
    cv::Vec3d translation;
};

inline CameraTransform inverseTransform(const CameraTransform& transform)
{
    const cv::Matx33d inverseRotation = transform.rotation.t();
    return CameraTransform(inverseRotation, -(inverseRotation * transform.translation));
}

inline CameraTransform composeTransforms(const CameraTransform& lhs, const CameraTransform& rhs)
{
    return CameraTransform(lhs.rotation * rhs.rotation,
                           lhs.rotation * rhs.translation + lhs.translation);
}

inline cv::Point3f transformPoint(const CameraTransform& transform, const cv::Point3f& point)
{
    const cv::Vec3d pointVec(point.x, point.y, point.z);
    const cv::Vec3d transformed = transform.rotation * pointVec + transform.translation;
    return cv::Point3f(static_cast<float>(transformed[0]),
                       static_cast<float>(transformed[1]),
                       static_cast<float>(transformed[2]));
}
}  // namespace ros_yolo
```

- [ ] **Step 5: Run the geometry test**

Run:

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make run_tests_two_stream_gtest_test_camera_geometry
```

Expected result: `test_camera_geometry` builds and passes.

- [ ] **Step 6: Manual review checkpoint**

Review only these files:

```bash
git -C /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS/src/two_stream diff -- include/ros_yolo/camera_geometry.h tests/test_camera_geometry.cpp CMakeLists.txt
```

Do not stage or commit.

---

### Task 2: Update Config Convention and Config Tests

**Files:**
- Modify: `ROS/src/two_stream/config/config.yaml`
- Modify: `ROS/src/two_stream/tests/test_read_config.cpp`

- [ ] **Step 1: Update package config comment and temporary IR translation**

In `ROS/src/two_stream/config/config.yaml`, replace:

```yaml
# Camera extrinsics use world_to_camera:
# X_camera = R_world_to_camera * X_world + t_world_to_camera
```

with:

```yaml
# Camera extrinsics use camera_to_body:
# X_body = R_body_camera * X_camera + t_body_camera
```

In the `cameras.ir` block, replace:

```yaml
    translation: [-0.036, 0.03, -0.009]
```

with:

```yaml
    translation: [0.036, -0.03, 0.009]
```

- [ ] **Step 2: Update inline test configs and expectations**

In `ROS/src/two_stream/tests/test_read_config.cpp`, replace every inline IR translation:

```yaml
    translation: [-0.036, 0.03, -0.009]
```

with:

```yaml
    translation: [0.036, -0.03, 0.009]
```

Replace every expectation of parsed IR translation values:

```cpp
EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[0], -0.036);
EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[1], 0.03);
EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[2], -0.009);
```

with:

```cpp
EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[0], 0.036);
EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[1], -0.03);
EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[2], 0.009);
```

- [ ] **Step 3: Run the config test**

Run:

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make run_tests_two_stream_gtest_test_read_config
```

Expected result: `test_read_config` passes.

- [ ] **Step 4: Manual review checkpoint**

Review only these files:

```bash
git -C /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS/src/two_stream diff -- config/config.yaml tests/test_read_config.cpp
```

Do not stage or commit.

---

### Task 3: Rename Geometry Members and Derive `T_ir_rgb`

**Files:**
- Modify: `ROS/src/two_stream/include/ros_yolo/yolo_node.h`
- Modify: `ROS/src/two_stream/src/yolo_node.cpp`

- [ ] **Step 1: Update `YoloNode` geometry members**

In `ROS/src/two_stream/include/ros_yolo/yolo_node.h`, replace:

```cpp
    cv::Matx33d rRgbWorld_;
    cv::Matx33d rIrWorld_;
    cv::Vec3d tRgbWorld_;
    cv::Vec3d tIrWorld_;
    cv::Matx33d r_;
    cv::Vec3d t_;
```

with:

```cpp
    cv::Matx33d rBodyRgb_;
    cv::Matx33d rBodyIr_;
    cv::Vec3d tBodyRgb_;
    cv::Vec3d tBodyIr_;
    cv::Matx33d rRgbBody_;
    cv::Matx33d rIrBody_;
    cv::Vec3d tRgbBody_;
    cv::Vec3d tIrBody_;
    cv::Matx33d rIrRgb_;
    cv::Vec3d tIrRgb_;
```

- [ ] **Step 2: Include the transform helper in the node implementation**

In `ROS/src/two_stream/src/yolo_node.cpp`, add this include after the existing node include:

```cpp
#include "ros_yolo/camera_geometry.h"
```

- [ ] **Step 3: Replace `loadCameraCalibration()`**

In `ROS/src/two_stream/src/yolo_node.cpp`, replace the full `YoloNode::loadCameraCalibration()` function with:

```cpp
void YoloNode::loadCameraCalibration()
{
    if (!configs_)
    {
        throw std::runtime_error("Cannot load camera calibration before config is initialized.");
    }

    const CameraCalibrationConfig& cameraConfig = configs_->camera_config;
    kRgb_ = matx33FromVector(cameraConfig.rgb.intrinsic);
    kIr_ = matx33FromVector(cameraConfig.ir.intrinsic);
    distRgb_ = distortionFromVector(cameraConfig.rgb.distortion);
    distIr_ = distortionFromVector(cameraConfig.ir.distortion);

    rBodyRgb_ = matx33FromVector(cameraConfig.rgb.rotation);
    rBodyIr_ = matx33FromVector(cameraConfig.ir.rotation);
    tBodyRgb_ = vec3FromVector(cameraConfig.rgb.translation);
    tBodyIr_ = vec3FromVector(cameraConfig.ir.translation);

    const CameraTransform bodyRgb(rBodyRgb_, tBodyRgb_);
    const CameraTransform bodyIr(rBodyIr_, tBodyIr_);
    const CameraTransform rgbBody = inverseTransform(bodyRgb);
    const CameraTransform irBody = inverseTransform(bodyIr);
    const CameraTransform irRgb = composeTransforms(irBody, bodyRgb);

    rRgbBody_ = rgbBody.rotation;
    tRgbBody_ = rgbBody.translation;
    rIrBody_ = irBody.rotation;
    tIrBody_ = irBody.translation;
    rIrRgb_ = irRgb.rotation;
    tIrRgb_ = irRgb.translation;

    kRgbTri_ = kRgb_;
    kIrTri_ = kIr_;
    rTri_ = rIrRgb_;
    tTri_ = tIrRgb_;
}
```

- [ ] **Step 4: Replace relative transform uses in `compute_F()` and `compute_H()`**

In `compute_F()`, replace:

```cpp
    const cv::Matx33d tCross(0.0, -t_[2], t_[1],
                          t_[2], 0.0, -t_[0],
                          -t_[1], t_[0], 0.0);
    const cv::Matx33d E_ = tCross * r_;
```

with:

```cpp
    const cv::Matx33d tCross(0.0, -tIrRgb_[2], tIrRgb_[1],
                          tIrRgb_[2], 0.0, -tIrRgb_[0],
                          -tIrRgb_[1], tIrRgb_[0], 0.0);
    const cv::Matx33d E_ = tCross * rIrRgb_;
```

In `compute_H()`, replace:

```cpp
        t_[0] * n[0], t_[0] * n[1], t_[0] * n[2],
        t_[1] * n[0], t_[1] * n[1], t_[1] * n[2],
        t_[2] * n[0], t_[2] * n[1], t_[2] * n[2]);

    H_ = kIr_ * (r_ - tOuter * (1.0 / refDepth)) * kRgb_.inv(cv::DECOMP_SVD);
```

with:

```cpp
        tIrRgb_[0] * n[0], tIrRgb_[0] * n[1], tIrRgb_[0] * n[2],
        tIrRgb_[1] * n[0], tIrRgb_[1] * n[1], tIrRgb_[1] * n[2],
        tIrRgb_[2] * n[0], tIrRgb_[2] * n[1], tIrRgb_[2] * n[2]);

    H_ = kIr_ * (rIrRgb_ - tOuter * (1.0 / refDepth)) * kRgb_.inv(cv::DECOMP_SVD);
```

- [ ] **Step 5: Replace relative transform uses in `updateRectification()`**

In `updateRectification()`, replace the rotation matrix construction:

```cpp
        << r_(0, 0), r_(0, 1), r_(0, 2),
           r_(1, 0), r_(1, 1), r_(1, 2),
           r_(2, 0), r_(2, 1), r_(2, 2));
    const cv::Mat translation = (cv::Mat_<double>(3, 1) << t_(0), t_(1), t_(2));
```

with:

```cpp
        << rIrRgb_(0, 0), rIrRgb_(0, 1), rIrRgb_(0, 2),
           rIrRgb_(1, 0), rIrRgb_(1, 1), rIrRgb_(1, 2),
           rIrRgb_(2, 0), rIrRgb_(2, 1), rIrRgb_(2, 2));
    const cv::Mat translation = (cv::Mat_<double>(3, 1) << tIrRgb_(0), tIrRgb_(1), tIrRgb_(2));
```

- [ ] **Step 6: Replace fallback reset assignments**

In both `catch` blocks in `syncImageCallback()`, replace:

```cpp
        rTri_ = r_;
        tTri_ = t_;
```

with:

```cpp
        rTri_ = rIrRgb_;
        tTri_ = tIrRgb_;
```

- [ ] **Step 7: Verify no old geometry member names remain**

Run:

```bash
grep -RIn "rRgbWorld_\\|rIrWorld_\\|tRgbWorld_\\|tIrWorld_\\|r_\\|t_" /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS/src/two_stream/src/yolo_node.cpp /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS/src/two_stream/include/ros_yolo/yolo_node.h
```

Expected result: no output for `rRgbWorld_`, `rIrWorld_`, `tRgbWorld_`, or `tIrWorld_`. The simple `r_` and `t_` alternatives may match unrelated local variables, so inspect any remaining hits and confirm no class member use remains.

- [ ] **Step 8: Run a compile check for the node target**

Run:

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make -DCMAKE_BUILD_TYPE=Release
```

Expected result: `ros_yolo_node` builds successfully.

- [ ] **Step 9: Manual review checkpoint**

Review only these files:

```bash
git -C /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS/src/two_stream diff -- include/ros_yolo/yolo_node.h src/yolo_node.cpp
```

Do not stage or commit.

---

### Task 4: Convert Final 3D Estimates to Body Coordinates

**Files:**
- Modify: `ROS/src/two_stream/src/yolo_node.cpp`

- [ ] **Step 1: Replace RGB-frame point handling before detection association**

In `syncImageCallback()`, find this block after `pointLeftVec` is computed:

```cpp
        cv::Point3f point3d(static_cast<float>(pointLeftVec[0]),
                            static_cast<float>(pointLeftVec[1]),
                            static_cast<float>(pointLeftVec[2]));
        if (!std::isfinite(point3d.x) || !std::isfinite(point3d.y) || !std::isfinite(point3d.z))
        {
            continue;
        }
        if (point3d.z <= 0.0f || point3d.z > 30.0f)
        {
            continue;
        }
```

Replace it with:

```cpp
        cv::Point3f point3dRgb(static_cast<float>(pointLeftVec[0]),
                               static_cast<float>(pointLeftVec[1]),
                               static_cast<float>(pointLeftVec[2]));
        if (!std::isfinite(point3dRgb.x) || !std::isfinite(point3dRgb.y) || !std::isfinite(point3dRgb.z))
        {
            continue;
        }
        if (point3dRgb.z <= 0.0f || point3dRgb.z > 30.0f)
        {
            continue;
        }

        const CameraTransform bodyRgb(rBodyRgb_, tBodyRgb_);
        const cv::Point3f point3dBody = transformPoint(bodyRgb, point3dRgb);
        if (!std::isfinite(point3dBody.x) || !std::isfinite(point3dBody.y) || !std::isfinite(point3dBody.z))
        {
            continue;
        }
```

- [ ] **Step 2: Store body-frame points after reprojection validation**

In the same loop, replace:

```cpp
                // 原始坐标系下的3D位置和矫正坐标系下的重投影误差
                pointsPerDetection[static_cast<size_t>(bestDetIdx)].push_back({point3d, pointReprojError});
```

with:

```cpp
                // body/IMU坐标系下的3D位置和矫正坐标系下的重投影误差
                pointsPerDetection[static_cast<size_t>(bestDetIdx)].push_back({point3dBody, pointReprojError});
```

- [ ] **Step 3: Compile the node target**

Run:

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make -DCMAKE_BUILD_TYPE=Release
```

Expected result: `ros_yolo_node` builds successfully.

- [ ] **Step 4: Manual review checkpoint**

Review only the 3D conversion diff:

```bash
git -C /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS/src/two_stream diff -- src/yolo_node.cpp
```

Confirm the RGB-frame depth gate still checks `point3dRgb.z`, and only the stored/displayed target point is converted to `point3dBody`.

Do not stage or commit.

---

### Task 5: Update Comments and Run Full Verification

**Files:**
- Modify: `ROS/src/two_stream/src/yolo_node.cpp`
- Modify: `ROS/src/two_stream/config/config.yaml`

- [ ] **Step 1: Clarify homography comments**

In `compute_H()`, replace:

```cpp
    //ROS_INFO("Homography initialized for RGB->IR alignment.");
```

with:

```cpp
    // H_ maps RGB image coordinates into IR image coordinates for alignment.
```

Above `const cv::Mat hIrToRgb = cv::Mat(H_.inv(cv::DECOMP_SVD));` in `remapDetectionsToOriginalRgb()`, add:

```cpp
    // Detections are produced in the IR/aligned image frame, so map their box
    // corners back into the original RGB image frame for feature association.
```

- [ ] **Step 2: Run package tests**

Run:

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make run_tests
```

Expected result: all `two_stream` tests pass, including:

```text
test_read_config
test_camera_geometry
test_yolov8_decode
test_detector_postprocess
```

- [ ] **Step 3: Run individual test binaries if the workspace produced them**

Run:

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
./devel/lib/two_stream/test_camera_geometry
./devel/lib/two_stream/test_read_config
```

Expected result: both binaries report all tests passed.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git -C /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS/src/two_stream diff -- include/ros_yolo/camera_geometry.h tests/test_camera_geometry.cpp CMakeLists.txt config/config.yaml tests/test_read_config.cpp include/ros_yolo/yolo_node.h src/yolo_node.cpp
```

Expected result:

- Config comments say `camera_to_body`.
- Config temporary IR translation is `[0.036, -0.03, 0.009]`.
- Tests expect `[0.036, -0.03, 0.009]`.
- Geometry helper derives `T_ir_rgb = [-0.036, 0.03, -0.009]`.
- `compute_H()` and `compute_F()` use `rIrRgb_` and `tIrRgb_`.
- `warpPerspective()` still uses `H_`.
- Detection remapping still uses `H_.inv()`.
- Stored target 3D points are body-frame points.

- [ ] **Step 5: Report verification**

Prepare a short completion report for the user with:

```text
Changed files:
- include/ros_yolo/camera_geometry.h
- tests/test_camera_geometry.cpp
- CMakeLists.txt
- config/config.yaml
- tests/test_read_config.cpp
- include/ros_yolo/yolo_node.h
- src/yolo_node.cpp

Verification:
- catkin_make run_tests: PASS
- ./devel/lib/two_stream/test_camera_geometry: PASS
- ./devel/lib/two_stream/test_read_config: PASS
```

If a command cannot run because dependencies or build artifacts are unavailable, report the exact command and the first relevant error line instead of marking it passed.

Do not stage or commit.
