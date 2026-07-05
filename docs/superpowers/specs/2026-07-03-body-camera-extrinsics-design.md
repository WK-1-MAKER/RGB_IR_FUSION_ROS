# Body Camera Extrinsics Design

## Context

`two_stream` currently parses each camera extrinsic from `config/config.yaml` as a
world-to-camera transform:

```text
X_camera = R_camera_world * X_world + t_camera_world
```

That convention matches OpenCV projection math, but it is inconsistent with the
LiDAR/IMU fusion convention used by FAST-LIO, where a sensor extrinsic is the
sensor pose in the IMU/body frame. The camera extrinsics in `two_stream` should
use the same multi-sensor convention before the package is fused with IMU,
LiDAR, or FAST-LIO outputs.

The current code derives a relative RGB-to-IR transform and uses it for:

- RGB-to-IR homography alignment before dual-stream detection.
- Inverse homography mapping of detection boxes back to RGB image coordinates.
- Fundamental matrix filtering.
- Stereo rectification and triangulation.

The migration must keep those internal geometry operations correct while
changing the public config convention.

## Goals

- Make camera extrinsics in `config.yaml` represent `T_body_camera`.
- Use a single convention only. Do not keep legacy `world_to_camera` config
  compatibility.
- Preserve the current RGB-IR geometry behavior with temporary calibration
  values.
- Keep `compute_H()` and `compute_F()` conceptually unchanged by feeding them
  the same relative transform they use today.
- Express final target 3D estimates in body/IMU coordinates.

## Non-Goals

- Do not make the homography reference depth or plane normal configurable in
  this change.
- Do not change the detector input pipeline or YOLO decode behavior.
- Do not introduce ROS TF publication or consumption.
- Do not estimate real camera-to-IMU calibration values. Temporary values will
  be used until measured extrinsics are available.

## Config Convention

The config comment should define camera extrinsics as camera-to-body:

```text
Camera extrinsics use camera_to_body:
X_body = R_body_camera * X_camera + t_body_camera
```

The configured matrices and translations are:

```text
T_body_rgb = (R_body_rgb, t_body_rgb)
T_body_ir  = (R_body_ir,  t_body_ir)
```

For the initial migration, `body` semantically means the real IMU/body frame,
but the current temporary values keep RGB numerically coincident with body:

```text
T_body_rgb = (I, [0, 0, 0])
T_body_ir  = (I, [0.036, -0.03, 0.009])
```

`T_body_ir` is the inverse of the current `T_ir_rgb` behavior. With the temporary
`T_body_rgb = I`, the derived relative transform remains:

```text
X_ir = I * X_rgb + [-0.036, 0.03, -0.009]
```

This preserves the existing alignment and triangulation geometry until real
camera-to-body measurements are provided.

## Transform Derivation

At load time, the node should store the configured body-frame camera poses and
also compute their inverse transforms:

```text
T_rgb_body = inverse(T_body_rgb)
T_ir_body  = inverse(T_body_ir)
```

For each camera:

```text
R_camera_body = R_body_camera^T
t_camera_body = -R_body_camera^T * t_body_camera
```

The stereo geometry needs the RGB-to-IR transform:

```text
T_ir_rgb = T_ir_body * T_body_rgb
```

Expanded:

```text
R_ir_rgb = R_ir_body * R_body_rgb
t_ir_rgb = R_ir_body * t_body_rgb + t_ir_body
```

The existing `r_` and `t_` members currently mean `R_ir_rgb` and `t_ir_rgb`.
The implementation should rename them to `rIrRgb_` and `tIrRgb_`, or otherwise
make that meaning explicit if a smaller patch keeps the old names.

Recommended member naming:

```text
rBodyRgb_, tBodyRgb_
rBodyIr_,  tBodyIr_
rRgbBody_, tRgbBody_
rIrBody_,  tIrBody_
rIrRgb_,   tIrRgb_
```

## Homography and Detection Box Mapping

`H_` should continue to represent an RGB-image-to-IR-image homography:

```text
rgbAligned = warpPerspective(rgbImage, H_, irImage.size())
```

`compute_H()` should still use the derived relative transform:

```text
H = K_ir * (R_ir_rgb - t_ir_rgb * n^T / refDepth) * K_rgb^-1
```

The existing reference depth and plane normal remain:

```text
refDepth = 3.0
n = [0, 0, 1]
```

Detection boxes are produced in the IR/aligned image coordinate frame after the
dual-stream detector sees `(rgbAligned, irImage)`. Mapping boxes back to the
original RGB image should continue to use:

```text
H^-1
```

The implementation should update comments and local names to describe this as
IR/aligned-to-RGB mapping, but the core logic in
`remapDetectionsToOriginalRgb()` should not need a direction change.

## Fundamental Matrix, Rectification, and Triangulation

`compute_F()` should continue to use the derived relative transform:

```text
E = [t_ir_rgb]x * R_ir_rgb
F = K_ir^-T * E * K_rgb^-1
```

`stereoRectify()` should still receive RGB as the left camera, IR as the right
camera, and the derived `R_ir_rgb`, `t_ir_rgb`.

Triangulation should continue to solve in the RGB camera frame internally:

```text
P1 = K_rgb * [I | 0]
P2 = K_ir  * [R_ir_rgb | t_ir_rgb]
```

After triangulation, convert the 3D point from RGB camera coordinates into
body/IMU coordinates before logging, visualizing, or otherwise treating it as
the target position:

```text
X_body = R_body_rgb * X_rgb + t_body_rgb
```

With the temporary `T_body_rgb = I`, the numeric output remains equal to the
old RGB-frame output. Once measured `T_body_rgb` values are supplied, the output
becomes a true body-frame estimate.

## Tests

Update `test_read_config` expectations to match the new temporary config:

```text
ir.translation = [0.036, -0.03, 0.009]
```

Add geometry coverage for the relative transform derivation. A minimal test case
should verify that:

```text
T_body_rgb = (I, [0, 0, 0])
T_body_ir  = (I, [0.036, -0.03, 0.009])
```

derives:

```text
T_ir_rgb = (I, [-0.036, 0.03, -0.009])
```

If the transform derivation remains private to `YoloNode`, add a small helper
with focused tests rather than testing through ROS node construction.

Run, when available:

```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make run_tests
```

At minimum, run the relevant `two_stream` test binary if the workspace has
already been built.

## Risks

- The homography remains a fixed-depth planar approximation. This migration does
  not improve depth-dependent RGB-IR alignment accuracy.
- If future measured `T_body_rgb` or `T_body_ir` values are entered with the old
  OpenCV convention, RGB-IR alignment and triangulation will be wrong. The config
  comment and variable names must make the direction unambiguous.
- Changing final 3D output to body coordinates changes its semantic frame. With
  the temporary identity RGB pose the numbers are unchanged, but future real
  extrinsics will change the displayed/logged coordinates as intended.
