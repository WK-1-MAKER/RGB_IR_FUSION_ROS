# Independent C++ BYTETracker3D

Approved scope: reproduce `ultralytics/trackers/byte_tracker_3d.py` and its
ByteTrack dependencies as a standalone C++14 header/source pair. The caller
constructs the input. Do not wire it into YoloNode, Detection, localization,
ROS messages, coordinate transformations, or z tracking. Do not commit code.

## Interface

`ByteTrack3DDetection` holds a double precision OpenCV rectangle, confidence,
integer class ID, and optional world XY (NaN by default). Each update receives
one vector, including an empty vector for an empty frame. Detection indices are
assigned from the input vector before confidence/geometry filtering.
`ByteTrack3DResult` contains the filtered rectangle, ID, confidence, class,
original detection index, and filtered/predicted world XY. `toArray()` exposes
the same ten columns as Python. IDs belong to an instance; reset restarts at 1.
Pimpl keeps track/filter/assignment details private and prevents accidental
copying of trajectory state. Expose latest removed IDs for lifecycle inspection.

## Behavior

Image filter: `[cx,cy,a,h,vx,vy,va,vh]`, original height-scaled process and
measurement covariances. Lost trajectories zero vh before image prediction.
Unconfirmed image states are not predicted. World filter: `[X,Y,Vx,Vy]`, dt=1,
constant acceleration process covariance, Joseph measurement covariance update.
All tracked, lost and unconfirmed world states predict once per frame and
expire when frames since the last real world observation exceed track_buffer.

High association uses IoU and optional score fusion; low association uses IoU
without score fusion. All three association stages fuse available world costs
and hard reject squared Mahalanobis distances above world_gate. Missing world
observations preserve the full image cost. Use global thresholded assignment,
with dummy unmatched costs equal to half the association threshold (equivalent
to LAPJV cost_limit), rather than greedy assignment or filtering a forced match.
Low/unconfirmed thresholds remain 0.5/0.7. New tracks after frame one need a
second high-score match for confirmation. Deduplicate tracked/lost pairs by raw
IoU distance <0.15, except pairs whose world states conflict. Retain the Python
pool ordering and removal timing, including its one-update delay before newly
removed tracks leave the lost pool; keep at most 1000 historical removed tracks.

Validate every input row before state mutation; confidence must be finite in
[0,1], class nonnegative, rectangle coordinates/extents finite. Drop non-positive
extents. A partly invalid world observation is entirely missing. Config noise
and gate are positive finite; weights/thresholds in [0,1], low <= high, buffer >=0.

## Verification

GTests cover 2D continuity, overlapping world separation, low-score association,
confirmation, missing/late/expired world measurements, reactivation, dedup,
thresholded global assignment, original row indices, reset, validation, and
numeric reference results. Compile independently of ROS/inference, build the
catkin library/test targets, and compare deterministic frame sequences against
the original Python tracker. Check existing user edits remain untouched.

## Usage

```cpp
#include "ros_yolo/byte_tracker_3d.h"

ros_yolo::BYTETracker3D tracker;
ros_yolo::ByteTrack3DDetection detection;
detection.box = cv::Rect2d(10, 20, 40, 60);
detection.score = 0.9;
detection.class_id = 0;
detection.position = cv::Point2d(2.0, 3.0);  // Optional fixed-reference XY, meters.
auto tracks = tracker.update({detection});
// Subsequent frames reuse tracker, including tracker.update({}) when empty.
```

Link the `byte_tracker_3d` library. Leave `position` at its default NaN when no
world observation is available. The tracker does not perform coordinate
transforms; the caller supplies target XY values in the fixed reference frame.
Each output's `detection_index` refers to the input vector of the current update;
`position` is the filtered/predicted XY.

## Validation results

- Standalone C++14 compile with `-Wall -Wextra -pedantic`: no warnings;
  all 19 tracker GTests pass.
- ROS catkin package build and all five enabled test suites pass (45 tests).
- Original Python versus C++: five deterministic sequences, 490 frames,
  including empty frames, overlapping boxes, missing/expired observations,
  mixed confidence, shuffled detection order, and lost-track recovery. IDs,
  original detection indices, scores, classes and per-frame removed IDs match.
  Maximum box-coordinate error: 2.184e-5 pixels (Python image storage includes
  float32 rounding); maximum world-XY error: 2.221e-16 meters.
- Independent read-only review found no important defects.
