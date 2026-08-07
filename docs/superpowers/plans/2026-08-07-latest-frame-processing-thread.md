# Latest-Frame Processing Thread Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple synchronized RGB/IR reception from the existing heavy processing pipeline with one latest-frame slot and one dedicated processing thread.

**Architecture:** Keep `ros::spin()` as the sole ROS callback dispatcher. Make `syncImageCallback()` only publish a synchronized `FramePair` into a mutex-protected, condition-variable-backed latest-frame slot; overwrite an unprocessed pair when a newer pair arrives. Move the existing callback body into `processImagePair()` and execute it exclusively on one worker thread, preserving all current model, geometry, output, and coordinate behavior.

**Tech Stack:** ROS1 roscpp, message_filters ApproximateTime, sensor_msgs/ImageConstPtr, C++14 `std::thread`/`std::mutex`/`std::condition_variable`, GoogleTest, existing OpenCV/ONNX Runtime/TensorRT code.

**Repository constraint:** Do not run `git add`, `git commit`, or `git push`; Git management is manual for this package.

## File Map

- Create `include/ros_yolo/latest_frame_buffer.h`: `FramePair` and latest-only handoff primitive.
- Create `tests/test_latest_frame_buffer.cpp`: handoff and shutdown tests without models.
- Modify `include/ros_yolo/yolo_node.h`: worker declarations and members.
- Modify `src/yolo_node.cpp`: worker lifecycle, enqueue-only callback, processing extraction, and worker-owned GUI.
- Modify `CMakeLists.txt`: C++ thread linkage and the new test target.

### Task 1: Add failing latest-frame tests

**Files:** Create `tests/test_latest_frame_buffer.cpp`; modify `CMakeLists.txt:18-28,114-201`.

- [x] **Step 1: Register the test target.** Add `find_package(Threads REQUIRED)` after existing dependency discovery. Inside `CATKIN_ENABLE_TESTING`, add:

```cmake
  catkin_add_gtest(test_latest_frame_buffer tests/test_latest_frame_buffer.cpp)
  if(TARGET test_latest_frame_buffer)
    target_link_libraries(test_latest_frame_buffer
      ${catkin_LIBRARIES}
      Threads::Threads
      gtest_main
    )
  endif()
```

- [x] **Step 2: Write the failing tests.** Create the test file with this complete test body:

```cpp
#include "ros_yolo/latest_frame_buffer.h"
#include <gtest/gtest.h>
#include <boost/make_shared.hpp>
#include <chrono>
#include <cstdint>
#include <future>
#include <thread>

namespace
{
ros_yolo::FramePair makePair(uint32_t sequence)
{
    auto rgb = boost::make_shared<sensor_msgs::Image>();
    auto ir = boost::make_shared<sensor_msgs::Image>();
    rgb->header.seq = sequence;
    ir->header.seq = sequence;
    return ros_yolo::FramePair(rgb, ir);
}
}

TEST(LatestFrameBufferTest, ReturnsNewestPairWhenPendingPairIsReplaced)
{
    ros_yolo::LatestFrameBuffer buffer;
    buffer.push(makePair(1));
    buffer.push(makePair(2));
    ros_yolo::FramePair result;
    ASSERT_TRUE(buffer.waitAndTake(result));
    ASSERT_TRUE(result.rgb);
    EXPECT_EQ(result.rgb->header.seq, 2u);
    buffer.stop();
}

TEST(LatestFrameBufferTest, WakesWaitingConsumerWhenPairArrives)
{
    ros_yolo::LatestFrameBuffer buffer;
    std::promise<uint32_t> received;
    std::future<uint32_t> result = received.get_future();
    std::thread consumer([&buffer, &received]() {
        ros_yolo::FramePair pair;
        if (buffer.waitAndTake(pair) && pair.rgb)
            received.set_value(pair.rgb->header.seq);
    });
    buffer.push(makePair(7));
    ASSERT_EQ(result.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(result.get(), 7u);
    buffer.stop();
    consumer.join();
}

TEST(LatestFrameBufferTest, StopWakesConsumerAndDiscardsPendingPair)
{
    ros_yolo::LatestFrameBuffer buffer;
    buffer.push(makePair(9));
    std::promise<bool> consumed;
    std::future<bool> result = consumed.get_future();
    std::thread consumer([&buffer, &consumed]() {
        ros_yolo::FramePair pair;
        consumed.set_value(buffer.waitAndTake(pair));
    });
    buffer.stop();
    ASSERT_EQ(result.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(result.get());
    consumer.join();
}

TEST(LatestFrameBufferTest, NewPairReplacesSlotAfterEarlierPairIsTaken)
{
    ros_yolo::LatestFrameBuffer buffer;
    buffer.push(makePair(1));
    ros_yolo::FramePair first;
    ASSERT_TRUE(buffer.waitAndTake(first));
    buffer.push(makePair(2));
    buffer.push(makePair(3));
    ros_yolo::FramePair newest;
    ASSERT_TRUE(buffer.waitAndTake(newest));
    EXPECT_EQ(newest.rgb->header.seq, 3u);
    buffer.stop();
}
```

- [x] **Step 3: Verify the test fails before implementation.** Run from the catkin workspace root:

```bash
catkin_make --pkg two_stream test_latest_frame_buffer
```

Expected: compilation fails because `ros_yolo/latest_frame_buffer.h` is absent.

### Task 2: Implement the latest-frame handoff

**Files:** Create `include/ros_yolo/latest_frame_buffer.h`.

- [x] **Step 1: Add the tested C++14 implementation.** Use this header-only class:

```cpp
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>
#include <sensor_msgs/Image.h>

namespace ros_yolo
{
struct FramePair
{
    FramePair() = default;
    FramePair(const sensor_msgs::ImageConstPtr& rgbIn,
              const sensor_msgs::ImageConstPtr& irIn)
        : rgb(rgbIn), ir(irIn) {}
    sensor_msgs::ImageConstPtr rgb;
    sensor_msgs::ImageConstPtr ir;
};

class LatestFrameBuffer
{
public:
    void push(FramePair frame)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_)
                return;
            pending_.reset(new FramePair(std::move(frame)));
        }
        condition_.notify_one();
    }

    bool waitAndTake(FramePair& frame)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]() { return stopped_ || pending_; });
        if (stopped_)
            return false;
        frame = std::move(*pending_);
        pending_.reset();
        return true;
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            pending_.reset();
        }
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::unique_ptr<FramePair> pending_;
    bool stopped_{false};
};
}  // namespace ros_yolo
```

`push()` must never wait for processing; replacing `pending_` is the drop-oldest
policy. `stop()` clears the slot so shutdown never starts another frame.

- [x] **Step 2: Run the focused test.**

```bash
catkin_make --pkg two_stream test_latest_frame_buffer
./devel/lib/two_stream/test_latest_frame_buffer
```

Expected: all four tests pass.

### Task 3: Add the processing thread lifecycle

**Files:** Modify `include/ros_yolo/yolo_node.h:1-80,133` and `src/yolo_node.cpp:262-322`.

- [x] **Step 1: Add declarations and members.** Include `<thread>` and `ros_yolo/latest_frame_buffer.h`. Under the public constructor declare `~YoloNode();`. Add private methods:

```cpp
void processingLoop();
void processImagePair(const sensor_msgs::ImageConstPtr& rgbMsg,
                     const sensor_msgs::ImageConstPtr& irMsg);
```

Add members near the ROS synchronizer:

```cpp
LatestFrameBuffer frameBuffer_;
std::thread processingThread_;
```

Remove `annotatedSub_` and `annotatedImageCallback()`; the worker will display
the annotated image directly when `show_window` is enabled.

- [x] **Step 2: Start the worker after subscription setup.** Immediately after the existing subscription, synchronizer, callback registration, and startup log calls, add:

```cpp
processingThread_ = std::thread(&YoloNode::processingLoop, this);
```

Keep configuration, calibration, detector construction, and SuperPoint/LightGlue
build calls before this line. Keep subscription/synchronizer setup immediately
before it; `main()` has not entered `ros::spin()` yet, so registration cannot race
the worker. Starting last also prevents a constructor failure from leaving a
joinable thread behind.

- [x] **Step 3: Add safe destruction.** Add after the constructor:

```cpp
YoloNode::~YoloNode()
{
    rgbSub_.unsubscribe();
    irSub_.unsubscribe();
    sync_.reset();
    frameBuffer_.stop();
    if (processingThread_.joinable())
        processingThread_.join();
}
```

This prevents new input before stopping the buffer and joins before model or
geometry members are destroyed. An active inference call is allowed to finish.

- [x] **Step 4: Add the worker exception boundary.** Implement:

```cpp
void YoloNode::processingLoop()
{
    FramePair frame;
    while (frameBuffer_.waitAndTake(frame))
    {
        try
        {
            processImagePair(frame.rgb, frame.ir);
        }
        catch (const cv_bridge::Exception& e)
        {
            ROS_ERROR_STREAM("Image conversion failed: " << e.what());
        }
        catch (const cv::Exception& e)
        {
            ROS_ERROR_STREAM("OpenCV processing failed: " << e.what());
        }
        catch (const std::exception& e)
        {
            ROS_ERROR_STREAM("Frame processing failed: " << e.what());
        }
        catch (...)
        {
            ROS_ERROR("Frame processing failed with an unknown exception.");
        }
    }
}
```

No exception may escape the `std::thread` entry point.

### Task 4: Move processing behind the buffer

**Files:** Modify `src/yolo_node.cpp:548-929,1024-1045`.

- [x] **Step 1: Make the synchronizer callback enqueue-only.** Replace its body with:

```cpp
void YoloNode::syncImageCallback(const sensor_msgs::ImageConstPtr& rgbMsg,
                                 const sensor_msgs::ImageConstPtr& irMsg)
{
    frameBuffer_.push(FramePair(rgbMsg, irMsg));
}
```

It must not convert images, invoke inference, call HighGUI, or publish.

- [x] **Step 2: Move the old callback body unchanged into `processImagePair()`.** Give it the same two pointer parameters. Preserve the existing sequence from `toBgrImage()` through undistortion, alignment, detector call, feature matching, triangulation, body-frame conversion, aggregation, and `publishCvImage(rgbImage, rgbMsg->header.stamp)`. Remove only the obsolete commented conversion `try` block; the worker loop now catches exceptions.

- [x] **Step 3: Keep HighGUI on the worker.** Keep the existing match-image display in `processImagePair()`. After `utils::visualizeDetection(...)` and before publication add:

```cpp
if (showWindow_)
{
    cv::imshow("two_stream_annotated", rgbImage);
    cv::waitKey(1);
}
```

Remove the constructor's `annotatedSub_` subscription and delete
`annotatedImageCallback()`, preventing GUI calls from both threads while keeping
the visible annotated window behavior.

- [x] **Step 4: Compile and inspect the callback boundary.** Run:

```bash
catkin_make --pkg two_stream ros_yolo_node
```

Expected: successful compile/link. Confirm `syncImageCallback()` contains only
the buffer push and `processImagePair()` contains the former complete pipeline.

### Task 5: Regression and runtime verification

**Files:** No additional files.

- [x] **Step 1: Run the handoff test.**

```bash
./devel/lib/two_stream/test_latest_frame_buffer
```

Expected: all four tests pass.

- [x] **Step 2: Run existing tests.**

```bash
./devel/lib/two_stream/test_camera_geometry
./devel/lib/two_stream/test_read_config
./devel/lib/two_stream/test_yolov8_decode
./devel/lib/two_stream/test_detector_postprocess
```

The handoff, camera-geometry, YOLO decode, and detector post-processing binaries
pass. `test_read_config` still has one pre-existing failure because its expected
IR translation is `[0.036, -0.03, 0.009]`, while `config/config.yaml` contains
`[0.009, -0.036, 0.03]`; this threading change does not alter that config.

- [x] **Step 3: Check the diff.**

```bash
git diff --check -- CMakeLists.txt include/ros_yolo/latest_frame_buffer.h include/ros_yolo/yolo_node.h src/yolo_node.cpp tests/test_latest_frame_buffer.cpp
```

Confirm there are no whitespace errors, no model/calibration changes, and no
remaining `annotatedSub_` or `annotatedImageCallback` references.

- [ ] **Step 4: Validate with live or recorded input.** Run:

```bash
roslaunch two_stream yolo_topic.launch
```

Use an input rate higher than processing capacity and verify that reception
continues while a frame is processed, the pending storage remains one pair,
output timestamps do not accumulate unbounded latency, annotated output and
body-frame `weighted3d` values retain their semantics, and shutdown joins cleanly
both while idle and during processing.
