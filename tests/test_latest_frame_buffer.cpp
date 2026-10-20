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
    auto lidar = boost::make_shared<sensor_msgs::PointCloud2>();
    rgb->header.seq = sequence;
    ir->header.seq = sequence;
    lidar->header.seq = sequence;
    return ros_yolo::FramePair(rgb, ir, lidar);
}
}  // namespace

TEST(LatestFrameBufferTest, ReturnsNewestPairWhenPendingPairIsReplaced)
{
    ros_yolo::LatestFrameBuffer buffer;
    buffer.push(makePair(1));
    buffer.push(makePair(2));

    ros_yolo::FramePair result;
    ASSERT_TRUE(buffer.waitAndTake(result));
    ASSERT_TRUE(result.rgb);
    ASSERT_TRUE(result.ir);
    ASSERT_TRUE(result.lidar);
    EXPECT_EQ(result.rgb->header.seq, 2u);
    EXPECT_EQ(result.ir->header.seq, 2u);
    EXPECT_EQ(result.lidar->header.seq, 2u);
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
        {
            received.set_value(pair.rgb->header.seq);
            return;
        }
        received.set_value(0);
    });

    buffer.push(makePair(7));
    const std::future_status status = result.wait_for(std::chrono::seconds(1));
    if (status != std::future_status::ready)
    {
        buffer.stop();
        consumer.join();
        FAIL() << "consumer was not woken by a new frame";
    }
    EXPECT_EQ(result.get(), 7u);
    buffer.stop();
    consumer.join();
}

TEST(LatestFrameBufferTest, StopWakesWaitingConsumer)
{
    ros_yolo::LatestFrameBuffer buffer;
    std::promise<bool> consumed;
    std::future<bool> result = consumed.get_future();
    std::thread consumer([&buffer, &consumed]() {
        ros_yolo::FramePair pair;
        consumed.set_value(buffer.waitAndTake(pair));
    });

    buffer.stop();
    const std::future_status status = result.wait_for(std::chrono::seconds(1));
    if (status != std::future_status::ready)
    {
        consumer.join();
        FAIL() << "consumer was not woken by stop";
    }
    EXPECT_FALSE(result.get());
    consumer.join();
}

TEST(LatestFrameBufferTest, StopDiscardsPendingPair)
{
    ros_yolo::LatestFrameBuffer buffer;
    buffer.push(makePair(9));
    buffer.stop();

    ros_yolo::FramePair result;
    EXPECT_FALSE(buffer.waitAndTake(result));
}

TEST(LatestFrameBufferTest, NewPairReplacesSlotAfterEarlierPairIsTaken)
{
    ros_yolo::LatestFrameBuffer buffer;
    buffer.push(makePair(1));
    ros_yolo::FramePair first;
    ASSERT_TRUE(buffer.waitAndTake(first));
    ASSERT_TRUE(first.rgb);
    EXPECT_EQ(first.rgb->header.seq, 1u);

    buffer.push(makePair(2));
    buffer.push(makePair(3));
    ros_yolo::FramePair newest;
    ASSERT_TRUE(buffer.waitAndTake(newest));
    ASSERT_TRUE(newest.rgb);
    EXPECT_EQ(newest.rgb->header.seq, 3u);
    buffer.stop();
}
