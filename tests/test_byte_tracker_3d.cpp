#include "ros_yolo/byte_tracker_3d.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
ros_yolo::ByteTrack3DDetection detection(double x1, double y1, double x2, double y2,
                                         double score, double world_x, double world_y)
{
    ros_yolo::ByteTrack3DDetection result;
    result.box = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
    result.score = score;
    result.position = cv::Point2d(world_x, world_y);
    return result;
}
}  // namespace

TEST(ByteTracker3DTest, CreatesTrackAndKeepsIdForWorldMeasurement)
{
    ros_yolo::BYTETracker3D tracker;
    const auto first = tracker.update({detection(10, 20, 50, 80, 0.9, 2.0, 3.0)});
    ASSERT_EQ(first.size(), 1u);
    EXPECT_EQ(first[0].track_id, 1);
    EXPECT_DOUBLE_EQ(first[0].position.x, 2.0);
    EXPECT_DOUBLE_EQ(first[0].position.y, 3.0);

    const auto second = tracker.update({detection(11, 20, 51, 80, 0.9, 2.1, 3.0)});
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second[0].track_id, 1);
    EXPECT_EQ(second[0].detection_index, 0u);
}

TEST(ByteTracker3DTest, WorldPositionDisambiguatesOverlappingBoxes)
{
    ros_yolo::BYTETracker3D tracker;
    const auto first = tracker.update({detection(10, 20, 50, 80, 0.9, 0.0, 0.0),
                                       detection(10, 20, 50, 80, 0.9, 20.0, 0.0)});
    ASSERT_EQ(first.size(), 2u);
    const auto second = tracker.update({detection(10, 20, 50, 80, 0.9, 20.0, 0.0),
                                        detection(10, 20, 50, 80, 0.9, 0.0, 0.0)});
    ASSERT_EQ(second.size(), 2u);
    EXPECT_EQ(second[0].track_id, first[0].track_id);
    EXPECT_EQ(second[0].detection_index, 1u);
    EXPECT_EQ(second[1].track_id, first[1].track_id);
    EXPECT_EQ(second[1].detection_index, 0u);
}

TEST(ByteTracker3DTest, MissingWorldMeasurementUsesPredictionAndResetRestartsIds)
{
    ros_yolo::ByteTrack3DConfig config;
    config.track_buffer = 1;
    ros_yolo::BYTETracker3D tracker(config);
    const auto first = tracker.update({detection(0, 0, 10, 10, 0.9, 1.0, 2.0)});
    ASSERT_EQ(first.size(), 1u);
    EXPECT_TRUE(std::isfinite(tracker.update({detection(0, 0, 10, 10, 0.9,
                                                std::numeric_limits<double>::quiet_NaN(),
                                                std::numeric_limits<double>::quiet_NaN())})[0].position.x));
    EXPECT_TRUE(std::isnan(tracker.update({detection(0, 0, 10, 10, 0.9,
                                               std::numeric_limits<double>::quiet_NaN(),
                                               std::numeric_limits<double>::quiet_NaN())})[0].position.x));
    tracker.reset();
    EXPECT_EQ(tracker.frameId(), 0);
    EXPECT_EQ(tracker.update({detection(0, 0, 10, 10, 0.9, 1.0, 2.0)})[0].track_id, 1);
}

TEST(ByteTracker3DTest, InvalidInputDoesNotAdvanceFrame)
{
    ros_yolo::BYTETracker3D tracker;
    tracker.update({detection(0, 0, 10, 10, 0.9, 1.0, 2.0)});
    const auto invalid = detection(0, 0, 10, 10, 1.2, 1.0, 2.0);
    EXPECT_THROW(tracker.update({invalid}), std::invalid_argument);
    EXPECT_EQ(tracker.frameId(), 1);
}

TEST(ByteTracker3DTest, LowScoresRetainOriginalIndicesAndUseWorldGate)
{
    ros_yolo::BYTETracker3D tracker;
    tracker.update({detection(10, 20, 50, 80, 0.9, 0, 0)});
    const auto result = tracker.update({detection(10, 20, 10, 80, 0.9, 999, 0),
                                        detection(10, 20, 50, 80, 0.15, 0.1, 0)});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].track_id, 1);
    EXPECT_EQ(result[0].detection_index, 1u);
    EXPECT_DOUBLE_EQ(result[0].score, 0.15);
    EXPECT_TRUE(tracker.update({detection(10, 20, 50, 80, 0.15, 100, 0)}).empty());
}

TEST(ByteTracker3DTest, NewTracksRequireConfirmationAndWorldGateApplies)
{
    ros_yolo::BYTETracker3D tracker;
    EXPECT_TRUE(tracker.update({}).empty());
    EXPECT_TRUE(tracker.update({detection(10, 20, 50, 80, 0.9, 0, 0)}).empty());
    EXPECT_TRUE(tracker.update({detection(10, 20, 50, 80, 0.9, 20, 0)}).empty());
    EXPECT_EQ(tracker.removedTrackIds(), std::vector<int>({1}));
    const auto confirmed = tracker.update({detection(10, 20, 50, 80, 0.9, 20.1, 0)});
    ASSERT_EQ(confirmed.size(), 1u);
    EXPECT_EQ(confirmed[0].track_id, 2);
    EXPECT_TRUE(tracker.removedTrackIds().empty());
}

TEST(ByteTracker3DTest, SpatialDedupPreservesDistinctLostTrackForRecovery)
{
    ros_yolo::BYTETracker3D tracker;
    tracker.update({detection(10, 20, 50, 80, 0.9, 0, 0)});
    EXPECT_TRUE(tracker.update({detection(10, 20, 50, 80, 0.9, 20, 0)}).empty());
    const auto recovered = tracker.update({detection(10, 20, 50, 80, 0.9, 0, 0)});
    ASSERT_EQ(recovered.size(), 1u);
    EXPECT_EQ(recovered[0].track_id, 1);
}

TEST(ByteTracker3DTest, LateWorldInitializationExpiryAndReinitializationKeepId)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ros_yolo::ByteTrack3DConfig config;
    config.track_buffer = 1;
    ros_yolo::BYTETracker3D tracker(config);
    auto no_world = detection(10, 20, 50, 80, 0.9, nan, nan);
    EXPECT_TRUE(std::isnan(tracker.update({no_world})[0].position.x));
    auto initialized = tracker.update({detection(10, 20, 50, 80, 0.9, 2, 3)});
    ASSERT_EQ(initialized.size(), 1u);
    EXPECT_DOUBLE_EQ(initialized[0].position.x, 2);
    EXPECT_TRUE(std::isfinite(tracker.update({no_world})[0].position.x));
    EXPECT_TRUE(std::isnan(tracker.update({no_world})[0].position.x));
    const auto recovered = tracker.update({detection(10, 20, 50, 80, 0.9, 100, 200)});
    ASSERT_EQ(recovered.size(), 1u);
    EXPECT_EQ(recovered[0].track_id, 1);
    EXPECT_DOUBLE_EQ(recovered[0].position.x, 100);
    EXPECT_DOUBLE_EQ(recovered[0].position.y, 200);
}

TEST(ByteTracker3DTest, EmptyFramesLoseAndExpireTracks)
{
    ros_yolo::ByteTrack3DConfig config;
    config.track_buffer = 1;
    ros_yolo::BYTETracker3D tracker(config);
    tracker.update({detection(0, 0, 10, 10, 0.9, 0, 0)});
    EXPECT_TRUE(tracker.update({}).empty());
    EXPECT_TRUE(tracker.update({}).empty());
    EXPECT_EQ(tracker.removedTrackIds(), std::vector<int>({1}));
    tracker.update({});  // Python removes newly expired tracks from lost one frame later.
    EXPECT_TRUE(tracker.update({detection(0, 0, 10, 10, 0.9, 0, 0)}).empty());
    const auto next = tracker.update({detection(0, 0, 10, 10, 0.9, 0, 0)});
    ASSERT_EQ(next.size(), 1u);
    EXPECT_EQ(next[0].track_id, 2);
}

TEST(ByteTracker3DTest, PureImageTrackingRecoversAfterMissAndIgnoresClassForMatching)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ros_yolo::BYTETracker3D tracker;
    tracker.update({detection(10, 20, 50, 80, 0.9, nan, nan)});
    tracker.update({detection(11, 20, 51, 80, 0.9, nan, nan)});
    EXPECT_EQ(tracker.update({detection(12, 20, 52, 80, 0.15, nan, nan)})[0].track_id, 1);
    EXPECT_TRUE(tracker.update({}).empty());
    auto det = detection(14, 20, 54, 80, 0.9, nan, nan);
    det.class_id = 3;
    const auto output = tracker.update({det});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].track_id, 1);
    EXPECT_EQ(output[0].class_id, 3);
    EXPECT_TRUE(std::isnan(output[0].position.x));
}

TEST(ByteTracker3DTest, TwoInstancesHaveIndependentIds)
{
    ros_yolo::BYTETracker3D a, b;
    EXPECT_EQ(a.update({detection(0, 0, 10, 10, 0.9, 0, 0)})[0].track_id, 1);
    EXPECT_EQ(b.update({detection(0, 0, 10, 10, 0.9, 0, 0)})[0].track_id, 1);
    b.reset();
    const auto output = a.update({detection(0, 0, 10, 10, 0.9, 0, 0),
                                  detection(100, 0, 110, 10, 0.9, 100, 0)});
    ASSERT_EQ(output.size(), 1u);  // Second target still unconfirmed.
    const auto confirmed = a.update({detection(0, 0, 10, 10, 0.9, 0, 0),
                                     detection(100, 0, 110, 10, 0.9, 100, 0)});
    ASSERT_EQ(confirmed.size(), 2u);
    EXPECT_EQ(confirmed[1].track_id, 2);
}

TEST(ByteTracker3DTest, NonfinitePositionsAreMissingAndInputIsUnchanged)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<ros_yolo::ByteTrack3DDetection> input{
        detection(0, 0, 10, 10, 0.9, nan, 1), detection(20, 0, 30, 10, 0.9, 0, inf)};
    ros_yolo::BYTETracker3D tracker;
    const auto output = tracker.update(input);
    ASSERT_EQ(output.size(), 2u);
    EXPECT_TRUE(std::isnan(output[0].position.x));
    EXPECT_TRUE(std::isnan(output[0].position.y));
    EXPECT_TRUE(std::isnan(output[1].position.x));
    EXPECT_EQ(input[0].position.y, 1);
    EXPECT_TRUE(std::isinf(input[1].position.y));
    const auto row = output[0].toArray();
    EXPECT_EQ(row[0], 0);
    EXPECT_EQ(row[2], 10);
    EXPECT_EQ(row[4], 1);
    EXPECT_EQ(row[7], 0);
    EXPECT_TRUE(std::isnan(row[8]));
}

TEST(ByteTracker3DTest, ValidatesWholeFrameBeforeAnyMutation)
{
    ros_yolo::BYTETracker3D tracker, reference;
    const auto valid = detection(10, 20, 50, 80, 0.9, 2, 3);
    tracker.update({valid});
    reference.update({valid});
    auto invalid = valid;
    invalid.box.x = std::numeric_limits<double>::infinity();
    EXPECT_THROW(tracker.update({valid, invalid}), std::invalid_argument);
    invalid = valid;
    invalid.class_id = -1;
    EXPECT_THROW(tracker.update({invalid}), std::invalid_argument);
    invalid = valid;
    invalid.score = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(tracker.update({invalid}), std::invalid_argument);
    EXPECT_EQ(tracker.frameId(), 1);
    const auto actual = tracker.update({valid})[0].toArray();
    const auto expected = reference.update({valid})[0].toArray();
    EXPECT_EQ(actual, expected);
}

TEST(ByteTracker3DTest, InvalidConfigurationIsRejected)
{
    for (int i = 0; i < 8; ++i)
    {
        ros_yolo::ByteTrack3DConfig config;
        if (i == 0) config.measurement_std = 0;
        if (i == 1) config.acceleration_std = -1;
        if (i == 2) config.initial_velocity_std = std::numeric_limits<double>::infinity();
        if (i == 3) config.world_weight = 1.1;
        if (i == 4) config.world_gate = std::numeric_limits<double>::quiet_NaN();
        if (i == 5) config.track_buffer = -1;
        if (i == 6) config.track_low_threshold = 0.5;
        if (i == 7) config.match_threshold = -0.1;
        EXPECT_THROW({ ros_yolo::BYTETracker3D tracker(config); }, std::invalid_argument);
    }
}

TEST(ByteTracker3DTest, ConfidenceBoundariesAndGeometryFilteringPreserveIndices)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ros_yolo::BYTETracker3D tracker;
    const auto output = tracker.update({detection(0, 0, 0, 10, 0.9, 0, 0),
                                        detection(0, 0, 10, 10, 0.1, nan, nan),
                                        detection(0, 0, 10, 10, 0.25, nan, nan)});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].detection_index, 2u);
    EXPECT_TRUE(tracker.update({detection(0, 0, 10, 10, 0.1, nan, nan)}).empty());
}

TEST(ByteTracker3DTest, FilterValuesMatchReferenceEquations)
{
    ros_yolo::BYTETracker3D tracker;
    tracker.update({detection(10, 20, 50, 80, 0.9, 2, 3)});
    const auto result = tracker.update({detection(11, 20, 51, 80, 0.9, 2.1, 3)});
    ASSERT_EQ(result.size(), 1u);
    // XYAH: initial position variance=36, velocity variance=(60/16)^2.
    EXPECT_NEAR(result[0].box.x, 10 + 59.0625 / 68.0625, 1e-10);
    // World: predicted variance=.09+1+.0025, measurement variance=.09.
    EXPECT_NEAR(result[0].position.x, 2 + 0.1 * 1.0925 / 1.1825, 1e-12);
}

TEST(ByteTracker3DTest, WorldGateUsesPredictedCovariancePlusMeasurementNoiseOnly)
{
    ros_yolo::BYTETracker3D tracker;
    tracker.update({detection(10, 20, 50, 80, 0.9, 0, 0)});
    // d^2=3.35^2/1.1825 > 9.21034; adding detection covariance again
    // would incorrectly admit this pair.
    EXPECT_TRUE(tracker.update({detection(10, 20, 50, 80, 0.9, 3.35, 0)}).empty());
}

TEST(ByteTracker3DTest, GlobalAssignmentRetainsBothTracksWhenGreedyWouldFail)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ros_yolo::ByteTrack3DConfig config;
    config.fuse_score = false;
    config.match_threshold = 0.5;
    ros_yolo::BYTETracker3D tracker(config);
    tracker.update({detection(0, 0, 10, 10, 0.9, nan, nan),
                    detection(3, 0, 13, 10, 0.9, nan, nan)});
    const auto result = tracker.update({detection(1, 0, 11, 10, 0.9, nan, nan),
                                        detection(-1.5, 0, 8.5, 10, 0.9, nan, nan)});
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].track_id, 1);
    EXPECT_EQ(result[0].detection_index, 1u);
    EXPECT_EQ(result[1].track_id, 2);
    EXPECT_EQ(result[1].detection_index, 0u);
}

TEST(ByteTracker3DTest, MissingObservationNeverCorrectsWorldStateToOrigin)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ros_yolo::BYTETracker3D tracker;
    tracker.update({detection(10, 20, 50, 80, 0.9, 10, 20)});
    const auto result = tracker.update({detection(10, 20, 50, 80, 0.9, nan, nan)});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result[0].position.x, 10);
    EXPECT_DOUBLE_EQ(result[0].position.y, 20);
}
