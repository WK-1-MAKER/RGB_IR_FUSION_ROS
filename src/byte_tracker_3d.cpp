#include "ros_yolo/byte_tracker_3d.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace ros_yolo
{
namespace
{
using Vec2 = Eigen::Vector2d;
using Vec4 = Eigen::Vector4d;
using Vec8 = Eigen::Matrix<double, 8, 1>;
using Mat2 = Eigen::Matrix2d;
using Mat4 = Eigen::Matrix4d;
using Mat8 = Eigen::Matrix<double, 8, 8>;

enum class State { New, Tracked, Lost, Removed };

struct WorldFilter
{
    explicit WorldFilter(const ByteTrack3DConfig& c)
    {
        measurement_noise = c.measurement_std * c.measurement_std * Mat2::Identity();
        initial_covariance.setZero();
        initial_covariance.diagonal() << c.measurement_std * c.measurement_std,
            c.measurement_std * c.measurement_std, c.initial_velocity_std * c.initial_velocity_std,
            c.initial_velocity_std * c.initial_velocity_std;
    }

    Mat4 motionFor(double dt) const
    {
        Mat4 result = Mat4::Identity();
        result(0, 2) = result(1, 3) = dt;
        return result;
    }

    Mat4 processNoiseFor(double dt, double accelerationStd) const
    {
        const Eigen::Matrix<double, 4, 2> acceleration =
            (Eigen::Matrix<double, 4, 2>() << 0.5 * dt * dt, 0.0,
                                               0.0, 0.5 * dt * dt,
                                               dt, 0.0,
                                               0.0, dt).finished();
        return accelerationStd * accelerationStd * (acceleration * acceleration.transpose());
    }

    Mat2 measurement_noise;
    Mat4 initial_covariance;
};

struct Track
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Track(const ByteTrack3DDetection& detection, std::size_t index)
        : box(detection.box), score(detection.score), class_id(detection.class_id), detection_index(index)
    {
        has_world_observation = std::isfinite(detection.position.x) && std::isfinite(detection.position.y);
        if (has_world_observation) world_observation << detection.position.x, detection.position.y;
    }

    cv::Rect2d currentBox() const
    {
        if (!image_initialized) return box;
        const double height = mean(3), width = mean(2) * height;
        return {mean(0) - width * 0.5, mean(1) - height * 0.5, width, height};
    }

    Vec4 measurement() const
    {
        return {box.x + box.width * 0.5, box.y + box.height * 0.5, box.width / box.height, box.height};
    }

    void updateWorld(const Track& detection, const WorldFilter& filter, int frame, double now)
    {
        if (!detection.has_world_observation) return;
        if (!world_initialized)
        {
            world_mean << detection.world_observation, 0.0, 0.0;
            world_covariance = filter.initial_covariance;
            world_initialized = true;
        }
        else
        {
            const Mat2 innovation = world_covariance.topLeftCorner<2, 2>() + filter.measurement_noise;
            const Eigen::Matrix<double, 4, 2> gain =
                innovation.ldlt().solve(world_covariance.leftCols<2>().transpose()).transpose();
            world_mean += gain * (detection.world_observation - world_mean.head<2>());
            Mat4 residual = Mat4::Identity();
            residual.leftCols<2>() -= gain;
            // Joseph form maintains a symmetric positive covariance during repeated corrections.
            world_covariance = residual * world_covariance * residual.transpose() +
                               gain * filter.measurement_noise * gain.transpose();
            world_covariance = ((world_covariance + world_covariance.transpose()) * 0.5).eval();
        }
        world_frame = frame;
        world_time = now;
    }

    void activate(int new_id, int frame, double now, const ByteTrack3DConfig& config,
                  const WorldFilter& filter)
    {
        id = new_id;
        mean.head<4>() = measurement();
        mean.tail<4>().setZero();
        Vec8 noise;
        const double height = box.height;
        noise << height / 10.0, height / 10.0, 1e-2, height / 10.0,
            height * config.initial_image_velocity_scale, height * config.initial_image_velocity_scale,
            config.aspect_acceleration_std, height * config.initial_image_velocity_scale;
        covariance = noise.cwiseProduct(noise).asDiagonal();
        image_initialized = true;
        state = State::Tracked;
        activated = frame == 1;
        frame_id = start_frame = frame;
        last_update_time = now;
        updateWorld(*this, filter, frame, now);
    }

    void predictImage(double dt, const ByteTrack3DConfig& config)
    {
        if (state != State::Tracked) mean(7) = 0.0;
        Mat8 motion = Mat8::Identity();
        for (int i = 0; i < 4; ++i) motion(i, i + 4) = dt;
        const double height = mean(3);
        mean = (motion * mean).eval();
        covariance = (motion * covariance * motion.transpose()).eval();
        Mat8 processNoise = Mat8::Zero();
        for (int i = 0; i < 4; ++i)
        {
            const double scale = i == 2 ? config.aspect_acceleration_std
                                         : config.image_acceleration_std * std::max(height, 1.0);
            const double spectralDensity = scale * scale;
            processNoise(i, i) = spectralDensity * dt * dt * dt / 3.0;
            processNoise(i, i + 4) = processNoise(i + 4, i) = spectralDensity * dt * dt / 2.0;
            processNoise(i + 4, i + 4) = spectralDensity * dt;
        }
        covariance += processNoise;
    }

    void predictWorld(const WorldFilter& filter, double now, double timeout, double dt, double accelerationStd)
    {
        if (!world_initialized) return;
        if (now - world_time > timeout)
        {
            world_initialized = false;
            world_frame = -1;
            return;
        }
        const Mat4 motion = filter.motionFor(dt);
        world_mean = (motion * world_mean).eval();
        world_covariance = (motion * world_covariance * motion.transpose() +
                            filter.processNoiseFor(dt, accelerationStd)).eval();
    }

    void update(const Track& detection, int frame, double now, const WorldFilter& filter)
    {
        const double height = mean(3);
        Vec4 noise;
        noise << height / 20.0, height / 20.0, 1e-1, height / 20.0;
        Mat4 innovation = covariance.topLeftCorner<4, 4>();
        innovation.diagonal() += noise.cwiseProduct(noise);
        const Eigen::Matrix<double, 8, 4> gain =
            innovation.ldlt().solve(covariance.leftCols<4>().transpose()).transpose();
        mean += gain * (detection.measurement() - mean.head<4>());
        covariance -= gain * innovation * gain.transpose();
        tracklet_len = state == State::Tracked ? tracklet_len + 1 : 0;
        state = State::Tracked;
        activated = true;
        frame_id = frame;
        last_update_time = now;
        score = detection.score;
        class_id = detection.class_id;
        detection_index = detection.detection_index;
        updateWorld(detection, filter, frame, now);
    }

    cv::Rect2d box;
    double score;
    int class_id;
    std::size_t detection_index;
    int id = 0;
    int frame_id = 0;
    int start_frame = 0;
    int world_frame = -1;
    double world_time = -std::numeric_limits<double>::infinity();
    double last_update_time = -std::numeric_limits<double>::infinity();
    int tracklet_len = 0;
    bool activated = false;
    State state = State::New;
    bool image_initialized = false;
    Vec8 mean = Vec8::Zero();
    Mat8 covariance = Mat8::Zero();
    bool has_world_observation = false;
    Vec2 world_observation = Vec2::Zero();
    bool world_initialized = false;
    Vec4 world_mean = Vec4::Zero();
    Mat4 world_covariance = Mat4::Zero();
};

// Shared identity across pools mirrors Python's lists of the same STrack objects.
// Allocate with Track's aligned operator new for Eigen under C++14.
using TrackPtr = std::shared_ptr<Track>;
using Tracks = std::vector<TrackPtr>;
using Costs = std::vector<std::vector<double>>;

double iou(const cv::Rect2d& a, const cv::Rect2d& b)
{
    const double width = std::max(0.0, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
    const double height = std::max(0.0, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
    const double intersection = width * height;
    const double union_area = a.area() + b.area() - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

Tracks join(const Tracks& a, const Tracks& b)
{
    Tracks result = a;
    std::unordered_set<int> ids;
    for (const auto& track : a) ids.insert(track->id);
    for (const auto& track : b) if (ids.insert(track->id).second) result.push_back(track);
    return result;
}

Tracks subtract(const Tracks& a, const Tracks& b)
{
    std::unordered_set<int> ids;
    for (const auto& track : b) ids.insert(track->id);
    Tracks result;
    for (const auto& track : a) if (!ids.count(track->id)) result.push_back(track);
    return result;
}

struct Assignment
{
    std::vector<std::pair<std::size_t, std::size_t>> matches;
    std::vector<std::size_t> unmatched_tracks, unmatched_detections;
};

// Hungarian assignment with explicit unmatched choices. Two dummy assignments
// cost threshold in total, matching LAPJV's cost_limit objective.
Assignment assign(const Costs& costs, std::size_t columns, double threshold)
{
    const std::size_t rows = costs.size();
    Assignment result;
    std::vector<bool> row_used(rows, false), column_used(columns, false);
    if (rows && columns)
    {
        const std::size_t size = rows + columns;
        Costs matrix(size, std::vector<double>(size, 0.0));
        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = 0; j < size; ++j)
                matrix[i][j] = (i < rows && j < columns)
                    ? (std::isfinite(costs[i][j]) && costs[i][j] <= threshold ? costs[i][j] : threshold + 1e6)
                    : ((i < rows || j < columns) ? threshold * 0.5 : 0.0);

        std::vector<double> u(size + 1), v(size + 1);
        std::vector<std::size_t> p(size + 1), way(size + 1);
        for (std::size_t i = 1; i <= size; ++i)
        {
            p[0] = i;
            std::size_t j0 = 0;
            std::vector<double> minv(size + 1, std::numeric_limits<double>::infinity());
            std::vector<bool> used(size + 1, false);
            do
            {
                used[j0] = true;
                const std::size_t i0 = p[j0];
                double delta = std::numeric_limits<double>::infinity();
                std::size_t j1 = 0;
                for (std::size_t j = 1; j <= size; ++j)
                    if (!used[j])
                    {
                        const double current = matrix[i0 - 1][j - 1] - u[i0] - v[j];
                        if (current < minv[j]) { minv[j] = current; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }
                for (std::size_t j = 0; j <= size; ++j)
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else minv[j] -= delta;
                j0 = j1;
            } while (p[j0] != 0);
            do
            {
                const std::size_t j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }
        for (std::size_t j = 1; j <= columns; ++j)
            if (p[j] >= 1 && p[j] <= rows && matrix[p[j] - 1][j - 1] <= threshold)
            {
                result.matches.emplace_back(p[j] - 1, j - 1);
                row_used[p[j] - 1] = column_used[j - 1] = true;
            }
        // Python applies LAPJV matches in ascending track-row order.
        std::sort(result.matches.begin(), result.matches.end());
    }
    for (std::size_t i = 0; i < rows; ++i) if (!row_used[i]) result.unmatched_tracks.push_back(i);
    for (std::size_t j = 0; j < columns; ++j) if (!column_used[j]) result.unmatched_detections.push_back(j);
    return result;
}

void validateConfig(const ByteTrack3DConfig& config)
{
    const auto positive = [](double v) { return std::isfinite(v) && v > 0.0; };
    const auto probability = [](double v) { return std::isfinite(v) && v >= 0.0 && v <= 1.0; };
    if (!positive(config.measurement_std) || !positive(config.acceleration_std) ||
        !positive(config.initial_velocity_std) || !positive(config.world_gate) ||
        !positive(config.image_acceleration_std) || !positive(config.aspect_acceleration_std) ||
        !positive(config.initial_image_velocity_scale) ||
        !probability(config.world_weight) || !probability(config.track_high_threshold) ||
        !probability(config.track_low_threshold) || config.track_low_threshold > config.track_high_threshold ||
        !probability(config.new_track_threshold) || !positive(config.match_threshold) ||
        config.match_threshold > 1.0 || config.track_buffer < 0)
        throw std::invalid_argument("invalid BYTETracker3D configuration");
    if (config.track_buffer_seconds < -1.0 || config.world_measurement_timeout_seconds < -1.0 ||
        (config.track_buffer_seconds >= 0.0 && !std::isfinite(config.track_buffer_seconds)) ||
        (config.world_measurement_timeout_seconds >= 0.0 && !std::isfinite(config.world_measurement_timeout_seconds)))
        throw std::invalid_argument("invalid BYTETracker3D expiry configuration");
}
}  // namespace

struct BYTETracker3D::Impl
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit Impl(const ByteTrack3DConfig& c) : config(c), world_filter(c) {}

    Costs costs(const Tracks& tracks, const Tracks& detections, bool fuse_score) const
    {
        Costs result(tracks.size(), std::vector<double>(detections.size()));
        for (std::size_t i = 0; i < tracks.size(); ++i)
            for (std::size_t j = 0; j < detections.size(); ++j)
            {
                const auto& track = *tracks[i];
                const auto& detection = *detections[j];
                const double overlap = iou(track.currentBox(), detection.currentBox());
                double cost = 1.0 - overlap * (fuse_score ? detection.score : 1.0);
                if (track.world_initialized && detection.has_world_observation)
                {
                    const Vec2 delta = detection.world_observation - track.world_mean.head<2>();
                    const Mat2 covariance = track.world_covariance.topLeftCorner<2, 2>() + world_filter.measurement_noise;
                    const double distance = delta.dot(covariance.ldlt().solve(delta));
                    cost = distance > config.world_gate ? std::numeric_limits<double>::infinity()
                        : (1.0 - config.world_weight) * cost + config.world_weight * (distance / config.world_gate);
                }
                result[i][j] = cost;
            }
        return result;
    }

    void deduplicate()
    {
        std::vector<bool> drop_tracked(tracked.size(), false), drop_lost(lost.size(), false);
        for (std::size_t i = 0; i < tracked.size(); ++i)
            for (std::size_t j = 0; j < lost.size(); ++j)
            {
                const Track& a = *tracked[i];
                const Track& b = *lost[j];
                if (1.0 - iou(a.currentBox(), b.currentBox()) >= 0.15) continue;
                if (a.world_initialized && b.world_initialized)
                {
                    const Vec2 delta = a.world_mean.head<2>() - b.world_mean.head<2>();
                    const Mat2 covariance = a.world_covariance.topLeftCorner<2, 2>() +
                                           b.world_covariance.topLeftCorner<2, 2>() + world_filter.measurement_noise;
                    if (delta.dot(covariance.ldlt().solve(delta)) > config.world_gate) continue;
                }
                if (a.frame_id - a.start_frame > b.frame_id - b.start_frame) drop_lost[j] = true;
                else drop_tracked[i] = true;
            }
        Tracks retained_tracked, retained_lost;
        for (std::size_t i = 0; i < tracked.size(); ++i) if (!drop_tracked[i]) retained_tracked.push_back(tracked[i]);
        for (std::size_t i = 0; i < lost.size(); ++i) if (!drop_lost[i]) retained_lost.push_back(lost[i]);
        tracked = std::move(retained_tracked);
        lost = std::move(retained_lost);
    }

    ByteTrack3DConfig config;
    WorldFilter world_filter;
    int frame_id = 0;
    int next_id = 1;
    double last_timestamp = -std::numeric_limits<double>::infinity();
    Tracks tracked, lost, removed;
    std::vector<int> removed_this_frame;
};

std::array<double, 10> ByteTrack3DResult::toArray() const
{
    return {box.x, box.y, box.x + box.width, box.y + box.height, static_cast<double>(track_id), score,
            static_cast<double>(class_id), static_cast<double>(detection_index), position.x, position.y};
}

BYTETracker3D::BYTETracker3D(const ByteTrack3DConfig& config)
{
    validateConfig(config);
    impl_.reset(new Impl(config));
}

BYTETracker3D::~BYTETracker3D() = default;

std::vector<ByteTrack3DResult> BYTETracker3D::update(const std::vector<ByteTrack3DDetection>& input)
{
    const double timestamp = std::isfinite(impl_->last_timestamp) ? impl_->last_timestamp + 1.0 : 1.0;
    return update(input, timestamp);
}

std::vector<ByteTrack3DResult> BYTETracker3D::update(
    const std::vector<ByteTrack3DDetection>& input, double timestamp_seconds)
{
    if (!std::isfinite(timestamp_seconds) || timestamp_seconds <= 0.0)
        throw std::invalid_argument("BYTETracker3D timestamp must be finite and positive");
    // Validate the complete frame before changing frame IDs, trajectories, or removal diagnostics.
    for (const auto& d : input)
        if (!std::isfinite(d.box.x) || !std::isfinite(d.box.y) || !std::isfinite(d.box.width) ||
            !std::isfinite(d.box.height) || !std::isfinite(d.score) || d.score < 0.0 || d.score > 1.0 || d.class_id < 0)
            throw std::invalid_argument("invalid BYTETracker3D detection");

    auto& tracker = *impl_;
    if (std::isfinite(tracker.last_timestamp) && timestamp_seconds <= tracker.last_timestamp)
        throw std::invalid_argument("BYTETracker3D timestamps must increase strictly");
    const double dt = std::isfinite(tracker.last_timestamp) ? timestamp_seconds - tracker.last_timestamp : 0.0;
    ++tracker.frame_id;
    tracker.last_timestamp = timestamp_seconds;
    tracker.removed_this_frame.clear();
    const double trackLifetime = tracker.config.track_buffer_seconds >= 0.0
        ? tracker.config.track_buffer_seconds : static_cast<double>(tracker.config.track_buffer);
    const double worldLifetime = tracker.config.world_measurement_timeout_seconds >= 0.0
        ? tracker.config.world_measurement_timeout_seconds : trackLifetime;

    Tracks newly_removed;
    const auto removeExpired = [&](Tracks& tracks) {
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](const TrackPtr& track) {
            if (timestamp_seconds - track->last_update_time > trackLifetime)
            {
                track->state = State::Removed;
                newly_removed.push_back(track);
                return true;
            }
            if (track->world_initialized && timestamp_seconds - track->world_time > worldLifetime)
            {
                track->world_initialized = false;
                track->world_frame = -1;
            }
            return false;
        }), tracks.end());
    };
    removeExpired(tracker.tracked);
    removeExpired(tracker.lost);

    Tracks high, low;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const auto& detection = input[i];
        if (detection.box.width <= 0.0 || detection.box.height <= 0.0) continue;
        if (detection.score >= tracker.config.track_high_threshold)
            high.emplace_back(new Track(detection, i));
        else if (detection.score > tracker.config.track_low_threshold)
            low.emplace_back(new Track(detection, i));
    }

    Tracks unconfirmed, confirmed;
    for (const auto& track : tracker.tracked) (track->activated ? confirmed : unconfirmed).push_back(track);
    Tracks pool = join(confirmed, tracker.lost);
    for (const auto& track : pool) track->predictImage(dt, tracker.config);
    for (const auto& track : join(pool, unconfirmed))
        track->predictWorld(tracker.world_filter, timestamp_seconds, worldLifetime, dt,
                            tracker.config.acceleration_std);

    Tracks activated, refound, newly_lost;
    const auto apply_match = [&](const TrackPtr& track, const TrackPtr& detection) {
        const bool was_tracked = track->state == State::Tracked;
        track->update(*detection, tracker.frame_id, timestamp_seconds, tracker.world_filter);
        (was_tracked ? activated : refound).push_back(track);
    };

    // Stage 1: confirmed + lost against high-confidence detections.
    const Assignment first = assign(tracker.costs(pool, high, tracker.config.fuse_score), high.size(),
                                    tracker.config.match_threshold);
    for (const auto& match : first.matches) apply_match(pool[match.first], high[match.second]);

    // Stage 2: only still-active unmatched tracks can consume low-confidence detections.
    Tracks remaining_active;
    for (const auto i : first.unmatched_tracks)
        if (pool[i]->state == State::Tracked) remaining_active.push_back(pool[i]);
    const Assignment second = assign(tracker.costs(remaining_active, low, false), low.size(), 0.5);
    for (const auto& match : second.matches) apply_match(remaining_active[match.first], low[match.second]);
    for (const auto i : second.unmatched_tracks)
    {
        remaining_active[i]->state = State::Lost;
        newly_lost.push_back(remaining_active[i]);
    }

    // Stage 3: unconfirmed tracks get one chance to match leftover high detections.
    Tracks remaining_high;
    for (const auto i : first.unmatched_detections) remaining_high.push_back(high[i]);
    const Assignment third = assign(tracker.costs(unconfirmed, remaining_high, tracker.config.fuse_score),
                                    remaining_high.size(), 0.7);
    for (const auto& match : third.matches) apply_match(unconfirmed[match.first], remaining_high[match.second]);
    for (const auto i : third.unmatched_tracks)
    {
        unconfirmed[i]->state = State::Removed;
        newly_removed.push_back(unconfirmed[i]);
    }
    for (const auto i : third.unmatched_detections)
    {
        auto& track = remaining_high[i];
        if (track->score < tracker.config.new_track_threshold) continue;
        track->activate(tracker.next_id++, tracker.frame_id, timestamp_seconds,
                        tracker.config, tracker.world_filter);
        activated.push_back(track);
    }

    tracker.tracked.erase(std::remove_if(tracker.tracked.begin(), tracker.tracked.end(),
                                        [](const TrackPtr& t) { return t->state != State::Tracked; }),
                          tracker.tracked.end());
    tracker.tracked = join(join(tracker.tracked, activated), refound);
    tracker.lost = subtract(tracker.lost, tracker.tracked);
    tracker.lost.insert(tracker.lost.end(), newly_lost.begin(), newly_lost.end());
    tracker.deduplicate();
    for (const auto& track : newly_removed) tracker.removed_this_frame.push_back(track->id);
    tracker.removed.insert(tracker.removed.end(), newly_removed.begin(), newly_removed.end());
    if (tracker.removed.size() > 1000) tracker.removed.erase(tracker.removed.begin(), tracker.removed.end() - 1000);

    std::vector<ByteTrack3DResult> output;
    for (const auto& track : tracker.tracked)
    {
        if (!track->activated && !tracker.config.return_unconfirmed_tracks) continue;
        ByteTrack3DResult result;
        result.box = track->currentBox();
        result.track_id = track->id;
        result.score = track->score;
        result.class_id = track->class_id;
        result.detection_index = track->detection_index;
        if (track->world_initialized)
        {
            result.position = {track->world_mean(0), track->world_mean(1)};
            result.velocity = {track->world_mean(2), track->world_mean(3)};
        }
        output.push_back(result);
    }
    return output;
}

void BYTETracker3D::reset()
{
    impl_->tracked.clear();
    impl_->lost.clear();
    impl_->removed.clear();
    impl_->removed_this_frame.clear();
    impl_->frame_id = 0;
    impl_->next_id = 1;
    impl_->last_timestamp = -std::numeric_limits<double>::infinity();
}

int BYTETracker3D::frameId() const { return impl_->frame_id; }
std::vector<int> BYTETracker3D::removedTrackIds() const { return impl_->removed_this_frame; }

}  // namespace ros_yolo
