#!/usr/bin/env python3
"""Collect ROS header timestamps and plot paired topic timelines."""

import argparse
import math
import statistics
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STREAM_LABELS = ("lidar", "odometry", "rgb", "ir")
DEFAULT_TOPICS = {
    "lidar": "/cloud_undistorted",
    "odometry": "/Odometry",
    "rgb": "/camera/color/image_raw",
    "ir": "/pub_ir",
}


def summarize(stamps):
    intervals_ms = [
        (current - previous) / 1e6
        for previous, current in zip(stamps, stamps[1:])
        if previous != 0 and current != 0
    ]
    valid_stamps = [stamp for stamp in stamps if stamp != 0]
    positive_intervals = [interval for interval in intervals_ms if interval > 0]
    backward_count = sum(interval < 0 for interval in intervals_ms)
    duplicate_count = sum(interval == 0 for interval in intervals_ms)
    rate_hz = None
    if positive_intervals and backward_count == 0 and duplicate_count == 0:
        rate_hz = 1000.0 / statistics.mean(positive_intervals)

    return {
        "count": len(stamps),
        "zero_count": len(stamps) - len(valid_stamps),
        "first_ns": valid_stamps[0] if valid_stamps else None,
        "last_ns": valid_stamps[-1] if valid_stamps else None,
        "duplicate_count": duplicate_count,
        "backward_count": backward_count,
        "min_interval_ms": min(intervals_ms) if intervals_ms else None,
        "median_interval_ms": statistics.median(intervals_ms) if intervals_ms else None,
        "max_interval_ms": max(intervals_ms) if intervals_ms else None,
        "rate_hz": rate_hz,
    }


def format_stamp(timestamp_ns):
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    return "{}.{:09d}".format(seconds, nanoseconds)


class TimestampCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._stamps = {label: [] for label in STREAM_LABELS}
        self._stopped = False

    def callback(self, message, label):
        timestamp_ns = message.header.stamp.to_nsec()
        with self._lock:
            if not self._stopped:
                self._stamps[label].append(timestamp_ns)

    def stop(self):
        with self._lock:
            self._stopped = True
            return {label: list(values) for label, values in self._stamps.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-topic", default=DEFAULT_TOPICS["lidar"])
    parser.add_argument("--odometry-topic", default=DEFAULT_TOPICS["odometry"])
    parser.add_argument("--rgb-topic", default=DEFAULT_TOPICS["rgb"])
    parser.add_argument("--ir-topic", default=DEFAULT_TOPICS["ir"])
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Capture duration in wall-clock seconds; 0 means until Ctrl+C.")
    parser.add_argument("--output", type=Path, default=Path("topic_timestamps.png"),
                        help="Output PNG path (default: topic_timestamps.png).")
    parser.add_argument("--queue-size", type=int, default=1000,
                        help="ROS subscriber queue size (default: 1000).")
    parser.add_argument("--buff-size-mb", type=int, default=16,
                        help="ROS TCP receive buffer in MiB (default: 16).")
    args = parser.parse_args(argv)

    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and nonnegative")
    if args.queue_size < 1:
        parser.error("--queue-size must be positive")
    if args.buff_size_mb < 1:
        parser.error("--buff-size-mb must be positive")

    args.topics = {
        "lidar": args.lidar_topic,
        "odometry": args.odometry_topic,
        "rgb": args.rgb_topic,
        "ir": args.ir_topic,
    }
    return args


def collect_timestamps(args):
    import rospy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, PointCloud2

    message_types = {
        "lidar": PointCloud2,
        "odometry": Odometry,
        "rgb": Image,
        "ir": Image,
    }
    collector = TimestampCollector()
    subscribers = []
    rospy.init_node("topic_timestamp_plotter", anonymous=True)
    try:
        for label in STREAM_LABELS:
            subscribers.append(rospy.Subscriber(
                args.topics[label], message_types[label], collector.callback,
                callback_args=label,
                queue_size=args.queue_size,
                buff_size=args.buff_size_mb * 1024 * 1024,
            ))

        deadline = time.monotonic() + args.duration if args.duration else None
        while not rospy.is_shutdown():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.1, remaining))
            else:
                time.sleep(0.1)
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        for subscriber in subscribers:
            subscriber.unregister()
    return collector.stop()


def print_summary(stamps, topics):
    print("Timestamp summary (header.stamp, ROS time):")
    for label in STREAM_LABELS:
        result = summarize(stamps[label])
        first = format_stamp(result["first_ns"]) if result["first_ns"] is not None else "n/a"
        last = format_stamp(result["last_ns"]) if result["last_ns"] is not None else "n/a"
        rate = "n/a" if result["rate_hz"] is None else "{:.3f} Hz".format(result["rate_hz"])
        interval = "n/a"
        if result["median_interval_ms"] is not None:
            interval = "{:.3f}/{:.3f}/{:.3f} ms (min/median/max)".format(
                result["min_interval_ms"],
                result["median_interval_ms"],
                result["max_interval_ms"],
            )
        print("  {} {}: count={}, zero={}, duplicate={}, backward={}, first={}, last={}, rate={}, dt={}".format(
            label.upper(), topics[label], result["count"], result["zero_count"],
            result["duplicate_count"], result["backward_count"], first, last, rate, interval,
        ))


def save_plot(stamps, topics, output_path):
    all_valid = [stamp for label in STREAM_LABELS for stamp in stamps[label] if stamp != 0]
    origin_ns = min(all_valid) if all_valid else 0
    groups = (("lidar", "odometry"), ("rgb", "ir"))
    colors = {"lidar": "#2563eb", "odometry": "#dc2626", "rgb": "#059669", "ir": "#d97706"}
    titles = ("LiDAR and Odometry", "RGB and IR")

    figure, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for axis, labels, title in zip(axes, groups, titles):
        y_positions = {labels[0]: 1, labels[1]: 0}
        for label in labels:
            relative_seconds = [
                (stamp - origin_ns) / 1e9
                for stamp in stamps[label]
                if stamp != 0
            ]
            axis.vlines(
                relative_seconds,
                y_positions[label] - 0.32,
                y_positions[label] + 0.32,
                color=colors[label],
                linewidth=1.2,
                label="{} ({})".format(label.upper(), topics[label]),
            )
        axis.set_yticks([y_positions[labels[0]], y_positions[labels[1]]])
        axis.set_yticklabels([labels[0].upper(), labels[1].upper()])
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
        axis.set_ylim(-0.6, 1.6)

    if all_valid:
        relative_min = (min(all_valid) - origin_ns) / 1e9
        relative_max = (max(all_valid) - origin_ns) / 1e9
        if relative_min == relative_max:
            axes[-1].set_xlim(relative_min - 0.5, relative_max + 0.5)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No nonzero timestamps received", transform=axis.transAxes,
                      ha="center", va="center", color="#6b7280")
        axes[-1].set_xlim(0, 1)

    origin_label = format_stamp(origin_ns) if all_valid else "n/a"
    axes[-1].set_xlabel(
        "ROS header.stamp offset from earliest message (s; origin {})".format(origin_label)
    )
    figure.suptitle("ROS Topic Timestamp Timeline")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    print("Collecting ROS header.stamp values. Press Ctrl+C to finish, or set --duration.")
    stamps = collect_timestamps(args)
    print_summary(stamps, args.topics)
    save_plot(stamps, args.topics, args.output)
    print("Saved timeline image: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
