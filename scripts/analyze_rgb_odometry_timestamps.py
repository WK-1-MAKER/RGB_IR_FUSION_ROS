#!/usr/bin/env python3
"""Compare ROS1 RGB/Odometry header stamps, reception timing and cache matching."""

import argparse
from bisect import bisect_left, bisect_right
import csv
import math
from pathlib import Path
import statistics
import threading
import time


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    left = int(index)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def nearest_index(sorted_values, target):
    right = bisect_left(sorted_values, target)
    candidates = [i for i in (right - 1, right) if 0 <= i < len(sorted_values)]
    return min(candidates, key=lambda i: (abs(sorted_values[i] - target), sorted_values[i]))


def analyze(records, wait_ms=30.0, cache_size=10):
    """Return full-capture and reception-neighbor pairs, plus delayed cache matches."""
    rgb = sorted((r for r in records if r['stream'] == 'rgb' and r['stamp_ns'] > 0),
                 key=lambda r: r['receive_ns'])
    odometry = sorted((r for r in records if r['stream'] == 'odometry' and r['stamp_ns'] > 0),
                      key=lambda r: r['receive_ns'])
    if not rgb or not odometry:
        return []
    rgb_by_stamp = sorted(rgb, key=lambda r: r['stamp_ns'])
    stamps = [r['stamp_ns'] for r in rgb_by_stamp]
    arrivals = [r['receive_ns'] for r in rgb]
    capture_end = max(r['receive_ns'] for r in records)
    pairs = []
    for anchor in odometry:
        stamp = anchor['stamp_ns']
        received = anchor['receive_ns']
        by_stamp = rgb_by_stamp[nearest_index(stamps, stamp)]
        by_receive = rgb[nearest_index(arrivals, received)]
        deadline = received + round(wait_ms * 1e6)
        end = bisect_right(arrivals, deadline)
        cached = rgb[max(0, end - cache_size):end]
        available = min(cached, key=lambda r: (abs(r['stamp_ns'] - stamp), r['stamp_ns'])) if cached else None
        pairs.append({
            'odometry_stamp_ns': stamp,
            'odometry_receive_ns': received,
            'nearest_rgb_stamp_ns': by_stamp['stamp_ns'],
            'nearest_delta_ms': (by_stamp['stamp_ns'] - stamp) / 1e6,
            'stamp_covered': stamps[0] <= stamp <= stamps[-1],
            'receive_neighbor_rgb_stamp_ns': by_receive['stamp_ns'],
            'receive_neighbor_delta_ms': (by_receive['stamp_ns'] - stamp) / 1e6,
            'receive_neighbor_arrival_delta_ms': (by_receive['receive_ns'] - received) / 1e6,
            'receive_covered': arrivals[0] <= received <= arrivals[-1],
            'deadline_covered': deadline <= capture_end,
            'cache_delta_ms': (available['stamp_ns'] - stamp) / 1e6 if available else None,
        })
    return pairs


def describe(name, values):
    if not values:
        print('  {}: 无可用样本'.format(name))
        return
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    print('  {}: n={}, mean={:.3f}, median={:.3f}, MAD={:.3f}, '
          'P5/P95={:.3f}/{:.3f}, min/max={:.3f}/{:.3f} ms'.format(
              name, len(values), statistics.mean(values), median, mad,
              percentile(values, .05), percentile(values, .95), min(values), max(values)))


def report(records, pairs, args):
    print('\n差值符号统一为 RGB header.stamp - Odometry header.stamp；正数表示 RGB 时间戳更晚。')
    for label, expected in (('rgb', args.rgb_hz), ('odometry', args.odometry_hz)):
        samples = sorted((r for r in records if r['stream'] == label), key=lambda r: r['receive_ns'])
        valid = [r for r in samples if r['stamp_ns'] > 0]
        intervals = [(b['stamp_ns'] - a['stamp_ns']) / 1e6
                     for a, b in zip(samples, samples[1:])
                     if a['stamp_ns'] > 0 and b['stamp_ns'] > 0]
        backward = sum(v < 0 for v in intervals)
        duplicate = sum(v == 0 for v in intervals)
        span = valid[-1]['stamp_ns'] - valid[0]['stamp_ns'] if len(valid) > 1 else 0
        rate = '{:.3f}'.format((len(valid) - 1) * 1e9 / span) if span > 0 and not backward else 'n/a'
        print('\n{}: count={}, zero={}, duplicate={}, backward={}, header频率={} Hz，预期={} Hz'.format(
            label.upper(), len(samples), len(samples) - len(valid), duplicate, backward, rate, expected))
        describe('相邻消息时间戳间隔', intervals)
        describe('回调接收 ROS 时间 - header.stamp（含传输/调度影响）',
                 [(r['receive_ros_ns'] - r['stamp_ns']) / 1e6 for r in valid if r['receive_ros_ns'] > 0])
        if backward:
            print('  检测到时间戳倒退：可能跨越回放循环/时钟重置，请分段重新采集。')

    print('\n离线最近时间戳配对（仅统计被 RGB 时间戳范围覆盖的 Odometry）：')
    covered = [p for p in pairs if p['stamp_covered']]
    deltas = [p['nearest_delta_ms'] for p in covered]
    describe('最近 RGB 差值', deltas)
    if deltas:
        passed = sum(abs(v) < 50 for v in deltas)
        print('  |差值| <50 ms: {}/{} ({:.1f}%)'.format(passed, len(deltas), 100 * passed / len(deltas)))
        if len(deltas) >= 10:
            middle = len(deltas) // 2
            early = statistics.median(deltas[:middle])
            late = statistics.median(deltas[middle:])
            print('  前/后半段差值中位数: {:.3f}/{:.3f} ms，变化={:.3f} ms'.format(
                early, late, late - early))
    print('  排除时间范围边缘/不重叠样本: {}'.format(len(pairs) - len(covered)))

    print('\n接收时刻邻近配对（用于观察表观偏移，不要求时间戳先对齐）：')
    received_pairs = [p for p in pairs if p['receive_covered']]
    describe('RGB - Odometry 时间戳差值', [p['receive_neighbor_delta_ms'] for p in received_pairs])
    describe('配对消息的本机接收时间差', [p['receive_neighbor_arrival_delta_ms'] for p in received_pairs])
    if len(received_pairs) >= 10:
        middle = len(received_pairs) // 2
        early = statistics.median(p['receive_neighbor_delta_ms'] for p in received_pairs[:middle])
        late = statistics.median(p['receive_neighbor_delta_ms'] for p in received_pairs[middle:])
        print('  前/后半段表观偏移中位数: {:.3f}/{:.3f} ms，变化={:.3f} ms'.format(early, late, late - early))

    ready = [p for p in pairs if p['deadline_covered']]
    passed = sum(p['cache_delta_ms'] is not None and abs(p['cache_delta_ms']) < 50 for p in ready)
    print('\n等待 {:.1f} ms、RGB 缓存 {} 帧时的匹配重建: {}/{} 通过 <50 ms'.format(
        args.wait_ms, args.cache_size, passed, len(ready)))
    print('  根据本脚本接收时刻重建；不模拟 YOLO 处理积压，也不代表另一节点的实际到达顺序。')
    print('\n理论参考：RGB {:.3f} ms/帧，Odometry {:.3f} ms/帧；连续 RGB 的最近帧理想上界约 {:.3f} ms。'.format(
        1000 / args.rgb_hz, 1000 / args.odometry_hz, 500 / args.rgb_hz))
    print('判断提示：最近差值前后半段稳定，可作为固定相位偏移的线索；漂移明显则需排查时钟漂移/重置。')
    print('该差值只确定到 RGB 帧周期的模：整帧（约 {:.3f} ms）偏移可能无法由最近帧配对分辨。'.format(
        1000 / args.rgb_hz))
    print('采样相位和传输延迟也会影响估计；仅凭异步话题不能唯一确认真实采集时钟偏移。')


class Collector:
    def __init__(self):
        self.lock = threading.Lock()
        self.records = []
        self.stopped = False

    def callback(self, message, stream):
        import rospy
        received = time.monotonic_ns()
        ros_received = rospy.Time.now().to_nsec()
        stamp = message.header.stamp.to_nsec()
        with self.lock:
            if not self.stopped:
                self.records.append({'stream': stream, 'stamp_ns': stamp,
                                     'receive_ns': received, 'receive_ros_ns': ros_received})

    def stop(self):
        with self.lock:
            self.stopped = True
            return list(self.records)


def collect(args):
    import rospy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    collector = Collector()
    subscribers = []
    rospy.init_node('rgb_odometry_timestamp_analyzer', anonymous=True)
    try:
        for label, topic, message_type in (('rgb', args.rgb_topic, Image),
                                            ('odometry', args.odometry_topic, Odometry)):
            subscribers.append(rospy.Subscriber(topic, message_type, collector.callback,
                                               callback_args=label, queue_size=1000,
                                               buff_size=16 * 1024 * 1024))
        deadline = time.monotonic() + args.duration if args.duration else None
        while not rospy.is_shutdown():
            remaining = deadline - time.monotonic() if deadline is not None else .1
            if remaining <= 0:
                break
            time.sleep(min(.1, remaining))
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        records = collector.stop()
        for subscriber in subscribers:
            subscriber.unregister()
    return records


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-topic', default='/camera/color/image_raw')
    parser.add_argument('--odometry-topic', default='/Odometry')
    parser.add_argument('--rgb-hz', type=float, default=15.0, help='Expected frequency; does not change publisher.')
    parser.add_argument('--odometry-hz', type=float, default=10.0)
    parser.add_argument('--duration', type=float, default=30.0, help='Wall seconds; 0 captures until Ctrl+C.')
    parser.add_argument('--wait-ms', type=float, default=30.0, help='Reception deadline for cache matching reconstruction.')
    parser.add_argument('--cache-size', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('rgb_odometry_timestamps.csv'))
    parser.add_argument('--pairs-output', type=Path, default=Path('rgb_odometry_differences.csv'))
    args = parser.parse_args(argv)
    for name in ('rgb_hz', 'odometry_hz', 'duration', 'wait_ms'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name.endswith('_hz') and value == 0):
            parser.error('{} must be finite and {}'.format(name, 'positive' if name.endswith('_hz') else 'nonnegative'))
    if args.cache_size < 1:
        parser.error('--cache-size must be positive')
    if args.output.resolve() == args.pairs_output.resolve():
        parser.error('--output and --pairs-output must be different paths')
    return args


def main(argv=None):
    args = parse_args(argv)
    print('订阅 RGB={}，Odometry={}；采集 {}，Ctrl+C 结束后输出统计与 CSV。'.format(
        args.rgb_topic, args.odometry_topic, '{} 秒'.format(args.duration) if args.duration else '直到 Ctrl+C'))
    records = collect(args)
    pairs = analyze(records, args.wait_ms, args.cache_size)
    save_csv(args.output, records, ['stream', 'stamp_ns', 'receive_ns', 'receive_ros_ns'])
    fields = ['odometry_stamp_ns', 'odometry_receive_ns', 'nearest_rgb_stamp_ns', 'nearest_delta_ms',
              'stamp_covered', 'receive_neighbor_rgb_stamp_ns', 'receive_neighbor_delta_ms',
              'receive_neighbor_arrival_delta_ms', 'receive_covered', 'deadline_covered', 'cache_delta_ms']
    save_csv(args.pairs_output, pairs, fields)
    report(records, pairs, args)
    print('\n原始数据: {}\n配对数据: {}'.format(args.output.resolve(), args.pairs_output.resolve()))


if __name__ == '__main__':
    main()
