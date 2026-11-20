"""Analyze ROS 2 MCAP LiDAR/stereo synchronization with Rerun (no ROS node)."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import rerun as rr
import rosbag2_py
import yaml
from message_filters import ApproximateTimeSynchronizer, SimpleFilter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topic_types = {
        t.name: get_message(t.type) for t in reader.get_all_topics_and_types()
    }
    while reader.has_next():
        topic, data, bag_timestamp_ns = reader.read_next()
        yield topic, deserialize_message(data, topic_types[topic]), bag_timestamp_ns


def header_ns(msg) -> int:
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def at_time(ns: int) -> None:
    # Preserve nanoseconds; converting epoch nanoseconds to float loses precision.
    rr.set_time("ros_time", timestamp=np.datetime64(int(ns), "ns"))


def output_path(value: str | None, config_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.parent.parent / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    bag_path = str(Path(config["bag_path"]).expanduser())
    topics = config["topics"]
    opts = config["sync"]
    out = config.get("output", {})
    slop = float(opts["slop"])
    queue_size = int(opts["queue_size"])
    if slop <= 0 or queue_size <= 0:
        raise ValueError("sync.slop and sync.queue_size must be positive")
    if len(set(topics.values())) != 3:
        raise ValueError("The three configured topic names must be distinct")

    rr.init("lidar_stereo_synchronization")
    rrd_path = output_path(out.get("rrd"), config_path)
    if rrd_path:
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(rrd_path))
    else:
        rr.spawn()

    # Static colors apply to the *time-series plot*, not the bottom Streams pins.
    for name, color in (("left", [50, 140, 240]), ("right", [245, 160, 55])):
        rr.log(
            f"offset_ms/{name}",
            rr.SeriesPoints(colors=color, names=f"{name} - LiDAR (ms)", markers="circle", marker_sizes=4),
            static=True,
        )

    filters = {name: SimpleFilter() for name in ("lidar", "left", "right")}
    synchronizer = ApproximateTimeSynchronizer(
        [filters["lidar"], filters["left"], filters["right"]],
        queue_size=queue_size,
        slop=slop,
    )
    rows = []
    matched_lidar = Counter()
    seen_lidar = Counter()
    counts = Counter()
    log_raw = bool(out.get("log_raw_events", True))
    log_matched = bool(out.get("log_matched_events", True))

    def on_match(lidar, left, right):
        times = {"lidar": header_ns(lidar), "left": header_ns(left), "right": header_ns(right)}
        matched_lidar[times["lidar"]] += 1
        match_id = len(rows) + 1
        left_delta = (times["left"] - times["lidar"]) / 1e6
        right_delta = (times["right"] - times["lidar"]) / 1e6
        spread = (max(times.values()) - min(times.values())) / 1e6
        rows.append({
            "match_id": match_id,
            "lidar_header_ns": times["lidar"],
            "left_header_ns": times["left"],
            "right_header_ns": times["right"],
            "left_minus_lidar_ms": left_delta,
            "right_minus_lidar_ms": right_delta,
            "triplet_spread_ms": spread,
        })

        # Plot at the LiDAR measurement time; positive offsets = camera later.
        at_time(times["lidar"])
        rr.log("offset_ms/left", rr.Scalars(left_delta))
        rr.log("offset_ms/right", rr.Scalars(right_delta))
        rr.log("offset_ms/triplet_spread", rr.Scalars(spread))

        if log_matched:
            # Three separate event rows at the *actual timestamps*, with the
            # same match ID so the triplet can be inspected in the Viewer.
            for sensor in ("lidar", "left", "right"):
                at_time(times[sensor])
                rr.log(
                    f"matched/{sensor}",
                    rr.TextLog(f"match {match_id}: {sensor}, stamp_ns={times[sensor]}"),
                )

    synchronizer.registerCallback(on_match)
    topic_to_sensor = {topic: sensor for sensor, topic in topics.items()}

    # SequentialReader preserves the bag's stored order; this is the feed order
    # seen by the offline message_filters implementation.
    for topic, msg, bag_ns in read_messages(bag_path):
        sensor = topic_to_sensor.get(topic)
        if sensor is None:
            continue
        ts = header_ns(msg)
        counts[sensor] += 1
        if sensor == "lidar":
            seen_lidar[ts] += 1
        if log_raw:
            at_time(ts)
            rr.log(f"raw/{sensor}", rr.TextLog(f"{sensor}: {ts}"))
        filters[sensor].signalMessage(msg)

    # Mark LiDAR messages that did not participate in any complete triplet.
    for ts, count in seen_lidar.items():
        for _ in range(max(0, count - matched_lidar[ts])):
            at_time(ts)
            rr.log("unmatched/lidar", rr.TextLog(f"No triplet for LiDAR stamp {ts}"))

    csv_path = output_path(out.get("csv", "sync_matches.csv"), config_path)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "match_id", "lidar_header_ns", "left_header_ns", "right_header_ns",
                "left_minus_lidar_ms", "right_minus_lidar_ms", "triplet_spread_ms",
            ])
            writer.writeheader()
            writer.writerows(rows)

    print(f"Topics seen: {dict(counts)}")
    print(f"Matched triplets: {len(rows)} / {counts['lidar']} LiDAR messages")
    print(f"Unmatched LiDAR: {sum(seen_lidar.values()) - sum(matched_lidar.values())}")
    print(f"CSV: {csv_path}")
    if rrd_path:
        print(f"RRD: {rrd_path}")
    # Ensure all buffered recording data is delivered before quitting.
    rr.shutdown()


if __name__ == "__main__":
    main()