"""Load a synchronized triplet from an MCAP without a ROS installation."""

from collections import defaultdict

import cv2
import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore

from visualize_match import OfflineTF, camera_parameters, header_ns, norm_frame


def load_triplet(bag_path, topics, info_topics, stamps, fixed_frame, tolerance_ms):
    store = get_typestore(Stores.ROS2_HUMBLE)
    sensors = {topic: name for name, topic in topics.items()}
    calibration = {topic: name for name, topic in info_topics.items()}
    selected, infos = {}, defaultdict(list)
    graph = OfflineTF()
    with bag_path.open('rb') as stream:
        reader = make_reader(stream)
        for _, channel, record in reader.iter_messages(
            topics=[*sensors, *calibration, '/tf', '/tf_static'], log_time_order=False
        ):
            topic = channel.topic
            if topic in sensors and sensors[topic] in selected:
                continue
            if topic in sensors:
                # A ROS 2 CDR Header starts with sec/nanosec after the encapsulation.
                raw = record.data
                order = 'little' if raw[1] & 1 else 'big'
                stamp = (int.from_bytes(raw[4:8], order, signed=True) * 10**9
                         + int.from_bytes(raw[8:12], order))
                if stamp != stamps[sensors[topic]]:
                    continue
            schema = reader.get_summary().schemas[channel.schema_id]
            msg_type = schema.name
            if msg_type not in store.types:
                msg_type = msg_type.replace('/', '/msg/', 1)
            msg = store.deserialize_cdr(record.data, msg_type)
            if topic in sensors:
                selected[sensors[topic]] = msg
            elif topic in calibration:
                infos[calibration[topic]].append(msg)
            else:
                graph.add(msg, is_static=topic == '/tf_static')

    missing = set(stamps) - selected.keys()
    if missing:
        raise ValueError(f'Match is missing {sorted(missing)} in {bag_path}')
    graph.finalize()
    frames = {name: norm_frame(msg.header.frame_id) for name, msg in selected.items()}
    world = norm_frame(fixed_frame) if fixed_frame else frames['left']
    poses = {
        name: graph.lookup(world, frame,
                           stamps[name] if fixed_frame else stamps['lidar'], tolerance_ms)
        for name, frame in frames.items()
    }

    cameras = {}
    for name in ('left', 'right'):
        if not infos[name]:
            raise ValueError(f'No calibration on {info_topics[name]}')
        info = min(infos[name], key=lambda msg: abs(header_ns(msg) - stamps[name]))
        if norm_frame(info.header.frame_id) != frames[name]:
            raise ValueError(f'{name} image and CameraInfo have different frames')
        cameras[name] = (selected[name], info, poses[name])
    return selected['lidar'], poses['lidar'], cameras, world, frames


def lidar_xyz(msg):
    fields = {field.name: field for field in msg.fields}
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [
            ('>' if msg.is_bigendian else '<') + {7: 'f4', 8: 'f8'}[fields[name].datatype]
            for name in ('x', 'y', 'z')
        ],
        'offsets': [fields[name].offset for name in ('x', 'y', 'z')],
        'itemsize': msg.point_step,
    })
    points = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                        strides=(msg.row_step, msg.point_step))
    return np.stack([points[name] for name in ('x', 'y', 'z')], axis=-1).reshape(-1, 3).astype(
        np.float32, copy=False
    )


def rectification_maps(info, raw_shape, height, width):
    """Build OpenCV rectification maps and the intrinsics of the output image."""
    raw_height, raw_width = raw_shape[:2]
    K, distortion = camera_parameters(info, raw_width, raw_height)
    sx, sy = width / raw_width, height / raw_height
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    output_K = resize @ K
    x, y = cv2.initUndistortRectifyMap(
        K, distortion if distortion.size else None, None, output_K,
        (width, height), cv2.CV_32FC1
    )
    return output_K.astype(np.float32), x, y


def remap_grid(info, raw_shape, height, width):
    """Build the static rectification/resize grid using pixel-center coordinates."""
    output_K, x, y = rectification_maps(info, raw_shape, height, width)
    raw_height, raw_width = raw_shape[:2]
    grid = np.stack(((2 * x + 1) / raw_width - 1, (2 * y + 1) / raw_height - 1), axis=-1)
    return output_K, grid
