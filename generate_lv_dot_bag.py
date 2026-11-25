"""Export a synchronized left-camera/LiDAR clip with SPNet depth as a ROS 1 bag."""

import argparse
from bisect import bisect_left
import csv
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from scipy.spatial.transform import Rotation
import yaml

from spnet_inputs import lidar_xyz, rectification_maps
from visualize_match import OfflineTF, header_ns, image_rgb, norm_frame


ROOT = Path(__file__).resolve().parent
COLOR_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/depth/image_rect_raw'
POSE_TOPIC = '/lv_dot/pose'
CLOUD_TYPE = 'sensor_msgs/msg/PointCloud2'
TF_TYPE = 'tf2_msgs/msg/TFMessage'
DEPTH_SCALE = 1000
MAX_UINT16 = np.iinfo(np.uint16).max


def read_matches(path, start, count):
    end = start + count
    with path.open(newline='') as stream:
        rows = sorted(
            (row for row in csv.DictReader(stream) if start <= int(row['match_id']) < end),
            key=lambda row: int(row['match_id']),
        )
    if [int(row['match_id']) for row in rows] != list(range(start, end)):
        raise ValueError(f'{path} does not contain exactly one row for each match {start}–{end - 1}')
    pairs = [(int(row['left_header_ns']), int(row['lidar_header_ns'])) for row in rows]
    if any(len(set(stamps)) != len(pairs) for stamps in zip(*pairs)):
        raise ValueError('Selected matches have repeated camera or LiDAR timestamps')
    return pairs


def load_geometry(bag, info_topic):
    graph = OfflineTF()
    calibrations = []
    connections = [c for c in bag.connections if c.topic in {info_topic, '/tf', '/tf_static'}]
    for connection, _, raw in bag.messages(connections=connections):
        message = bag.deserialize(raw, connection.msgtype)
        if connection.topic == info_topic:
            calibrations.append(message)
        else:
            graph.add(message, is_static=connection.topic == '/tf_static')
    if not calibrations:
        raise ValueError(f'No CameraInfo messages on {info_topic}')
    if not graph.dynamic:
        raise ValueError('No dynamic TF recorded for robot localization')
    calibrations.sort(key=header_ns)
    graph.finalize()
    return graph, calibrations


def capture_stamp(raw):
    # ROS 2 stores sec and nanosec at the start of these Header-bearing CDR messages.
    order = 'little' if raw[1] & 1 else 'big'
    return int.from_bytes(raw[4:8], order, signed=True) * 10**9 + int.from_bytes(raw[8:12], order)


def load_completion(config, device_name):
    checkout = ROOT / Path(config['spnet_checkout']).expanduser()
    checkpoint = ROOT / Path(config.get('checkpoint') or f"checkpoints/{config['model']}_300.pth").expanduser()
    if not (checkout / 'src/networks.py').is_file() or not checkpoint.is_file():
        raise ValueError(f'SPNet checkout or checkpoint missing ({checkout}, {checkpoint}). '
                         'Set the paths in the SPNet config or run scripts/run_spnet.sh.')
    import torch
    from benchmark_spnet import project_sparse
    from detection_models import load_spnet, perform_spnet

    if device_name == 'auto':
        device_name = f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    sys.path.insert(0, str(checkout.resolve()))
    try:
        model = load_spnet({**config, 'checkpoint': checkpoint}, device, ROOT)
    finally:
        sys.path.pop(0)
    max_depth = float(config['max_depth_m'])

    class NoTiming:
        def stage(self, _name, *, cuda=False):
            return nullcontext()

    timing = NoTiming()

    @torch.inference_mode()
    def complete(rgb, points, intrinsics, camera_from_lidar):
        height, width = rgb.shape[:2]
        sparse = project_sparse(
            torch.as_tensor(points, device=device),
            torch.as_tensor(camera_from_lidar, dtype=torch.float32, device=device),
            torch.as_tensor(intrinsics, device=device),
            height, width, max_depth,
        )
        if not sparse.any().item():
            raise ValueError('No LiDAR points project into the camera')
        depth = perform_spnet(model, rgb, sparse, max_depth, timing)
        return depth.cpu().numpy()

    print(f"Running SPNet {config['model']} on {device}", flush=True)
    return complete


def depth_millimetres(depth):
    valid = np.isfinite(depth) & (depth > 0) & (depth <= MAX_UINT16 / DEPTH_SCALE)
    return np.rint(np.where(valid, depth, 0) * DEPTH_SCALE).astype('<u2')


def write_clip(writer, bag, complete, sensor, pairs, resolution, fixed, body, tolerance):
    width, height = resolution
    lidar_topic = sensor['topics']['lidar']
    camera_topic = sensor['topics']['left']
    info_topic = sensor['visualization']['camera_info_topics']['left']
    graph, calibrations = load_geometry(bag, info_topic)
    calibration_stamps = [header_ns(info) for info in calibrations]

    ros1 = get_typestore(Stores.ROS1_NOETIC)
    ros1.register(get_types_from_msg('geometry_msgs/TransformStamped[] transforms', TF_TYPE))
    types = ros1.types
    output_types = {
        COLOR_TOPIC: 'sensor_msgs/msg/Image',
        DEPTH_TOPIC: 'sensor_msgs/msg/Image',
        '/camera/color/camera_info': 'sensor_msgs/msg/CameraInfo',
        lidar_topic: CLOUD_TYPE,
        POSE_TOPIC: 'geometry_msgs/msg/PoseStamped',
        '/tf': TF_TYPE,
        '/tf_static': TF_TYPE,
    }
    outputs = {
        topic: writer.add_connection(topic, kind, typestore=ros1, latching=1 if topic == '/tf_static' else None)
        for topic, kind in output_types.items()
    }

    def write(topic, bag_stamp, message):
        connection = outputs[topic]
        writer.write(connection, bag_stamp, ros1.serialize_ros1(message, connection.msgtype))

    def header(stamp, frame, seq=0):
        return types['std_msgs/msg/Header'](seq, types['builtin_interfaces/msg/Time'](*divmod(stamp, 10**9)), frame)

    def transform(stamp, parent, child, matrix):
        translation = types['geometry_msgs/msg/Vector3'](*matrix[:3, 3])
        rotation = types['geometry_msgs/msg/Quaternion'](*Rotation.from_matrix(matrix[:3, :3]).as_quat())
        return types['geometry_msgs/msg/TransformStamped'](
            header(stamp, parent), child, types['geometry_msgs/msg/Transform'](translation, rotation),
        )

    def write_image(topic, bag_stamp, image_header, array, encoding):
        image = np.ascontiguousarray(array)
        write(topic, bag_stamp, types['sensor_msgs/msg/Image'](
            image_header, height, width, encoding, 0, image.strides[0], image.view(np.uint8).ravel(),
        ))

    camera_matches = dict(pairs)
    lidar_matches = {lidar: camera for camera, lidar in pairs}
    first = min(min(pair) for pair in pairs)
    last = max(max(pair) for pair in pairs)
    pose_margin_ns = int(tolerance * 1_000_000)
    pending_images, pending_clouds = {}, {}
    seen_images, seen_clouds, seen_poses = set(), set(), set()
    calibration = None
    current_rectification = None
    camera_frame = lidar_frame = None
    scan_count = image_count = 0
    # Find which dynamic TF edges to use as the pose publication clock. For this
    # recording, odom is connected to base_link through odom -> base_footprint.
    pose_edges = {edge for edge in graph.dynamic if fixed in edge}

    sources = [c for c in bag.connections if c.topic in {lidar_topic, camera_topic, '/tf'}]
    available = {c.topic for c in sources}
    if missing := {lidar_topic, camera_topic, '/tf'} - available:
        raise ValueError(f'Missing input topics: {sorted(missing)}')
    for connection, bag_stamp, raw in bag.messages(connections=sources):
        topic = connection.topic
        if topic == '/tf':
            message = bag.deserialize(raw, connection.msgtype)
            for entry in message.transforms:
                edge = norm_frame(entry.header.frame_id), norm_frame(entry.child_frame_id)
                stamp = header_ns(entry)
                if (edge not in pose_edges or stamp in seen_poses
                        or not first - pose_margin_ns <= stamp <= last + pose_margin_ns):
                    continue
                pose_matrix = graph.lookup(fixed, body, stamp, tolerance)
                tf = transform(stamp, fixed, body, pose_matrix)
                pose = types['geometry_msgs/msg/Pose'](
                    types['geometry_msgs/msg/Point'](*pose_matrix[:3, 3]), tf.transform.rotation,
                )
                write(POSE_TOPIC, bag_stamp, types['geometry_msgs/msg/PoseStamped'](tf.header, pose))
                write('/tf', bag_stamp, types[TF_TYPE]([tf]))
                seen_poses.add(stamp)
            continue

        stamp = capture_stamp(raw)
        if topic == lidar_topic:
            if not first <= stamp <= last or stamp in seen_clouds:
                continue
            seen_clouds.add(stamp)
            writer.write(outputs[lidar_topic], bag_stamp, bag.typestore.cdr_to_ros1(raw, CLOUD_TYPE))
            scan_count += 1
            if stamp not in lidar_matches:
                continue
            pending_clouds[stamp] = bag.deserialize(raw, CLOUD_TYPE)
            image_stamp = lidar_matches[stamp]
        else:
            if stamp not in camera_matches or stamp in seen_images:
                continue
            seen_images.add(stamp)
            pending_images[stamp] = bag.deserialize(raw, connection.msgtype)
            image_stamp = stamp

        cloud_stamp = camera_matches[image_stamp]
        if image_stamp not in pending_images or cloud_stamp not in pending_clouds:
            continue
        image = pending_images.pop(image_stamp)
        cloud = pending_clouds.pop(cloud_stamp)
        image_frame = norm_frame(image.header.frame_id)
        cloud_frame = norm_frame(cloud.header.frame_id)
        rgb = image_rgb(image)
        index = bisect_left(calibration_stamps, image_stamp)
        candidates = range(max(0, index - 1), min(index + 1, len(calibrations)))
        info = calibrations[min(candidates, key=lambda i: abs(calibration_stamps[i] - image_stamp))]
        if norm_frame(info.header.frame_id) != image_frame:
            raise ValueError('The left image and CameraInfo have different frame IDs')
        rectification_key = (rgb.shape, info.width, info.height, info.distortion_model, tuple(info.k), tuple(info.d))
        if current_rectification != rectification_key:
            intrinsics, xmap, ymap = rectification_maps(info, rgb.shape, height, width)
            current_rectification = rectification_key
        rgb = cv2.remap(rgb, xmap, ymap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        if calibration is None:
            camera_frame, lidar_frame = image_frame, cloud_frame
            body_from_camera = graph.lookup(body, camera_frame, image_stamp, tolerance)
            body_from_lidar = graph.lookup(body, lidar_frame, cloud_stamp, tolerance)
            calibration = intrinsics
            static = [
                transform(0, body, camera_frame, body_from_camera),
                transform(0, body, lidar_frame, body_from_lidar),
            ]
            # Upstream LV-DOT labels its world-frame visualization messages "map".
            if fixed != 'map':
                static.append(transform(0, 'map', fixed, np.eye(4)))
            write('/tf_static', bag_stamp, types[TF_TYPE](static))
        elif image_frame != camera_frame or cloud_frame != lidar_frame or not np.array_equal(calibration, intrinsics):
            raise ValueError('The sensor frames or rectified intrinsics changed within this clip')

        fixed_from_camera = graph.lookup(fixed, camera_frame, image_stamp, tolerance)
        fixed_from_lidar = graph.lookup(fixed, lidar_frame, cloud_stamp, tolerance)
        camera_from_lidar = np.linalg.solve(fixed_from_camera, fixed_from_lidar)
        try:
            depth = depth_millimetres(complete(rgb, lidar_xyz(cloud), intrinsics, camera_from_lidar))
        except ValueError as error:
            raise ValueError(f'Camera timestamp {image_stamp}: {error}') from error
        image_header = header(image_stamp, camera_frame, image_count)
        write_image(COLOR_TOPIC, bag_stamp, image_header, rgb[..., ::-1], 'bgr8')
        write_image(DEPTH_TOPIC, bag_stamp, image_header, depth, '16UC1')
        projection = np.column_stack((intrinsics, np.zeros(3))).ravel().astype(np.float64)
        write('/camera/color/camera_info', bag_stamp, types['sensor_msgs/msg/CameraInfo'](
            image_header, height, width, 'plumb_bob', np.zeros(5), intrinsics.ravel().astype(np.float64),
            np.eye(3).ravel(), projection, 0, 0, types['sensor_msgs/msg/RegionOfInterest'](0, 0, 0, 0, False),
        ))
        image_count += 1
        if image_count % 25 == 0:
            print(f'Wrote {image_count}/{len(pairs)} RGB/depth pairs', flush=True)

    if image_count != len(pairs):
        missing_images = camera_matches.keys() - seen_images
        missing_clouds = lidar_matches.keys() - seen_clouds
        raise ValueError(f'Wrote {image_count}/{len(pairs)} RGB/depth pairs. '
                         f'Missing {len(missing_images)} camera and {len(missing_clouds)} LiDAR timestamps. '
                         'Check that the match CSV corresponds to this recording.')
    if not seen_poses:
        raise ValueError(f'No localization samples for {fixed} -> {body} within the replay clip')

    pinhole = [float(calibration[i, j]) for i, j in [(0, 0), (1, 1), (0, 2), (1, 2)]]
    params = {
        'localization_mode': 0,
        'pose_topic': POSE_TOPIC,
        'color_image_topic': COLOR_TOPIC,
        'depth_image_topic': DEPTH_TOPIC,
        'lidar_pointcloud_topic': lidar_topic,
        'depth_scale_factor': DEPTH_SCALE,
        'depth_intrinsics': pinhole,
        'color_intrinsics': pinhole,
        'image_cols': width,
        'image_rows': height,
        'body_to_camera_depth': body_from_camera.ravel().tolist(),
        'body_to_camera_color': body_from_camera.ravel().tolist(),
        'body_to_lidar': body_from_lidar.ravel().tolist(),
    }
    return params, image_count, scan_count, len(seen_poses)


def export_bag(path, bag_path, complete, sensor, pairs, resolution, fixed, body, tolerance):
    params_path = path.with_suffix('.detector.yaml')
    for destination in (path, params_path):
        if destination.exists():
            raise FileExistsError(f'Output already exists: {destination}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.lv-dot-', dir=path.parent) as directory:
        staged_bag = Path(directory) / path.name
        staged_params = Path(directory) / params_path.name
        with AnyReader([bag_path]) as bag, Writer(staged_bag) as writer:
            params, image_count, scan_count, pose_count = write_clip(
                writer, bag, complete, sensor, pairs, resolution, fixed, body, tolerance,
            )
        staged_params.write_text(yaml.safe_dump(params, sort_keys=False))
        os.link(staged_bag, path)
        try:
            os.link(staged_params, params_path)
        except OSError:
            path.unlink()
            raise
    print(f'Wrote {path} with {image_count} RGB/depth pairs, {scan_count} scans, and {pose_count} poses.')
    print(f'LV-DOT sensor parameters: {params_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/spnet.yaml')
    parser.add_argument('--bag', type=Path, help='Override the corrected MCAP from the configuration')
    parser.add_argument('--output', type=Path, required=True, help='Output ROS 1 .bag path')
    parser.add_argument('--start-match', type=int, help='First match ID (defaults to the SPNet config)')
    parser.add_argument('--frames', type=int, default=50, help='Number of consecutive matched RGB/depth pairs')
    parser.add_argument('--device', default='auto', help='PyTorch device (for example cuda:0 or cpu)')
    parser.add_argument('--fixed-frame', default='odom', help='TF parent frame for the exported pose')
    parser.add_argument('--body-frame', default='base_link', help='Robot frame for poses and LV-DOT extrinsics')
    parser.add_argument('--tf-max-age-ms', type=float, default=20, help='Maximum gap to the nearest TF sample')
    args = parser.parse_args()

    try:
        config_path = args.config.expanduser().resolve()
        config = yaml.safe_load(config_path.read_text())
        sensor = yaml.safe_load((config_path.parent / config['sensor_config']).read_text())
        bag_path = ROOT / Path(args.bag or config.get('bag_path') or sensor['bag_path']).expanduser()
        match_path = ROOT / Path(config.get('match_csv') or sensor['output']['csv']).expanduser()
        start = args.start_match if args.start_match is not None else int(config['match_id'])
        width, height = map(int, config['resolution_wh'])
        if min(width, height) < 32 or width % 32 or height % 32:
            raise ValueError('Both resolution_wh dimensions must be positive multiples of 32')
        if start < 1 or args.frames < 1:
            raise ValueError('The starting match and frame count must be positive')
        if not np.isfinite(config['max_depth_m']) or config['max_depth_m'] <= 0:
            raise ValueError('max_depth_m must be positive and finite')
        if not np.isfinite(args.tf_max_age_ms) or args.tf_max_age_ms <= 0:
            raise ValueError('--tf-max-age-ms must be positive and finite')
        if args.output.suffix != '.bag':
            raise ValueError('--output must end in .bag')
        if not bag_path.is_file():
            raise ValueError(f'Input MCAP missing: {bag_path}')
        pairs = read_matches(match_path, start, args.frames)
        complete = load_completion(config, args.device)
        export_bag(args.output.expanduser(), bag_path, complete, sensor, pairs, (width, height),
                   norm_frame(args.fixed_frame), norm_frame(args.body_frame), args.tf_max_age_ms)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')


if __name__ == '__main__':
    main()
