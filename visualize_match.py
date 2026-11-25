"""Inspect one synchronized MCAP LiDAR + stereo match in Rerun, offline.

Requires: ROS 2 Jazzy Python environment, numpy, PyYAML, opencv-python, rerun-sdk,
           sensor_msgs_py, rosbag2_py, scipy (for rotations).
No ROS node, subscriptions, tf2 listener, or depth estimator required.
"""

import argparse
from bisect import bisect_left, insort_right
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yaml
from scipy.spatial.transform import Rotation


def header_ns(msg):
    s = msg.header.stamp
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


def norm_frame(frame):
    return str(frame).lstrip('/')


def output_path(value, config_path):
    if not value:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else config_path.parent.parent / p


def read_messages(input_bag):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag), storage_id='mcap'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'
        ),
    )
    topic_types = {
        t.name: get_message(t.type) for t in reader.get_all_topics_and_types()
    }
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic in topic_types:
            yield topic, deserialize_message(data, topic_types[topic]), stamp


def matrix_from_tf(transform):
    q = transform.rotation
    t = transform.translation
    rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class OfflineTF:
    """TF graph for a selected instant; nearest dynamic sample within tolerance.

    T_parent_child maps coordinates in child into parent. Dynamic TF is NOT
    interpolated here; nearest-sample approximation is explicitly bounded.
    """

    def __init__(self):
        self.static = {}
        self.dynamic = defaultdict(list)

    def add(self, tf_msg, is_static):
        for entry in tf_msg.transforms:
            parent = norm_frame(entry.header.frame_id)
            child = norm_frame(entry.child_frame_id)
            if not parent or not child or parent == child:
                continue
            value = matrix_from_tf(entry.transform)
            key = (parent, child)
            if is_static:
                self.static[key] = value
            else:
                stamp = header_ns(entry)
                history = self.dynamic[key]
                if not history or stamp >= history[-1][0]:
                    history.append((stamp, value))
                else:
                    insort_right(history, (stamp, value), key=lambda pair: pair[0])

    def finalize(self):
        for values in self.dynamic.values():
            values.sort(key=lambda pair: pair[0])

    def lookup(self, target, source, at_ns, tolerance_ms):
        target, source = norm_frame(target), norm_frame(source)
        if source == target:
            return np.eye(4)
        edges = dict(self.static)
        skipped = []
        for key, values in self.dynamic.items():
            if key in edges or not values:
                continue
            index = bisect_left(values, at_ns, key=lambda pair: pair[0])
            candidates = [i for i in (index - 1, index) if 0 <= i < len(values)]
            selected = min(candidates, key=lambda i: abs(values[i][0] - at_ns))
            delta_ms = abs(values[selected][0] - at_ns) / 1e6
            if delta_ms <= tolerance_ms:
                edges[key] = values[selected][1]
            else:
                skipped.append((key, delta_ms))
        adjacency = defaultdict(list)
        for (parent, child), T_parent_child in edges.items():
            adjacency[child].append((parent, T_parent_child))
            adjacency[parent].append((child, np.linalg.inv(T_parent_child)))
        todo = deque([(source, np.eye(4))])
        visited = {source}
        while todo:
            current, T_current_source = todo.popleft()
            for neighbor, T_neighbor_current in adjacency[current]:
                if neighbor in visited:
                    continue
                result = T_neighbor_current @ T_current_source
                if neighbor == target:
                    return result
                visited.add(neighbor)
                todo.append((neighbor, result))
        available = sorted({f for pair in edges for f in pair})
        near = [f'{a}->{b}: {delta:.1f}ms' for (a, b), delta in skipped[:8]]
        raise RuntimeError(
            f'Cannot resolve TF {source!r} -> {target!r} at {at_ns}. '
            f'Available frames: {available}. Out-of-tolerance dynamic edges: {near}. '
            'Record /tf and /tf_static, or provide a valid fixed transform.'
        )


def image_rgb(msg):
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError('Compressed image is empty')
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Cannot decode compressed image')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def camera_parameters(info, width, height):
    K = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    D = np.asarray(info.d, dtype=np.float64)
    if (int(info.width), int(info.height)) != (width, height):
        raise ValueError(
            f'CameraInfo {info.width}x{info.height} does not match image {width}x{height}. '
            'Ensure CameraInfo corresponds to the selected image.'
        )
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        raise ValueError('Invalid focal lengths in CameraInfo K')
    # This expects RAW, unrectified images and a supported OpenCV distortion model.
    model = str(info.distortion_model)
    if model not in ('', 'plumb_bob', 'rational_polynomial'):
        raise ValueError(f'Unsupported distortion model {model!r}; adapt rectification first.')
    return K, D


def camera_calibration(info, rgb):
    K, D = camera_parameters(info, rgb.shape[1], rgb.shape[0])
    # cv2.undistort removes lens distortion, retaining the original K.
    undistorted = cv2.undistort(rgb, K, D if D.size else None, None, K)
    return K, undistorted


def cloud_xyz(msg, max_points):
    from sensor_msgs_py import point_cloud2

    # read_points_numpy is available in current ROS 2 sensor_msgs_py.
    if hasattr(point_cloud2, 'read_points_numpy'):
        pts = point_cloud2.read_points_numpy(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        )
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    else:
        pts = np.asarray(
            list(point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )), dtype=np.float64
        ).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > max_points:
        pts = pts[::int(np.ceil(len(pts) / max_points))]
    return pts


def transform_xyz(xyz, T):
    return xyz @ T[:3, :3].T + T[:3, 3]


def print_camera_geometry(left, right, platform_pose, cal, rgb):
    """Compare the logged RDF pinholes with section 2, Fig. 6 of the platform PDF."""
    def report(label, expected, actual, unit, angular=False):
        if not np.isfinite(actual):
            print(f'  {label}: expected {expected:g}{unit}, actual unavailable')
            return
        delta = actual - expected
        if angular:
            delta = (delta + 180) % 360 - 180
        print(f'  {label}: expected {expected:g}{unit}, actual {actual:.2f}{unit}, '
              f'delta {delta:+.2f}{unit}')

    baseline = right[:3, 3] - left[:3, 3]
    print('Camera geometry (PDF Fig. 6, positive yaw to the right)')
    report('Camera center spacing', 27, 100 * np.linalg.norm(baseline), ' cm')
    if platform_pose is None:
        # Estimate a mount frame independently of the cameras' outward headings.
        up = -(left[:3, 1] + right[:3, 1])
        forward = np.cross(up, baseline)
        print('  Estimated headings assume the baseline points right and mean camera down is vertical.')
    else:
        forward = platform_pose[:3, 0]
        up = platform_pose[:3, 2]
    lateral = np.cross(forward, up)
    headings = []
    for sensor, pose, expected in [('left', left, -32.5), ('right', right, 32.5)]:
        heading = np.nan
        if min(np.linalg.norm(forward), np.linalg.norm(lateral)) > 1e-8:
            axis = pose[:3, 2]
            x = axis @ forward / np.linalg.norm(forward)
            y = axis @ lateral / np.linalg.norm(lateral)
            if np.hypot(x, y) > 1e-8:
                heading = np.degrees(np.arctan2(y, x))
        headings.append(heading)
        report(f'{sensor} yaw', expected, heading, '°', angular=True)
        K, _ = cal[sensor]
        w = rgb[sensor].shape[1]
        # Fig. 6 uses the nominal BRIO FOV in plan view.
        hfov = np.degrees(np.arctan2(w - K[0, 2], K[0, 0]) +
                          np.arctan2(K[0, 2], K[0, 0]))
        report(f'{sensor} horizontal FOV (PDF plan-view assumption)', 90, hfov, '°')
    separation = (headings[1] - headings[0] + 180) % 360 - 180
    report('Right minus left yaw', 65, separation, '°', angular=True)


def project_cloud(xyz_cam, K, width, height, max_depth):
    z = xyz_cam[:, 2]
    visible = np.isfinite(xyz_cam).all(axis=1) & (z > 0.1) & (z < max_depth)
    points = xyz_cam[visible]
    depth = points[:, 2]
    uv = (points[:, :2] / depth[:, None]) @ K[:2, :2].T + K[:2, 2]
    valid = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return uv[valid].astype(np.float32), depth[valid]


def depth_colors(depth, max_depth):
    # BGR from OpenCV colormap -> RGB for Rerun
    scaled = np.clip(depth / max_depth * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled.reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0, ::-1].copy()


def find_match(csv_file, match_id):
    with Path(csv_file).open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if int(row['match_id']) == match_id:
                return {
                    'lidar': int(row['lidar_header_ns']),
                    'left': int(row['left_header_ns']),
                    'right': int(row['right_header_ns']),
                }
    raise ValueError(f'Match ID {match_id} not found in {csv_file}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    vis = cfg.get('visualization', {})
    match_id = int(vis['match_id'])
    if match_id < 1:
        raise ValueError('visualization.match_id must be >= 1')
    match_csv = output_path(cfg.get('output', {}).get('csv', 'sync_matches.csv'), config_path)
    if match_csv is None:
        raise ValueError('output.csv must point to the sync_analysis.py CSV')
    stamps = find_match(match_csv, match_id)
    bag_path = Path(cfg['bag_path']).expanduser()
    topics = cfg['topics']
    info_topics = vis.get('camera_info_topics', {
        'left': '/camera_left/camera_info',
        'right': '/camera_right/camera_info',
    })
    max_points = int(vis.get('max_lidar_points', 120_000))
    max_depth = float(vis.get('max_projection_depth_m', 80.0))
    tf_tolerance = float(vis.get('tf_max_age_ms', 100.0))
    if max_points <= 0 or max_depth <= 0 or tf_tolerance < 0:
        raise ValueError('Invalid visualization parameters')
    selected = {}
    infos = defaultdict(list)
    tf_graph = OfflineTF()
    print(f'Finding match {match_id} in {bag_path} ...')
    topic_to_sensor = {topic: sensor for sensor, topic in topics.items()}
    info_to_sensor = {topic: sensor for sensor, topic in info_topics.items()}
    for topic, msg, _bag_stamp in read_messages(bag_path):
        if topic == '/tf_static':
            tf_graph.add(msg, is_static=True)
        elif topic == '/tf':
            tf_graph.add(msg, is_static=False)
        elif topic in info_to_sensor:
            infos[info_to_sensor[topic]].append((header_ns(msg), msg))
        elif topic in topic_to_sensor:
            sensor = topic_to_sensor[topic]
            if sensor not in selected and header_ns(msg) == stamps[sensor]:
                selected[sensor] = msg
    tf_graph.finalize()
    missing = sorted(set(stamps) - set(selected))
    if missing:
        raise RuntimeError(f'Missing messages from MCAP: {missing}; check bag and CSV paths')

    rgb = {}
    cal = {}
    for sensor in ('left', 'right'):
        candidates = infos[sensor]
        if not candidates:
            raise RuntimeError(f'No CameraInfo messages on {info_topics[sensor]}')
        # Select nearest calibration by header stamp; zero-stamp calibration is common.
        chosen_ns, info = min(
            candidates, key=lambda pair: abs(pair[0] - stamps[sensor])
        )
        if chosen_ns and abs(chosen_ns - stamps[sensor]) > 1_000_000_000:
            print(f'WARNING: {sensor} CameraInfo is >1 s from image timestamp; inspect calibration changes.')
        im = image_rgb(selected[sensor])
        K, rgb[sensor] = camera_calibration(info, im)
        cal[sensor] = (K, info)
        image_frame = norm_frame(selected[sensor].header.frame_id)
        info_frame = norm_frame(info.header.frame_id)
        if image_frame != info_frame:
            raise RuntimeError(
                f'{sensor} image frame {image_frame!r} != CameraInfo frame {info_frame!r}'
            )

    lidar_frame = norm_frame(selected['lidar'].header.frame_id)
    left_frame = norm_frame(selected['left'].header.frame_id)
    right_frame = norm_frame(selected['right'].header.frame_id)
    lidar_time = stamps['lidar']
    fixed_frame = vis.get('fixed_frame')
    if fixed_frame:
        world_frame = norm_frame(fixed_frame)
        # Motion-aware when TF is recorded: poses evaluated at each sensor's time.
        T_world_lidar = tf_graph.lookup(world_frame, lidar_frame, stamps['lidar'], tf_tolerance)
        T_world_left = tf_graph.lookup(world_frame, left_frame, stamps['left'], tf_tolerance)
        T_world_right = tf_graph.lookup(world_frame, right_frame, stamps['right'], tf_tolerance)
        print('Using fixed frame and per-message TF timestamps:', world_frame)
    else:
        world_frame = left_frame
        # A single left optical reference instant; appropriate for a rigid, static rig.
        T_world_lidar = tf_graph.lookup(left_frame, lidar_frame, lidar_time, tf_tolerance)
        T_world_left = np.eye(4)
        T_world_right = tf_graph.lookup(left_frame, right_frame, lidar_time, tf_tolerance)
        print('Using left optical frame at LiDAR time. Moving platforms need a fixed frame + dynamic TF.')

    platform_frame = vis.get('platform_frame', 'base_link')
    try:
        platform_pose = tf_graph.lookup(world_frame, platform_frame, lidar_time, tf_tolerance)
        print(f'Platform heading reference: {platform_frame} at LiDAR time (+X forward, +Z up)')
    except RuntimeError:
        print(f'Platform frame {platform_frame!r} unavailable at LiDAR time.')
        platform_pose = None
    print_camera_geometry(T_world_left, T_world_right, platform_pose, cal, rgb)

    xyz_lidar = cloud_xyz(selected['lidar'], max_points)
    xyz_world = transform_xyz(xyz_lidar, T_world_lidar)
    if not len(xyz_world):
        raise RuntimeError('Selected LiDAR point cloud has no valid XYZ points')

    rr.init(f'visualize_match_{match_id}')
    rrd_path = output_path(vis.get('rrd'), config_path)
    if rrd_path:
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(rrd_path))
    else:
        rr.spawn()
    rr.set_time('ros_time', timestamp=np.datetime64(lidar_time, 'ns'))
    rr.log('scene', rr.ViewCoordinates.RDF, static=True)
    rr.log('scene/lidar_points', rr.Points3D(
        xyz_world.astype(np.float32), colors=[75, 210, 250], radii=0.035
    ))

    for sensor, T_world_cam, color in (
        ('left', T_world_left, [45, 155, 250]),
        ('right', T_world_right, [255, 160, 55]),
    ):
        K, _info = cal[sensor]
        im = rgb[sensor]
        H, W = im.shape[:2]
        camera_path = f'scene/{sensor}/pose/camera'
        rr.log(f'scene/{sensor}/pose', rr.Transform3D(
            translation=T_world_cam[:3, 3],
            quaternion=rr.Quaternion(xyzw=Rotation.from_matrix(T_world_cam[:3, :3]).as_quat()),
        ))
        rr.log(camera_path, rr.Pinhole(
            image_from_camera=K, resolution=[W, H],
            image_plane_distance=float(vis.get('image_plane_distance_m', 1.0)),
        ))
        rr.log(f'{camera_path}/image', rr.Image(im))

        T_cam_world = np.linalg.inv(T_world_cam)
        xyz_cam = transform_xyz(xyz_world, T_cam_world)
        uv, depth = project_cloud(xyz_cam, K, W, H, max_depth)
        # 2D overlay in the *same camera* with LiDAR points colored by distance.
        # These are undistorted images, not raw distorted pixels.
        rr.log(f'overlays/{sensor}', rr.Image(im))
        if len(uv):
            rr.log(f'overlays/{sensor}/points', rr.Points2D(
                uv, colors=depth_colors(depth, max_depth), radii=float(vis.get('overlay_radius_px', 2.0))
            ))
        print(f'{sensor}: {len(uv)} projected points in image, frame={norm_frame(selected[sensor].header.frame_id)}')

    print(f'Match {match_id}:')
    for sensor in ('lidar', 'left', 'right'):
        print(f'  {sensor}: stamp_ns={stamps[sensor]}  frame={norm_frame(selected[sensor].header.frame_id)}')
    print(f'  left-lidar: {(stamps["left"] - lidar_time)/1e6:+.3f} ms')
    print(f'  right-lidar: {(stamps["right"] - lidar_time)/1e6:+.3f} ms')
    if rrd_path:
        print(f'RRD: {rrd_path}')
    # Explicit views keep each image and its projected 2D points together.
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial3DView(name='3D sensor alignment', origin='scene'),
        rrb.Spatial2DView(name='Left LiDAR overlay', origin='overlays/left'),
        rrb.Spatial2DView(name='Right LiDAR overlay', origin='overlays/right'),
    ))
    rr.disconnect()


if __name__ == '__main__':
    main()
