"""ROS 2 message adapters. Conversion functions return loggable Rerun archetypes."""
import numpy as np
import rerun as rr


def xyz(v):
    return [v.x, v.y, v.z]


def quaternion(q):
    return rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w])


def pose(msg, *, parent_frame, child_frame):
    """geometry_msgs/Pose. Supply a unique child frame for this estimate."""
    return rr.Transform3D(translation=xyz(msg.position), rotation=quaternion(msg.orientation),
                          parent_frame=parent_frame, child_frame=child_frame, axis_length=0.3)


def pose_stamped(msg, *, child_frame):
    return pose(msg.pose, parent_frame=msg.header.frame_id, child_frame=child_frame)


def odometry(msg):
    """Pose component of nav_msgs/Odometry."""
    return pose(msg.pose.pose, parent_frame=msg.header.frame_id, child_frame=msg.child_frame_id)


def transform(msg):
    """geometry_msgs/TransformStamped, as found in TFMessage.transforms."""
    t = msg.transform
    return rr.Transform3D(translation=xyz(t.translation), rotation=quaternion(t.rotation),
                          parent_frame=msg.header.frame_id, child_frame=msg.child_frame_id)


def pointcloud(msg):
    """sensor_msgs/PointCloud2 -> finite XYZ points. Other channels are omitted."""
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    fields = {f.name: f for f in msg.fields}
    columns = []
    for name in ("x", "y", "z"):
        f = fields.get(name)
        if f is None or f.count != 1 or f.datatype not in types:
            raise ValueError(f"Expected scalar numeric PointCloud2 field {name!r}")
        dtype = np.dtype((">" if msg.is_bigendian else "<") + types[f.datatype])
        if msg.width * msg.height == 0:
            return rr.Points3D(np.empty((0, 3), dtype=np.float32))
        values = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                            offset=f.offset, strides=(msg.row_step, msg.point_step))
        columns.append(values.ravel())
    points = np.column_stack(columns)
    return rr.Points3D(points[np.isfinite(points).all(axis=1)])


def laser_scan(msg):
    r = np.asarray(msg.ranges)
    angle = msg.angle_min + np.arange(r.size) * msg.angle_increment
    keep = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
    r, angle = r[keep], angle[keep]
    return rr.Points3D(np.column_stack((r*np.cos(angle), r*np.sin(angle), np.zeros_like(r))))


def path(msg):
    """nav_msgs/Path, whose positions must all use the path frame."""
    if any(p.header.frame_id not in ("", msg.header.frame_id) for p in msg.poses):
        raise ValueError("Transform path poses into a common frame first")
    points = [xyz(p.pose.position) for p in msg.poses]
    return rr.LineStrips3D([points] if points else [])


def image(msg, bridge):
    """sensor_msgs/Image -> RGB image using a cv_bridge.CvBridge instance."""
    return rr.Image(bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"))


def set_stamp(stamp):
    ns = stamp.sec * 1_000_000_000 + stamp.nanosec
    rr.set_time("ros_time", duration=np.timedelta64(ns, "ns"))


def log_stamped(entity, msg, data, *, spatial=False):
    """Use spatial=True for geometry expressed in header.frame_id."""
    set_stamp(msg.header.stamp)
    if spatial:
        rr.log(entity, data, rr.CoordinateFrame(frame=msg.header.frame_id))
    else:
        rr.log(entity, data)
