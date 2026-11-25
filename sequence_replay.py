"""Stream recorded sensor arrivals and assemble ready matches for inference."""

from bisect import bisect_right, insort_right
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import perf_counter_ns, sleep
from types import SimpleNamespace

from mcap.reader import NonSeekingReader
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rosbags.typesys import Stores, get_typestore

from spnet_inputs import lidar_xyz
from visualize_match import OfflineTF, header_ns, norm_frame


SENSORS = ('left', 'right', 'lidar')
CAMERAS = ('left', 'right')
NS_PER_SECOND = 10**9


@dataclass
class SensorSample:
    path: Path
    stamp_ns: int
    frame: str
    arrival_ns: int


class MatchAssembler:
    def __init__(self, match_csv, spool, cameras, world_frame, tf_max_age_ms):
        self.spool = spool
        self.cameras = cameras
        self.world_frame = norm_frame(world_frame)
        self.tf_max_age_ms = tf_max_age_ms
        self.match_ids = []
        self.by_stamp = {sensor: {} for sensor in SENSORS}
        with match_csv.open(newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                match_id = int(row['match_id'])
                self.match_ids.append(match_id)
                for sensor in SENSORS:
                    stamp = int(row[f'{sensor}_header_ns'])
                    if stamp in self.by_stamp[sensor]:
                        raise ValueError(f'Duplicate {sensor} timestamp in the match CSV: {stamp}')
                    self.by_stamp[sensor][stamp] = match_id
        if not self.match_ids or len(self.match_ids) != len(set(self.match_ids)):
            raise ValueError('The match CSV must contain unique match IDs')

        self.pending = {}
        self.complete = set()
        self.submitted = set()
        self.waiting = {}
        self.calibrations = {name: {} for name in CAMERAS}
        self.transforms = OfflineTF()

    def add_calibration(self, name, message):
        frame = norm_frame(message.header.frame_id)
        history = self.calibrations[name].setdefault(frame, [])
        info = SimpleNamespace(width=message.width, height=message.height,
                               k=message.k.tolist(), d=message.d.tolist(),
                               distortion_model=message.distortion_model)
        insort_right(history, (header_ns(message), info), key=lambda entry: entry[0])

    def calibration_for(self, name, sample):
        history = self.calibrations[name].get(sample.frame)
        if not history:
            raise RuntimeError(f'missing {name} CameraInfo for {sample.frame}')
        index = bisect_right(history, sample.stamp_ns, key=lambda entry: entry[0])
        return history[max(0, index - 1)][1]

    def record_sample(self, match_id, name, message, payload, arrival_ns):
        if match_id is None or match_id in self.submitted:
            return
        samples = self.pending.setdefault(match_id, {})
        if name in samples:
            return
        path = self.spool / f'input-{match_id}-{name}'
        if name == 'lidar':
            with path.open('wb') as file:
                np.save(file, payload, allow_pickle=False)
        else:
            path.write_bytes(payload)
        samples[name] = SensorSample(path, header_ns(message), norm_frame(message.header.frame_id), arrival_ns)
        if len(samples) == len(SENSORS):
            self.complete.add(match_id)

    def ready_jobs(self, origin_ns):
        for match_id in sorted(self.complete):
            samples = self.pending[match_id]
            try:
                calibrations = {name: self.calibration_for(name, samples[name]) for name in self.cameras}
                poses = {}
                for name in ('lidar', *self.cameras):
                    sample = samples[name]
                    poses[name] = self.transforms.lookup(
                        self.world_frame, sample.frame, sample.stamp_ns, self.tf_max_age_ms,
                    )
            except RuntimeError as error:
                self.waiting[match_id] = str(error)
                continue

            self.waiting.pop(match_id, None)
            self.pending.pop(match_id)
            self.complete.remove(match_id)
            self.submitted.add(match_id)
            yield {
                'match_id': match_id, 'samples': samples, 'calibrations': calibrations,
                'poses': poses, 'world_frame': self.world_frame, 'origin_ns': origin_ns,
                'ready_ns': perf_counter_ns(),
            }

    def incomplete_matches(self):
        incomplete = []
        for match_id in self.match_ids:
            if match_id in self.submitted:
                continue
            missing = [name for name in SENSORS if name not in self.pending.get(match_id, {})]
            reason = f'missing {", ".join(missing)}' if missing else self.waiting[match_id]
            incomplete.append({'match_id': match_id, 'reason': reason})
        return incomplete


def set_time(recording, replay_ns, match_id=None, capture_ns=None):
    recording.reset_time()
    recording.set_time('replay', duration=np.timedelta64(replay_ns, 'ns'))
    if match_id is not None:
        recording.set_time('match', sequence=match_id)
    if capture_ns is not None:
        recording.set_time('sensor_capture', timestamp=np.datetime64(capture_ns, 'ns'))


def log_sensor(recording, assembler, name, message, payload, replay_ns, match_id):
    stamp = header_ns(message)
    set_time(recording, replay_ns, match_id, stamp)
    if name in CAMERAS:
        media_type = 'image/png' if payload.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg'
        recording.log(f'input/{name}', rr.EncodedImage(contents=payload, media_type=media_type))
        return

    frame = norm_frame(message.header.frame_id)
    try:
        pose = assembler.transforms.lookup(assembler.world_frame, frame, stamp, assembler.tf_max_age_ms)
    except RuntimeError:
        # Preserve the arriving cloud in sensor coordinates until a world pose is available.
        recording.log('scene/measured_lidar', rr.Clear(recursive=False))
        recording.log('input/lidar_native', rr.Points3D(payload, radii=0.005))
        return
    points = (payload @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    recording.log('scene/measured_lidar', rr.Points3D(points, colors=[65, 220, 255], radii=0.005))


def log_result(recording, path, elapsed_ns, match_id, capture_ns, timing, max_depth):
    set_time(recording, elapsed_ns, match_id, capture_ns)
    for name, value in timing.items():
        recording.log(f'timing/{name}', rr.Scalars([value]))
    with np.load(path, allow_pickle=False) as data:
        for name in CAMERAS:
            if f'{name}_rgb' not in data:
                continue
            rgb = data[f'{name}_rgb']
            recording.log(f'output/{name}/rgb', rr.Image(rgb))
            for kind in ('sparse', 'depth'):
                recording.log(f'output/{name}/{kind}', rr.DepthImage(
                    data[f'{name}_{kind}'], meter=1, depth_range=[0, max_depth],
                ))
            xyxy = data[f'{name}_xyxy']
            if len(xyxy):
                recording.log(f'output/{name}/rgb/yolo', rr.Boxes2D(
                    mins=xyxy[:, :2], sizes=xyxy[:, 2:] - xyxy[:, :2], labels=data[f'{name}_labels'],
                ))
            else:
                recording.log(f'output/{name}/rgb/yolo', rr.Clear(recursive=False))
            strips = []
            for corners in data[f'{name}_corners']:
                strips.extend([corners[[0, 1, 2, 3, 0]], corners[[4, 5, 6, 7, 4]]])
                strips.extend(corners[[i, i + 4]] for i in range(4))
            color = [255, 183, 76] if name == 'left' else [182, 133, 255]
            entity = f'scene/detections/{name}'
            if strips:
                recording.log(entity, rr.LineStrips3D(strips, colors=color))
                recording.log(f'{entity}/labels', rr.Points3D(
                    data[f'{name}_centers'], colors=color, radii=0.005,
                    labels=data[f'{name}_box_labels'],
                ))
            else:
                recording.log(entity, rr.Clear(recursive=True))


def result_publisher(recording, results_queue, jobs_queue, max_depth):
    try:
        while (item := results_queue.get()) is not None:
            path, elapsed_ns, match_id, capture_ns, timing = item
            log_result(recording, path, elapsed_ns, match_id, capture_ns, timing, max_depth)
            path.unlink()
    except Exception as error:
        jobs_queue.put(('error', f'Rerun result publishing failed: {error}'))


def recording_blueprint(cameras, live):
    views = [rrb.Spatial3DView(name='Live LiDAR and 3D detections', origin='scene')]
    for name in cameras:
        views.append(rrb.Horizontal(
            rrb.Spatial2DView(name='Incoming RGB', origin=f'input/{name}'),
            rrb.Spatial2DView(name='YOLO result', origin=f'output/{name}/rgb'),
            rrb.Spatial2DView(name='SPNet depth', origin=f'output/{name}/depth'),
            name=name.title(),
        ))
    views.append(rrb.TimeSeriesView(name='Pipeline timing', origin='timing'))
    time_panel = rrb.TimePanel(timeline='replay', play_state='following' if live else 'playing')
    return rrb.Blueprint(rrb.Tabs(*views), time_panel, auto_views=False)


def stream_bag(bag_path, sensor_config, assembler, recording, jobs_queue, rate):
    store = get_typestore(Stores.ROS2_HUMBLE)
    sensors = {topic: name for name, topic in sensor_config['topics'].items()}
    infos = {topic: name for name, topic in sensor_config['visualization']['camera_info_topics'].items()}
    topics = [*sensors, *infos, '/tf', '/tf_static']
    messages_by_topic = Counter()
    arrival_lateness_ms = []
    first_log_ns = None
    previous_log_ns = None
    out_of_order = 0
    origin_ns = perf_counter_ns()
    with bag_path.open('rb') as file:
        for schema, channel, record in NonSeekingReader(file).iter_messages(topics=topics, log_time_order=False):
            topic = channel.topic
            if first_log_ns is None:
                first_log_ns = record.log_time
                origin_ns = perf_counter_ns()
            if previous_log_ns is not None and record.log_time < previous_log_ns:
                out_of_order += 1
            previous_log_ns = record.log_time
            expected_ns = origin_ns
            if rate:
                expected_ns += round((record.log_time - first_log_ns) / rate)
                remaining_ns = expected_ns - perf_counter_ns()
                if remaining_ns > 0:
                    sleep(remaining_ns / NS_PER_SECOND)
            arrival_ns = perf_counter_ns()
            messages_by_topic[topic] += 1

            if schema is None:
                raise ValueError(f'MCAP topic {topic} has no ROS message schema')
            msg_type = schema.name
            if msg_type not in store.types:
                msg_type = msg_type.replace('/', '/msg/', 1)
            message = store.deserialize_cdr(record.data, msg_type)
            if topic in sensors:
                name = sensors[topic]
                stamp = header_ns(message)
                match_id = assembler.by_stamp[name].get(stamp)
                if name == 'lidar':
                    points = lidar_xyz(message)
                    payload = points[np.isfinite(points).all(axis=1)]
                else:
                    payload = bytes(message.data)
                log_sensor(recording, assembler, name, message, payload, arrival_ns - origin_ns, match_id)
                if rate:
                    arrival_lateness_ms.append((arrival_ns - expected_ns) / 1e6)
                assembler.record_sample(match_id, name, message, payload, arrival_ns)
            elif topic in infos:
                assembler.add_calibration(infos[topic], message)
            else:
                assembler.transforms.add(message, is_static=topic == '/tf_static')
            for job in assembler.ready_jobs(origin_ns):
                jobs_queue.put(('match', job))

    replay_done_ns = perf_counter_ns()
    return {
        'messages_by_topic': dict(messages_by_topic),
        'recorded_seconds': (previous_log_ns - first_log_ns) / NS_PER_SECOND if first_log_ns is not None else 0,
        'replay_seconds': (replay_done_ns - origin_ns) / NS_PER_SECOND,
        'replay_origin_ns': origin_ns,
        'replay_done_ns': replay_done_ns,
        'out_of_order_messages': out_of_order,
        'sensor_arrival_lateness_ms': arrival_lateness_ms,
        'expected_matches': len(assembler.match_ids),
        'incomplete_matches': assembler.incomplete_matches(),
    }


def replay_process(bag_path, match_csv, sensor_config, settings, spool, output, viewer, rate,
                   max_depth, jobs_queue, results_queue):
    try:
        rr.set_strict_mode(True)
        assembler = MatchAssembler(
            match_csv, spool, settings['cameras'], settings['world_frame'],
            sensor_config['visualization']['tf_max_age_ms'],
        )
        with rr.RecordingStream('Detection sequence') as recording:
            sinks = [rr.FileSink(output)]
            if viewer:
                sinks.append(rr.GrpcSink(viewer))
            recording.set_sinks(*sinks, default_blueprint=recording_blueprint(settings['cameras'], bool(viewer)))
            recording.log('scene', rr.ViewCoordinates.FLU, static=True)
            publisher = Thread(target=result_publisher, daemon=True,
                               args=(recording, results_queue, jobs_queue, max_depth))
            publisher.start()
            stats = stream_bag(bag_path, sensor_config, assembler, recording, jobs_queue, rate)
            jobs_queue.put(('end', stats))
            publisher.join()
            recording.flush()
            recording.disconnect()
        jobs_queue.put(('saved', None))
    except Exception as error:
        jobs_queue.put(('error', str(error)))
