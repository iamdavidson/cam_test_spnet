"""Copy a ROS 2 MCAP, setting base_link -> camera_right X to the measured left X.

Run: python3 correct_camera_bag.py [--config config/config.yaml]
Dependencies: mcap, rosbags, PyYAML.
"""
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml
from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer
from rosbags.typesys import Stores, get_typestore


def correct_bag(source):
    target = source.with_name(f'{source.stem}_corrected.mcap')
    if target.is_file():
        print(f'Skipped correction. Output already exists: {target}')
        return
    store = get_typestore(Stores.ROS2_HUMBLE)
    tf_type = 'tf2_msgs/msg/TFMessage'

    def camera_entries(msg, name):
        return (tf for tf in msg.transforms if tf.header.frame_id.lstrip('/') == 'base_link'
                and tf.child_frame_id.lstrip('/') == name)

    with source.open('rb') as src:
        reader = make_reader(src)
        left_x = set()
        for _, _, record in reader.iter_messages(topics=['/tf_static']):
            msg = store.deserialize_cdr(record.data, tf_type)
            left_x.update(tf.transform.translation.x for tf in camera_entries(msg, 'camera_left'))
        if len(left_x) != 1:
            raise ValueError(f'Expected one static left camera X coordinate, found {left_x}')
        forward = left_x.pop()
        summary = reader.get_summary()
        if summary is None or not summary.channels:
            raise ValueError('Expected a ROS bag MCAP with a channel summary')

        with NamedTemporaryFile(dir=source.parent, suffix='.mcap') as tmp:
            writer = Writer(tmp, compression=CompressionType.NONE)
            writer.start(profile=reader.get_header().profile)
            schemas = {s.id: writer.register_schema(s.name, s.encoding, s.data)
                       for s in summary.schemas.values()}
            channels = {c.id: writer.register_channel(c.topic, c.message_encoding,
                                                      schemas.get(c.schema_id, 0), c.metadata)
                        for c in summary.channels.values()}
            previous = set()
            for _, channel, record in reader.iter_messages(log_time_order=False):
                data = record.data
                if channel.topic == '/tf_static':
                    msg = store.deserialize_cdr(data, tf_type)
                    entries = list(camera_entries(msg, 'camera_right'))
                    for tf in entries:
                        previous.add(tf.transform.translation.x)
                        tf.transform.translation.x = forward
                    if entries:
                        data = bytes(store.serialize_cdr(msg, tf_type))
                writer.add_message(channels[channel.id], record.log_time, data,
                                   record.publish_time, record.sequence)
            if not previous:
                raise ValueError('Missing static base_link -> camera_right transform')
            for m in reader.iter_metadata():
                writer.add_metadata(m.name, m.metadata)
            for a in reader.iter_attachments():
                writer.add_attachment(a.create_time, a.log_time, a.name, a.media_type, a.data)
            writer.finish()
            tmp.flush()
            os.link(tmp.name, target)  # Publish the finished copy without overwriting.
    print(f'Right camera X: {sorted(previous)} -> {forward:g} m. Saved {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/config.yaml'))
    args = parser.parse_args()
    with args.config.open() as config:
        correct_bag(Path(yaml.safe_load(config)['bag_path']).expanduser())
