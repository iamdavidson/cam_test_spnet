#!/usr/bin/env python3
"""Overlay 3D detections from selected benchmark runs in one Rerun recording."""

import argparse
import colorsys
from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation


@dataclass
class Run:
    report_path: Path
    match: dict
    detections: dict

    @property
    def name(self):
        if self.report_path.name == 'report.json':
            return self.report_path.parent.name
        return self.report_path.stem


def read_run(path):
    path = path.expanduser()
    report_path = (path / 'report.json' if path.is_dir() else path).resolve()
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        match = report['match']
        detections = report['detections']
        if not match['world_frame'] or not match['header_stamps_ns'] or not isinstance(detections, dict):
            raise ValueError('missing world frame, sensor timestamps, or detections')
        for camera in detections.values():
            if not isinstance(camera['boxer_3d'], list):
                raise ValueError('Boxer detections must be a list')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{report_path}: expected a detection benchmark report ({error})') from error
    return Run(report_path, match, detections)


def selection_paths(selection_file):
    selection_file = selection_file.expanduser()
    entries = json.loads(selection_file.read_text(encoding='utf-8'))
    if not isinstance(entries, list) or any(not isinstance(entry, str) or not entry for entry in entries):
        raise ValueError(f'{selection_file}: expected a JSON list of run or report paths')
    return [selection_file.parent / Path(entry).expanduser() for entry in entries]


def align_run(reference, run, shared_world):
    """Return the transform from the run's frame to the first run's frame."""
    reference_match = reference.match
    run_match = run.match
    same_capture = (reference_match['header_stamps_ns'] == run_match['header_stamps_ns']
                    and reference_match['frames'] == run_match['frames'])
    same_world = reference_match['world_frame'] == run_match['world_frame']
    if not same_capture:
        if shared_world and same_world:
            return np.eye(4)
        raise ValueError(f'{run.name}: different sensor capture from {reference.name}. '
                         'Use --shared-world only when both runs use the same fixed world frame.')
    if same_world:
        return np.eye(4)

    # Both runs observe this camera at the same instant, so its pose ties the frames together.
    reference_cameras = reference_match['cameras']
    run_cameras = run_match['cameras']
    transforms = []
    for camera_name in sorted(reference_cameras.keys() & run_cameras.keys()):
        reference_from_camera = np.asarray(reference_cameras[camera_name]['world_from_camera'])
        run_from_camera = np.asarray(run_cameras[camera_name]['world_from_camera'])
        transforms.append(reference_from_camera @ np.linalg.inv(run_from_camera))
    if not transforms:
        raise ValueError(f'{run.name}: different world frame and no shared camera for alignment')
    if not all(np.allclose(pose, transforms[0], atol=1e-4, rtol=0) for pose in transforms):
        raise ValueError(f'{run.name}: the shared cameras disagree about the frame alignment')
    return transforms[0]


def run_colors(count):
    if count > 256**3:
        raise ValueError('Too many runs to assign each one a unique RGB color')
    colors = []
    used = set()
    hue = 0.55
    for _ in range(count):
        # Rotate around the color wheel so consecutive runs get visibly different colors.
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        color = tuple(round(channel * 255) for channel in rgb)
        if color in used:
            color_number = int.from_bytes(bytes(color), byteorder='big')
            while color in used:
                color_number = (color_number + 1) % 256**3
                color = tuple(color_number.to_bytes(3, byteorder='big'))
        colors.append(color)
        used.add(color)
        hue = (hue + 0.618034) % 1
    return colors


def write_camera_boxes(recording, entity, boxes, color, reference_from_run):
    centers, sizes, rotations, labels = [], [], [], []
    for box in boxes:
        corners = np.asarray(box['corners_m'], dtype=float)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError(f'{entity}: expected eight finite 3D corners')
        corners = corners @ reference_from_run[:3, :3].T + reference_from_run[:3, 3]
        # In the saved corner order, corners 1, 3 and 4 share an edge with corner 0.
        axes = corners[[1, 3, 4]] - corners[0]
        size = np.linalg.norm(axes, axis=1)
        if np.any(size <= 0):
            raise ValueError(f'{entity}: found a box with a zero-length edge')
        orientation = (axes / size[:, None]).T
        centers.append(corners.mean(axis=0))
        sizes.append(size)
        rotations.append(Rotation.from_matrix(orientation).as_quat())
        labels.append(f"{box['label']} {box['boxer_score']:.2f}")

    if boxes:
        recording.log(entity, rr.Boxes3D(
            centers=centers, sizes=sizes, quaternions=rotations,
            colors=[color], labels=labels, show_labels=False,
        ), static=True)


def write_comparison(output, runs, transforms, colors):
    legend = [f"Reference frame: {runs[0].match['world_frame']}",
              'Toggle run layers in the Rerun Blueprint panel.', '']
    with rr.RecordingStream('3D detection comparison') as recording:
        recording.save(str(output))
        recording.log('comparison', rr.ViewCoordinates.FLU, static=True)

        for index, (run, transform, color) in enumerate(zip(runs, transforms, colors), start=1):
            run_name = rr.escape_entity_path_part(f'{index:02d} {run.name}')
            box_count = 0
            for camera_name, camera in run.detections.items():
                boxes = camera['boxer_3d']
                camera_entity = rr.escape_entity_path_part(camera_name)
                write_camera_boxes(recording, f'comparison/{run_name}/{camera_entity}', boxes, color, transform)
                box_count += len(boxes)
            color_hex = '#' + ''.join(f'{channel:02X}' for channel in color)
            legend.extend([f'{index:02d}  {color_hex}  {box_count} boxes', f'    {run.name}', ''])

        recording.log('legend', rr.TextDocument('\n'.join(legend)), static=True)
        recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(name='Overlaid 3D detections', origin='comparison'),
            rrb.TextDocumentView(name='Selected runs', origin='legend'),
            column_shares=[3, 2],
        )))
        recording.flush()
    return legend[3:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='*', type=Path, help='run directories or report.json files')
    parser.add_argument('--selection', type=Path, help='JSON list of paths relative to the selection file')
    parser.add_argument('--output', type=Path, default=Path('comparison.rrd'))
    parser.add_argument('--shared-world', action='store_true',
                        help='allow different captures already expressed in the same fixed world frame')
    args = parser.parse_args()

    try:
        paths = list(args.runs)
        if args.selection:
            paths.extend(selection_paths(args.selection))
        runs_by_path = {}
        for path in paths:
            run = read_run(path)
            runs_by_path[run.report_path] = run
        runs = list(runs_by_path.values())
        if len(runs) < 2:
            raise ValueError('Select at least two different detection runs')
        transforms = [align_run(runs[0], run, args.shared_world) for run in runs]

        output = args.output.expanduser().absolute()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'{output} already exists. Choose another --output path.')
        rr.set_strict_mode(True)
        with TemporaryDirectory(dir=output.parent) as directory:
            temporary_output = Path(directory) / 'comparison.rrd'
            legend = write_comparison(temporary_output, runs, transforms, run_colors(len(runs)))
            os.link(temporary_output, output)  # Publish the finished file without replacing another run.
    except (KeyError, OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')

    print('\n'.join(legend))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
