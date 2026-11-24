"""Benchmark YOLO → SPNet → Boxer on a synchronized camera/LiDAR match."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import yaml

from benchmark_spnet import transform
from detection_models import (
    Detections2D,
    load_boxer,
    load_spnet,
    load_yolo,
    make_sparse_range_map,
    perform_boxer,
    perform_spnet,
    perform_yolo,
    resized_intrinsics,
)
from detection_profiling import Profiler, summarize
from report_printer import print_report
from spnet_inputs import lidar_xyz, load_triplet
from visualize_match import camera_parameters, find_match, image_rgb


ROOT = Path(__file__).resolve().parent


@dataclass
class CameraInput:
    message: object
    maps: tuple[np.ndarray, np.ndarray]
    input_wh: tuple[int, int]
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    camera_from_lidar: np.ndarray


@dataclass
class Scene:
    cameras: dict[str, CameraInput]
    lidar_points: torch.Tensor
    world_from_lidar: torch.Tensor
    metadata: dict


@dataclass
class CameraResult:
    rgb_image: np.ndarray
    detections_2d: Detections2D
    sparse_range_map: torch.Tensor
    dense_range_map: torch.Tensor
    detections_3d: list[dict]


def run_camera(camera, lidar_points, yolo_model, spnet_model, boxer_model, config, profiler):
    """Run all three models once for one camera image."""
    with profiler.measure() as (timing, input_metrics):
        rgb_image = read_camera_image(camera, timing)

    with profiler.measure() as (timing, yolo_metrics):
        detections_2d = perform_yolo(yolo_model, rgb_image, config['yolo'], lidar_points.device, timing)

    spnet_settings = config['spnet']
    with profiler.measure() as (timing, spnet_metrics):
        sparse_range_map = make_sparse_range_map(
            lidar_points, camera, rgb_image.shape, spnet_settings, timing,
        )
        dense_range_map = perform_spnet(
            spnet_model, rgb_image, sparse_range_map, spnet_settings['max_depth_m'], timing,
        )

    with profiler.measure() as (timing, boxer_metrics):
        detections_3d = perform_boxer(
            boxer_model, detections_2d, dense_range_map, rgb_image, camera, config['boxer'], timing,
        )

    result = CameraResult(
        rgb_image=rgb_image,
        detections_2d=detections_2d,
        sparse_range_map=sparse_range_map,
        dense_range_map=dense_range_map,
        detections_3d=detections_3d,
    )
    metrics = {
        'Camera input': input_metrics,
        'YOLO': yolo_metrics,
        'SPNet': spnet_metrics,
        'Boxer': boxer_metrics,
    }
    return result, metrics


def read_camera_image(camera, timing):
    with timing.stage('Decode compressed image'):
        rgb_image = image_rgb(camera.message)

    with timing.stage('Rectify and resize RGB'):
        height, width = rgb_image.shape[:2]
        if (width, height) != camera.input_wh:
            raise ValueError('CameraInfo dimensions differ from the decoded image')
        map_x, map_y = camera.maps
        rgb_image = cv2.remap(
            rgb_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
    return rgb_image


def run_match(scene, yolo_model, spnet_model, boxer_model, config, profiler):
    """One pass processes each selected camera once."""
    started = profiler.start()
    results = {}
    camera_metrics = {}
    peak_mib = 0
    for name, camera in scene.cameras.items():
        result, metrics = run_camera(
            camera, scene.lidar_points, yolo_model, spnet_model, boxer_model, config, profiler,
        )
        results[name] = result
        camera_metrics[name] = metrics
        for stage in metrics.values():
            peak_mib = max(peak_mib, stage['cuda_peak_allocated_mib'])

    overall = profiler.finish(started, peak_mib * 2**20)
    return results, {'overall': overall, 'cameras': camera_metrics}


def load_scene(config_path, settings, device):
    sensors = yaml.safe_load((config_path.parent / settings['sensor_config']).read_text())
    bag = ROOT / Path(settings['bag_path'] or sensors['bag_path']).expanduser()
    csv = ROOT / Path(settings['match_csv']).expanduser()
    stamps = find_match(csv, settings['match_id'])
    vis = sensors['visualization']
    lidar_msg, lidar_pose, cameras, world, frames = load_triplet(
        bag, sensors['topics'], vis['camera_info_topics'], stamps,
        settings['world_frame'], vis['tf_max_age_ms'],
    )
    width, height = settings['image_wh']
    selected = {}
    for name in settings['cameras']:
        message, info, pose = cameras[name]
        input_wh = int(info.width), int(info.height)
        intrinsics, distortion = camera_parameters(info, *input_wh)
        rectified_intrinsics = resized_intrinsics(
            intrinsics, width / input_wh[0], height / input_wh[1],
        )
        maps = cv2.initUndistortRectifyMap(
            intrinsics, distortion if distortion.size else None, None, rectified_intrinsics,
            (width, height), cv2.CV_32FC1,
        )
        selected[name] = CameraInput(
            message=message,
            maps=maps,
            input_wh=input_wh,
            intrinsics=rectified_intrinsics.astype(np.float32),
            world_from_camera=pose.astype(np.float32),
            camera_from_lidar=np.linalg.solve(pose, lidar_pose).astype(np.float32),
        )
    lidar = lidar_xyz(lidar_msg)
    lidar = lidar[np.isfinite(lidar).all(axis=1)]
    if not len(lidar):
        raise ValueError('The selected LiDAR scan has no finite points')
    metadata = {
        'match_id': settings['match_id'], 'bag': str(bag), 'match_csv': str(csv),
        'header_stamps_ns': stamps, 'frames': frames, 'world_frame': world,
        'lidar_points': len(lidar),
        'cameras': {name: {
            'intrinsics': camera.intrinsics.tolist(),
            'world_from_camera': camera.world_from_camera.tolist(),
        } for name, camera in selected.items()},
    }
    return Scene(selected, torch.as_tensor(lidar, device=device),
                 torch.as_tensor(lidar_pose, dtype=torch.float32, device=device), metadata)


def write_scene(path, scene, results, max_depth):
    lidar_world = transform(scene.lidar_points, scene.world_from_lidar).cpu().numpy()
    colors = {'left': [255, 183, 76], 'right': [182, 133, 255]}
    with rr.RecordingStream(f"Detection match {scene.metadata['match_id']}") as recording:
        recording.save(str(path))
        recording.log('scene', rr.ViewCoordinates.FLU, static=True)
        recording.log('scene/measured_lidar',
                      rr.Points3D(lidar_world, colors=[65, 220, 255], radii=0.005), static=True)
        views = [rrb.Spatial3DView(name='LiDAR and Boxer detections', origin='scene')]
        for name, result in results.items():
            recording.log(f'{name}/rgb', rr.Image(result.rgb_image), static=True)
            for label, depth in [('sparse', result.sparse_range_map),
                                 ('completed', result.dense_range_map)]:
                recording.log(f'{name}/{label}', rr.DepthImage(depth.cpu().numpy(), meter=1,
                              depth_range=[0, max_depth]), static=True)
            xyxy = result.detections_2d.xyxy.numpy()
            if len(xyxy):
                labels = [f'{label} {score:.2f}' for label, score in
                          zip(result.detections_2d.labels, result.detections_2d.scores.tolist())]
                recording.log(f'{name}/rgb/yolo', rr.Boxes2D(
                    mins=xyxy[:, :2], sizes=xyxy[:, 2:] - xyxy[:, :2], labels=labels,
                ), static=True)
            for index, box in enumerate(result.detections_3d):
                corners = np.array(box['corners_m'])
                strips = [corners[[0, 1, 2, 3, 0]], corners[[4, 5, 6, 7, 4]]]
                strips += [corners[[i, i + 4]] for i in range(4)]
                entity = f'scene/{name}/box_{index}'
                recording.log(entity, rr.LineStrips3D(strips, colors=colors[name]), static=True)
                label = f"{box['label']} {box['boxer_score']:.2f} ({name})"
                recording.log(f'{entity}/label', rr.Points3D(
                    [box['center_m']], labels=[label], radii=0.005, colors=colors[name],
                ), static=True)
            views.append(rrb.Horizontal(
                rrb.Spatial2DView(name='YOLO detections', origin=f'{name}/rgb'),
                rrb.Spatial2DView(name='LiDAR depth', origin=f'{name}/sparse'),
                rrb.Spatial2DView(name='SPNet depth', origin=f'{name}/completed'),
                name=name.title(),
            ))
        recording.send_blueprint(rrb.Blueprint(rrb.Tabs(*views)))
        recording.flush()


def validate(config):
    pipeline, spnet, boxer = (config[name] for name in ('pipeline', 'spnet', 'boxer'))
    cameras = pipeline['cameras']
    if not cameras or len(set(cameras)) != len(cameras) or set(cameras) - {'left', 'right'}:
        raise ValueError('pipeline.cameras must select left, right, or both')
    if not pipeline['world_frame']:
        raise ValueError('pipeline.world_frame must name a frame with +Z up')
    if pipeline['iterations'] < 1 or pipeline['warmup'] < 0 or pipeline['resource_sample_ms'] <= 0:
        raise ValueError('Specify positive iterations and resource_sample_ms, and nonnegative warmup')
    if len(pipeline['image_wh']) != 2 or min(pipeline['image_wh']) < 1:
        raise ValueError('pipeline.image_wh must contain positive width and height')
    if len(spnet['resolution_wh']) != 2 or any(n < 32 or n % 32 for n in spnet['resolution_wh']):
        raise ValueError('SPNet width and height must be positive multiples of 32')
    if spnet['model'] not in ('Tiny', 'Small', 'Base', 'Large'):
        raise ValueError('Choose SPNet Tiny, Small, Base, or Large')
    for model in (spnet, boxer):
        max_depth = model['max_depth_m']
        if not np.isfinite(max_depth) or max_depth <= 0:
            raise ValueError('Model depth limits must be positive and finite')
    if boxer['precision'] not in ('float32', 'bfloat16'):
        raise ValueError('Boxer precision must be float32 or bfloat16')


def benchmark(scene, yolo_model, spnet_model, boxer_model, config, profiler):
    settings = config['pipeline']
    first_results, first_pass = run_match(scene, yolo_model, spnet_model, boxer_model, config, profiler)
    del first_results

    for _ in range(settings['warmup']):
        run_match(scene, yolo_model, spnet_model, boxer_model, config, profiler)

    samples = []
    for iteration in range(settings['iterations']):
        results, metrics = run_match(scene, yolo_model, spnet_model, boxer_model, config, profiler)
        samples.append(metrics)
        if iteration < settings['iterations'] - 1:
            del results

    camera_summary = {}
    for name in scene.cameras:
        camera_summary[name] = {}
        for stage in samples[0]['cameras'][name]:
            stage_samples = [sample['cameras'][name][stage] for sample in samples]
            camera_summary[name][stage] = summarize(stage_samples)
    summary = {
        'overall': summarize([sample['overall'] for sample in samples]),
        'cameras': camera_summary,
    }
    return results, {'first_pass': first_pass, 'summary': summary, 'samples': samples}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/detection.yaml')
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    validate(config)
    settings = config['pipeline']

    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Run on a CUDA machine with the benchmark dependencies installed.')
    device = torch.device(f"cuda:{settings['gpu']}")
    torch.cuda.set_device(device)
    if config['boxer']['precision'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        parser.error('This GPU does not support Boxer bfloat16')
    revisions = {}
    for name, model_file in [('spnet', 'src/networks.py'), ('boxer', 'boxernet/boxernet.py')]:
        checkout = (ROOT / Path(config[name]['checkout']).expanduser()).resolve()
        if not (checkout / model_file).is_file():
            parser.error(f'Missing {name} checkout at {checkout}. See DETECTION_BENCHMARK.md.')
        sys.path.insert(0, str(checkout))
        revisions[name] = subprocess.check_output(['git', '-C', checkout, 'rev-parse', 'HEAD'],
                                                  text=True).strip()

    with Profiler(device, settings['resource_sample_ms']) as profiler:
        started = profiler.start()
        print(f"Loading match {settings['match_id']} and models on {torch.cuda.get_device_name(device)}",
              flush=True)
        with profiler.measure() as (_, load_match_metrics):
            scene = load_scene(config_path, settings, device)
        with profiler.measure() as (_, load_yolo_metrics):
            yolo_model = load_yolo(config['yolo'], device, ROOT)
        with profiler.measure() as (_, load_spnet_metrics):
            spnet_model = load_spnet(config['spnet'], device, ROOT)
        with profiler.measure() as (_, load_boxer_metrics):
            boxer_model = load_boxer(config['boxer'], device, ROOT)

        print(f"Running first pass, {settings['warmup']} warmups and {settings['iterations']} timed passes",
              flush=True)
        results, metrics = benchmark(scene, yolo_model, spnet_model, boxer_model, config, profiler)

        now = datetime.now(timezone.utc)
        run_name = f"{now:%H-%M-%S}_detection_match{settings['match_id']}_{now:%Y-%m-%d}_{uuid4().hex[:8]}"
        directory = ROOT / Path(settings['runs_dir']).expanduser() / run_name
        directory.mkdir(parents=True)
        with profiler.measure() as (_, export_metrics):
            write_scene(directory / 'scene.rrd', scene, results, config['spnet']['max_depth_m'])
        overall = profiler.run_totals(started)

    setup = {
        'Load match': load_match_metrics,
        'Load YOLO': load_yolo_metrics,
        'Load SPNet': load_spnet_metrics,
        'Load Boxer': load_boxer_metrics,
        'Write scene.rrd': export_metrics,
    }
    report = {
        'schema_version': 1, 'settings': config, 'match': scene.metadata,
        'gpu': torch.cuda.get_device_name(device), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'upstream_commits': revisions,
        'boxer_depth_source': 'SPNet optical Z in metres, pooled directly into Boxer patches',
        'overall_run': overall, 'setup_and_export': setup, **metrics,
        'detections': {name: {
            'yolo_2d': result.detections_2d.records(), 'boxer_3d': result.detections_3d,
        } for name, result in results.items()},
    }
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print_report(report)
    print(f'\nArtifacts in {directory}')


if __name__ == '__main__':
    main()
