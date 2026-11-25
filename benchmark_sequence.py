"""Replay a full MCAP and run the detection pipeline on its established matches."""

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
import yaml

from benchmark_detection import ROOT, configure_models, prepare_scene, run_match, summarize_matches, validate
from detection_models import load_boxer, load_spnet, load_yolo
from detection_profiling import Profiler, measure_gpu_activity, measure_wall_time
from report_printer import print_report
from sequence_replay import replay_process


def prepare_job(job, settings, device, calibration_cache):
    samples = job['samples']
    lidar = np.load(samples['lidar'].path, allow_pickle=False)
    cameras = {}
    for name in settings['cameras']:
        image = SimpleNamespace(data=samples[name].path.read_bytes())
        cameras[name] = (image, job['calibrations'][name], job['poses'][name])
    metadata = {
        'match_id': job['match_id'], 'world_frame': job['world_frame'],
        'header_stamps_ns': {name: sample.stamp_ns for name, sample in samples.items()},
        'frames': {name: sample.frame for name, sample in samples.items()},
    }
    scene = prepare_scene(lidar, job['poses']['lidar'], cameras, metadata, settings, device, calibration_cache)
    for sample in samples.values():
        sample.path.unlink()
    return scene


def result_payload(results):
    payload = {}
    for name, result in results.items():
        payload[f'{name}_rgb'] = result.rgb_image
        payload[f'{name}_sparse'] = result.sparse_range_map.cpu().numpy()
        payload[f'{name}_depth'] = result.dense_range_map.cpu().numpy()
        payload[f'{name}_xyxy'] = result.detections_2d.xyxy.numpy()
        payload[f'{name}_labels'] = np.asarray(result.detections_2d.labels, dtype=str)
        boxes = result.detections_3d
        payload[f'{name}_corners'] = np.asarray([box['corners_m'] for box in boxes], dtype=np.float32)
        payload[f'{name}_centers'] = np.asarray([box['center_m'] for box in boxes], dtype=np.float32)
        payload[f'{name}_box_labels'] = np.asarray(
            [f"{box['label']} {box['boxer_score']:.2f}" for box in boxes], dtype=str,
        )
    return payload


def next_replay_message(process, jobs_queue):
    while True:
        try:
            kind, data = jobs_queue.get(timeout=0.25)
        except Empty:
            if not process.is_alive():
                raise RuntimeError(f'Replay process exited before finishing (exit code {process.exitcode})')
            continue
        if kind == 'error':
            raise RuntimeError(data)
        return kind, data


def run_sequence(process, jobs_queue, results_queue, spool, config, models, profiler, memory_baseline):
    settings = config['pipeline']
    yolo, spnet, boxer = models
    calibration_cache = {}
    samples, matches, resource_scopes = [], [], []
    scene = None
    while True:
        kind, job = next_replay_message(process, jobs_queue)
        if kind == 'end':
            replay = job
            break
        if kind != 'match':
            raise RuntimeError(f'Unexpected replay message: {kind}')

        begin_ns = perf_counter_ns()
        resources_started = profiler.start_pass()
        with profiler.measure('Prepare match') as (_, preparation):
            scene = prepare_job(job, settings, profiler.device, calibration_cache)
        prep_peak = profiler.pass_cuda_peak_bytes
        results, metrics = run_match(scene, yolo, spnet, boxer, config, profiler)
        completion_ns = perf_counter_ns()
        gpu_peak = max(prep_peak, profiler.pass_cuda_peak_bytes)
        resource_scopes.append(profiler.finish(resources_started, gpu_peak, memory_baseline=memory_baseline))

        last_input_ns = max(sample.arrival_ns for sample in job['samples'].values())
        origin_ns = job['origin_ns']
        match = {
            'match_id': job['match_id'],
            'header_stamps_ns': scene.metadata['header_stamps_ns'],
            'last_input_replay_ns': last_input_ns - origin_ns,
            'ready_replay_ns': job['ready_ns'] - origin_ns,
            'started_replay_ns': begin_ns - origin_ns,
            'completed_replay_ns': completion_ns - origin_ns,
            'input_wait_ms': (job['ready_ns'] - last_input_ns) / 1e6,
            'queue_wait_ms': (begin_ns - job['ready_ns']) / 1e6,
            'prepare_ms': preparation['wall_ms'],
            'inference_ms': metrics['overall']['wall_ms'],
            'end_to_end_ms': (completion_ns - last_input_ns) / 1e6,
            'detection_counts': {
                name: {'yolo_2d': len(result.detections_2d.labels), 'boxer_3d': len(result.detections_3d)}
                for name, result in results.items()
            },
        }
        timing = {key: match[key] for key in ('queue_wait_ms', 'inference_ms', 'end_to_end_ms')}
        result_path = spool / f"result-{match['match_id']}.npz"
        np.savez(result_path, **result_payload(results))
        results_queue.put((result_path, match['completed_replay_ns'], match['match_id'],
                           scene.metadata['header_stamps_ns']['lidar'], timing))
        match['result_handoff_ms'] = (perf_counter_ns() - completion_ns) / 1e6
        samples.append(metrics)
        matches.append(match)
        del results
        if len(matches) % 100 == 0:
            print(f'Processed {len(matches)} matches · latest queue wait {match["queue_wait_ms"]:.1f} ms',
                  flush=True)

    results_queue.put(None)
    kind, _ = next_replay_message(process, jobs_queue)
    if kind != 'saved':
        raise RuntimeError(f'Unexpected end of replay: {kind}')
    return scene, samples, matches, replay, resource_scopes


def latency_summary(values):
    if not values:
        return None
    return {
        'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
        'max': max(values),
    }


def replay_summary(replay, matches):
    # Count matches waiting for the worker, excluding the match currently being processed.
    queue_events = []
    for match in matches:
        queue_events.append((match['ready_replay_ns'], 1))
        queue_events.append((match['started_replay_ns'], -1))
    pending = max_backlog = 0
    for _, delta in sorted(queue_events, key=lambda event: (event[0], -event[1])):
        pending += delta
        max_backlog = max(max_backlog, pending)

    last_completion_ns = max((match['completed_replay_ns'] for match in matches), default=0)
    replay_elapsed_ns = replay['replay_done_ns'] - replay['replay_origin_ns']
    elapsed_s = max(last_completion_ns, replay_elapsed_ns) / 1e9
    return {
        'completed_matches': len(matches),
        'max_backlog': max_backlog,
        'drain_seconds': max(0, last_completion_ns - replay_elapsed_ns) / 1e9,
        'throughput_matches_per_second': len(matches) / elapsed_s if elapsed_s > 0 else 0,
        'arrival_lateness_ms': latency_summary(replay['sensor_arrival_lateness_ms']),
        **{key: latency_summary([match[key] for match in matches])
           for key in ('input_wait_ms', 'queue_wait_ms', 'prepare_ms', 'inference_ms',
                       'end_to_end_ms', 'result_handoff_ms')},
    }


def combine_resources(scopes):
    active_ms = sum(scope['wall_ms'] for scope in scopes)
    cpu_ms = sum(scope['cpu_ms'] for scope in scopes)
    return {
        'wall_ms': active_ms, 'cpu_ms': cpu_ms,
        'cpu_percent_one_core': 100 * cpu_ms / active_ms,
        'extra_ram_mib': max(scope['extra_ram_mib'] for scope in scopes),
        'extra_vram_mib': max(scope['extra_vram_mib'] for scope in scopes),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/detection.yaml')
    parser.add_argument('--rate', type=float, default=1.0, help='replay multiplier (0 runs without pacing)')
    parser.add_argument('--viewer', nargs='?', const='rerun+http://127.0.0.1:9876/proxy',
                        help='also stream to a live Rerun viewer, optionally at this URL')
    args = parser.parse_args()
    if not np.isfinite(args.rate) or args.rate < 0:
        parser.error('--rate must be finite and nonnegative')
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    validate(config)
    settings = config['pipeline']
    sensor_config = yaml.safe_load((config_path.parent / settings['sensor_config']).read_text())
    bag_path = ROOT / Path(settings['bag_path'] or sensor_config['bag_path']).expanduser()
    match_csv = ROOT / Path(settings['match_csv']).expanduser()
    if not bag_path.is_file() or not match_csv.is_file():
        parser.error(f'Missing MCAP or match CSV: {bag_path}, {match_csv}')
    try:
        device, revisions = configure_models(config)
    except ValueError as error:
        parser.error(str(error))

    now = datetime.now(timezone.utc)
    run_name = f'detection_sequence_{now:%H-%M-%S}_{now:%Y-%m-%d}_{uuid4().hex[:8]}'
    directory = ROOT / Path(settings['runs_dir']).expanduser() / run_name
    directory.mkdir(parents=True)
    started_ns = perf_counter_ns()
    setup = {}

    context = mp.get_context('spawn')
    jobs_queue, results_queue = context.Queue(), context.Queue()
    with TemporaryDirectory(prefix='.replay-', dir=directory) as temporary:
        spool = Path(temporary)
        with Profiler(device, settings['resource_sample_ms']) as profiler:
            baseline = profiler.start_pass()
            print(f'Loading models on {torch.cuda.get_device_name(device)}', flush=True)
            with profiler.measure('Load detection models'):
                with measure_wall_time(setup, 'Load YOLO'):
                    yolo = load_yolo(config['yolo'], device, ROOT)
                with measure_wall_time(setup, 'Load SPNet'):
                    spnet = load_spnet(config['spnet'], device, ROOT)
                with measure_wall_time(setup, 'Load Boxer'):
                    boxer = load_boxer(config['boxer'], device, ROOT)
            model_resources = profiler.finish(baseline, profiler.pass_cuda_peak_bytes)

            process = context.Process(target=replay_process, args=(
                bag_path, match_csv, sensor_config, settings, spool, directory / 'scene.rrd',
                args.viewer, args.rate, config['spnet']['max_depth_m'], jobs_queue, results_queue,
            ))
            print(f'Starting replay at {args.rate:g}×' if args.rate else 'Starting unpaced replay', flush=True)
            process.start()
            try:
                scene, samples, matches, replay, scopes = run_sequence(
                    process, jobs_queue, results_queue, spool, config, (yolo, spnet, boxer), profiler, baseline,
                )
            finally:
                if process.exitcode is None:
                    process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
                    process.join()
                for channel in (jobs_queue, results_queue):
                    channel.cancel_join_thread()
                    channel.close()

    summary = summarize_matches(samples) if samples else {'overall': {}, 'cameras': {}}
    gpu_activity = {'unavailable': 'No complete match to trace'}
    if scene is not None:
        gpu_activity = measure_gpu_activity(
            profiler, lambda: run_match(scene, yolo, spnet, boxer, config, profiler),
            summary['cameras'],
        )
        gpu_activity['match_id'] = scene.metadata['match_id']

    report = {
        'schema_version': 1, 'benchmark': 'detection_sequence', 'settings': config,
        'bag': str(bag_path), 'match_csv': str(match_csv), 'rate': args.rate,
        'gpu': torch.cuda.get_device_name(device), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'upstream_commits': revisions,
        'overall_run': {'wall_ms': (perf_counter_ns() - started_ns) / 1e6},
        'setup_and_export': setup, 'detection_resources': combine_resources([model_resources, *scopes]),
        'gpu_activity': gpu_activity, 'replay': replay, 'sequence': replay_summary(replay, matches),
        'summary': summary, 'matches': matches, 'samples': samples,
    }
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print_report(report)
    print(f'\nArtifacts in {directory}')


if __name__ == '__main__':
    main()
