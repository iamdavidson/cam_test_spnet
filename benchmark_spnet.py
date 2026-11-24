"""Benchmark SPNet depth completion for one synchronized stereo/LiDAR match."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as F
import yaml

from spnet_inputs import lidar_xyz, load_triplet, remap_grid
from visualize_match import find_match, header_ns, image_rgb


ROOT = Path(__file__).resolve().parent
VARIANTS = ('Tiny', 'Small', 'Base', 'Large')


def ms(start):
    return (perf_counter() - start) * 1000


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def project_sparse(points, camera_from_lidar, K, height, width, max_depth):
    xyz = transform(points, camera_from_lidar)
    projected = xyz @ K.T
    z = xyz[:, 2]
    uv = torch.round(projected[:, :2] / z[:, None])
    u, v = uv.unbind(dim=1)
    valid = ((z > 0) & (z <= max_depth) & (u >= 0) & (u < width)
             & (v >= 0) & (v < height))
    # Route invalid samples to a discarded slot. Keep the nearest optical Z per pixel.
    index = torch.where(valid, v * width + u, height * width).long()
    sparse = z.new_full((height * width + 1,), torch.inf)
    sparse.scatter_reduce_(0, index, torch.where(valid, z, torch.inf), reduce='amin')
    sparse = sparse[:-1].reshape(height, width)
    return torch.where(sparse.isfinite(), sparse, 0)


def virtual_points(depth, rgb, sparse, K, world_from_camera, stride, max_depth):
    height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(0, height, stride, device=depth.device),
        torch.arange(0, width, stride, device=depth.device),
        indexing='ij',
    )
    z = depth[y, x]
    valid = (z > 0) & (z <= max_depth) & (sparse[y, x] == 0)
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).to(depth.dtype)
    rays = pixels @ torch.linalg.inv_ex(K).inverse.T
    xyz = transform((rays * z[..., None])[valid], world_from_camera)
    colors = (rgb[0, :, y, x].permute(1, 2, 0)[valid] * 255).round().byte()
    return xyz, colors


def stage(events, name):
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    events.append((name, event))


def run_image(model, message, cloud, K, grid, camera_from_lidar, max_depth, device):
    height, width = grid.shape[1:3]
    start = perf_counter()
    rgb_cpu = image_rgb(message)
    decode_ms = ms(start)
    events = []
    stage(events, 'start')
    rgb = torch.from_numpy(rgb_cpu).to(device)
    stage(events, 'rgb_h2d')
    rgb = rgb.permute(2, 0, 1).unsqueeze(0).float().div_(255)
    rgb = F.grid_sample(rgb, grid, mode='bilinear', padding_mode='border', align_corners=False)
    stage(events, 'rgb_rectify_resize_normalize')
    sparse = project_sparse(cloud, camera_from_lidar, K, height, width, max_depth)
    raw = sparse[None, None] / max_depth
    mask = (raw > 0).float()
    stage(events, 'lidar_project_rasterize_normalize')
    pred = model(rgb, raw, mask)
    stage(events, 'spnet_forward')
    depth = pred[0, 0] * max_depth
    depth = torch.where(depth.isfinite() & (depth > 0) & (depth <= max_depth), depth, 0)
    stage(events, 'depth_to_meters_and_filter')
    events[-1][1].synchronize()
    timing = {'jpeg_decode_cpu_ms': decode_ms, 'image_to_depth_wall_ms': ms(start)}
    timing['image_to_depth_cuda_ms'] = events[0][1].elapsed_time(events[-1][1])
    for (_, begin), (name, end) in zip(events, events[1:]):
        timing[f'{name}_cuda_ms'] = begin.elapsed_time(end)
    return timing, (rgb, sparse, depth)


def summarize(samples):
    return {
        key: {name: float(fn([sample[key] for sample in samples])) for name, fn in (
            ('min', np.min), ('mean', np.mean), ('median', np.median),
            ('p95', lambda values: np.percentile(values, 95)), ('max', np.max)
        )}
        for key in samples[0]
    }


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_camera(model, camera, cloud, lidar_pose, settings, device):
    message, info, world_from_camera = camera
    height, width, max_depth, stride, warmup, iterations = settings
    start = perf_counter()
    rgb_cpu = image_rgb(message)
    K_np, grid_np = remap_grid(info, rgb_cpu.shape, height, width)
    camera_from_lidar = np.linalg.solve(world_from_camera, lidar_pose)
    geometry_cpu_ms = ms(start)
    start = perf_counter()
    K = torch.as_tensor(K_np, device=device)
    grid = torch.as_tensor(grid_np, device=device)[None]
    pose = torch.as_tensor(camera_from_lidar, dtype=torch.float32, device=device)
    world_pose = torch.as_tensor(world_from_camera, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    geometry_h2d_ms = ms(start)
    args = (model, message, cloud, K, grid, pose, max_depth, device)

    torch.cuda.reset_peak_memory_stats(device)
    first, first_data = run_image(*args)
    if not first_data[1].any().item():
        raise RuntimeError('No LiDAR samples project into this camera at the requested depth range')
    del first_data
    for _ in range(warmup):
        run_image(*args)

    samples = []
    for _ in range(iterations):
        timing, data = run_image(*args)
        samples.append(timing)
        del data

    peak_memory = torch.cuda.max_memory_allocated(device)
    # Capture one separate inference for export, keeping export allocations out of timing.
    export_inference, (rgb, sparse, depth) = run_image(*args)
    events = []
    stage(events, 'start')
    xyz, colors = virtual_points(depth, rgb, sparse, K, world_pose, stride, max_depth)
    stage(events, 'virtual_backproject')
    events[-1][1].synchronize()
    start = perf_counter()
    image = (rgb[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    sparse_np, depth_np, xyz_np, colors_np = [
        tensor.cpu().numpy() for tensor in (sparse, depth, xyz, colors)
    ]
    export_ms = ms(start)
    sparse_count = int(np.count_nonzero(sparse_np))
    result = {
        'geometry_setup_cpu_ms': geometry_cpu_ms,
        'geometry_h2d_wall_ms': geometry_h2d_ms,
        'camera_info_stamp_ns': header_ns(info),
        'input_resolution_wh': [int(info.width), int(info.height)],
        'intrinsics': K_np.tolist(),
        'world_from_camera': world_from_camera.tolist(),
        'first_pass_ms': first,
        'samples_ms': samples,
        'summary_ms': summarize(samples),
        'peak_cuda_allocated_bytes': peak_memory,
        'export': {
            'image_to_depth_wall_ms': export_inference['image_to_depth_wall_ms'],
            'virtual_backproject_cuda_ms': events[0][1].elapsed_time(events[1][1]),
            'image_pack_and_d2h_wall_ms': export_ms,
            'sparse_depth_pixels': sparse_count,
            'estimated_depth_pixels': int(np.count_nonzero(depth_np)),
            'virtual_points': len(xyz_np),
        },
    }
    return result, (image, sparse_np, depth_np, xyz_np, colors_np)


def write_rrd(path, cloud, captures, match_id, max_depth, axes):
    with rr.RecordingStream(f'SPNet match {match_id}') as recording:
        recording.save(str(path))
        recording.log('scene', axes, static=True)
        recording.log('scene/measured_lidar', rr.Points3D(
            cloud, colors=[65, 220, 255], radii=0.035
        ), static=True)
        camera_views = []
        for name, (image, sparse, depth, xyz, colors) in captures.items():
            recording.log(f'scene/virtual_{name}', rr.Points3D(
                xyz, colors=colors, radii=0.012
            ), static=True)
            recording.log(f'{name}/rgb', rr.Image(image), static=True)
            for kind, data in [('sparse', sparse), ('estimate', depth)]:
                recording.log(f'{name}/{kind}', rr.DepthImage(
                    data, meter=1, depth_range=[0, max_depth]
                ), static=True)
            camera_views.append(rrb.Horizontal(*[
                rrb.Spatial2DView(name=label, origin=f'{name}/{entity}')
                for entity, label in [('rgb', 'RGB'), ('sparse', 'LiDAR depth'),
                                      ('estimate', 'SPNet depth')]
            ], name=name.title()))
        recording.send_blueprint(rrb.Blueprint(rrb.Tabs(
            rrb.Spatial3DView(name='Measured LiDAR and virtual points', origin='scene'),
            *camera_views,
        )))
        recording.flush()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/spnet.yaml')
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())
    source = yaml.safe_load((config_path.parent / cfg['sensor_config']).read_text())
    match_id = int(cfg['match_id'])
    variant = cfg['model']
    width, height = map(int, cfg['resolution_wh'])
    max_depth = float(cfg['max_depth_m'])
    stride, warmup, iterations = (int(cfg[key]) for key in ('virtual_stride', 'warmup', 'iterations'))
    if variant not in VARIANTS:
        parser.error(f'model must be one of {", ".join(VARIANTS)}')
    if min(width, height) < 32 or width % 32 or height % 32:
        parser.error('resolution_wh dimensions must be positive multiples of 32')
    if (not np.isfinite(max_depth) or max_depth <= 0 or stride < 1
            or warmup < 0 or iterations < 1):
        parser.error('Invalid max_depth_m, virtual_stride, warmup or iterations')
    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Check nvidia-smi and use the server installer.')
    device = torch.device(f'cuda:{int(cfg["gpu"])}')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    def path(value):
        return ROOT / Path(value).expanduser()

    upstream = path(cfg['spnet_checkout']).resolve()
    checkpoint = path(cfg.get('checkpoint') or f'checkpoints/{variant}_300.pth')
    if not checkpoint.is_file():
        parser.error(f'Checkpoint missing: {checkpoint}. Run scripts/run_spnet.sh.')
    bag = path(cfg.get('bag_path') or source['bag_path'])
    csv = path(cfg.get('match_csv') or source['output']['csv'])
    if not bag.is_file():
        parser.error(f'Corrected MCAP missing: {bag}. Set bag_path in {config_path}.')
    sys.path.insert(0, str(upstream))
    from config import Configs
    from src.networks import V2Net

    started = perf_counter()
    stamps = find_match(csv, match_id)
    vis = source['visualization']
    start = perf_counter()
    lidar, lidar_pose, cameras, world, frames = load_triplet(
        bag, source['topics'], vis['camera_info_topics'], stamps,
        vis.get('fixed_frame'), float(vis['tf_max_age_ms'])
    )
    bag_ms = ms(start)
    start = perf_counter()
    cloud_cpu = lidar_xyz(lidar)
    parse_ms = ms(start)
    start = perf_counter()
    cloud = torch.as_tensor(cloud_cpu, device=device)
    cloud = cloud[cloud.isfinite().all(dim=1)]
    lidar_gpu_pose = torch.as_tensor(lidar_pose, dtype=torch.float32, device=device)
    world_cloud = transform(cloud, lidar_gpu_pose)
    torch.cuda.synchronize(device)
    upload_ms = ms(start)
    if not len(cloud):
        raise RuntimeError('Selected LiDAR cloud has no finite points')

    start = perf_counter()
    architecture = Configs(gpus_num=1, model_type=variant)
    model = V2Net(architecture.dims, architecture.depths, architecture.dp_rate, 'CNX')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights['network'])
    del weights
    model.to(device).eval()
    torch.cuda.synchronize(device)
    model_ms = ms(start)

    print(f'SPNet {variant}, {width}x{height}, match {match_id}, {torch.cuda.get_device_name(device)}',
          flush=True)
    results, captures = {}, {}
    for name, camera in cameras.items():
        print(f'Benchmarking {name} camera ...', flush=True)
        results[name], captures[name] = run_camera(
            model, camera, cloud, lidar_pose,
            (height, width, max_depth, stride, warmup, iterations), device
        )
    now = datetime.now(timezone.utc)
    run_name = f'spnet_{now:%H-%M-%S}_{now:%Y-%m-%d}_{variant}_{width}x{height}_match{match_id}_{uuid4().hex[:8]}'
    directory = path(cfg['runs_dir']) / run_name
    directory.mkdir(parents=True)
    checkpoint_sha256 = file_sha256(checkpoint)
    start = perf_counter()
    write_rrd(directory / 'scene.rrd', world_cloud.cpu().numpy(), captures, match_id, max_depth,
              rr.ViewCoordinates.FLU if vis.get('fixed_frame') else rr.ViewCoordinates.RDF)
    rrd_ms = ms(start)

    report = {
        'schema_version': 1,
        'settings': cfg,
        'model': {
            'variant': variant, 'checkpoint': str(checkpoint),
            'checkpoint_sha256': checkpoint_sha256,
            'parameters': sum(p.numel() for p in model.parameters()),
            'spnet_commit': subprocess.check_output(
                ['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True
            ).strip(),
        },
        'environment': {
            'device': str(device), 'gpu': torch.cuda.get_device_name(device),
            'total_gpu_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'rerun': rr.__version__,
            'python': sys.version, 'precision': 'fp32', 'tf32': False, 'cudnn_benchmark': False,
        },
        'match': {
            'id': match_id, 'bag': str(bag), 'csv': str(csv),
            'header_stamps_ns': stamps, 'frames': frames, 'world_frame': world,
            'fixed_frame': vis.get('fixed_frame'), 'world_from_lidar': lidar_pose.tolist(),
            'camera_minus_lidar_ms': {
                name: (stamps[name] - stamps['lidar']) / 1e6 for name in cameras
            },
            'measured_points': len(cloud),
        },
        'run_wall_ms': {
            'mcap_lookup_and_tf': bag_ms, 'pointcloud_unpack_cpu': parse_ms,
            'lidar_upload_filter_transform': upload_ms, 'model_load': model_ms,
            'lidar_download_and_rrd_write': rrd_ms,
            'elapsed_before_report_write': ms(started),
        },
        'cameras': results,
    }
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Artifacts: {directory}')


if __name__ == '__main__':
    main()
