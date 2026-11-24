"""Load and run the three models used by the detection benchmark."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_spnet import project_sparse


@dataclass
class Detections2D:
    """YOLO boxes in rectified image pixels, with one score and label per box."""

    xyxy: torch.Tensor
    scores: torch.Tensor
    labels: list[str]

    def records(self):
        records = []
        for label, score, box in zip(self.labels, self.scores.tolist(), self.xyxy.tolist()):
            records.append({'label': label, 'score': score, 'xyxy_pixels': box})
        return records


def load_yolo(settings, device, root):
    from ultralytics import YOLO

    weights = Path(settings['weights']).expanduser()
    is_local_path = weights.is_absolute() or weights.parent != Path('.')
    if is_local_path or (root / weights).is_file():
        weights = root / weights
        if not weights.is_file():
            raise FileNotFoundError(f'YOLO checkpoint missing: {weights}')

    model = YOLO(str(weights), task='detect')
    model.to(device)
    return model


def load_spnet(settings, device, root):
    from config import Configs
    from src.networks import V2Net

    config = Configs(gpus_num=1, model_type=settings['model'])
    model = V2Net(config.dims, config.depths, config.dp_rate, config.norm_type)
    checkpoint = settings['checkpoint'] or f"checkpoints/{settings['model']}_300.pth"
    checkpoint = root / Path(checkpoint).expanduser()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights['network'])
    return model.to(device).eval()


def load_boxer(settings, device, root):
    from boxernet.boxernet import BoxerNet

    checkout = root / Path(settings['checkout']).expanduser()
    checkpoint = settings['checkpoint'] or 'boxernet_hw960in2x6d768-c88128f8.ckpt'
    checkpoint = Path(checkpoint).expanduser()
    if checkpoint.parent == Path('.'):
        checkpoint = checkout / 'ckpts' / checkpoint
    checkpoint = root / checkpoint

    model = BoxerNet.load_from_checkpoint(str(checkpoint), device='cuda')
    model.device = device
    image_size = settings['image_size'] or model.hw
    if not isinstance(image_size, int) or image_size < 16 or image_size % 16:
        raise ValueError('Boxer image_size must be a square side length divisible by 16')
    return model


def perform_yolo(model, rgb_image, settings, device, timing):
    """Detect 2D objects and return their boxes in the original image coordinates."""
    with timing.stage('RGB to BGR'):
        bgr_image = np.ascontiguousarray(rgb_image[..., ::-1])

    with timing.stage('Ultralytics predict', cuda=True):
        result = model.predict(
            bgr_image, device=device, imgsz=settings['image_size'],
            conf=settings['confidence'], iou=settings['iou_threshold'],
            classes=settings['classes'], max_det=settings['max_detections'],
            half=settings['fp16'], rect=True, verbose=False, save=False,
        )[0]
    for stage, elapsed_ms in result.speed.items():
        timing.host_ms[f'YOLO {stage}'] = elapsed_ms

    with timing.stage('Read boxes and labels'):
        boxes = result.boxes.cpu()
        labels = [result.names[index] for index in boxes.cls.int().tolist()]
    return Detections2D(boxes.xyxy, boxes.conf, labels)


def make_sparse_range_map(lidar_points, camera, image_shape, settings, timing):
    """Project LiDAR onto the SPNet image grid and keep the nearest depth per pixel."""
    image_height, image_width = image_shape[:2]
    width, height = settings['resolution_wh']
    device = lidar_points.device

    with timing.stage('Project LiDAR', cuda=True):
        intrinsics = resized_intrinsics(
            camera.intrinsics, width / image_width, height / image_height,
        )
        intrinsics = torch.as_tensor(intrinsics, device=device)
        camera_from_lidar = torch.as_tensor(camera.camera_from_lidar, device=device)
        sparse_range_map = project_sparse(
            lidar_points, camera_from_lidar, intrinsics, height, width, settings['max_depth_m'],
        )
    return sparse_range_map


def perform_spnet(model, rgb_image, sparse_range_map, max_depth_m, timing):
    """Complete sparse optical-axis depth, preserving measured pixels."""
    with timing.stage('Upload and resize RGB', cuda=True):
        rgb = rgb_tensor(rgb_image, sparse_range_map.device)
        rgb = F.interpolate(rgb, size=sparse_range_map.shape, mode='bilinear', align_corners=False)

    with timing.stage('Normalize sparse depth', cuda=True):
        # The model expects a batch and a channel dimension before height and width.
        normalized_depth = sparse_range_map[None, None] / max_depth_m
        measured_pixels = sparse_range_map > 0
        mask = measured_pixels[None, None].float()

    with timing.stage('SPNet forward', cuda=True):
        prediction = model(rgb, normalized_depth, mask)

    with timing.stage('Convert to metres and retain measured depth', cuda=True):
        predicted_depth_m = prediction[0, 0] * max_depth_m
        valid = predicted_depth_m.isfinite() & (predicted_depth_m > 0) & (predicted_depth_m <= max_depth_m)
        predicted_depth_m = torch.where(valid, predicted_depth_m, 0)
        dense_range_map = torch.where(measured_pixels, sparse_range_map, predicted_depth_m)
    return dense_range_map


def perform_boxer(model, detections_2d, dense_range_map, rgb_image, camera, settings, timing):
    """Lift YOLO boxes to 3D using the SPNet depth map directly."""
    if not detections_2d.labels:
        timing.notes.append('Boxer inference skipped because YOLO returned zero boxes')
        return []

    with timing.stage('Align RGB, depth and 2D prompts', cuda=True):
        inputs, aligned_depth = prepare_boxer_inputs(
            rgb_image, dense_range_map, detections_2d, camera,
            settings['image_size'] or model.hw, settings['max_depth_m'],
        )

    use_bfloat16 = settings['precision'] == 'bfloat16'
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
        features = encode_dense_depth(model, inputs, aligned_depth, timing)
        with timing.stage('Lift prompts to oriented boxes', cuda=True):
            output = model.query(inputs, features)

    with timing.stage('Filter and read back 3D boxes', cuda=True):
        detections_3d = read_boxer_boxes(output['obbs_pr_w'][0], detections_2d, settings['confidence'])
    return detections_3d


def prepare_boxer_inputs(rgb_image, depth_m, detections_2d, camera, image_size, max_depth_m):
    """Resize and pad RGB, depth, boxes, and calibration onto the same square."""
    from utils.tw.camera import CameraTW
    from utils.tw.pose import PoseTW

    device = depth_m.device
    height, width = rgb_image.shape[:2]
    scale = image_size / max(height, width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    bottom = image_size - resized_height - top
    right = image_size - resized_width - left
    padding = (left, right, top, bottom)
    size = (resized_height, resized_width)
    scale_x = resized_width / width
    scale_y = resized_height / height

    rgb = rgb_tensor(rgb_image, device)
    rgb = F.interpolate(rgb, size=size, mode='bilinear', align_corners=False)
    rgb = F.pad(rgb, padding, value=114 / 255)

    depth_m = torch.where(depth_m <= max_depth_m, depth_m, 0)
    depth_m = F.interpolate(depth_m[None, None], size=size, mode='nearest-exact')
    depth_m = F.pad(depth_m, padding)

    intrinsics = resized_intrinsics(camera.intrinsics, scale_x, scale_y, left, top)
    boxer_camera = CameraTW.from_parameters(
        width=image_size, height=image_size,
        fx=float(intrinsics[0, 0]), fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]), cy=float(intrinsics[1, 2]),
        dist_params=torch.empty(0),
    )
    # Treat the optical camera itself as the rig, with identity camera-from-rig.
    world_from_camera = PoseTW.from_matrix(torch.from_numpy(camera.world_from_camera))

    box_coordinates = detections_2d.xyxy
    box_scale = box_coordinates.new_tensor([scale_x, scale_y, scale_x, scale_y])
    box_offset = box_coordinates.new_tensor([left, top, left, top])
    resized_boxes = box_coordinates * box_scale + box_offset
    # Boxer expects x1, x2, y1, y2 in pixel-center coordinates.
    box_prompts = resized_boxes[:, [0, 2, 1, 3]] - 0.5

    inputs = {
        'img0': rgb,
        'cam0': boxer_camera.to(device).float().unsqueeze(0),
        'T_world_rig0': world_from_camera.to(device).float().unsqueeze(0),
        'bb2d': box_prompts.to(device).unsqueeze(0),
    }
    return inputs, depth_m


def encode_dense_depth(model, inputs, depth_m, timing):
    """Run the Boxer encoder with depth tokens pooled directly from the image."""
    from boxernet.boxernet import batch_dino, generate_plucker_encoding
    from utils.gravity import gravity_align_T_world_cam

    patch_size = model.dino.patch_size
    with timing.stage('Pool metric depth per patch', cuda=True):
        patch_depth = depth_patches(depth_m, patch_size)

    with timing.stage('DINOv3 image encoder', cuda=True):
        image_features = batch_dino(model.dino, inputs['img0'])

    with timing.stage('Camera rays and feature attention', cuda=True):
        camera = inputs['cam0']
        world_from_camera = inputs['T_world_rig0'] @ camera.T_camera_rig.inverse()
        world_from_voxel = gravity_align_T_world_cam(world_from_camera, z_grav=True)
        inputs['T_world_voxel0'] = world_from_voxel

        # Each patch gets one image token, one depth value, and optionally a ray.
        image_tokens = image_features.flatten(2).transpose(1, 2)
        depth_tokens = patch_depth.flatten(2).transpose(1, 2)
        tokens = [image_tokens, depth_tokens]
        if model.with_ray:
            batch, _, rows, cols = image_features.shape
            voxel_from_camera = world_from_voxel.inverse() @ world_from_camera
            ray_tokens = generate_plucker_encoding(
                batch, rows, cols, patch_size, camera, voxel_from_camera,
            )
            tokens.append(ray_tokens)

        encoded = model.input2emb(torch.cat(tokens, dim=-1))
        if model.in_depth:
            encoded = model.self_attn(encoded)
    return {'input_enc': encoded}


def depth_patches(depth_m, patch_size):
    """Match Boxer's mean of valid optical Z, with -1 for an empty patch."""
    valid = depth_m.isfinite() & (depth_m > 0)
    valid_depth = torch.where(valid, depth_m, 0)
    # A divisor of 1 gives the sum in each patch instead of its average.
    depth_sum = F.avg_pool2d(valid_depth, patch_size, divisor_override=1)
    valid_count = F.avg_pool2d(valid.float(), patch_size, divisor_override=1)
    mean_depth = depth_sum / valid_count.clamp_min(1)
    return torch.where(valid_count > 0, mean_depth, -1)


def read_boxer_boxes(boxes, detections_2d, confidence):
    boxes = boxes.float().cpu()
    scores = boxes.prob.squeeze(-1).numpy()
    corners = boxes.bb3corners_world.numpy()

    detections_3d = []
    for label, yolo_score, score, box in zip(
        detections_2d.labels, detections_2d.scores.tolist(), scores, corners,
    ):
        if not np.isfinite(score) or score < confidence or not np.isfinite(box).all():
            continue

        axes = box[[1, 3, 4]] - box[0]
        detections_3d.append({
            'label': label,
            'yolo_score': yolo_score,
            'boxer_score': float(score),
            'center_m': box.mean(axis=0).tolist(),
            'size_xyz_m': np.linalg.norm(axes, axis=1).tolist(),
            'corners_m': box.tolist(),
        })
    return detections_3d


def rgb_tensor(rgb_image, device):
    """Convert an H × W × 3 uint8 image to a 1 × 3 × H × W float tensor."""
    rgb = torch.from_numpy(rgb_image).to(device)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)
    return rgb.float().div_(255)


def resized_intrinsics(intrinsics, scale_x, scale_y, left=0, top=0):
    """Scale focal lengths and move the principal point to the new pixel grid."""
    resized = intrinsics.copy()
    resized[0] *= scale_x
    resized[1] *= scale_y
    resized[0, 2] += (scale_x - 1) / 2 + left
    resized[1, 2] += (scale_y - 1) / 2 + top
    return resized
