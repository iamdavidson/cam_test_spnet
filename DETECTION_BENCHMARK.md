# YOLO → SPNet → Boxer benchmark

Run one synchronized LiDAR scan and the selected cameras through the three models on CUDA. Each camera produces independent 2D and 3D detections in a shared frame. The default is match 130, YOLO26n, SPNet Tiny, and the published BoxerNet checkpoint. Boxer was trained for indoor scenes, so these outdoor results measure integration and latency. They have no ground-truth accuracy metrics.

## Setup

From the project directory on a Linux CUDA machine with Python 3.12 and Git:

```bash
python3.12 -m venv .venv-detection
source .venv-detection/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-detection.txt
mkdir -p .deps checkpoints
git clone https://github.com/Wang-xjtu/SPNet.git .deps/SPNet
git -C .deps/SPNet checkout b836bd044517b33d3737094acd6a1f09c2362f04
git clone https://github.com/facebookresearch/boxer.git .deps/boxer
gdown 1ivmCX-i9lej4uJhT0Yyk2Nq9ZmoQlsB9 -O checkpoints/Tiny_300.pth
mkdir -p .deps/boxer/ckpts
for weight in boxernet_hw960in2x6d768-c88128f8.ckpt dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth; do
  curl --fail --location "https://huggingface.co/facebook/boxer/resolve/main/$weight" \
    --output ".deps/boxer/ckpts/$weight"
done
```

Reuse existing checkouts and weights by setting their paths in [config/detection.yaml](config/detection.yaml). Ultralytics downloads the default YOLO weights on first use. For an offline benchmark, provide a local YOLO checkpoint ahead of time. A first-time weight download counts toward setup time. If OpenCV cannot import `libGL.so.1`, install the distribution's OpenGL runtime library (e.g. `libgl1` on Ubuntu).

Set `pipeline.bag_path` to the corrected MCAP that corresponds to `sync_matches.csv`. `pipeline.world_frame` must exist in TF and have +Z up. The default uses the robot's `base_link` frame, assuming a level robot. Use a gravity-aligned odometry or map frame if available and the robot pitches or rolls.

The benchmark selects YOLO26's NMS-free head with `yolo.nms: false`. Set `yolo.nms: true` to use standard NMS and apply `yolo.iou_threshold`. [Ultralytics detection modes](https://docs.ultralytics.com/guides/end2end-detection)

```bash
python benchmark_detection.py --config config/detection.yaml
```

## Depth handoff

Each camera uses one rectified image coordinate system. YOLO supplies 2D box prompts. SPNet projects the measured LiDAR into its own image grid and predicts depth, keeping the original measurements where available. The Boxer adapter aligns the RGB, depth, box prompts, and camera intrinsics to Boxer's square input. It pools valid optical-axis Z in metres into the mean-depth token for each DINO patch, with `-1` for empty patches. It calls Boxer's existing DINO, ray encoding, attention, and 3D head with the published weights. The adapter skips point-cloud construction and re-rasterization. Dense, completed depth changes the samples that contribute to patch means compared with a sparse point cloud. [Upstream Boxer encoder](https://github.com/facebookresearch/boxer/blob/main/boxernet/boxernet.py#L598-L647)

## Results

The benchmark prints median and p95 match latency, pipeline-only resource usage, and per-camera stage times and detection counts. A pass runs all selected cameras once in sequence. The same match is repeated for measurement. `report.json` contains the first pass, timed samples, resource measurements, and box coordinates. Total invocation time includes bag lookup, model loading, GPU tracing, and Rerun export. If YOLO returns no boxes, Boxer is marked as skipped.

Run names start with the benchmark name, followed by the UTC time and date, for example `detection_13-25-08_2026-11-24_match130_2af3519f`. To rename existing detection and SPNet run directories on the server, run the following from the repository. Pass your configured runs directory if it differs from `runs/`.

```bash
bash scripts/rename_runs.sh --dry-run runs
bash scripts/rename_runs.sh runs
```

Use the same printer to open one or more saved reports or an older detection benchmark log. For standalone use, install `rich==14.1.0` in your Python environment.

```bash
python report_printer.py runs/<run>/report.json
python report_printer.py --details runs/<run>/report.json
python report_printer.py ~/Downloads/log.log
```

- **Wall time** includes CPU work and a CUDA synchronization at each method boundary. Internal CUDA timings measure CUDA stream elapsed time. Ultralytics reports its own preprocessing, inference, and postprocessing times.
- **Detection CPU** is process CPU time from the start of model loading through the last inference pass. 100% means one full core on average.
- **Detection RAM** is peak process resident memory (RSS) above the baseline taken after MCAP loading. `pipeline.resource_sample_ms` controls RSS sampling. The increase can be small when the allocator reuses already resident memory.
- **Detection VRAM** is peak live PyTorch CUDA allocation above that same post-MCAP baseline. It includes all loaded GPU models and inference allocations.
- **GPU active** is the fraction of a separate traced pass occupied by CUDA kernels, copies, or memory sets from this benchmark process. Overlapping activity counts once. The trace has its own wall time and can be unavailable if PyTorch cannot collect CUDA activity.

The headline detection figures cover model loading, the first pass, warmup, and timed inference. The MCAP load and Rerun export report wall time only. Use `--details` on the report printer to see the extra memory and CPU use per timed pass and stage, measured from the start of that pass or stage.

Open the output with `rerun runs/<run>/scene.rrd`. The 3D view shows the measured LiDAR with a 0.005 m radius and Boxer's oriented 3D boxes. Each camera tab shows the YOLO overlay, sparse LiDAR depth, and completed SPNet depth.

## Compare 3D detections

Select two or more detection runs by directory or `report.json` path. Shell globs work for selecting several runs by time or match.

```bash
python scripts/compare_detections.py runs/detection_*_match130_*
rerun comparison.rrd
```

For a comparison you want to repeat, save a JSON list of run paths and use `python scripts/compare_detections.py --selection selection.json`. Relative paths in the list are interpreted from the directory containing `selection.json`. Use `--output NAME.rrd` to save another comparison.

Each run has one color for all its boxes, including boxes from both cameras. Toggle individual run layers in the Rerun Blueprint panel. The first run sets the reference frame. Runs of the same sensor capture are aligned through a shared camera pose when their world frames differ. For different captures that already share a fixed world frame, such as the same map origin, pass `--shared-world`.
