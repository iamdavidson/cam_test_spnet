# SPNet CUDA benchmark

Run [upstream SPNet](https://github.com/Wang-xjtu/SPNet) on one matched LiDAR scan and both camera images. Each camera gets its own projected LiDAR depth map and a separate batch-size-one forward pass. The default is SPNet-Tiny at 512 × 288 on match 130.

## 1. Put the project and bag on the GPU server

The commands below assume a Linux CUDA server with Git, Python 3.12 with venv support, and a working NVIDIA driver. Replace `you@gpu-server` with your SSH destination. From your local computer, in the project directory:

```bash
scp cam_test_spnet.bundle you@gpu-server:~/
scp /home/karnelius/datasets/compressed/rosbag2_2026_05_12-16_58_33_0_corrected.mcap you@gpu-server:~/
ssh you@gpu-server
git clone ~/cam_test_spnet.bundle ~/cam_test
cd ~/cam_test
nvidia-smi
```

The bundle contains `sync_matches.csv`. Use the corrected MCAP corresponding to that CSV. The benchmark selects messages by the exact header timestamps listed in the chosen row.

If you publish the bundled repo to your own Git remote, clone from that remote instead. To publish, clone the bundle locally, then run `git remote set-url origin YOUR_REPO_URL` and `git push -u origin main` in that clone.

## 2. Configure and run

On the server, copy the example configuration and set the bag path:

```bash
cp config/spnet.yaml config/spnet.server.yaml
nano config/spnet.server.yaml
```

Set `bag_path` to `~/rosbag2_2026_05_12-16_58_33_0_corrected.mcap` if you copied it to your home directory. Set `match_id` from the first CSV column. Choose `Tiny`, `Small`, `Base`, or `Large`, then set `resolution_wh` in width, height order. Both dimensions must be multiples of 32 because of the upstream model's five downsampling stages and skip connections. The output has the same resolution as the input. Use the same aspect ratio as the source if you want to preserve image proportions. Increase resolution or decrease `virtual_stride` to sample more virtual points. Large models at high resolutions require more VRAM.

```bash
./scripts/run_spnet.sh config/spnet.server.yaml
```

The script creates a virtual environment, installs CUDA-enabled PyTorch and the benchmark dependencies, clones SPNet, downloads the published checkpoint for the selected model, and runs headlessly. ROS 2 installation is unnecessary. The default PyTorch wheel uses CUDA 12.8. For a compatible alternate wheel, set `TORCH_INDEX_URL`, such as `https://download.pytorch.org/whl/cu126`, and use a fresh environment via `SPNET_VENV`. To select a Python executable use `PYTHON_BIN`.

If Google Drive blocks an automated download, use the variant link in the [upstream checkpoint table](https://github.com/Wang-xjtu/SPNet#test) and transfer the file to `~/cam_test/checkpoints/<model>_300.pth`. You can also set `checkpoint` to an existing trusted SPNet checkpoint with the upstream `network` state dictionary format.

Once that model's checkpoint and dependencies are installed, rerun the benchmark directly after changing the match or resolution:

```bash
.venv-spnet/bin/python benchmark_spnet.py --config config/spnet.server.yaml
```

Run the installer again when changing to a variant whose checkpoint is missing.

## 3. Copy and inspect the results

Each invocation saves a new folder under `runs/` and prints its full path. Names start with `spnet`, followed by the UTC time and date, for example `spnet_20-43-39_2026-11-20_Tiny_512x288_match130_65c5df33`. To rename existing directories on the server, preview with `bash scripts/rename_runs.sh --dry-run runs` and apply with `bash scripts/rename_runs.sh runs`. From your local computer, substitute the printed run directory name:

```bash
mkdir -p runs
scp -r you@gpu-server:~/cam_test/runs/RUN_DIRECTORY ./runs/
rerun ./runs/RUN_DIRECTORY/scene.rrd
```

Use Rerun 0.43.2 to match the recorder. The recording opens with a 3D view of the measured LiDAR cloud and both sets of RGB-colored virtual points in a common frame. The **Left** and **Right** tabs show the resized, undistorted image, sparse LiDAR depth, and estimated metric depth on the same pixel grid. Virtual points are back-projected from valid predictions at pixels without a LiDAR measurement. Their stride is for export only. The default uses the left optical frame and rigid transforms at LiDAR time. If accurate dynamic transforms are available, set `visualization.fixed_frame` in the sensor config to use each sensor's capture time.

`report.json` contains model and checkpoint provenance, GPU and software versions, match timestamps and camera geometry, and per-camera timing samples and summaries. The first pass is reported separately for each camera. The right camera's first pass can benefit from caches warmed by the left. The measured samples follow the configured warmups.

- **Image-to-depth wall time** starts at compressed image decoding and ends when the GPU depth map is ready. It includes CPU dispatch and synchronization, and uses the scan and camera geometry already prepared for this match.
- **CUDA event times** cover RGB upload, rectification, resize, normalization, LiDAR transform and nearest-point rasterization, SPNet forward, and conversion to metric depth. The model runs in FP32 with batch size one.
- **Setup and export times** include CPU bag scanning, TF and calibration setup, GPU uploads, model loading, back-projection, output readback, and recording serialization. Virtual point generation is timed on a separate export pass.
- **Peak allocation** is PyTorch CUDA allocated memory, including resident model weights. It is measured across each camera's first pass, warmups, and timed passes.

The 100 m default follows the upstream outdoor normalization scale. Depth is optical-axis Z. Zero denotes missing or invalid depth. The recording provides a visual comparison with measured LiDAR, which also supplies the model input.

## Export a ROS 1 bag for LV-DOT

Use the SPNet environment and checkpoint from the setup above. Run this from the
repository root on the GPU machine. Use a checkout that includes the exporter,
the corrected MCAP, and its matching `sync_matches.csv`.

```bash
.venv-spnet/bin/python generate_lv_dot_bag.py \
  --config config/spnet.yaml \
  --bag ~/rosbag2_2026_05_12-16_58_33_0_corrected.mcap \
  --start-match 130 --frames 50 \
  --output runs/lv_dot_test.bag
```

The exporter uses the left camera and the Velodyne scans. It writes aligned BGR
and millimetre-depth images, rectified camera calibration, the original LiDAR
scans in the selected time span, and poses reconstructed from
`odom → base_footprint → base_link`. It preserves the sensor capture timestamps
in the message headers. Set `resolution_wh` in the SPNet config to control the
shared image size.

Merge the keys from `runs/lv_dot_test.detector.yaml` into LV-DOT's
`onboard_detector/cfg/detector_param.yaml` so the detector uses the exported
topics, calibration, and pose mode. Keep the other detector settings from your
checkout. Mount the output directory at `/data` in the Noetic container. With
`roscore` running, set `rosparam set /use_sim_time true`, launch
`roslaunch onboard_detector run_detector.launch`, and replay the bag in a
second shell with `rosbag play --clock --pause /data/lv_dot_test.bag`.
