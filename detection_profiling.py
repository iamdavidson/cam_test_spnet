"""Wall time, CUDA stage timing, and process resource measurements."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter, process_time

import numpy as np
import psutil
import torch


TRACE_PREFIX = "detection/"


@dataclass
class StageTimes:
    host_ms: dict = field(default_factory=dict)
    cuda_events: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @contextmanager
    def stage(self, name, *, cuda=False):
        if cuda:
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            yield
            end.record()
            self.cuda_events[name] = (begin, end)
        else:
            started = perf_counter()
            yield
            self.host_ms[name] = (perf_counter() - started) * 1000

    def report(self):
        return {
            'host_ms': self.host_ms,
            'cuda_stream_ms': {
                name: start.elapsed_time(end) for name, (start, end) in self.cuda_events.items()
            },
        }


class Profiler:
    """Measure process resources relative to the start of each pipeline scope."""

    def __init__(self, device, interval_ms):
        self.device = device
        self.interval = interval_ms / 1000
        self.process = psutil.Process()
        self.samples = []
        self.pass_cuda_peak_bytes = 0
        self.trace_gpu = False
        self.stop = Event()

    def __enter__(self):
        self.thread = Thread(target=self.sample_ram, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def sample_ram(self):
        while not self.stop.is_set():
            self.samples.append((perf_counter(), self.process.memory_info().rss))
            self.stop.wait(self.interval)

    def start(self):
        return (
            perf_counter(), process_time(), self.process.memory_info().rss,
            torch.cuda.memory_allocated(self.device), len(self.samples),
        )

    def start_pass(self):
        start = self.start()
        self.pass_cuda_peak_bytes = torch.cuda.memory_allocated(self.device)
        return start

    def finish(self, start, cuda_peak_bytes):
        wall_start, cpu_start, rss_start, cuda_start, sample_start = start
        wall_end, cpu_end = perf_counter(), process_time()
        elapsed = wall_end - wall_start
        rss_peak = max(rss_start, self.process.memory_info().rss)
        for timestamp, rss in self.samples[sample_start:]:
            if wall_start <= timestamp <= wall_end:
                rss_peak = max(rss_peak, rss)
        cpu_seconds = cpu_end - cpu_start
        return {
            'wall_ms': elapsed * 1000,
            'cpu_ms': cpu_seconds * 1000,
            'cpu_percent_one_core': cpu_seconds / elapsed * 100,
            'extra_ram_mib': (rss_peak - rss_start) / 2**20,
            'extra_vram_mib': (cuda_peak_bytes - cuda_start) / 2**20,
        }

    def gpu_scope(self, name):
        if self.trace_gpu:
            return torch.profiler.record_function(TRACE_PREFIX + name)
        return nullcontext()

    @contextmanager
    def measure(self, name):
        torch.cuda.reset_peak_memory_stats(self.device)
        start = self.start()
        timing, result = StageTimes(), {}
        with self.gpu_scope(name):
            yield timing, result
            torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        self.pass_cuda_peak_bytes = max(self.pass_cuda_peak_bytes, peak)
        result.update(self.finish(start, peak), stages=timing.report(), notes=timing.notes)


def gpu_trace_summary(events, camera_stages, device_index):
    """Measure the fraction of each traced scope occupied by this process's GPU work."""
    scopes = {}
    gpu_intervals = []
    for event in events:
        if event.get('ph') != 'X':
            continue
        category = event.get('cat')
        name = event.get('name', '')
        if category == 'user_annotation' and name.startswith(TRACE_PREFIX):
            scopes[name.removeprefix(TRACE_PREFIX)] = (event['ts'], event['ts'] + event['dur'])
        if category in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            if event.get('args', {}).get('device', device_index) == device_index:
                gpu_intervals.append((event['ts'], event['ts'] + event['dur']))

    if not gpu_intervals:
        raise ValueError('CUDA trace contains no device activity')
    gpu_intervals.sort()
    # Merge concurrent kernels and copies so their shared time is counted only once.
    merged_intervals = []
    for start, end in gpu_intervals:
        if merged_intervals and start <= merged_intervals[-1][1]:
            previous_start, previous_end = merged_intervals[-1]
            merged_intervals[-1] = (previous_start, max(previous_end, end))
        else:
            merged_intervals.append((start, end))

    def scope_metrics(name):
        if name not in scopes:
            raise ValueError(f'CUDA trace is missing the {name} scope')
        scope_start, scope_end = scopes[name]
        busy_us = sum(max(0, min(end, scope_end) - max(start, scope_start))
                      for start, end in merged_intervals)
        wall_us = scope_end - scope_start
        return {
            'active_ms': busy_us / 1000,
            'traced_wall_ms': wall_us / 1000,
            'active_percent': 100 * busy_us / wall_us if wall_us > 0 else None,
        }

    overall = scope_metrics('Pipeline')
    if overall['active_ms'] == 0:
        raise ValueError('CUDA trace contains no device activity inside the pipeline')
    cameras = {}
    for camera, stages in camera_stages.items():
        cameras[camera] = {stage: scope_metrics(f'{camera}/{stage}') for stage in stages}
    return {'overall': overall, 'cameras': cameras}


def measure_gpu_activity(profiler, run_pass, camera_stages):
    """Use a separate CUPTI pass so tracing does not affect benchmark timings."""
    cpu = torch.profiler.ProfilerActivity.CPU
    cuda = torch.profiler.ProfilerActivity.CUDA
    if cuda not in torch.profiler.supported_activities():
        return {'unavailable': 'PyTorch CUDA activity tracing is unavailable'}

    profiler.trace_gpu = True
    try:
        with torch.profiler.profile(activities=[cpu, cuda]) as trace:
            run_pass()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu-trace.json'
            trace.export_chrome_trace(str(path))
            events = json.loads(path.read_text())['traceEvents']
        return gpu_trace_summary(events, camera_stages, profiler.device.index)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        return {'unavailable': str(error)}
    finally:
        profiler.trace_gpu = False


@contextmanager
def measure_wall_time(results, label):
    start = perf_counter()
    yield
    results[label] = {'wall_ms': (perf_counter() - start) * 1000}


def summarize(samples):
    metrics = [name for name in samples[0] if name not in {'stages', 'notes'}]
    summary = {}
    for name in metrics:
        values = [sample[name] for sample in samples if sample[name] is not None]
        summary[name] = {
            'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)),
            'max': max(values),
        } if values else None
    summary['stages'] = {}
    for clock in ('host_ms', 'cuda_stream_ms'):
        stages = {}
        for sample in samples:
            for stage, elapsed in sample.get('stages', {}).get(clock, {}).items():
                stages.setdefault(stage, []).append(elapsed)
        summary['stages'][clock] = {
            name: {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
                   'samples': len(values)}
            for name, values in stages.items()
        }
    return summary
