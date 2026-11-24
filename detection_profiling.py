"""Wall time, CUDA stage timing, and process resource measurements."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import resource
from threading import Event, Thread
from time import perf_counter, process_time

import numpy as np
import psutil
from pynvml import NVMLError
import torch


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
    """Sample process RSS and device utilization while measuring named intervals."""

    def __init__(self, device, interval_ms):
        self.device = device
        self.interval = interval_ms / 1000
        self.process = psutil.Process()
        self.samples = []
        self.peak_cuda_bytes = 0
        self.stop = Event()

    def __enter__(self):
        self.thread = Thread(target=self.sample_resources, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def sample_resources(self):
        has_nvml = True
        while not self.stop.is_set():
            now = perf_counter()
            rss = self.process.memory_info().rss
            utilization = None
            if has_nvml:
                try:
                    utilization = torch.cuda.utilization(self.device)
                except (ImportError, RuntimeError, OSError, NVMLError):
                    has_nvml = False
            self.samples.append((now, rss, utilization))
            self.stop.wait(self.interval)

    def start(self):
        return (
            perf_counter(), process_time(), self.process.memory_info().rss,
            torch.cuda.memory_allocated(self.device), len(self.samples),
        )

    def finish(self, start, cuda_peak):
        wall_start, cpu_start, rss_start, cuda_start, sample_start = start
        wall_end, cpu_end = perf_counter(), process_time()
        elapsed = wall_end - wall_start
        samples = [sample for sample in self.samples[sample_start:] if wall_start <= sample[0] <= wall_end]
        rss_peak = max([rss_start, self.process.memory_info().rss, *(sample[1] for sample in samples)])
        gpu_samples = [sample[2] for sample in samples if sample[2] is not None]
        return {
            'wall_ms': elapsed * 1000,
            'cpu_ms': (cpu_end - cpu_start) * 1000,
            'cpu_percent_one_core': (cpu_end - cpu_start) / elapsed * 100,
            'ram_rss_peak_mib': rss_peak / 2**20,
            'ram_rss_above_start_mib': (rss_peak - rss_start) / 2**20,
            'cuda_peak_allocated_mib': cuda_peak / 2**20,
            'cuda_above_start_mib': (cuda_peak - cuda_start) / 2**20,
            'gpu_device_busy_percent': float(np.mean(gpu_samples)) if gpu_samples else None,
            'gpu_samples': len(gpu_samples),
        }

    @contextmanager
    def measure(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        start = self.start()
        timing, result = StageTimes(), {}
        yield timing, result
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        self.peak_cuda_bytes = max(self.peak_cuda_bytes, peak)
        result.update(self.finish(start, peak), stages=timing.report(), notes=timing.notes)

    def run_totals(self, start):
        result = self.finish(start, self.peak_cuda_bytes)
        rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result['ram_rss_peak_mib'] = rss_peak / 2**20
        result['ram_rss_above_start_mib'] = (rss_peak - start[2]) / 2**20
        return result


def summarize(samples):
    metrics = [name for name in samples[0] if name not in {'stages', 'notes', 'gpu_samples'}]
    summary = {}
    for name in metrics:
        values = [sample[name] for sample in samples if sample[name] is not None]
        summary[name] = {
            'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)),
        } if values else None
    gpu_samples = sum(sample['gpu_samples'] for sample in samples)
    summary['gpu_samples'] = gpu_samples
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
