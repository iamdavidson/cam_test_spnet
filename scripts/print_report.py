#!/usr/bin/env python3
"""Display an SPNet benchmark report in the terminal."""

import argparse
import json
import os
from pathlib import Path
import re
import sys


STYLES = {
    "heading": "1;36",
    "latency": "1;32",
    "tail": "1;33",
    "memory": "1;35",
    "muted": "2",
}
STATISTICS = ("min", "mean", "median", "p95", "max")
WALL_LATENCY = "image_to_depth_wall_ms"
CUDA_LATENCY = "image_to_depth_cuda_ms"
JSON_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"(\s*:)?|\b(?:true|false|null)\b|'
    r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
)


def styled(text, style, color):
    return f"\033[{STYLES[style]}m{text}\033[0m" if color else text


def timing_label(key):
    for suffix, label in (("_wall_ms", " (wall)"), ("_cuda_ms", " (CUDA)"),
                          ("_cpu_ms", " (CPU)"), ("_ms", "")):
        if key.endswith(suffix):
            return key.removesuffix(suffix).replace("_", " ").title() + label
    return key.replace("_", " ").title()


def milliseconds(value):
    return "—" if value is None else f"{value:,.3f}"


def count(value):
    return "—" if value is None else f"{value:,}"


def show_camera(name, camera, color):
    print(styled(f"\n{name.title()} camera", "heading", color))
    peak = camera.get("peak_cuda_allocated_bytes")
    if peak is not None:
        memory = styled(f"{peak / 1024**2:,.1f} MiB", "memory", color)
        print(f"  Peak CUDA allocation  {memory}")

    first = camera.get("first_pass_ms", {})
    summary = camera.get("summary_ms", {})
    metrics = dict.fromkeys((WALL_LATENCY, CUDA_LATENCY, *summary, *first))
    rows = [(key, timing_label(key)) for key in metrics if key in summary or key in first]
    if rows:
        width = max(len(label) for _, label in rows)
        columns = ("First", "Min", "Mean", "Median", "P95", "Max")
        header = f"  {'Timing (ms)':<{width}}  " + " ".join(f"{label:>9}" for label in columns)
        print(styled(header, "muted", color))
        for key, label in rows:
            stats = summary.get(key, {})
            values = [first.get(key), *(stats.get(stat) for stat in STATISTICS)]
            cells = [f"{milliseconds(value):>9}" for value in values]
            if key in (WALL_LATENCY, CUDA_LATENCY):
                label = styled(f"{label:<{width}}", "latency", color)
                cells[4] = styled(cells[4], "latency", color)
                cells[5] = styled(cells[5], "tail", color)
            else:
                label = f"{label:<{width}}"
            print(f"  {label}  " + " ".join(cells))

    export = camera.get("export", {})
    if export:
        sparse = count(export.get("sparse_depth_pixels"))
        estimated = count(export.get("estimated_depth_pixels"))
        virtual = count(export.get("virtual_points"))
        print(f"  Export  {sparse} sparse pixels  ·  {estimated} estimated pixels  ·  {virtual} virtual points")


def show_benchmark(report, color):
    settings = report.get("settings", {})
    model = report.get("model", {})
    environment = report.get("environment", {})
    match = report.get("match", {})
    resolution = " × ".join(map(str, settings.get("resolution_wh") or [])) or "—"

    print(styled(f"SPNet {model.get('variant', 'benchmark')}  ·  Match {match.get('id', '—')}",
                 "heading", color))
    print(f"  GPU  {environment.get('gpu', '—')}  ·  Resolution  {resolution}")
    print(f"  Samples per camera  {settings.get('iterations', '—')}  ·  Warmups  {settings.get('warmup', '—')}")
    print(f"  LiDAR points  {count(match.get('measured_points'))}")
    for name, camera in report["cameras"].items():
        show_camera(name, camera, color)

    timings = report.get("run_wall_ms", {})
    if timings:
        print(styled("\nRun timings", "heading", color))
        width = max(len(timing_label(key)) for key in timings)
        for key, value in timings.items():
            row = f"  {timing_label(key):<{width}}  {milliseconds(value):>10} ms"
            print(styled(row, "latency", color) if key == "elapsed_before_report_write" else row)


def show_json(data, color):
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    if color:
        def highlight(match):
            token = match.group()
            if match.group(1):
                return styled(token, "heading", color)
            style = "latency" if token.startswith('"') else "memory"
            return styled(token, style, color)

        formatted = JSON_TOKEN.sub(highlight, formatted)
    print(formatted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="path to report.json")
    parser.add_argument("--full", action="store_true", help="print every field as formatted JSON")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    args = parser.parse_args()

    try:
        data = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

    color = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    try:
        if not args.full and isinstance(data, dict) and isinstance(data.get("cameras"), dict):
            show_benchmark(data, color)
        else:
            show_json(data, color)
    except (AttributeError, TypeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: invalid benchmark report ({error}). Use --full to inspect it.\n")


if __name__ == "__main__":
    main()
