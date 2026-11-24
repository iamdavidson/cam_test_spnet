"""Print a compact terminal report for the YOLO → SPNet → Boxer benchmark."""

import argparse
import json
from pathlib import Path
import re

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


STAGES = ('Camera input', 'YOLO', 'SPNet', 'Boxer')
MISSING = '—'


def latency(milliseconds):
    if milliseconds is None:
        return MISSING
    if milliseconds >= 1000:
        return f'{milliseconds / 1000:.2f} s'
    return f'{milliseconds:.2f} ms'


def memory(mib):
    if mib is None:
        return MISSING
    if mib >= 1024:
        return f'{mib / 1024:.2f} GiB'
    return f'{mib:.0f} MiB'


def overview(report):
    match = report['match']['match_id']
    gpu = report['gpu']
    timed = report['summary']['overall']['wall_ms']
    run = report['overall_run']
    camera_count = len(report['summary']['cameras'])
    camera_label = 'camera' if camera_count == 1 else 'cameras'

    content = Text(f'Match {match}  ·  {gpu}', style='bold')
    content.append('\n\n')
    content.append(latency(timed['median']), style='bold bright_green')
    content.append(f' median per match ({camera_count} {camera_label})\n')
    content.append(latency(timed['p95']), style='bright_yellow')
    content.append(' p95 per match\n\n')
    content.append(memory(run['cuda_peak_allocated_mib']), style='bright_magenta')
    content.append(' peak VRAM    ')
    content.append(memory(run['ram_rss_peak_mib']), style='bright_cyan')
    content.append(' peak RAM')
    content.append(f"\n{latency(run['wall_ms'])} total incl. setup and export", style='dim')
    return Panel.fit(
        content, title='YOLO → SPNet → Boxer', title_align='left',
        border_style='bright_blue', padding=(1, 1),
    )


def camera_table(cameras, counts):
    table = Table(
        box=box.SIMPLE, header_style='bold bright_cyan', padding=(0, 1), collapse_padding=True,
    )
    table.add_column('Stage · median')
    for name in cameras:
        table.add_column(name.title(), justify='right', no_wrap=True)

    for stage in STAGES:
        cells = []
        for name, metrics in cameras.items():
            if stage == 'Boxer' and name in counts and counts[name][0] == 0:
                cells.append(Text('Skipped', style='dim'))
                continue
            stage_time = metrics.get(stage, {}).get('wall_ms', {}).get('median')
            cells.append(latency(stage_time))
        table.add_row(stage, *cells, end_section=stage == 'Boxer')

    table.add_row(
        '2D / 3D boxes',
        *(f'{counts[name][0]} / {counts[name][1]}' if name in counts else MISSING for name in cameras),
        style='bold',
    )
    return table


def print_details(report, console):
    first_pass = report.get('first_pass', {}).get('overall', {}).get('wall_ms')
    if first_pass is not None:
        console.print(Text(f'First pass  {latency(first_pass)}', style='bold'))

    setup = report.get('setup_and_export', {})
    if setup:
        table = Table('Setup and export', 'Wall time', box=box.SIMPLE, header_style='bold cyan')
        for label, metrics in setup.items():
            table.add_row(label, latency(metrics['wall_ms']))
        console.print(table)

    for name, stages in report['summary']['cameras'].items():
        table = Table(box=box.SIMPLE, header_style='bold cyan')
        table.add_column(f'{name.title()} camera')
        table.add_column('Median', justify='right')
        table.add_column('p95', justify='right')
        for stage, metrics in stages.items():
            wall_time = metrics['wall_ms']
            table.add_row(stage, latency(wall_time['median']), latency(wall_time['p95']), style='bold')
            for clock, subtasks in metrics.get('stages', {}).items():
                for label, times in subtasks.items():
                    source = 'CUDA' if clock == 'cuda_stream_ms' else 'wall'
                    table.add_row(
                        f'  {label} ({source})', latency(times['median']), latency(times['p95']),
                    )
        console.print(table)


def print_report(report, *, console=None, details=False):
    if console is None:
        console = Console()

    counts = report.get('detection_counts', {}).copy()
    for name, detections in report.get('detections', {}).items():
        counts[name] = (len(detections['yolo_2d']), len(detections['boxer_3d']))

    console.print(overview(report))
    console.print(camera_table(report['summary']['cameras'], counts))

    if details:
        print_details(report, console)


def read_report(path):
    text = path.read_text(encoding='utf-8').lstrip()
    if text.startswith(('{', '[')):
        report = json.loads(text)
        if not isinstance(report, dict) or 'summary' not in report or 'overall_run' not in report:
            raise ValueError('Expected a detection benchmark report.json')
        return report
    return read_legacy_log(text)


def read_legacy_log(text):
    """Read the timings and counts from the previous detection console format."""
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    headers = list(re.finditer(r'^YOLO → SPNet → Boxer \| Match (\d+) \| (.+)$', text, re.MULTILINE))
    if not headers:
        raise ValueError('Expected a detection report.json or a complete detection benchmark log')
    header = headers[-1]
    text = text[header.end():]
    timed = re.search(r'^ +Timed pass +([\d.]+) ms median, ([\d.]+) ms p95', text, re.MULTILINE)
    total = re.search(
        r'^ +Total +([\d.]+) ms \| CPU [^\n]+? \| RAM ([\d.]+) MiB [^\n]+? \| VRAM ([\d.]+) MiB',
        text, re.MULTILINE,
    )
    if not (timed and total):
        raise ValueError('The last detection benchmark in this log is incomplete')

    report = {
        'match': {'match_id': int(header[1])},
        'gpu': header[2].strip(),
        'overall_run': {
            'wall_ms': float(total[1]),
            'ram_rss_peak_mib': float(total[2]),
            'cuda_peak_allocated_mib': float(total[3]),
        },
        'summary': {'overall': {'wall_ms': times(timed[1], timed[2])}, 'cameras': {}},
        'detection_counts': {},
        'setup_and_export': {},
    }
    camera_metrics = None
    stage_metrics = None
    for line in text.splitlines():
        camera = re.fullmatch(r'(\w+) \| (\d+) 2D boxes, (\d+) 3D boxes', line)
        if camera:
            name = camera[1].lower()
            camera_metrics = {}
            report['summary']['cameras'][name] = camera_metrics
            report['detection_counts'][name] = (int(camera[2]), int(camera[3]))
            stage_metrics = None
            continue

        stage = re.match(r'^  (\S.*?) +([\d.]+) ms median, ([\d.]+) ms p95', line)
        if camera_metrics is not None and stage and stage[1] in STAGES:
            stage_metrics = {'wall_ms': times(stage[2], stage[3]), 'stages': {}}
            camera_metrics[stage[1]] = stage_metrics
            continue

        subtask = re.fullmatch(r'    (.+?) +([\d.]+) ms median, ([\d.]+) ms p95 \((wall|CUDA stream)\)', line)
        if stage_metrics is not None and subtask:
            clock = 'cuda_stream_ms' if subtask[4] == 'CUDA stream' else 'host_ms'
            stage_metrics['stages'].setdefault(clock, {})[subtask[1]] = times(subtask[2], subtask[3])
            continue

        startup = re.match(r'^  (Load .+?|Write scene\.rrd|First pass) +([\d.]+) ms \|', line)
        if startup:
            label, elapsed = startup.groups()
            if label == 'First pass':
                report['first_pass'] = {'overall': {'wall_ms': float(elapsed)}}
            else:
                report['setup_and_export'][label] = {'wall_ms': float(elapsed)}

    return report


def times(median, p95):
    return {'median': float(median), 'p95': float(p95)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+', help='report.json files, run directories, or legacy logs')
    parser.add_argument('--details', action='store_true', help='show startup and internal stage timings')
    parser.add_argument('--color', choices=('auto', 'always', 'never'), default='auto')
    args = parser.parse_args()

    force_terminal = {'auto': None, 'always': True, 'never': False}[args.color]
    console = Console(force_terminal=force_terminal)
    for index, path in enumerate(args.reports):
        path = path.expanduser()
        if path.is_dir():
            path /= 'report.json'
        try:
            report = read_report(path)
            if len(args.reports) > 1:
                if index:
                    console.print()
                console.print(Text(str(path), style='dim'))
            print_report(report, console=console, details=args.details)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            parser.exit(1, f'{parser.prog}: {path}: {error}\n')


if __name__ == '__main__':
    main()
