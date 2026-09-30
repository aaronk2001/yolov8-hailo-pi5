"""Synthetic latency/throughput benchmark for HailoInference.run().

Feeds random uint8 frames (fixed seed) at the model's input size, discards the
warmup iterations, and times only HailoInference.run() with time.perf_counter().
No camera is involved, so this measures the Hailo path alone.

Only one process can hold /dev/hailo0, so stop anything else using the NPU first:
    python benchmarks/run_benchmark.py --label baseline --out benchmarks/results-$(date +%F).md
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from hailo_inference import HailoInference  # noqa: E402


def hailo_firmware():
    try:
        out = subprocess.run(['hailortcli', 'fw-control', 'identify'],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 'unknown'
    for line in out.splitlines():
        if line.startswith('Firmware Version:'):
            return line.split(':', 1)[1].strip()
    return 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/yolov8n.hef')
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label', default='run')
    parser.add_argument('--out', help='markdown file to append the result row to')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hailo = HailoInference(args.model)
    shape = (hailo.input_height, hailo.input_width, 3)

    times_ms = []
    try:
        for i in range(args.warmup + args.frames):
            frame = rng.integers(0, 256, size=shape, dtype=np.uint8)
            input_data = hailo.preprocess(frame)
            t0 = time.perf_counter()
            hailo.run(input_data)
            dt = (time.perf_counter() - t0) * 1000
            if i >= args.warmup:
                times_ms.append(dt)
    finally:
        hailo.release()

    t = np.array(times_ms)
    stats = {
        'mean': t.mean(), 'min': t.min(), 'p50': np.percentile(t, 50),
        'p95': np.percentile(t, 95), 'p99': np.percentile(t, 99), 'max': t.max(),
    }
    fps = 1000.0 / stats['mean']
    row = (f"| {args.label} | {fps:.2f} | {stats['mean']:.2f} | {stats['p50']:.2f} | "
           f"{stats['p95']:.2f} | {stats['p99']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |")

    print(f"frames={args.frames} warmup={args.warmup} seed={args.seed} input={shape[1]}x{shape[0]}")
    print(f"FPS={fps:.2f} mean={stats['mean']:.2f}ms p50={stats['p50']:.2f}ms "
          f"p95={stats['p95']:.2f}ms p99={stats['p99']:.2f}ms")

    if args.out:
        new_file = not os.path.exists(args.out)
        with open(args.out, 'a') as f:
            if new_file:
                f.write(f"# Benchmark results — {datetime.date.today().isoformat()}\n\n")
                f.write(f"- Host: {platform.node()} ({platform.machine()}, {platform.platform()})\n")
                f.write(f"- Hailo firmware: {hailo_firmware()}\n")
                f.write(f"- Model: `{args.model}`, input {shape[1]}x{shape[0]}x3 uint8\n")
                f.write(f"- Frame source: synthetic random noise, seed {args.seed} "
                        f"(no camera)\n")
                f.write(f"- {args.frames} timed frames after {args.warmup} warmup; "
                        f"time.perf_counter() around HailoInference.run()\n")
                f.write("- FPS = 1000 / mean latency (sequential, batch 1)\n")
                f.write(f"- Command: `python benchmarks/run_benchmark.py --label <label> "
                        f"--out {args.out}`\n\n")
                f.write("| Variant | FPS | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms |\n")
                f.write("|---|---|---|---|---|---|---|---|\n")
            f.write(row + "\n")
        print(f"appended to {args.out}")


if __name__ == '__main__':
    main()
