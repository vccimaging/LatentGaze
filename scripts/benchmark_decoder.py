"""Measure the CPU forward-pass latency of the measurement decoder.

This measures software inference only. Optical sensing, ADC/FPGA readout, data transfer,
and eye reconstruction are outside the timed region.

    python scripts/benchmark_decoder.py weights/masks_decoder.pt --output runs/decoder_latency.json
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latentgaze.models import load_decoder


def cpu_name():
    path = Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument('--measurements', help='Optional prototype NPZ containing measures (N, 16).')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', help='Optional JSON report path.')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.threads < 1:
        parser.error('warmup must be nonnegative; iterations and threads must be positive')

    torch.set_num_threads(args.threads)
    model = load_decoder(args.checkpoint, map_location='cpu').eval()
    width = model.norm.normalized_shape[0]
    if args.measurements:
        with np.load(args.measurements) as data:
            values = np.asarray(data['measures'], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != width or not len(values) or not np.isfinite(values).all():
            parser.error(f'measures must be a nonempty finite array with shape (N, {width})')
        samples = torch.from_numpy(values)
    else:
        samples = torch.zeros(1, width)
    inputs = [row.unsqueeze(0) for row in samples]

    for i in range(args.warmup):
        model(inputs[i % len(inputs)])
    elapsed = []
    for i in range(args.iterations):
        value = inputs[i % len(inputs)]
        start = perf_counter_ns()
        model(value)
        elapsed.append((perf_counter_ns() - start) / 1e6)

    report = {
        'scope': 'CPU decoder forward pass only; excludes hardware acquisition and data transfer',
        'checkpoint': args.checkpoint,
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'measurement_source': args.measurements or 'zero-valued synthetic input',
        'input_dimensions': width,
        'parameters': sum(p.numel() for p in model.parameters()),
        'batch_size': 1,
        'cpu': cpu_name(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'threads': torch.get_num_threads(),
        'warmup': args.warmup,
        'iterations': args.iterations,
        'mean_ms': float(np.mean(elapsed)),
        'median_ms': float(np.median(elapsed)),
        'p95_ms': float(np.percentile(elapsed, 95)),
        'std_ms': float(np.std(elapsed)),
    }
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
