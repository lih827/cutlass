#!/usr/bin/env python3
"""Estimate Qwen2.5 forward latency from measured GEMM best times.

This intentionally reports a GEMM-only lower bound: no fusion, Batch=1,
ordinary GEMMs for attention, FP16 inputs/output/accumulation, and only the
last-position LM Head.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PRESETS = {
    "0.5b": (896, 14, 2, 64, 4864, 151936, 24),
    "1.5b": (1536, 12, 2, 128, 8960, 151936, 28),
    "3b": (2048, 16, 2, 128, 11008, 151936, 36),
    "7b": (3584, 28, 4, 128, 18944, 152064, 28),
    "14b": (5120, 40, 8, 128, 13824, 152064, 48),
    "32b": (5120, 40, 8, 128, 27648, 152064, 64),
    "72b": (8192, 64, 8, 128, 29568, 152064, 80),
}
CASE_RE = re.compile(r"MxNxK=(\d+)x(\d+)x(\d+)\s*$")
BEST_RE = re.compile(r"^Best configuration:")
TIME_RE = re.compile(r"^\s*avg_time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*$")


def parse_best_times(path: Path) -> dict[tuple[int, int, int], float]:
    current_shape = None
    awaiting_time = False
    times: dict[tuple[int, int, int], float] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CASE_RE.search(raw)
        if match:
            current_shape = tuple(map(int, match.groups()))
            awaiting_time = False
            continue
        if current_shape and BEST_RE.match(raw):
            awaiting_time = True
            continue
        match = TIME_RE.match(raw)
        if current_shape and awaiting_time and match:
            times[current_shape] = float(match.group(1))
            awaiting_time = False
    if not times:
        raise ValueError(f"no complete Best configuration/avg_time records in {path}")
    return times


def weighted_shapes(model: str, stage: str, length: int):
    h, heads, kv_heads, d, intermediate, vocab, layers = PRESETS[model]
    token_m = length if stage == "prefill" else 1
    attention_m = heads * token_m if stage == "prefill" else heads
    attention_n = length
    rows = [
        ("Q + Attention Out", (token_m, h, h), 2 * layers),
        ("K + V", (token_m, kv_heads * d, h), 2 * layers),
        ("Attention QK^T", (attention_m, attention_n, d), layers),
        ("Attention PV", (attention_m, d, attention_n), layers),
        ("MLP Up + Gate", (token_m, intermediate, h), 2 * layers),
        ("MLP Down", (token_m, h, intermediate), layers),
        ("LM Head (last token only)", (1, vocab, h), 1),
    ]
    return layers, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--model", default="7b", choices=PRESETS)
    parser.add_argument("--stage", required=True, choices=("prefill", "decode"))
    parser.add_argument("--length", required=True, type=int,
                        help="Prefill prompt S or Decode KV length L")
    args = parser.parse_args()
    if args.length <= 0:
        parser.error("--length must be positive")

    try:
        times = parse_best_times(args.log)
        layers, rows = weighted_shapes(args.model, args.stage, args.length)
        missing = [(name, shape) for name, shape, _ in rows if shape not in times]
        if missing:
            text = ", ".join(f"{name}={m}x{n}x{k}" for name, (m, n, k) in missing)
            raise ValueError(f"required shapes are missing from log: {text}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    total_ms = 0.0
    print(f"Qwen2.5-{args.model} {args.stage.capitalize()} GEMM-only estimate")
    print(f"Batch=1, length={args.length}, layers={layers}")
    print("operation                         MxNxK                 calls    time/call(ms)    subtotal(ms)")
    for name, shape, calls in rows:
        time_ms = times[shape]
        subtotal = calls * time_ms
        total_ms += subtotal
        shape_text = "x".join(map(str, shape))
        print(f"{name:<33} {shape_text:<21} {calls:>5} {time_ms:>16.6f} {subtotal:>15.6f}")
    print(f"GEMM-only lower-bound latency: {total_ms:.6f} ms")
    if args.stage == "decode":
        print(f"GEMM-only upper-bound throughput: {1000.0 / total_ms:.3f} token/s")
    print("Excluded: embedding, RMSNorm, RoPE, softmax/mask, activation, residual, memory/KV handling, launch gaps and communication.")
    print("FP16 accumulation is a BF16 performance proxy, not a numerical BF16 equivalent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
