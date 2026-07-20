#!/usr/bin/env python3
"""Generate a globally deduplicated Qwen2.5 GEMM shape manifest."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PRESETS = {
    "0.5b": (896, 14, 2, 64, 4864, 151936),
    "1.5b": (1536, 12, 2, 128, 8960, 151936),
    "3b": (2048, 16, 2, 128, 11008, 151936),
    "7b": (3584, 28, 4, 128, 18944, 152064),
    "14b": (5120, 40, 8, 128, 13824, 152064),
    "32b": (5120, 40, 8, 128, 27648, 152064),
    "72b": (8192, 64, 8, 128, 29568, 152064),
}
DECODE_LENGTHS = (128, 256, 512, 1024, 2048)
PREFILL_LENGTHS = (128, 256, 512, 1024, 2048, 129, 130, 132, 136)


def parse_models(value: str) -> list[str]:
    models = list(PRESETS) if value.lower() == "all" else value.lower().split(",")
    invalid = [model for model in models if model not in PRESETS]
    if invalid:
        raise ValueError(f"Unsupported models: {','.join(invalid)}")
    return list(dict.fromkeys(models))


def add(records, shape, model, stage, operation, length):
    item = records[shape]
    item["models"].add(model)
    item["stages"].add(stage)
    item["operations"].add(operation)
    item["lengths"].add(f"{'L' if stage == 'Decode' else 'S'}={length}")


def enumerate_shapes(models: list[str], batch: int):
    records = defaultdict(lambda: {
        "models": set(), "stages": set(), "operations": set(), "lengths": set()
    })
    for model in models:
        h, heads, kv_heads, head_dim, intermediate, vocab = PRESETS[model]
        token_m = batch
        base = {
            "Q / Attention Out": (token_m, h, h),
            "K / V": (token_m, kv_heads * head_dim, h),
            "MLP Up / MLP Gate": (token_m, intermediate, h),
            "MLP Down": (token_m, h, intermediate),
            "LM Head": (token_m, vocab, h),
        }
        for operation, shape in base.items():
            add(records, shape, model, "Decode", operation, "-")
        for length in DECODE_LENGTHS:
            add(records, (batch * heads, length, head_dim), model, "Decode", "Attention QK^T", length)
            add(records, (batch * heads, head_dim, length), model, "Decode", "Attention PV", length)
        for length in PREFILL_LENGTHS:
            token_m = batch * length
            attention_m = batch * heads * length
            prefill = {
                "Q / Attention Out": (token_m, h, h),
                "K / V": (token_m, kv_heads * head_dim, h),
                "MLP Up / MLP Gate": (token_m, intermediate, h),
                "MLP Down": (token_m, h, intermediate),
                "LM Head": (token_m, vocab, h),
                "Attention QK^T": (attention_m, length, head_dim),
                "Attention PV": (attention_m, head_dim, length),
            }
            for operation, shape in prefill.items():
                add(records, shape, model, "Prefill", operation, length)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("--batch must be positive")
    try:
        models = parse_models(args.models)
    except ValueError as error:
        parser.error(str(error))
    records = enumerate_shapes(models, args.batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("m", "n", "k", "models", "stages", "operations", "lengths"))
        for (m, n, k), item in sorted(records.items()):
            writer.writerow((m, n, k, ";".join(sorted(item["models"])),
                             ";".join(sorted(item["stages"])),
                             ";".join(sorted(item["operations"])),
                             ";".join(sorted(item["lengths"]))))
    print(f"Generated {len(records)} unique shapes for {','.join(models)} in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
