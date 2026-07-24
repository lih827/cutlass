#!/usr/bin/env python3
"""Select median CUTLASS winners and generate exact-shape optimal dispatch."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RECORD = re.compile(r"^CUTLASS_CANDIDATE\s+(.+)$")
FIELD = re.compile(r"(\w+)=([^\s]+)")
REQUIRED = (
    "m", "n", "k", "trans", "batch_count", "source", "name", "accumulator",
    "layout_a", "layout_b", "layout_c",
    "align_a", "align_b", "align_c", "tb_m", "tb_n", "tb_k",
    "warp_m", "warp_n", "warp_k", "swizzle", "stages", "split_k",
    "valid", "avg_time_ms", "gflops",
)


def config_key(fields: dict[str, str]) -> tuple[str, ...]:
    return tuple(fields[name] for name in REQUIRED[3:-2])


def parse_logs(paths: list[Path]):
    samples = defaultdict(list)
    configurations = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = RECORD.match(line)
            if not match:
                continue
            fields = dict(FIELD.findall(match.group(1)))
            missing = [name for name in REQUIRED if name not in fields]
            if missing:
                raise ValueError(f"Incomplete record in {path}: missing {','.join(missing)}")
            if fields["valid"] != "1":
                continue
            shape = (*tuple(int(fields[name]) for name in ("m", "n", "k")),
                     fields["trans"], int(fields["batch_count"]))
            key = (shape, config_key(fields))
            samples[key].append((float(fields["avg_time_ms"]), float(fields["gflops"])))
            configurations[key] = fields
    if not samples:
        raise ValueError("No CUTLASS_CANDIDATE records found")
    return samples, configurations


def load_manifest(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return {(int(row["m"]), int(row["n"]), int(row["k"]), row["trans"],
                 int(row["batch_count"])): row
                for row in csv.DictReader(stream)}


def load_cublaslt(paths: list[Path]):
    values = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                shape = (*tuple(int(row[name]) for name in ("m", "n", "k")),
                         row.get("trans", "NN"), int(row.get("batch_count", 1)))
                values[shape] = row
    return values


def swizzle_type(name: str) -> str:
    if name == "Identity":
        return "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>"
    if name == "Stream-K":
        return "cutlass::gemm::threadblock::ThreadblockSwizzleStreamK"
    raise ValueError(f"Unsupported swizzle in measured result: {name}")

def existing_entries(path: Path | None, excluded_type: str) -> list[str]:
    if path is None or not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries = []
    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        if not stripped.startswith(("GEMM_OPTIMAL_ENTRY(", "GEMM_OPTIMAL_ENTRY_EX(")):
            index += 1
            continue
        block = [lines[index]]
        while not block[-1].rstrip().endswith(")") and index + 1 < len(lines):
            index += 1
            block.append(lines[index])
        first_argument = block[0].split("(", 1)[1].split(",", 1)[0].strip()
        if first_argument != excluded_type:
            entries.append("\n".join(block))
        index += 1
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cublaslt-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--gpu", default="unknown")
    parser.add_argument("--cuda", default="unknown")
    parser.add_argument("--arch", default="unknown")
    parser.add_argument("--accumulator", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--existing-optimal", type=Path)
    args = parser.parse_args()

    samples, configurations = parse_logs(args.log)
    expected_accumulator = {"fp16": "half", "fp32": "float"}[args.accumulator]
    measured_accumulators = {
        fields["accumulator"] for fields in configurations.values()
    }
    if measured_accumulators != {expected_accumulator}:
        raise ValueError(
            f"Expected only {expected_accumulator} records, got "
            f"{sorted(measured_accumulators)}"
        )
    manifest = load_manifest(args.manifest)
    cublaslt = load_cublaslt(args.cublaslt_report)
    by_shape = defaultdict(list)
    for key, measured in samples.items():
        shape, _ = key
        fields = configurations[key]
        median_time = statistics.median(value[0] for value in measured)
        median_gflops = statistics.median(value[1] for value in measured)
        by_shape[shape].append((median_time, -len(measured), median_gflops, fields, measured))

    missing = sorted(set(manifest) - set(by_shape))
    if missing:
        preview = ", ".join("x".join(map(str, shape)) for shape in missing[:10])
        raise ValueError(f"Missing measured shapes ({len(missing)}): {preview}")

    winners = []
    for shape in sorted(manifest):
        maximum_samples = max(-candidate[1] for candidate in by_shape[shape])
        if maximum_samples != len(args.log):
            raise ValueError(
                f"No configuration completed every round for shape {shape}; "
                f"best coverage is {maximum_samples}/{len(args.log)}"
            )
        stable = [candidate for candidate in by_shape[shape]
                  if -candidate[1] == maximum_samples]
        median_time, neg_count, median_gflops, fields, measured = min(
            stable, key=lambda candidate: candidate[0])
        winner = dict(fields)
        winner.update({
            "median_time_ms": median_time,
            "median_gflops": median_gflops,
            "samples": len(measured),
            "models": manifest[shape]["models"],
            "stages_used": manifest[shape]["stages"],
            "operations": manifest[shape]["operations"],
            "lengths": manifest[shape]["lengths"],
            "configuration_origin": "direct:measured-cutlass-candidate",
            "selection_origin": "derived:minimum-median-time-across-rounds",
        })
        lt = cublaslt.get(shape, {})
        winner["cublaslt_gflops"] = lt.get("cublaslt_gflops", "")
        if winner["cublaslt_gflops"]:
            winner["cutlass_over_cublaslt"] = median_gflops / float(winner["cublaslt_gflops"])
        else:
            winner["cutlass_over_cublaslt"] = ""
        winners.append(winner)

    lines = [
        "// Generated by generate_optimal_configurations.py; do not edit.",
        f"// GPU: {args.gpu}", f"// CUDA: {args.cuda}", f"// Arch: {args.arch}",
        f"// Models: {args.models}", f"// Accumulator: {args.accumulator}",
        f"// Unique shapes: {len(winners)}", "",
    ]
    accumulator_type = {
        "fp16": "cutlass::half_t",
        "fp32": "float",
    }[args.accumulator]
    preserved_entries = existing_entries(args.existing_optimal, accumulator_type)
    for row in winners:
        lines.extend([
            f"// {row['models']} | {row['stages_used']} | {row['operations']} | "
            f"source={row['source']} median_gflops={float(row['median_gflops']):.4f}",
            f"GEMM_OPTIMAL_ENTRY_TRANS_EX({accumulator_type}, "
            f"{row['m']}, {row['n']}, {row['k']}, \"{row['trans']}\", "
            f"{row['batch_count']},",
            f"    {row['layout_a']}, {row['layout_b']}, {row['layout_c']}, "
            f"{row['align_a']}, {row['align_b']}, {row['align_c']},",
            f"    {row['tb_m']}, {row['tb_n']}, {row['tb_k']}, "
            f"{row['warp_m']}, {row['warp_n']}, {row['warp_k']},",
            f"    {swizzle_type(row['swizzle'])}, {row['stages']}, {row['split_k']})",
            "",
        ])
    if preserved_entries:
        lines.extend([
            "// Preserved mappings for other accumulator types.",
            *preserved_entries,
            "",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    report_fields = [
        "m", "n", "k", "trans", "batch_count", "models", "stages_used",
        "operations", "lengths",
        "source", "name", "accumulator", "layout_a", "layout_b", "layout_c",
        "align_a", "align_b", "align_c", "tb_m", "tb_n", "tb_k",
        "warp_m", "warp_n", "warp_k", "swizzle", "stages", "split_k",
        "samples", "median_time_ms", "median_gflops", "cublaslt_gflops",
        "cutlass_over_cublaslt", "configuration_origin", "selection_origin",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(winners)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu": args.gpu, "cuda": args.cuda, "arch": args.arch,
        "models": args.models, "accumulator": args.accumulator,
        "unique_shapes": len(winners),
        "logs": [str(path) for path in args.log],
        "manifest": str(args.manifest),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    print(f"Generated {len(winners)} optimal mappings in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
