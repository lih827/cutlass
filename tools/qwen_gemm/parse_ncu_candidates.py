#!/usr/bin/env python3
"""Parse NCU captures and emit NCU-exact CUTLASS completion candidates."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


# Verified for names such as:
# cutlass_80_wmma_tensorop_h161616gemm_32x32_128x2_nn_align8
# The schema encodes CTA MxN, then CTA K x pipeline stages.
KERNEL_SCHEMA = re.compile(
    r"cutlass_.*(?:wmma|tensorop).*gemm_"
    r"(?P<tb_m>\d+)x(?P<tb_n>\d+)_"
    r"(?P<tb_k>\d+)x(?P<stages>\d+)_"
    r"(?P<trans>[nt]{2})_align(?P<align>\d+)",
    re.IGNORECASE,
)
FIELD_NORMALIZE = re.compile(r"[^a-z0-9]+")
INTEGER = re.compile(r"\d+")


def normalized(value: str) -> str:
    return FIELD_NORMALIZE.sub("_", value.strip().lower()).strip("_")


def read_ncu_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines)
         if "Kernel Name" in line and ("Metric Name" in line or "ID" in line)),
        None,
    )
    if header_index is None:
        return []
    return list(csv.DictReader(lines[header_index:]))


def field(row: dict[str, str], *names: str) -> str:
    values = {normalized(key): value for key, value in row.items() if key}
    for name in names:
        if normalized(name) in values:
            return (values[normalized(name)] or "").strip()
    return ""


def parse_vector(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in INTEGER.findall(value))


def capture_kernels(path: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in read_ncu_rows(path):
        name = field(row, "Kernel Name")
        if not name:
            continue
        # The timed profiler launches the same kernel repeatedly. Group by
        # demangled name so repeated launches do not create duplicate templates.
        key = ("", name)
        record = grouped.setdefault(key, {
            "kernel_name": name, "block_threads": 0,
            "grid": "", "registers_per_thread": "", "shared_memory_bytes": "",
        })
        metric = field(row, "Metric Name")
        value = field(row, "Metric Value", "Value")
        metric_key = normalized(metric)
        if metric_key in ("launch_block_size", "launch_block_size_threads"):
            vector = parse_vector(value)
            record["block_threads"] = (
                vector[0] * vector[1] * vector[2] if len(vector) >= 3
                else (vector[0] if vector else 0)
            )
        elif metric_key in ("launch_grid_size", "launch_grid_size_blocks"):
            record["grid"] = value
        elif "registers_per_thread" in metric_key:
            record["registers_per_thread"] = value
        elif "shared_mem" in metric_key and "block" in metric_key:
            record["shared_memory_bytes"] = value

        # Some NCU CSV versions expose launch columns rather than metric rows.
        block = field(row, "Block Size", "Block")
        if block and not record["block_threads"]:
            vector = parse_vector(block)
            record["block_threads"] = (
                vector[0] * vector[1] * vector[2] if len(vector) >= 3
                else (vector[0] if vector else 0)
            )
    return list(grouped.values())


def layout_and_alignment(
    m: int, n: int, k: int, trans: str
) -> tuple[str, str, str, int, int, int]:
    # Same cuBLAS-column-major mapping used by gemm.cu.
    layout_a = ("cutlass::layout::ColumnMajor"
                if trans[0] == "N" else "cutlass::layout::RowMajor")
    layout_b = ("cutlass::layout::ColumnMajor"
                if trans[1] == "N" else "cutlass::layout::RowMajor")
    layout_c = "cutlass::layout::ColumnMajor"
    contiguous_a = m if layout_a.endswith("ColumnMajor") else k
    contiguous_b = k if layout_b.endswith("ColumnMajor") else n

    def alignment(extent: int) -> int:
        for width in (8, 4, 2):
            if extent % width == 0:
                return width
        return 1

    return (layout_a, layout_b, layout_c, alignment(contiguous_a),
            alignment(contiguous_b), alignment(m))


def warp_candidates(
    tb_m: int, tb_n: int, tb_k: int, block_threads: int
) -> list[tuple[int, int, int]]:
    if block_threads <= 0 or block_threads % 32:
        return []
    expected_warps = block_threads // 32
    choices: list[tuple[int, int, int]] = []
    # Deliberately bounded SM80+ TensorOp shapes. CTA and stages remain exact;
    # only WarpShape is enumerated as CUTLASS completion information.
    for wm in (16, 32, 64, 128):
        for wn in (16, 32, 64, 128):
            # CUTLASS's standard GEMM templates partition the CTA across M/N;
            # keep Warp K equal to the NCU CTA K instead of inventing a K warp
            # partition that cannot be inferred from block threads.
            for wk in (tb_k,):
                if tb_m % wm or tb_n % wn or tb_k % wk:
                    continue
                warps = (tb_m // wm) * (tb_n // wn) * (tb_k // wk)
                if warps == expected_warps and 1 <= warps <= 8:
                    choices.append((wm, wn, wk))
    return sorted(choices, key=lambda item: (
        abs(item[0] - item[1]), -item[0] * item[1], item))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--accumulator", choices=("fp16", "fp32"), required=True)
    args = parser.parse_args()

    with args.capture_index.open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))

    output_rows: list[dict[str, object]] = []
    captures: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        kernels = capture_kernels(Path(case["raw_csv"]))
        parsed_for_case = 0
        for kernel in kernels:
            match = KERNEL_SCHEMA.search(str(kernel["kernel_name"]))
            if not match:
                skipped.append({
                    "case_id": case["case_id"],
                    "kernel_name": str(kernel["kernel_name"]),
                    "reason": "unverified-kernel-name-schema",
                })
                continue
            values = {name: int(match.group(name))
                      for name in ("tb_m", "tb_n", "tb_k", "stages", "align")}
            trans = match.group("trans").upper()
            if trans != case["trans"]:
                skipped.append({
                    "case_id": case["case_id"],
                    "kernel_name": str(kernel["kernel_name"]),
                    "reason": f"kernel-trans-{trans}-does-not-match-{case['trans']}",
                })
                continue
            warps = warp_candidates(
                values["tb_m"], values["tb_n"], values["tb_k"],
                int(kernel["block_threads"]))
            if not warps:
                skipped.append({
                    "case_id": case["case_id"],
                    "kernel_name": str(kernel["kernel_name"]),
                    "reason": "no-legal-warp-shape-for-ncu-block-threads",
                })
                continue
            m, n, k = (int(case[name]) for name in ("m", "n", "k"))
            la, lb, lc, aa, ab, ac = layout_and_alignment(m, n, k, trans)
            for wm, wn, wk in warps:
                output_rows.append({
                    "accumulator": args.accumulator,
                    "m": m, "n": n, "k": k, "trans": trans,
                    "batch_count": int(case["batch_count"]),
                    "layout_a": la, "layout_b": lb, "layout_c": lc,
                    "align_a": aa, "align_b": ab, "align_c": ac,
                    "ncu_tb_m": values["tb_m"], "ncu_tb_n": values["tb_n"],
                    "ncu_tb_k": values["tb_k"],
                    "warp_m": wm, "warp_n": wn, "warp_k": wk,
                    "swizzle": "Identity", "ncu_stages": values["stages"],
                    "split_k": 1,
                })
            captures.append({
                "case_id": case["case_id"],
                "kernel_name": kernel["kernel_name"],
                "kernel_name_origin": "direct:ncu",
                "cta_origin": "parsed:verified-kernel-name-schema",
                "stages_origin": "parsed:verified-kernel-name-schema",
                "block_threads": kernel["block_threads"],
                "block_threads_origin": "direct:ncu",
                "warp_shape_origin": "enumerated:legal-cutlass-candidates",
                **values,
            })
            parsed_for_case += 1
        if not parsed_for_case:
            skipped.append({
                "case_id": case["case_id"], "kernel_name": "",
                "reason": "no-supported-gemm-kernel-for-case",
            })

    if not output_rows:
        raise ValueError("No supported NCU kernels produced CUTLASS candidates")
    fields = list(output_rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {
        "capture_index": str(args.capture_index),
        "candidate_rows": len(output_rows),
        "parsed_kernels": captures,
        "skipped": skipped,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    print(f"Generated {len(output_rows)} NCU-exact candidate rows")
    print(f"Metadata and skipped kernels: {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
