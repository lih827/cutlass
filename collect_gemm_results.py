#!/usr/bin/env python3
"""Parse run_gemm.sh output and update only the CUTLASS columns."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


CASE_RE = re.compile(
    r"^\[(?P<index>\d+)/(?P<total>\d+)\]\s+"
    r"L=(?P<length>\d+)\s+op=(?P<operation>.+?)\s+"
    r"MxNxK=(?P<m>\d+)x(?P<n>\d+)x(?P<k>\d+)\s*$"
)
BEST_RE = re.compile(r"^Best configuration:\s*(?P<configuration>.+?)\s*$")
GFLOPS_RE = re.compile(r"^\s*gflops:\s*(?P<gflops>[0-9]+(?:\.[0-9]+)?)\s*$")

COLUMNS = [
    "case_id",
    "context_length",
    "operation",
    "m",
    "n",
    "k",
    "cutlass_best_config",
    "cutlass_gflops",
    "hgemm_custom_best_config",
    "hgemm_custom_gflops",
    "hgemm_cuda_best_config",
    "hgemm_cuda_gflops",
    "hgemm_custom_vs_cutlass_speedup",
    "hgemm_cuda_vs_cutlass_speedup",
    "manual_selection_notes",
]

LEGACY_COLUMNS = [
    "case_id", "context_length", "operation", "m", "n", "k",
    "native_best_config", "native_gflops", "custom_best_config",
    "custom_gflops", "custom_vs_native_speedup",
]

NV_COLUMNS = [
    "case_id", "context_length", "operation", "m", "n", "k",
    "nv_best_config", "nv_gflops", "custom_best_config", "custom_gflops",
    "custom_vs_nv_speedup", "custom_selection_notes",
]


@dataclass(frozen=True)
class ParsedResult:
    context_length: int
    operation: str
    m: int
    n: int
    k: int
    best_configuration: str
    gflops: float

    @property
    def case_id(self) -> str:
        normalized_op = re.sub(r"[^a-z0-9]+", "_", self.operation.lower()).strip("_")
        return f"l{self.context_length}_{normalized_op}_{self.m}x{self.n}x{self.k}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse CUTLASS gemm output and update the CUTLASS result columns. "
            "HGEMM (self-developed/CUDA) results are selected and entered manually."
        )
    )
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("gemm_performance_comparison.csv")
    )
    return parser.parse_args()


def parse_log(path: Path) -> list[ParsedResult]:
    current_case: dict[str, str] | None = None
    pending_configuration: str | None = None
    results: list[ParsedResult] = []

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            case_match = CASE_RE.match(line)
            if case_match:
                current_case = case_match.groupdict()
                pending_configuration = None
                continue

            best_match = BEST_RE.match(line)
            if best_match and current_case:
                pending_configuration = best_match.group("configuration")
                continue

            gflops_match = GFLOPS_RE.match(line)
            if gflops_match and current_case and pending_configuration:
                results.append(
                    ParsedResult(
                        context_length=int(current_case["length"]),
                        operation=current_case["operation"],
                        m=int(current_case["m"]),
                        n=int(current_case["n"]),
                        k=int(current_case["k"]),
                        best_configuration=pending_configuration,
                        gflops=float(gflops_match.group("gflops")),
                    )
                )
                pending_configuration = None

    if not results:
        raise ValueError(
            f"No complete cases found in {path}. "
            "The log must contain run_gemm.sh case headers and gemm's Best configuration/GFLOPS output."
        )
    return results


def load_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames in (LEGACY_COLUMNS, NV_COLUMNS):
            migrated = {}
            for old in reader:
                row = {column: "" for column in COLUMNS}
                row.update({key: old.get(key, "") for key in COLUMNS if key in old})
                row["cutlass_best_config"] = old.get("nv_best_config", old.get("native_best_config", ""))
                row["cutlass_gflops"] = old.get("nv_gflops", old.get("native_gflops", ""))
                row["hgemm_custom_best_config"] = old.get("custom_best_config", "")
                row["hgemm_custom_gflops"] = old.get("custom_gflops", "")
                row["hgemm_custom_vs_cutlass_speedup"] = old.get(
                    "custom_vs_nv_speedup", old.get("custom_vs_native_speedup", "")
                )
                row["manual_selection_notes"] = old.get("custom_selection_notes", "")
                migrated[row["case_id"]] = row
            return migrated
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        return {row["case_id"]: row for row in reader}


def update_rows(
    rows: dict[str, dict[str, str]],
    results: list[ParsedResult],
) -> None:
    for result in results:
        row = rows.setdefault(result.case_id, {column: "" for column in COLUMNS})
        row.update(
            {
                "case_id": result.case_id,
                "context_length": str(result.context_length),
                "operation": result.operation,
                "m": str(result.m),
                "n": str(result.n),
                "k": str(result.k),
                "cutlass_best_config": result.best_configuration,
                "cutlass_gflops": f"{result.gflops:.4f}",
            }
        )
        cutlass = float(row["cutlass_gflops"]) if row["cutlass_gflops"] else 0.0
        custom = float(row["hgemm_custom_gflops"]) if row["hgemm_custom_gflops"] else 0.0
        cuda = float(row["hgemm_cuda_gflops"]) if row["hgemm_cuda_gflops"] else 0.0
        row["hgemm_custom_vs_cutlass_speedup"] = (
            f"{custom / cutlass:.4f}" if cutlass > 0.0 and custom > 0.0 else ""
        )
        row["hgemm_cuda_vs_cutlass_speedup"] = (
            f"{cuda / cutlass:.4f}" if cutlass > 0.0 and cuda > 0.0 else ""
        )


def write_table(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            int(row["context_length"]),
            row["operation"],
            int(row["m"]),
            int(row["n"]),
            int(row["k"]),
        ),
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)


def main() -> int:
    args = parse_args()
    try:
        parsed = parse_log(args.log)
        rows = load_existing(args.output)
        update_rows(rows, parsed)
        write_table(args.output, rows)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Updated {len(parsed)} CUTLASS cases in {args.output.resolve()}; "
        "HGEMM (self-developed/CUDA) columns were preserved for manual selection."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
