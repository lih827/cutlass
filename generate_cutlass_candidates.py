#!/usr/bin/env python3
"""Generate exact-shape CUTLASS dispatch from measured cuBLASLt winners."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

BEST = re.compile(r"CUBLASLT_BEST\s+(.+)$")
FIELD = re.compile(r"(\w+)=([^\s]+)")

# Stable cuBLASLt tile IDs used by Ampere-era kernels.
TILES = {
    1:(8,8), 2:(8,16), 3:(16,8), 4:(8,32), 5:(16,16), 6:(32,8),
    7:(8,64), 8:(16,32), 9:(32,16), 10:(64,8), 11:(32,32),
    12:(32,64), 13:(64,32), 14:(32,128), 15:(64,64), 16:(128,32),
    17:(64,128), 18:(128,64), 19:(64,256), 20:(128,128),
    21:(256,64), 22:(64,512), 23:(128,256), 24:(256,128),
    25:(512,64), 26:(64,96), 27:(96,64), 28:(96,128), 29:(128,160),
    30:(160,128), 31:(192,128), 32:(128,192), 33:(128,96),
    34:(32,256), 35:(256,32),
}

# Legal, deliberately bounded SM80 CUTLASS CTA catalog and corresponding warp M/N.
CATALOG = {
    (32,32):(32,32), (32,64):(32,64), (64,32):(64,32),
    (32,128):(32,64), (64,64):(32,64), (128,32):(64,32),
    (64,128):(32,64), (128,64):(64,32), (64,256):(32,64),
    (128,128):(64,64), (256,64):(64,32), (64,512):(64,64),
    (128,256):(64,64), (256,128):(64,64), (512,64):(64,64),
    (32,256):(32,64), (256,32):(64,32),
}


def nearest_tile(tile: tuple[int, int]) -> tuple[int, int]:
    return min(CATALOG, key=lambda t: abs(math.log2(t[0] / tile[0])) + abs(math.log2(t[1] / tile[1])))


def validate_catalog() -> None:
    for (tbm, tbn), (wm, wn) in CATALOG.items():
        if tbm % wm or tbn % wn or wm < 32 or wn < 32:
            raise ValueError(f"Illegal CUTLASS tile/warp mapping: {(tbm,tbn)} -> {(wm,wn)}")
        warps = (tbm // wm) * (tbn // wn)
        if not 1 <= warps <= 8:
            raise ValueError(f"CUTLASS mapping has {warps} warps: {(tbm,tbn)} -> {(wm,wn)}")


def stage_spec(stage_id: int) -> tuple[int, int, str]:
    if 1 <= stage_id <= 24:
        count = ((stage_id - 1) % 6) + 1
        # cuBLASLt's first stage field describes its internal stage byte/depth
        # class, not CUTLASS ThreadblockShape::kK. Keep the verified SM80 K=32.
        # One buffer is invalid for CUTLASS MmaMultistage and very deep pipelines
        # can exceed shared-memory limits for large CTA tiles.
        stages = min(4, max(2, count))
        reason = "exact-stage-count"
        if count < 2: reason = "cutlass-min-stage-2"
        elif count > 4: reason = "cutlass-max-stage-4"
        return 32, stages, reason
    return 32, 3, "fallback-stage"


def alignment(m: int, k: int) -> tuple[int, int, int, bool]:
    if m == 1:
        return 8, 8, 8, True
    if m % 8 == 0 and k % 8 == 0: return 8, 8, 8, True
    if m % 4 == 0 and k % 8 == 0: return 4, 8, 4, True
    if m % 2 == 0 and k % 8 == 0: return 2, 8, 2, True
    return 1, 1, 1, False


def parse_log(path: Path) -> dict[tuple[int,int,int], dict[str,str]]:
    records = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = BEST.search(line)
        if not match: continue
        fields = dict(FIELD.findall(match.group(1)))
        shape = tuple(int(fields[x]) for x in ("m", "n", "k"))
        records[shape] = fields
    if not records:
        raise ValueError(f"No CUBLASLT_BEST records found in {path}")
    return records


def main() -> int:
    validate_catalog()
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path, action="append",
        help="cuBLASLt log; repeat for Decode and Prefill")
    parser.add_argument("--output", type=Path,
        default=Path("examples/gemm/cublaslt_generated_candidates.inc"))
    parser.add_argument("--report", type=Path, default=Path("cublaslt_cutlass_mapping.csv"))
    args = parser.parse_args()
    records = {}
    for log in args.log:
        records.update(parse_log(log))
    rows, cases = [], []
    for (m,n,k), fields in sorted(records.items()):
        source = TILES.get(int(fields["tile_id"]))
        if source is None:
            source = (128,256) if n >= 2*m else ((256,128) if m >= 2*n else (128,128))
            reason = "unknown-tile-id"
        else:
            reason = "exact" if source in CATALOG else "nearest-legal-tile"
        tbm,tbn = source if source in CATALOG else nearest_tile(source)
        wm,wn = CATALOG[(tbm,tbn)]
        tbk,stages,stage_reason = stage_spec(int(fields["stages_id"]))
        aa,ab,ac,async_ok = alignment(m,k)
        if not async_ok:
            tbk,stages,stage_reason = 32,2,"alignment1-sync-override"
        wk = min(32,tbk)
        split_k = max(1, int(fields.get("split_k", "0")))
        layout = ("LayoutAM1, LayoutBM1, LayoutCM1" if m == 1 else
                  "LayoutAAttention, LayoutBAttention, LayoutCAttention")
        name = f"cuBLASLt-derived {tbm}x{tbn}x{tbk} S{stages}"
        cases.append(
            f"  if (options.m == {m} && options.n == {n} && options.k == {k}) {{\n"
            f"    return profile_cublaslt_template<{layout}, {aa}, {ab}, {ac},\n"
            f"        cutlass::gemm::GemmShape<{tbm}, {tbn}, {tbk}>,\n"
            f"        cutlass::gemm::GemmShape<{wm}, {wn}, {wk}>, {stages}>(\"{name}\", {split_k}, options);\n"
            f"  }}")
        rows.append({"m":m,"n":n,"k":k,"algo_id":fields["algo_id"],
            "cublaslt_tile_id":fields["tile_id"],"cublaslt_tile":f"{source[0]}x{source[1]}",
            "cublaslt_stages_id":fields["stages_id"],"cutlass_threadblock":f"{tbm}x{tbn}x{tbk}",
            "cutlass_warp":f"{wm}x{wn}x{wk}","cutlass_stages":stages,
            "cutlass_split_k":split_k,"alignment":f"{aa}/{ab}/{ac}",
            "tile_mapping":reason,"stage_mapping":stage_reason})
    content = "// Generated by generate_cutlass_candidates.py; do not edit.\n"
    content += "int profile_cublaslt_generated_candidate(Options const &options) {\n"
    content += "\n".join(cases) + "\n  return -1;\n}\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Generated {len(rows)} shape mappings in {args.output} and {args.report}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
