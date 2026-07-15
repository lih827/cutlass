#!/usr/bin/env bash
set -euo pipefail

model="7b"
iterations=20
arch="sm_89"
log_dir="cublaslt_tuning"

usage() {
  cat <<'EOF'
Profile Decode and Prefill with cuBLASLt, generate exact-shape CUTLASS
templates, and rebuild GEMM with the narrowed configuration set.

Usage: ./tune_cutlass_from_cublaslt.sh [options]
  --model MODEL       Qwen2.5 model size (default: 7b)
  --iterations N      Timed iterations per cuBLASLt candidate (default: 20)
  --arch ARCH         CUDA architecture passed to build_gemm.sh (default: sm_89)
  --log-dir DIR       Tuning artifacts directory (default: cublaslt_tuning)
  --help              Show this help
EOF
}

while (($#)); do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --iterations) iterations="$2"; shift 2 ;;
    --arch) arch="$2"; shift 2 ;;
    --log-dir) log_dir="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$log_dir"

# Never let a stale generated mapping break the baseline build. Preserve it as
# a tuning artifact instead of deleting it permanently.
generated_file="examples/gemm/cublaslt_generated_candidates.inc"
generated_tmp="$log_dir/cublaslt_generated_candidates.inc.new"
previous_generated="$log_dir/cublaslt_generated_candidates.inc.previous"
failed_generated="$log_dir/cublaslt_generated_candidates.inc.failed"

rm -f "$generated_tmp" "$failed_generated"
if [[ -f "$generated_file" ]]; then
  cp "$generated_file" "$previous_generated"
  rm -f "$generated_file"
fi

# First build deliberately excludes generated dispatch. It creates both the
# baseline CUTLASS executable and cuBLASLt profiler.
./build_gemm.sh --arch "$arch"

./run_gemm.sh --backend cublaslt --model "$model" --stage decode \
  --iterations "$iterations" | tee "$log_dir/cublaslt_decode.log"
./run_gemm.sh --backend cublaslt --model "$model" --stage prefill \
  --iterations "$iterations" | tee "$log_dir/cublaslt_prefill.log"

python3 generate_cutlass_candidates.py \
  --log "$log_dir/cublaslt_decode.log" \
  --log "$log_dir/cublaslt_prefill.log" \
  --output "$generated_tmp" \
  --report "$log_dir/cublaslt_cutlass_mapping.csv"

# Install the generated source only after generation succeeds. If its compile
# fails, retain it for diagnosis and rebuild the baseline executable so the
# normal build/run path remains usable.
mv "$generated_tmp" "$generated_file"
if ! ./build_gemm.sh --arch "$arch"; then
  echo "Generated CUTLASS templates failed to compile; restoring baseline build." >&2
  mv "$generated_file" "$failed_generated"
  ./build_gemm.sh --arch "$arch"
  echo "Failed generated source: $failed_generated" >&2
  exit 1
fi

echo "Tuning complete. Mapping: $log_dir/cublaslt_cutlass_mapping.csv"
