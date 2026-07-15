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

# First build creates both the baseline CUTLASS executable and cuBLASLt profiler.
./build_gemm.sh --arch "$arch"

./run_gemm.sh --backend cublaslt --model "$model" --stage decode \
  --iterations "$iterations" | tee "$log_dir/cublaslt_decode.log"
./run_gemm.sh --backend cublaslt --model "$model" --stage prefill \
  --iterations "$iterations" | tee "$log_dir/cublaslt_prefill.log"

python3 generate_cutlass_candidates.py \
  --log "$log_dir/cublaslt_decode.log" \
  --log "$log_dir/cublaslt_prefill.log" \
  --output examples/gemm/cublaslt_generated_candidates.inc \
  --report "$log_dir/cublaslt_cutlass_mapping.csv"

# Rebuild so only generated exact-shape configurations are instantiated and used.
./build_gemm.sh --arch "$arch"

echo "Tuning complete. Mapping: $log_dir/cublaslt_cutlass_mapping.csv"
