#!/usr/bin/env bash

set -euo pipefail

arch="sm_89"
nvcc_command="${NVCC:-nvcc}"
concise_log=0
timer="cuda-event"
accumulator="fp32"
optimal_only=0
ncu_exact=0
skip_verification=0

usage() {
  cat <<'EOF'
Build examples/gemm/gemm.cu from the CUTLASS repository root.

Usage:
  ./build_gemm.sh [options]

Options:
  --arch ARCH       CUDA architecture, for example sm_80 or sm_89 (default: sm_89)
  --nvcc PATH       nvcc executable (default: $NVCC or nvcc)
  --concise-log     Compile gemm to print only the winning candidate details
  --timer TIMER     cuda-event or chrono (default: cuda-event)
  --accumulator TYPE  fp16 or fp32 accumulation (default: fp32)
  --optimal-only    Run one preselected template for mapped exact M/N/K shapes
  --ncu-exact       Use only exact NCU CTA M/N/K and stages mappings; no fallback
  --skip-verification  Skip reference GEMM, result copies, and comparison
  --help            Show this help
EOF
}

while (($# > 0)); do
  case "$1" in
    --arch)
      [[ $# -ge 2 ]] || { echo "Missing value for --arch" >&2; exit 2; }
      arch="$2"
      shift 2
      ;;
    --nvcc)
      [[ $# -ge 2 ]] || { echo "Missing value for --nvcc" >&2; exit 2; }
      nvcc_command="$2"
      shift 2
      ;;
    --concise-log)
      concise_log=1
      shift
      ;;
    --timer)
      [[ $# -ge 2 ]] || { echo "Missing value for --timer" >&2; exit 2; }
      timer="$2"
      shift 2
      ;;
    --accumulator)
      [[ $# -ge 2 ]] || { echo "Missing value for --accumulator" >&2; exit 2; }
      accumulator="$2"
      shift 2
      ;;
    --optimal-only)
      optimal_only=1
      shift
      ;;
    --ncu-exact)
      ncu_exact=1
      shift
      ;;
    --skip-verification)
      skip_verification=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$timer" == "cuda-event" || "$timer" == "chrono" ]] || {
  echo "Unsupported timer: $timer (expected cuda-event or chrono)" >&2
  exit 2
}

[[ "$accumulator" == "fp16" || "$accumulator" == "fp32" ]] || {
  echo "Unsupported accumulator: $accumulator (expected fp16 or fp32)" >&2
  exit 2
}

if ((optimal_only == 1 && ncu_exact == 1)); then
  echo "--optimal-only and --ncu-exact are mutually exclusive." >&2
  exit 2
fi

cutlass_root="$(pwd)"
source_file="$cutlass_root/examples/gemm/gemm.cu"
output_file="$cutlass_root/examples/gemm/gemm"
cublaslt_source="$cutlass_root/examples/gemm/cublaslt_profiler.cu"
cublaslt_output="$cutlass_root/examples/gemm/cublaslt_profiler"

if [[ ! -f "$cutlass_root/include/cutlass/cutlass.h" ]]; then
  echo "Run this script from the CUTLASS repository root." >&2
  echo "Missing: $cutlass_root/include/cutlass/cutlass.h" >&2
  exit 2
fi

if [[ ! -f "$source_file" ]]; then
  echo "GEMM source not found: $source_file" >&2
  exit 2
fi

echo "CUTLASS root: $cutlass_root"
echo "CUDA arch: $arch"
echo "Source: $source_file"
echo "Output: $output_file"

extra_nvcc_flags=()
if [[ "$accumulator" == "fp32" ]]; then
  extra_nvcc_flags+=("-DGEMM_ACCUMULATOR_TYPE=float")
else
  extra_nvcc_flags+=("-DGEMM_ACCUMULATOR_TYPE=cutlass::half_t")
fi
cublaslt_nvcc_flags=()
if [[ "$accumulator" == "fp32" ]]; then
  cublaslt_nvcc_flags+=("-DGEMM_CUBLASLT_FP32_ACCUMULATOR=1")
else
  cublaslt_nvcc_flags+=("-DGEMM_CUBLASLT_FP32_ACCUMULATOR=0")
fi
if [[ "$timer" == "chrono" ]]; then
  extra_nvcc_flags+=("-DGEMM_USE_CHRONO=1")
fi
if ((optimal_only == 1)); then
  extra_nvcc_flags+=("-DGEMM_OPTIMAL_ONLY=1")
  echo "Candidate mode: optimal-only for mapped shapes"
  echo "Optimal source: examples/gemm/optimal_configurations.inc"
else
  echo "Candidate mode: compare all applicable candidates"
fi
if ((ncu_exact == 1)); then
  ncu_exact_file="$cutlass_root/examples/gemm/ncu_exact_configurations.inc"
  if [[ ! -f "$ncu_exact_file" ]]; then
    echo "NCU-exact configuration not found: $ncu_exact_file" >&2
    echo "Generate it with tools/qwen_gemm/generate_ncu_exact_configurations.py." >&2
    exit 2
  fi
  extra_nvcc_flags+=("-DGEMM_NCU_EXACT_ONLY=1")
  echo "Candidate mode: NCU-exact only (strict match, no fallback)"
  echo "NCU exact source: examples/gemm/ncu_exact_configurations.inc"
fi
if ((skip_verification == 1)); then
  extra_nvcc_flags+=("-DGEMM_SKIP_VERIFICATION=1")
  echo "Verification: disabled"
else
  echo "Verification: enabled"
fi
echo "Timer: $timer"
echo "Accumulator: $accumulator (A/B/C/D remain fp16)"
if ((concise_log == 1)); then
  extra_nvcc_flags+=("-DGEMM_CONCISE_LOG=1")
  echo "Log mode: concise (winning configuration only)"
else
  echo "Log mode: full candidate details"
fi

"$nvcc_command" \
  -std=c++17 \
  -arch="$arch" \
  -I"$cutlass_root/include" \
  -I"$cutlass_root/tools/util/include" \
  "${extra_nvcc_flags[@]}" \
  "$source_file" \
  -o "$output_file"

chmod +x "$output_file"
echo "Build succeeded: $output_file"

if [[ -f "$cublaslt_source" ]]; then
  "$nvcc_command" -std=c++17 -arch="$arch" "$cublaslt_source" \
    "${cublaslt_nvcc_flags[@]}" \
    -lcublasLt -lcublas -o "$cublaslt_output"
  chmod +x "$cublaslt_output"
  echo "Build succeeded: $cublaslt_output"
fi
