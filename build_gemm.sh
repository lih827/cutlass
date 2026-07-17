#!/usr/bin/env bash

set -euo pipefail

arch="sm_89"
nvcc_command="${NVCC:-nvcc}"
concise_log=0
timer="cuda-event"

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
if [[ "$timer" == "chrono" ]]; then
  extra_nvcc_flags+=("-DGEMM_USE_CHRONO=1")
fi
echo "Timer: $timer"
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
    -lcublasLt -lcublas -o "$cublaslt_output"
  chmod +x "$cublaslt_output"
  echo "Build succeeded: $cublaslt_output"
fi
