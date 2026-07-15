#!/usr/bin/env bash

set -euo pipefail

arch="sm_89"
nvcc_command="${NVCC:-nvcc}"

usage() {
  cat <<'EOF'
Build examples/gemm/gemm.cu from the CUTLASS repository root.

Usage:
  ./build_gemm.sh [options]

Options:
  --arch ARCH       CUDA architecture, for example sm_80 or sm_89 (default: sm_89)
  --nvcc PATH       nvcc executable (default: $NVCC or nvcc)
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

"$nvcc_command" \
  -std=c++17 \
  -arch="$arch" \
  -I"$cutlass_root/include" \
  -I"$cutlass_root/tools/util/include" \
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
