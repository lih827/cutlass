#!/usr/bin/env bash
set -euo pipefail

bash_command="${BASH:-bash}"
python_command="${PYTHON:-python3}"
models="all"
arch="sm_89"
rounds=3
iterations=100
batch=1
accumulator="fp32"
log_dir=""
skip_validation=0
ncu_exact_csv=""
capture_ncu=0
ncu_command="${NCU:-ncu}"

usage() {
  cat <<'EOF'
Tune one target GPU for all standard Qwen2.5 model sizes.

Usage: bash tune_optimal_cutlass.sh [options]
  --models LIST       all or comma-separated sizes (default: all)
  --arch ARCH         CUDA architecture passed to build_gemm.sh (default: sm_89)
  --rounds N          CUTLASS measurement rounds (default: 3)
  --iterations N      Timed iterations per candidate (default: 100)
  --batch N           Decode/Prefill batch size (default: 1)
  --accumulator TYPE  fp16 or fp32 tuning and optimal mapping (default: fp32)
  --log-dir DIR       Artifacts directory (default: outputs/qwen_gemm/optimal_tuning/TYPE)
  --skip-validation   Do not run the final optimal-only coverage pass
  --ncu-exact-csv FILE  Add NCU-exact configurations to the measured candidates
  --capture-ncu       Run NCU on the target machine and generate candidates
  --ncu PATH          NCU executable used by --capture-ncu (default: $NCU or ncu)
  --help              Show this help
EOF
}

while (($#)); do
  case "$1" in
    --models) models="$2"; shift 2 ;;
    --arch) arch="$2"; shift 2 ;;
    --rounds) rounds="$2"; shift 2 ;;
    --iterations) iterations="$2"; shift 2 ;;
    --batch) batch="$2"; shift 2 ;;
    --accumulator) accumulator="$2"; shift 2 ;;
    --log-dir) log_dir="$2"; shift 2 ;;
    --skip-validation) skip_validation=1; shift ;;
    --ncu-exact-csv) ncu_exact_csv="$2"; shift 2 ;;
    --capture-ncu) capture_ncu=1; shift ;;
    --ncu) ncu_command="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$accumulator" == "fp16" || "$accumulator" == "fp32" ]] || {
  echo "Unsupported accumulator: $accumulator (expected fp16 or fp32)" >&2
  exit 2
}
if ((capture_ncu == 1)) && [[ -n "$ncu_exact_csv" ]]; then
  echo "--capture-ncu and --ncu-exact-csv are mutually exclusive." >&2
  exit 2
fi
[[ -n "$log_dir" ]] || log_dir="outputs/qwen_gemm/optimal_tuning/$accumulator"

for value in "$rounds" "$iterations" "$batch"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "rounds, iterations, and batch must be positive integers" >&2; exit 2; }
done

mkdir -p "$log_dir"
manifest="$log_dir/qwen2_5_all_shapes.csv"
cublaslt_inc="examples/gemm/cublaslt_generated_candidates.inc"
ncu_exact_inc="examples/gemm/ncu_exact_configurations.inc"
optimal_inc="examples/gemm/optimal_configurations.inc"
cublaslt_tmp="$log_dir/cublaslt_generated_candidates.inc.new"
optimal_tmp="$log_dir/optimal_configurations.inc.new"
cublaslt_report="$log_dir/cublaslt_cutlass_mapping.csv"
optimal_report="$log_dir/optimal_configurations_report.csv"
metadata="$log_dir/optimal_configurations_metadata.json"
ncu_capture_dir="$log_dir/ncu_capture"
ncu_csv="$log_dir/ncu_exact_candidates.csv"
ncu_metadata="$log_dir/ncu_exact_metadata.json"
previous_cublaslt="$log_dir/cublaslt_generated_candidates.inc.previous"
previous_optimal="$log_dir/optimal_configurations.inc.previous"

restore_on_failure() {
  local status="$1"
  if ((status == 0)); then return; fi
  echo "Tuning failed; restoring previously active generated mappings." >&2
  if [[ -f "$previous_cublaslt" ]]; then cp "$previous_cublaslt" "$cublaslt_inc"; else rm -f "$cublaslt_inc"; fi
  if [[ -f "$previous_optimal" ]]; then cp "$previous_optimal" "$optimal_inc"; else rm -f "$optimal_inc"; fi
}
trap 'restore_on_failure $?' EXIT

"$python_command" tools/qwen_gemm/generate_qwen_shapes.py \
  --models "$models" --batch "$batch" --output "$manifest"
shape_count=$(($(wc -l < "$manifest") - 1))
((shape_count > 0)) || { echo "Shape manifest is empty" >&2; exit 1; }

if [[ -f "$cublaslt_inc" ]]; then cp "$cublaslt_inc" "$previous_cublaslt"; else rm -f "$previous_cublaslt"; fi
if [[ -f "$optimal_inc" ]]; then cp "$optimal_inc" "$previous_optimal"; else rm -f "$previous_optimal"; fi
rm -f "$cublaslt_inc" "$optimal_inc" "$cublaslt_tmp" "$optimal_tmp"

if [[ -n "$ncu_exact_csv" ]]; then
  [[ -f "$ncu_exact_csv" ]] || {
    echo "NCU-exact CSV not found: $ncu_exact_csv" >&2
    exit 2
  }
  "$python_command" tools/qwen_gemm/generate_ncu_exact_configurations.py \
    --input "$ncu_exact_csv" --output "$ncu_exact_inc"
  echo "NCU-exact candidates: generated from $ncu_exact_csv"
elif ((capture_ncu == 0)) && [[ -f "$ncu_exact_inc" ]]; then
  echo "NCU-exact candidates: using existing $ncu_exact_inc"
elif ((capture_ncu == 0)); then
  echo "NCU-exact candidates: none"
fi

echo "[1/6] Build baseline CUTLASS and cuBLASLt profiler"
"$bash_command" ./build_gemm.sh --arch "$arch" --accumulator "$accumulator" --concise-log --skip-verification

if ((capture_ncu == 1)); then
  echo "[NCU] Capture actual cuBLASLt kernels and generate NCU-exact candidates"
  "$bash_command" tools/qwen_gemm/capture_ncu_candidates.sh \
    --manifest "$manifest" --output-dir "$ncu_capture_dir" \
    --executable "$(pwd)/examples/gemm/cublaslt_profiler" \
    --iterations 1 --ncu "$ncu_command"
  "$python_command" tools/qwen_gemm/parse_ncu_candidates.py \
    --capture-index "$ncu_capture_dir/capture_index.csv" \
    --output "$ncu_csv" --metadata "$ncu_metadata" \
    --accumulator "$accumulator"
  "$python_command" tools/qwen_gemm/generate_ncu_exact_configurations.py \
    --input "$ncu_csv" --output "$ncu_exact_inc"
fi

run_manifest() {
  local backend="$1"
  local executable="$2"
  local log="$3"
  local include_alpha_beta="$4"
  local index=0
  : > "$log"
  while IFS=',' read -r m n k trans batch_count operand_a operand_b shape_models stages operations lengths; do
    [[ "$m" == "m" ]] && continue
    index=$((index + 1))
    {
      printf '\n[%d/%d] backend=%s models=%s stages=%s lengths=%s ops=%s A=%s B=%s trans=%s batchCount=%s MxNxK=%sx%sx%s\n' \
        "$index" "$shape_count" "$backend" "$shape_models" "$stages" "$lengths" "$operations" \
        "$operand_a" "$operand_b" "$trans" "$batch_count" "$m" "$n" "$k"
      command=("$executable" "--m=$m" "--n=$n" "--k=$k"
               "--iterations=$iterations" "--trans=$trans"
               "--batch-count=$batch_count")
      if [[ "$include_alpha_beta" == "yes" ]]; then
        command+=("--alpha=1.0" "--beta=0.0" "--operation=${operations// /_}"
                  "--operand-a=$operand_a" "--operand-b=$operand_b")
      fi
      "${command[@]}"
    } 2>&1 | tee -a "$log"
  done < "$manifest"
}

echo "[2/6] Profile actual cuBLASLt winners for $shape_count unique shapes"
cublaslt_log="$log_dir/cublaslt_all.log"
run_manifest "cublaslt" "$(pwd)/examples/gemm/cublaslt_profiler" "$cublaslt_log" "no"
"$python_command" tools/qwen_gemm/generate_cutlass_candidates.py \
  --log "$cublaslt_log" --output "$cublaslt_tmp" --report "$cublaslt_report" \
  --accumulator "$accumulator"
mv "$cublaslt_tmp" "$cublaslt_inc"

echo "[3/6] Build CUTLASS with baseline, cuBLASLt-derived, and available NCU-exact candidates"
"$bash_command" ./build_gemm.sh --arch "$arch" --accumulator "$accumulator" --concise-log --skip-verification

echo "[4/6] Measure every CUTLASS candidate for $rounds rounds"
cutlass_log_args=()
for ((round=1; round<=rounds; ++round)); do
  log="$log_dir/cutlass_round_${round}.log"
  run_manifest "cutlass-round-$round" "$(pwd)/examples/gemm/gemm" "$log" "yes"
  cutlass_log_args+=(--log "$log")
done

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
cuda_version=$("${NVCC:-nvcc}" --version 2>/dev/null | tail -n 1 || true)
[[ -n "$gpu_name" ]] || gpu_name="unknown"
[[ -n "$cuda_version" ]] || cuda_version="unknown"

echo "[5/6] Generate target-specific optimal mapping and rebuild"
"$python_command" tools/qwen_gemm/generate_optimal_configurations.py \
  "${cutlass_log_args[@]}" --manifest "$manifest" \
  --cublaslt-report "$cublaslt_report" --output "$optimal_tmp" \
  --report "$optimal_report" --metadata "$metadata" --models "$models" \
  --gpu "$gpu_name" --cuda "$cuda_version" --arch "$arch" \
  --accumulator "$accumulator" --existing-optimal "$previous_optimal"
mv "$optimal_tmp" "$optimal_inc"
"$bash_command" ./build_gemm.sh --arch "$arch" --accumulator "$accumulator" --optimal-only --concise-log --skip-verification

if ((skip_validation == 0)); then
  echo "[6/6] Validate complete optimal-only coverage"
  validation_log="$log_dir/optimal_validation.log"
  run_manifest "optimal-validation" "$(pwd)/examples/gemm/gemm" "$validation_log" "yes"
  if grep -q "Fallback-random\|no exact M/N/K mapping" "$validation_log"; then
    echo "Optimal validation used a fallback template" >&2
    exit 1
  fi
else
  echo "[6/6] Validation skipped by request"
fi

echo "Tuning complete"
echo "  Shapes: $shape_count"
echo "  Generated optimal mapping: $optimal_inc"
echo "  Report: $optimal_report"
echo "  Metadata: $metadata"
