#!/usr/bin/env bash

set -uo pipefail

original_args=("$@")
script_path="${BASH_SOURCE[0]}"
bash_command="${BASH:-bash}"
model="7b"
stage=""
backend="cutlass"
iterations=20
alpha="1.0"
beta="0.0"
batch=1
lengths_csv=""
executable=""
dry_run=0

override_h=""
override_heads=""
override_kv_heads=""
override_head_dim=""
override_intermediate=""
override_vocab=""

usage() {
  cat <<'EOF'
Run deduplicated Qwen2.5 Decode or Prefill GEMM cases.

Usage:
  ./run_gemm.sh [options]

Options:
  --model MODEL            0.5b, 1.5b, 3b, 7b, 14b, 32b, or 72b (default: 7b)
  --stage STAGE            all, decode, or prefill (default: all)
  --backend BACKEND        cutlass or cublaslt (default: cutlass)
  --iterations N           Timed iterations per kernel (default: 20)
  --alpha VALUE            GEMM alpha (default: 1.0)
  --beta VALUE             GEMM beta (default: 0.0)
  --batch N                Batch size (default: 1)
  --lengths CSV            Decode L or Prefill S values
  --executable PATH        GEMM executable path
  --h N                    Override hidden size
  --heads N                Override attention head count
  --kv-heads N             Override KV head count
  --head-dim N             Override attention head dimension
  --intermediate N         Override MLP intermediate size
  --vocab N                Override vocabulary size
  --dry-run                Print commands without executing them
  --help                   Show this help
EOF
}

require_value() {
  if (($# < 2)); then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while (($# > 0)); do
  case "$1" in
    --model)        require_value "$@"; model="$2"; shift 2 ;;
    --stage)        require_value "$@"; stage="$2"; shift 2 ;;
    --backend)      require_value "$@"; backend="$2"; shift 2 ;;
    --iterations)   require_value "$@"; iterations="$2"; shift 2 ;;
    --alpha)        require_value "$@"; alpha="$2"; shift 2 ;;
    --beta)         require_value "$@"; beta="$2"; shift 2 ;;
    --batch)        require_value "$@"; batch="$2"; shift 2 ;;
    --lengths)      require_value "$@"; lengths_csv="$2"; shift 2 ;;
    --executable)   require_value "$@"; executable="$2"; shift 2 ;;
    --h)            require_value "$@"; override_h="$2"; shift 2 ;;
    --heads)        require_value "$@"; override_heads="$2"; shift 2 ;;
    --kv-heads)     require_value "$@"; override_kv_heads="$2"; shift 2 ;;
    --head-dim)     require_value "$@"; override_head_dim="$2"; shift 2 ;;
    --intermediate) require_value "$@"; override_intermediate="$2"; shift 2 ;;
    --vocab)        require_value "$@"; override_vocab="$2"; shift 2 ;;
    --dry-run)      dry_run=1; shift ;;
    --help|-h)      usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

stage="${stage,,}"
backend="${backend,,}"
[[ "$backend" == "cutlass" || "$backend" == "cublaslt" ]] || {
  echo "Unsupported backend: $backend (expected cutlass or cublaslt)" >&2; exit 2;
}

# With no stage (or explicit --stage all), invoke the same validated path for
# Decode and Prefill. The appended option wins over a preceding --stage all.
if [[ -z "$stage" || "$stage" == "all" ]]; then
  overall_status=0
  "$bash_command" "$script_path" "${original_args[@]}" --stage decode || overall_status=$?
  "$bash_command" "$script_path" "${original_args[@]}" --stage prefill || overall_status=$?
  exit "$overall_status"
fi

case "$stage" in
  decode)
    [[ -n "$lengths_csv" ]] || lengths_csv="128,256,512,1024,2048"
    ;;
  prefill)
    [[ -n "$lengths_csv" ]] || lengths_csv="128,256,512,1024,2048,129,130,132,136"
    ;;
  *) echo "Unsupported stage: $stage (expected all, decode, or prefill)" >&2; exit 2 ;;
esac

if ! [[ "$iterations" =~ ^[1-9][0-9]*$ && "$batch" =~ ^[1-9][0-9]*$ ]]; then
  echo "iterations and batch must be positive integers" >&2
  exit 2
fi

case "${model,,}" in
  0.5b) h=896;  heads=14; kv_heads=2; head_dim=64;  intermediate=4864;  vocab=151936 ;;
  1.5b) h=1536; heads=12; kv_heads=2; head_dim=128; intermediate=8960;  vocab=151936 ;;
  3b)   h=2048; heads=16; kv_heads=2; head_dim=128; intermediate=11008; vocab=151936 ;;
  7b)   h=3584; heads=28; kv_heads=4; head_dim=128; intermediate=18944; vocab=152064 ;;
  14b)  h=5120; heads=40; kv_heads=8; head_dim=128; intermediate=13824; vocab=152064 ;;
  32b)  h=5120; heads=40; kv_heads=8; head_dim=128; intermediate=27648; vocab=152064 ;;
  72b)  h=8192; heads=64; kv_heads=8; head_dim=128; intermediate=29568; vocab=152064 ;;
  *) echo "Unsupported model: $model" >&2; exit 2 ;;
esac

[[ -n "$override_h" ]] && h="$override_h"
[[ -n "$override_heads" ]] && heads="$override_heads"
[[ -n "$override_kv_heads" ]] && kv_heads="$override_kv_heads"
[[ -n "$override_head_dim" ]] && head_dim="$override_head_dim"
[[ -n "$override_intermediate" ]] && intermediate="$override_intermediate"
[[ -n "$override_vocab" ]] && vocab="$override_vocab"

if [[ -z "$executable" ]]; then
  if [[ "$backend" == "cublaslt" ]]; then
    executable="$(pwd)/examples/gemm/cublaslt_profiler"
  else
    executable="$(pwd)/examples/gemm/gemm"
  fi
fi

if ((dry_run == 0)) && [[ ! -x "$executable" ]]; then
  echo "Executable is missing or not executable: $executable" >&2
  exit 2
fi

IFS=',' read -r -a lengths <<< "$lengths_csv"
base_operations=(
  "Q"
  "K"
  "V"
  "Attention Out"
  "MLP Up"
  "MLP Gate"
  "MLP Down"
  "LM Head"
)
attention_operations=("Attention QK^T" "Attention PV")

for length in "${lengths[@]}"; do
  if ! [[ "$length" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid sequence length: $length" >&2
    exit 2
  fi
done

shape_for_decode_op() {
  local operation="$1"
  local context_length="$2"
  local token_m="$batch"
  local attention_m=$((batch * heads))

  case "$operation" in
    "Q")              m=$token_m;     n=$h;                     k=$h ;;
    "K"|"V")         m=$token_m;     n=$((kv_heads * head_dim)); k=$h ;;
    "Attention QK^T") m=$attention_m; n=$context_length;        k=$head_dim ;;
    "Attention PV")   m=$attention_m; n=$head_dim;              k=$context_length ;;
    "Attention Out")  m=$token_m;     n=$h;                     k=$h ;;
    "MLP Up"|"MLP Gate") m=$token_m; n=$intermediate; k=$h ;;
    "MLP Down")       m=$token_m;     n=$h;                     k=$intermediate ;;
    "LM Head")        m=$token_m;     n=$vocab;                 k=$h ;;
    *) echo "Unsupported operation: $operation" >&2; return 2 ;;
  esac
}

shape_for_prefill_op() {
  local operation="$1"
  local sequence_length="$2"
  local token_m=$((batch * sequence_length))
  local attention_m=$((batch * heads * sequence_length))

  case "$operation" in
    "Q")              m=$token_m;     n=$h;                      k=$h ;;
    "K"|"V")         m=$token_m;     n=$((kv_heads * head_dim)); k=$h ;;
    "Attention QK^T") m=$attention_m; n=$sequence_length;        k=$head_dim ;;
    "Attention PV")   m=$attention_m; n=$head_dim;               k=$sequence_length ;;
    "Attention Out")  m=$token_m;     n=$h;                      k=$h ;;
    "MLP Up"|"MLP Gate") m=$token_m; n=$intermediate; k=$h ;;
    "MLP Down")       m=$token_m;     n=$h;                      k=$intermediate ;;
    "LM Head")        m=$token_m;     n=$vocab;                  k=$h ;;
    *) echo "Unsupported operation: $operation" >&2; return 2 ;;
  esac
}

declare -A shape_to_slot=()
declare -a case_m=() case_n=() case_k=() case_operations=() case_lengths=()

add_unique_case() {
  local operation="$1"
  local context_length="$2"
  local key="${m}x${n}x${k}"
  local slot

  if [[ -n "${shape_to_slot[$key]+present}" ]]; then
    slot="${shape_to_slot[$key]}"
    if [[ " / ${case_operations[$slot]} / " != *" / $operation / "* ]]; then
      case_operations[$slot]+=" / $operation"
    fi
    if [[ ",${case_lengths[$slot]}," != *",$context_length,"* ]]; then
      case_lengths[$slot]+=",$context_length"
    fi
    return
  fi

  slot=${#case_m[@]}
  shape_to_slot[$key]=$slot
  case_m[$slot]=$m
  case_n[$slot]=$n
  case_k[$slot]=$k
  case_operations[$slot]="$operation"
  case_lengths[$slot]="$context_length"
}

if [[ "$stage" == "decode" ]]; then
  # L does not affect these GEMMs. Generate them once, then merge equal M/N/K
  # shapes such as Q/Attention Out and K/V.
  for operation in "${base_operations[@]}"; do
    shape_for_decode_op "$operation" 1
    add_unique_case "$operation" "-"
  done

  # Only attention GEMMs depend on decode context length L.
  for context_length in "${lengths[@]}"; do
    for operation in "${attention_operations[@]}"; do
      shape_for_decode_op "$operation" "$context_length"
      add_unique_case "$operation" "$context_length"
    done
  done
else
  # In Prefill, S participates in token M and attention M/N/K.
  for sequence_length in "${lengths[@]}"; do
    for operation in "${base_operations[@]}" "${attention_operations[@]}"; do
      shape_for_prefill_op "$operation" "$sequence_length"
      add_unique_case "$operation" "$sequence_length"
    done
  done
fi

total_cases=${#case_m[@]}
failed_cases=0
case_index=0

echo "Model: Qwen2.5-$model"
echo "Stage: ${stage^}"
echo "Backend: $backend"
echo "Parameters: H=$h, heads=$heads, kv_heads=$kv_heads, head_dim=$head_dim, intermediate=$intermediate, vocab=$vocab, batch=$batch"
echo "Sequence lengths: ${lengths[*]}"
echo "Unique GEMM cases (deduplicated by M/N/K): $total_cases"

for slot in "${!case_m[@]}"; do
    ((case_index += 1))
    m=${case_m[$slot]}
    n=${case_n[$slot]}
    k=${case_k[$slot]}
    operations_label=${case_operations[$slot]}
    lengths_label=${case_lengths[$slot]}
    command=("$executable" "--m=$m" "--n=$n" "--k=$k" "--iterations=$iterations")
    if [[ "$backend" == "cutlass" ]]; then
      command+=("--alpha=$alpha" "--beta=$beta")
    fi

    printf '\n[%d/%d] stage=%s length=%s ops=%s MxNxK=%dx%dx%d\n' \
      "$case_index" "$total_cases" "${stage^}" "$lengths_label" "$operations_label" "$m" "$n" "$k"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if ((dry_run == 1)); then
      continue
    fi

    "${command[@]}"
    exit_code=$?
    if ((exit_code != 0)); then
      ((failed_cases += 1))
      echo "FAILED: exit_code=$exit_code stage=${stage^} length=$lengths_label ops=$operations_label MxNxK=${m}x${n}x${k}" >&2
    fi
done

printf '\n%s GEMM summary\n' "${stage^}"
echo "  total: $total_cases"
if ((dry_run == 1)); then
  echo "  dry-run: commands generated but not executed"
  exit 0
fi

echo "  passed: $((total_cases - failed_cases))"
echo "  failed: $failed_cases"
((failed_cases == 0))
