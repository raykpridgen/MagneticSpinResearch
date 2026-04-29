#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN="${REPO_ROOT}/build/operators_cpp_sweep"
DATA_DIR="${REPO_ROOT}/data"

mkdir -p "${DATA_DIR}"

DEFAULT_SWEEP_ARGS=(
  --bz-min -10
  --bz-max 10
  --bz-step 0.1
  --no-plot
  --no-show
)

SIZES_SPEC=""
NUCLEI_LIST="1,2,3,4"
UNIFORM_J_NUMERATOR=1
SWEEP_ARGS=()

print_usage() {
  cat <<'EOF'
Usage:
  benchmark_backends.sh [script options] [-- sweep options]

Script options:
  --sizes "<spec>"            Semicolon-separated j-numerator lists.
                              Example: "1;1,1;1,1,1;1,1,1,1"
  --nuclei-list "<list>"      Comma-separated nuclei counts for uniform spins.
                              Example: "1,2,3,4" (default)
  --j-numerator <int>         Repeated j numerator used with --nuclei-list (default 1)
  --help                      Show this message

Sweep options:
  Passed through to operators_cpp_sweep.
  If omitted, defaults are: --bz-min -10 --bz-max 10 --bz-step 0.1

Examples:
  benchmark_backends.sh
  benchmark_backends.sh --nuclei-list "1,2,3,4,5" -- --bz-min -5 --bz-max 5 --bz-step 0.05
  benchmark_backends.sh --sizes "1;1,3;3,3,3" -- --ks 4e6 --kd 1e6
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sizes)
      SIZES_SPEC="$2"
      shift 2
      ;;
    --nuclei-list)
      NUCLEI_LIST="$2"
      shift 2
      ;;
    --j-numerator)
      UNIFORM_J_NUMERATOR="$2"
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    --)
      shift
      SWEEP_ARGS=("$@")
      break
      ;;
    *)
      SWEEP_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#SWEEP_ARGS[@]} -eq 0 ]]; then
  SWEEP_ARGS=("${DEFAULT_SWEEP_ARGS[@]}")
fi

if [[ " ${SWEEP_ARGS[*]} " != *" --no-plot "* ]]; then
  SWEEP_ARGS+=(--no-plot)
fi
if [[ " ${SWEEP_ARGS[*]} " != *" --no-show "* ]]; then
  SWEEP_ARGS+=(--no-show)
fi

echo "Building operators_cpp binary..."
make -C "${REPO_ROOT}" operators-cpp

build_uniform_j_csv() {
  local nuclei_count="$1"
  local numerator="$2"
  local j_csv=""
  local idx
  for ((idx = 0; idx < nuclei_count; ++idx)); do
    if [[ ${idx} -gt 0 ]]; then
      j_csv+=","
    fi
    j_csv+="${numerator}"
  done
  printf '%s\n' "${j_csv}"
}

declare -a SIZE_NAMES=()
declare -a SIZE_JCSV=()
declare -a SIZE_HNUM=()

if [[ -n "${SIZES_SPEC}" ]]; then
  IFS=';' read -r -a raw_sizes <<< "${SIZES_SPEC}"
  for j_csv in "${raw_sizes[@]}"; do
    j_csv="${j_csv// /}"
    if [[ -z "${j_csv}" ]]; then
      continue
    fi
    IFS=',' read -r -a js <<< "${j_csv}"
    nuclei_count="${#js[@]}"
    SIZE_NAMES+=("n${nuclei_count}_${j_csv//,/x}")
    SIZE_JCSV+=("${j_csv}")
    SIZE_HNUM+=("${nuclei_count}")
  done
else
  IFS=',' read -r -a nuclei_counts <<< "${NUCLEI_LIST}"
  for nuclei_count in "${nuclei_counts[@]}"; do
    nuclei_count="${nuclei_count// /}"
    if [[ -z "${nuclei_count}" ]]; then
      continue
    fi
    j_csv="$(build_uniform_j_csv "${nuclei_count}" "${UNIFORM_J_NUMERATOR}")"
    SIZE_NAMES+=("n${nuclei_count}_j${UNIFORM_J_NUMERATOR}")
    SIZE_JCSV+=("${j_csv}")
    SIZE_HNUM+=("${nuclei_count}")
  done
fi

if [[ ${#SIZE_NAMES[@]} -eq 0 ]]; then
  echo "No valid size configurations were provided."
  exit 1
fi

run_case_ms() {
  local mode_flag="$1"
  local out_csv="$2"
  local j_csv="$3"
  local h_number="$4"
  local log_path="$5"
  local status=0

  local start_ns end_ns elapsed_ms
  start_ns="$(date +%s%N)"
  "${BIN}" "${SWEEP_ARGS[@]}" "${mode_flag}" --j-numerators "${j_csv}" --h-number "${h_number}" --out "${out_csv}" >"${log_path}" 2>&1 || status=$?
  end_ns="$(date +%s%N)"

  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
  printf '%s %s\n' "${status}" "${elapsed_ms}"
}

echo
echo "Sweep args: ${SWEEP_ARGS[*]}"

printf '%-18s %-8s %-8s %-10s\n' "problem_size" "cpu_ms" "gpu_ms" "speedup"

for idx in "${!SIZE_NAMES[@]}"; do
  size_name="${SIZE_NAMES[$idx]}"
  j_csv="${SIZE_JCSV[$idx]}"
  h_number="${SIZE_HNUM[$idx]}"
  cpu_csv="${DATA_DIR}/operators_cpp_sweep_cpu_${size_name}.csv"
  gpu_csv="${DATA_DIR}/operators_cpp_sweep_gpu_${size_name}.csv"
  cpu_log="${DATA_DIR}/operators_cpp_sweep_cpu_${size_name}.log"
  gpu_log="${DATA_DIR}/operators_cpp_sweep_gpu_${size_name}.log"

  read -r cpu_status cpu_ms <<< "$(run_case_ms "--cpu" "${cpu_csv}" "${j_csv}" "${h_number}" "${cpu_log}")"
  if [[ "${cpu_status}" -ne 0 ]]; then
    printf '%-18s %-8s %-8s %-10s\n' "${size_name}" "FAIL" "-" "-"
    echo "CPU run failed for ${size_name}."
    echo "See log: ${cpu_log}"
    exit 1
  fi

  read -r gpu_status gpu_ms <<< "$(run_case_ms "--gpu" "${gpu_csv}" "${j_csv}" "${h_number}" "${gpu_log}")"
  if [[ "${gpu_status}" -ne 0 ]]; then
    printf '%-18s %-8s %-8s %-10s\n' "${size_name}" "${cpu_ms}" "FAIL" "-"
    echo "GPU run failed for ${size_name} (likely CUDA unavailable)."
    echo "See log: ${gpu_log}"
    exit 1
  fi

  speedup="$(awk "BEGIN { if (${gpu_ms} > 0) printf \"%.2fx\", ${cpu_ms}/${gpu_ms}; else print \"inf\" }")"
  printf '%-18s %-8s %-8s %-10s\n' "${size_name}" "${cpu_ms}" "${gpu_ms}" "${speedup}"
done

echo
echo "Done. CSV outputs are in ${DATA_DIR}:"
echo "  operators_cpp_sweep_cpu_*"
echo "  operators_cpp_sweep_gpu_*"
  echo
  echo "Done. Outputs:"
  echo "  CPU: ${CPU_CSV}"
  echo "  GPU: ${GPU_CSV}"
else
  echo "GPU run failed (likely no CUDA device or CUDA backend unavailable)."
  exit 1
fi
