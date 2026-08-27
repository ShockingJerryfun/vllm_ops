#!/usr/bin/env bash
set -euo pipefail

# One invocation measures one exact stage in one exact phase.  This script is
# intentionally conservative because server 84 is shared.
mode=${RCC_MODE:?RCC_MODE must be eager or graph}
phase=${RCC_PHASE:?RCC_PHASE is required}
stage_id=${RCC_STAGE_ID:?RCC_STAGE_ID is required}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
case "$mode" in
  eager) root=/home/fj/vllm_ops_eager ;;
  graph) root=/home/fj/vllm_ops ;;
  *) echo "unsupported RCC_MODE: $mode" >&2; exit 40 ;;
esac
rt=$root/.runtime_build/runtime
run_id=${RUN_ID:-qwen3_${mode}_${phase}_stage${stage_id}_$(date +%Y%m%dT%H%M%S%z)}
run_dir=$root/.runtime_build/runs/$run_id
container=${CONTAINER_NAME:-qwen3_container_fj}
module=/home/fj/project_archive/retired_20260825_vllm_ops_migration/qwen3_operator_dispatch_readcorecycle_v1/runs/r3_overlay_pilot_20260824T173046+0800/module/pmccntr_el0_enable.ko
cpu=249
cpu_frequency_hz=$(($(cat /sys/devices/system/cpu/$cpu/cpufreq/cpuinfo_max_freq) * 1000))

if [ "$(git -C "$root" rev-parse HEAD)" != "$expected_commit" ]; then
  echo "unexpected commit in $root" >&2
  exit 41
fi
dirty=$(git -C "$root" status --porcelain | grep -v '^?? \.runtime_build/' || true)
if [ -n "$dirty" ]; then
  echo "repository has tracked modifications" >&2
  printf '%s\n' "$dirty" >&2
  exit 41
fi
if [ ! -x "$rt/lib/libvllm_read_core_cycle.so" ]; then
  echo "missing ReadCoreCycle runtime" >&2
  exit 42
fi
if [ -e "$run_dir" ]; then
  echo "run directory already exists: $run_dir" >&2
  exit 42
fi
if [ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" != true ]; then
  echo "measurement container is not running" >&2
  exit 43
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU occupied; refusing to measure" >&2
  exit 44
fi
if [ -d /sys/module/pmccntr_el0_enable ]; then
  echo "PMU module already loaded; refusing to reuse unknown state" >&2
  exit 45
fi

mkdir -p "$run_dir/triton_cache"
cleanup() {
  if [ -d /sys/module/pmccntr_el0_enable ]; then
    rmmod pmccntr_el0_enable || true
  fi
}
trap cleanup EXIT INT TERM
insmod "$module" cpu=$cpu enable_counter=1 exclude_kernel=0

{
  echo "mode=$mode"
  echo "phase=$phase"
  echo "stage_id=$stage_id"
  echo "cpu=$cpu"
  echo "governor=$(cat /sys/devices/system/cpu/$cpu/cpufreq/scaling_governor)"
  echo "cur_freq=$(cat /sys/devices/system/cpu/$cpu/cpufreq/scaling_cur_freq)"
  echo "max_freq=$(cat /sys/devices/system/cpu/$cpu/cpufreq/cpuinfo_max_freq)"
  echo "commit=$(git -C "$root" rev-parse HEAD)"
  sha256sum "$rt/lib/libvllm_read_core_cycle.so"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,utilization.gpu --format=csv,noheader
} > "$run_dir/environment_before.txt"

docker exec \
  -e PYTHONPATH="$rt" \
  -e LD_LIBRARY_PATH="$rt/torch/lib:$rt/lib:/opt/vllm/lib/python3.13/site-packages/nvidia/nccl/lib:/usr/local/cuda-13.3/targets/sbsa-linux/lib" \
  -e LD_PRELOAD="$rt/lib/libvllm_read_core_cycle.so" \
  -e VLLM_RCC_RUNTIME="$rt/lib/libvllm_read_core_cycle.so" \
  -e VLLM_RCC_INCLUDE_DIR="$root/csrc" \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e USE_LIBUV=0 \
  -e VLLM_CACHE_ROOT="$root/.runtime_build/caches/exact_stage_v2" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TRITON_CACHE_DIR="$run_dir/triton_cache" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e RCC_REPO="$root" \
  -e RCC_RUN_DIR="$run_dir" \
  -e RCC_PHASE="$phase" \
  -e RCC_STAGE_ID="$stage_id" \
  -e RCC_CONTEXT_ID="${RCC_CONTEXT_ID:-700101}" \
  -e RCC_TARGET_STEP="${RCC_TARGET_STEP:-0}" \
  -e RCC_EXPECTED_COUNT="${RCC_EXPECTED_COUNT:-}" \
  -e RCC_CPU_FREQUENCY_HZ="$cpu_frequency_hz" \
  -e RCC_INPUT_TOKENS="${INPUT_TOKENS:-128}" \
  -e RCC_MAX_TOKENS="${MAX_TOKENS:-4}" \
  -e RCC_KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1258291200}" \
  "$container" bash -lc \
  "taskset -c $cpu python -u $root/tools/run_qwen3_rcc_measurement.py" \
  2>&1 | tee "$run_dir/model_run.log"

rmmod pmccntr_el0_enable
trap - EXIT INT TERM
if [ -d /sys/module/pmccntr_el0_enable ]; then
  echo "PMU module remained loaded" >&2
  exit 46
fi
{
  echo "cur_freq=$(cat /sys/devices/system/cpu/$cpu/cpufreq/scaling_cur_freq)"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
} > "$run_dir/environment_after.txt"
echo "RUN_DIR=$run_dir"
