#!/usr/bin/env bash
set -euo pipefail

# Measure one exact Graph stage in one exact execution phase. ReadCoreCycle is
# the only stage timer; all environment and provenance collection stays outside
# the armed interval.
phase=${RCC_PHASE:?RCC_PHASE is required}
stage_id=${RCC_STAGE_ID:?RCC_STAGE_ID is required}
case "$phase" in
  graph-capture-full|graph-prefill|graph-replay) ;;
  *) echo "unsupported Graph phase: $phase" >&2; exit 40 ;;
esac

task_root=/home/fj/vllm_ops
source_root=$task_root/source/vllm
torch_source=$task_root/source/pytorch
flash_source=$task_root/source/flash_attention
runtime_root=$task_root/build/runtime
measurement_python=$task_root/tools/run_stage.py
collector_python=$task_root/tools/collect_readcorecycle.py
source_manifest=$task_root/evidence/source_manifest.sha256
torch_cuda_binary=$runtime_root/torch/lib/libtorch_cuda.so
torch_python_binary=$runtime_root/torch/lib/libtorch_python.so
custom_binary=$runtime_root/vllm/_C_stable_libtorch.abi3.so
flash_binary=$runtime_root/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
static_launcher=$runtime_root/torch/_inductor/runtime/static_triton_launcher.py
cache_tag=${RCC_CACHE_TAG:-graph_stage}
cache_root=$task_root/build/caches/$cache_tag
triton_cache=$cache_root/triton
run_id=${RUN_ID:-qwen3_${phase}_stage${stage_id}_$(date +%Y%m%dT%H%M%S%z)}
run_dir=$task_root/runs/$run_id
container=${CONTAINER_NAME:-qwen3_container_fj}
module=/home/fj/project_archive/retired_20260825_vllm_ops_migration/qwen3_operator_dispatch_readcorecycle_v1/runs/r3_overlay_pilot_20260824T173046+0800/module/pmccntr_el0_enable.ko
cpu=249

read_cpu_attr() {
  local attribute=$1
  local path=/sys/devices/system/cpu/cpu$cpu/cpufreq/$attribute
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if [ -r "$path" ]; then
      cat "$path"
      return 0
    fi
    sleep 0.2
  done
  echo "CPU frequency attribute unavailable: $path" >&2
  return 1
}

cpu_max_khz=$(read_cpu_attr cpuinfo_max_freq)
cpu_frequency_hz=${RCC_CPU_FREQUENCY_HZ:-$((cpu_max_khz * 1000))}
cpu_governor=$(read_cpu_attr scaling_governor)
if [ "$cpu_governor" != performance ]; then
  echo "CPU$cpu governor is not performance: $cpu_governor" >&2
  exit 40
fi

test "$(git -C "$source_root" rev-parse HEAD)" = \
  568afb3a13806beb53bb2e6bd518269357b237c0
test "$(git -C "$torch_source" rev-parse HEAD)" = \
  70d99e998b4955e0049d13a98d77ae1b14db1f45
test "$(git -C "$flash_source" rev-parse HEAD)" = \
  caaa4eb59845388a20b1f435ecaafb4bd9517ad8
git -C "$source_root" diff --check
git -C "$torch_source" diff --check
git -C "$flash_source" diff --check

for evidence_file in \
  "$runtime_root/lib/libvllm_read_core_cycle.so" \
  "$torch_cuda_binary" \
  "$torch_python_binary" \
  "$custom_binary" \
  "$flash_binary" \
  "$static_launcher" \
  "$measurement_python" \
  "$collector_python" \
  "$source_manifest" \
  "$module"; do
  if [ ! -f "$evidence_file" ]; then
    echo "missing measurement provenance file: $evidence_file" >&2
    exit 42
  fi
done
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

mkdir -p "$run_dir" "$triton_cache"
cleanup() {
  if [ -d /sys/module/pmccntr_el0_enable ]; then
    rmmod pmccntr_el0_enable || true
  fi
}
trap cleanup EXIT INT TERM
insmod "$module" cpu=$cpu enable_counter=1 exclude_kernel=0

{
  echo "phase=$phase"
  echo "stage_id=$stage_id"
  echo "cpu=$cpu"
  echo "governor=$cpu_governor"
  echo "cur_freq=$(read_cpu_attr scaling_cur_freq)"
  echo "max_freq=$cpu_max_khz"
  echo "cpufreq_reference_hz=$cpu_frequency_hz"
  echo "authoritative_time=paired_CNTVCT_delta_over_CNTFRQ"
  echo "rcc_capacity=${RCC_CAPACITY}"
  echo "triton_cache=$triton_cache"
  echo "vllm_commit=$(git -C "$source_root" rev-parse HEAD)"
  echo "torch_commit=$(git -C "$torch_source" rev-parse HEAD)"
  echo "flash_commit=$(git -C "$flash_source" rev-parse HEAD)"
  sha256sum \
    "$runtime_root/lib/libvllm_read_core_cycle.so" \
    "$torch_cuda_binary" \
    "$torch_python_binary" \
    "$custom_binary" \
    "$flash_binary" \
    "$static_launcher" \
    "$measurement_python" \
    "$collector_python" \
    "$source_manifest"
  nvidia-smi \
    --query-gpu=index,name,pci.bus_id,memory.used,utilization.gpu \
    --format=csv,noheader
} > "$run_dir/environment_before.txt"

docker exec \
  -e PYTHONPATH="$runtime_root" \
  -e LD_LIBRARY_PATH="$runtime_root/torch/lib:$runtime_root/lib:/opt/vllm/lib/python3.13/site-packages/nvidia/nccl/lib:/usr/local/cuda-13.3/targets/sbsa-linux/lib" \
  -e LD_PRELOAD="$runtime_root/lib/libvllm_read_core_cycle.so" \
  -e VLLM_RCC_RUNTIME="$runtime_root/lib/libvllm_read_core_cycle.so" \
  -e VLLM_RCC_INCLUDE_DIR="$source_root/csrc" \
  -e VLLM_RCC_STAGE_ID="$stage_id" \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e USE_LIBUV=0 \
  -e VLLM_CACHE_ROOT="$cache_root" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TRITON_CACHE_DIR="$triton_cache" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}" \
  -e RCC_REPO="$source_root" \
  -e RCC_RUN_DIR="$run_dir" \
  -e RCC_PHASE="$phase" \
  -e RCC_STAGE_ID="$stage_id" \
  -e RCC_CONTEXT_ID="${RCC_CONTEXT_ID:-700101}" \
  -e RCC_TARGET_STEP="${RCC_TARGET_STEP:-0}" \
  -e RCC_WINDOW_SCOPE="${RCC_WINDOW_SCOPE:-single-step}" \
  -e RCC_EXPECTED_COUNT="${RCC_EXPECTED_COUNT:-}" \
  -e RCC_CAPACITY="${RCC_CAPACITY:?RCC_CAPACITY is required}" \
  -e RCC_INSTRUMENTATION_ACTIVE="${RCC_INSTRUMENTATION_ACTIVE:-1}" \
  -e RCC_CPU_FREQUENCY_HZ="$cpu_frequency_hz" \
  -e RCC_INPUT_TOKENS="${INPUT_TOKENS:-128}" \
  -e RCC_MAX_TOKENS="${MAX_TOKENS:-4}" \
  -e RCC_KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1258291200}" \
  -e RCC_EXPECTED_TORCH_CUDA_BINARY="$torch_cuda_binary" \
  -e RCC_EXPECTED_TORCH_PYTHON_BINARY="$torch_python_binary" \
  -e RCC_TORCH_CUDA_STAGING_BINARY="$torch_cuda_binary" \
  -e RCC_EXPECTED_CUSTOM_BINARY="$custom_binary" \
  -e RCC_EXPECTED_FLASH_ATTENTION_BINARY="$flash_binary" \
  -e RCC_PROBE_SOURCE_MANIFEST="$source_manifest" \
  -e RCC_EXPECTED_STATIC_TRITON_LAUNCHER="$static_launcher" \
  -e RCC_COLLECTOR_PATH="$collector_python" \
  -e RCC_TRITON_STATIC_BUILD_MAP="$cache_root/rcc_static_build_map.jsonl" \
  -e RCC_TRITON_STATIC_RUNTIME_MAP="$run_dir/rcc_static_runtime_map.jsonl" \
  "$container" bash -lc "
  set -euo pipefail
  audio_path=/opt/vllm/lib/python3.13/site-packages/torchaudio
  audio_backup=/tmp/torchaudio.rcc.\$\$
  restore_audio() {
    if [ -d \"\$audio_backup\" ]; then
      mv \"\$audio_backup\" \"\$audio_path\"
    fi
  }
  trap restore_audio EXIT INT TERM
  if [ -d \"\$audio_path\" ]; then
    mv \"\$audio_path\" \"\$audio_backup\"
  fi
  taskset -c $cpu python -u $measurement_python
  " 2>&1 | tee "$run_dir/model_run.log"

rmmod pmccntr_el0_enable
trap - EXIT INT TERM
if [ -d /sys/module/pmccntr_el0_enable ]; then
  echo "PMU module remained loaded" >&2
  exit 46
fi
{
  echo "cur_freq=$(read_cpu_attr scaling_cur_freq)"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
} > "$run_dir/environment_after.txt"
echo "RUN_DIR=$run_dir"
