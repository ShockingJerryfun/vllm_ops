#!/usr/bin/env bash
set -euo pipefail

task_root=/home/fj/vllm_ops
vllm_root=/home/fj/vllm_ops/source/vllm
torch_root=/home/fj/vllm_ops/source/pytorch
flash_root=/home/fj/vllm_ops/source/flash_attention
build_root=/home/fj/vllm_ops/build/vllm_build
runtime_root=/home/fj/vllm_ops/build/runtime
cmake_bin=/home/fj/vllm_ops/build/buildtools/bin/cmake
evidence_root=$task_root/evidence/build
jobs=${QWEN3_BUILD_JOBS:-96}

mkdir -p "$evidence_root"

git -C "$vllm_root" merge-base --is-ancestor \
  568afb3a13806beb53bb2e6bd518269357b237c0 HEAD
test "$(git -C "$flash_root" rev-parse HEAD)" = \
  caaa4eb59845388a20b1f435ecaafb4bd9517ad8
git -C "$vllm_root" diff --check
git -C "$flash_root" diff --check

cat > "$evidence_root/vllm_forced_rebuild_paths.txt" <<'EOF'
CMakeLists.txt
csrc/libtorch_stable/activation_kernels.cu
csrc/libtorch_stable/cache_kernels.cu
csrc/libtorch_stable/layernorm_kernels.cu
csrc/libtorch_stable/pos_encoding_kernels.cu
vllm/compilation/cuda_graph.py
EOF

cat > "$evidence_root/flash_forced_rebuild_paths.txt" <<'EOF'
csrc/flash_attn/flash_api.cpp
csrc/flash_attn/src/flash_fwd_launch_template.h
EOF

rebuild_marker=$evidence_root/vllm_forced_rebuild_applied
if [ ! -f "$rebuild_marker" ]; then
  while IFS= read -r relative_path; do
    test -f "$vllm_root/$relative_path"
    touch "$vllm_root/$relative_path"
  done < "$evidence_root/vllm_forced_rebuild_paths.txt"

  while IFS= read -r relative_path; do
    test -f "$flash_root/$relative_path"
    touch "$flash_root/$relative_path"
  done < "$evidence_root/flash_forced_rebuild_paths.txt"
  date -Is > "$rebuild_marker"
fi

export PYTHONPATH=/home/fj/vllm_ops/build/buildtools:$torch_root
export LD_LIBRARY_PATH=$torch_root/torch/lib:$runtime_root/lib:/opt/vllm/lib/python3.13/site-packages/nvidia/nccl/lib:/usr/local/cuda-13.3/targets/sbsa-linux/lib
export TORCH_INSTALL_PREFIX=$torch_root/torch

generation_marker=$evidence_root/vllm_generated_sources_applied
generation_cache_reset=()
if [ ! -f "$generation_marker" ]; then
  generation_cache_reset=(
    -U MARLIN_GEN_SCRIPT_HASH_AND_ARCH
    -U MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH
  )
fi

"$cmake_bin" -S "$vllm_root" -B "$build_root" -G Ninja \
  "${generation_cache_reset[@]}" \
  -U TORCH_LIBRARY \
  -U c10_LIBRARY \
  -U C10_CUDA_LIBRARY \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CXX_FLAGS="-DVLLM_RCC_PROFILE=1 -I/home/fj/vllm_ops/source/vllm/csrc" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -L$runtime_root/lib -Wl,-rpath,$runtime_root/lib -lvllm_read_core_cycle -Wl,--as-needed" \
  -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--no-as-needed -L$runtime_root/lib -Wl,-rpath,$runtime_root/lib -lvllm_read_core_cycle -Wl,--as-needed" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
  -DTorch_DIR="$torch_root/torch/share/cmake/Torch" \
  -DVLLM_PYTHON_EXECUTABLE=/opt/vllm/bin/python

test -s "$vllm_root/csrc/libtorch_stable/quantization/marlin/kernel_selector.h"
test -s "$vllm_root/csrc/libtorch_stable/moe/marlin_moe_wna16/kernel_selector.h"
date -Is > "$generation_marker"

if grep -Fq 'read_core_cycle_runtime.cpp' "$build_root/compile_commands.json"; then
  echo "duplicate ReadCoreCycle runtime source found in vLLM build graph" >&2
  exit 1
fi

"$cmake_bin" --build "$build_root" \
  --target _C_stable_libtorch _vllm_fa2_C -j "$jobs"

stable_binary=$build_root/_C_stable_libtorch.abi3.so
flash_binary=$build_root/vllm-flash-attn/_vllm_fa2_C.abi3.so
test -s "$stable_binary"
test -s "$flash_binary"

{
  echo "vllm_commit=$(git -C "$vllm_root" rev-parse HEAD)"
  echo "flash_commit=$(git -C "$flash_root" rev-parse HEAD)"
  echo "jobs=$jobs"
  sha256sum \
    "$stable_binary" \
    "$flash_binary" \
    "$vllm_root/csrc/libtorch_stable/quantization/marlin/kernel_selector.h" \
    "$vllm_root/csrc/libtorch_stable/moe/marlin_moe_wna16/kernel_selector.h" \
    "$runtime_root/lib/libvllm_read_core_cycle.so"
  readelf -d "$stable_binary" | grep -F libvllm_read_core_cycle
  readelf -d "$flash_binary" | grep -F libvllm_read_core_cycle
} > "$evidence_root/vllm_build.txt"
