#!/usr/bin/env bash
set -euo pipefail

task_root=/home/fj/vllm_ops
source_root=/home/fj/vllm_ops/source/pytorch
build_root=$source_root/build
runtime_root=/home/fj/vllm_ops/build/runtime
cmake_bin=/home/fj/vllm_ops/build/buildtools/bin/cmake
evidence_root=$task_root/evidence/build
jobs=${QWEN3_BUILD_JOBS:-96}

export PYTHONPATH=/home/fj/vllm_ops/build/buildtools${PYTHONPATH:+:$PYTHONPATH}
export CPATH=/home/fj/vllm_ops/source/vllm/csrc${CPATH:+:$CPATH}

mkdir -p "$evidence_root"

expected_commit=70d99e998b4955e0049d13a98d77ae1b14db1f45
actual_commit=$(git -C "$source_root" rev-parse HEAD)
test "$actual_commit" = "$expected_commit"
git -C "$source_root" diff --check

test "$(sha256sum /home/fj/vllm_ops/source/vllm/csrc/read_core_cycle_runtime.h | cut -d' ' -f1)" = \
  "$(sha256sum "$task_root/tools/readcorecycle/read_core_cycle_runtime.h" | cut -d' ' -f1)"
test "$(sha256sum "$runtime_root/lib/libvllm_read_core_cycle.so" | cut -d' ' -f1)" = \
  "$(sha256sum "$task_root/build/runtime/lib/libvllm_read_core_cycle.so" | cut -d' ' -f1)"

cat > "$evidence_root/torch_forced_rebuild_paths.txt" <<'EOF'
aten/src/ATen/core/List_inl.h
aten/src/ATen/core/dispatch/Dispatcher.h
aten/src/ATen/EmptyTensor.cpp
aten/src/ATen/cuda/CUDABlas.cpp
aten/src/ATen/cuda/CUDAGraph.cpp
aten/src/ATen/cuda/EmptyTensor.cpp
aten/src/ATen/native/Copy.cpp
aten/src/ATen/native/cuda/Blas.cpp
aten/src/ATen/native/cuda/Copy.cu
aten/src/ATen/native/cuda/CUDALoops.cuh
aten/src/ATen/native/cuda/FillKernel.cu
aten/src/ATen/native/cuda/IndexKernelUtils.cu
aten/src/ATen/native/cuda/Indexing.cu
aten/src/ATen/native/cuda/Loops.cuh
aten/src/ATen/native/cuda/TensorFactories.cu
c10/cuda/CUDACachingAllocator.cpp
torch/_inductor/runtime/static_triton_launcher.py
torch/csrc/cuda/Graph.cpp
torch/csrc/inductor/static_launcher/cuda.cpp
EOF

rebuild_marker=$evidence_root/torch_forced_rebuild_applied
if [ ! -f "$rebuild_marker" ]; then
  while IFS= read -r relative_path; do
    test -f "$source_root/$relative_path"
    touch "$source_root/$relative_path"
  done < "$evidence_root/torch_forced_rebuild_paths.txt"
  date -Is > "$rebuild_marker"
fi

"$cmake_bin" -S "$source_root" -B "$build_root" -G Ninja \
  -DCMAKE_CXX_FLAGS="-DVLLM_RCC_PROFILE=1 -I/home/fj/vllm_ops/source/vllm/csrc" \
  -DCMAKE_CUDA_FLAGS="-DVLLM_RCC_PROFILE=1 -I/home/fj/vllm_ops/source/vllm/csrc" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -L$runtime_root/lib -Wl,-rpath,$runtime_root/lib -lvllm_read_core_cycle -Wl,--as-needed" \
  -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--no-as-needed -L$runtime_root/lib -Wl,-rpath,$runtime_root/lib -lvllm_read_core_cycle -Wl,--as-needed" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -L$runtime_root/lib -Wl,-rpath,$runtime_root/lib -lvllm_read_core_cycle -Wl,--as-needed"

if grep -Fq 'read_core_cycle_runtime.cpp' "$build_root/compile_commands.json"; then
  echo "duplicate ReadCoreCycle runtime source found in PyTorch build graph" >&2
  exit 1
fi

"$cmake_bin" --build "$build_root" --target install -j "$jobs"

test -s "$source_root/torch/lib/libtorch_cuda.so"
test -s "$source_root/torch/lib/libtorch_python.so"
torch_c_extension=$(find "$source_root/build" -path '*/torch/_C*.so' \
  -type f -print -quit)
test -n "$torch_c_extension"
install -m 0755 "$torch_c_extension" \
  "$source_root/torch/$(basename "$torch_c_extension")"

PYTHONPATH="$source_root" \
LD_LIBRARY_PATH="$source_root/torch/lib:$runtime_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD="$runtime_root/lib/libvllm_read_core_cycle.so" \
/opt/vllm/bin/python - <<'PY' > "$evidence_root/torch_import_smoke.txt"
from pathlib import Path

import torch

print(f"torch={Path(torch.__file__).resolve()}")
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cmake_prefix_path={torch.utils.cmake_prefix_path}")
PY

{
  echo "commit=$actual_commit"
  echo "jobs=$jobs"
  echo "cmake=$($cmake_bin --version | head -1)"
  echo "ninja=$(/opt/vllm/bin/ninja --version)"
  sha256sum \
    "$source_root/torch/lib/libtorch_cuda.so" \
    "$source_root/torch/lib/libtorch_python.so" \
    "$source_root/torch/$(basename "$torch_c_extension")" \
    "$runtime_root/lib/libvllm_read_core_cycle.so"
  readelf -d "$source_root/torch/lib/libtorch_cuda.so" | grep -F libvllm_read_core_cycle
  readelf -d "$source_root/torch/lib/libtorch_python.so" | grep -F libvllm_read_core_cycle
} > "$evidence_root/torch_build.txt"
