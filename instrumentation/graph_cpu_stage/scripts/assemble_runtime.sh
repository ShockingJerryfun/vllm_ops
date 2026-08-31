#!/usr/bin/env bash
set -euo pipefail

task_root=/home/fj/vllm_ops
overlay_root=/home/fj/vllm_ops/source/vllm
torch_root=/home/fj/vllm_ops/source/pytorch
vllm_build=/home/fj/vllm_ops/build/vllm_build
runtime_root=$task_root/build/runtime
evidence_root=$task_root/evidence/runtime

mkdir -p "$runtime_root" "$runtime_root/lib" "$evidence_root"

test -s "$torch_root/torch/lib/libtorch_cuda.so"
test -s "$torch_root/torch/lib/libtorch_python.so"
test -s "$vllm_build/_C_stable_libtorch.abi3.so"
test -s "$vllm_build/vllm-flash-attn/_vllm_fa2_C.abi3.so"
test -s "$runtime_root/lib/libvllm_read_core_cycle.so"

for target in torch functorch torchgen vllm; do
  rm -rf "$runtime_root/$target"
done
cp -a "$torch_root/torch" "$runtime_root/torch"
cp -a "$torch_root/functorch" "$runtime_root/functorch"
cp -a "$torch_root/torchgen" "$runtime_root/torchgen"
cp -a "$overlay_root/vllm" "$runtime_root/vllm"

install -m 0755 \
  "$vllm_build/_C_stable_libtorch.abi3.so" \
  "$runtime_root/vllm/_C_stable_libtorch.abi3.so"
mkdir -p "$runtime_root/vllm/vllm_flash_attn"
install -m 0755 \
  "$vllm_build/vllm-flash-attn/_vllm_fa2_C.abi3.so" \
  "$runtime_root/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"

find "$runtime_root" -type d -name __pycache__ -prune -exec rm -rf {} +

export PYTHONPATH=$runtime_root
export LD_LIBRARY_PATH=$runtime_root/torch/lib:$runtime_root/lib:/opt/vllm/lib/python3.13/site-packages/nvidia/nccl/lib:/usr/local/cuda-13.3/targets/sbsa-linux/lib
export LD_PRELOAD=$runtime_root/lib/libvllm_read_core_cycle.so

/opt/vllm/bin/python - <<'PY' > "$evidence_root/import_smoke.txt"
import importlib
from pathlib import Path

import torch

modules = [
    "vllm._C_stable_libtorch",
    "vllm.vllm_flash_attn._vllm_fa2_C",
]
print(f"torch={Path(torch.__file__).resolve()}")
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
for name in modules:
    module = importlib.import_module(name)
    print(f"{name}={Path(module.__file__).resolve()}")
PY

for binary in \
  "$runtime_root/torch/lib/libtorch_cuda.so" \
  "$runtime_root/torch/lib/libtorch_python.so" \
  "$runtime_root/vllm/_C_stable_libtorch.abi3.so" \
  "$runtime_root/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"; do
  readelf -d "$binary" | grep -F libvllm_read_core_cycle
done > "$evidence_root/readcorecycle_linkage.txt"

sha256sum \
  "$runtime_root/lib/libvllm_read_core_cycle.so" \
  "$runtime_root/torch/lib/libtorch_cuda.so" \
  "$runtime_root/torch/lib/libtorch_python.so" \
  "$runtime_root/vllm/_C_stable_libtorch.abi3.so" \
  "$runtime_root/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so" \
  > "$evidence_root/runtime_binaries.sha256"

printf 'assembled_at=%s\n' "$(date -Is)" > "$evidence_root/assembly.txt"
du -sh "$runtime_root" >> "$evidence_root/assembly.txt"
