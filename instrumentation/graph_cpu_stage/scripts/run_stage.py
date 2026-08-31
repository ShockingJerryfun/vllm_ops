#!/usr/bin/env python3
"""Run one exact Qwen3 CPU-dispatch stage inside one model execution window.

This driver deliberately starts the recorder for exactly one stage and never
synchronizes CUDA.  It is meant to run inside the vLLM runtime container; the
shell wrapper owns PMU setup, CPU pinning, and shared-server preflight checks.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapped_realpaths() -> set[Path]:
    paths: set[Path] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1])
        try:
            paths.add(path.resolve(strict=True))
        except FileNotFoundError:
            continue
    return paths


repo = Path(_required("RCC_REPO"))
run_dir = Path(_required("RCC_RUN_DIR"))
phase = _required("RCC_PHASE")
stage_id = int(_required("RCC_STAGE_ID"))
context_id = int(os.environ.get("RCC_CONTEXT_ID", "700101"))
target_step = int(os.environ.get("RCC_TARGET_STEP", "0"))
expected_count_text = os.environ.get("RCC_EXPECTED_COUNT", "")
expected_count = int(expected_count_text) if expected_count_text else None
capacity = int(_required("RCC_CAPACITY"))
if capacity <= 0 or capacity > 4096:
    raise RuntimeError(f"RCC_CAPACITY must be in [1, 4096], got {capacity}")
if expected_count is not None and capacity < expected_count:
    raise RuntimeError(
        f"RCC_CAPACITY {capacity} is smaller than expected_count {expected_count}"
    )
window_scope = os.environ.get("RCC_WINDOW_SCOPE", "single-step")
if window_scope not in {"single-step", "full-request"}:
    raise RuntimeError(f"unsupported RCC_WINDOW_SCOPE: {window_scope}")
input_tokens = int(os.environ.get("RCC_INPUT_TOKENS", "128"))
max_tokens = int(os.environ.get("RCC_MAX_TOKENS", "4"))
kv_cache_memory_bytes = int(os.environ.get("RCC_KV_CACHE_MEMORY_BYTES", "1258291200"))
cpu_frequency_hz = int(_required("RCC_CPU_FREQUENCY_HZ"))
instrumentation_active = os.environ.get("RCC_INSTRUMENTATION_ACTIVE", "1") != "0"
expected_torch_cuda_binary = Path(
    _required("RCC_EXPECTED_TORCH_CUDA_BINARY")
).resolve(strict=True)
expected_torch_python_binary = Path(
    _required("RCC_EXPECTED_TORCH_PYTHON_BINARY")
).resolve(strict=True)
torch_cuda_staging_binary = Path(
    _required("RCC_TORCH_CUDA_STAGING_BINARY")
).resolve(strict=True)
expected_custom_binary = Path(
    _required("RCC_EXPECTED_CUSTOM_BINARY")
).resolve(strict=True)
expected_flash_attention_binary = Path(
    _required("RCC_EXPECTED_FLASH_ATTENTION_BINARY")
).resolve(strict=True)
probe_source_manifest = Path(
    _required("RCC_PROBE_SOURCE_MANIFEST")
).resolve(strict=True)
expected_static_triton_launcher = Path(
    _required("RCC_EXPECTED_STATIC_TRITON_LAUNCHER")
).resolve(strict=True)
is_capture = phase.startswith("graph-capture-")

if phase not in {
    "graph-capture-full",
    "graph-prefill",
    "graph-replay",
}:
    raise RuntimeError(f"unsupported RCC_PHASE: {phase}")

collector_path = Path(
    os.environ.get(
        "RCC_COLLECTOR_PATH",
        str(repo / ".runtime_build/finegrain_tools/collect_readcorecycle.py"),
    )
)
spec = importlib.util.spec_from_file_location("rcc_collector", collector_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load collector: {collector_path}")
collector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector_module)
controller = collector_module.RccWorkerExtension()

import transformers.utils as transformers_utils  # noqa: E402
import transformers.utils.import_utils as transformers_import_utils  # noqa: E402

for optional_probe in ("is_torchaudio_available", "is_torchvision_available"):
    setattr(transformers_import_utils, optional_probe, lambda: False)
    setattr(transformers_utils, optional_probe, lambda: False)

import torch  # noqa: E402

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.worker.gpu.model_runner import (  # noqa: E402
    GPUModelRunner as GPUModelRunnerV2,
)
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager  # noqa: E402
from vllm.v1.worker.gpu_model_runner import (  # noqa: E402
    GPUModelRunner as GPUModelRunnerV1,
)

custom_extension = __import__(
    "vllm._C_stable_libtorch", fromlist=["_C_stable_libtorch"]
)
custom_extension_path = Path(custom_extension.__file__).resolve(strict=True)
if custom_extension_path != expected_custom_binary:
    raise RuntimeError(
        "custom extension import path mismatch: "
        f"expected {expected_custom_binary}, got {custom_extension_path}"
    )
flash_attention_extension = __import__(
    "vllm.vllm_flash_attn._vllm_fa2_C", fromlist=["_vllm_fa2_C"]
)
flash_attention_extension_path = Path(
    flash_attention_extension.__file__
).resolve(strict=True)
if flash_attention_extension_path != expected_flash_attention_binary:
    raise RuntimeError(
        "FlashAttention extension import path mismatch: "
        f"expected {expected_flash_attention_binary}, "
        f"got {flash_attention_extension_path}"
    )

result: dict[str, Any] = {
    "site_id": 700,
    "stage_id": stage_id,
    "phase": phase,
    "target_step": target_step,
    "expected_count": expected_count,
    "capacity": capacity,
    "instrumentation_active": instrumentation_active,
    "window_scope": window_scope,
    "cpu": 249,
    "model": "/home/model/Qwen3-8B-Instruct",
    "input_tokens": input_tokens,
    "max_tokens": max_tokens,
    "torch_version": torch.__version__,
    "torch_path": torch.__file__,
    "vllm_path": vllm.__file__,
    "custom_extension_path": str(custom_extension_path),
    "flash_attention_extension_path": str(flash_attention_extension_path),
    "torch_cuda_runtime_staging_copy": {
        "path": str(torch_cuda_staging_binary),
        "sha256": _sha256(torch_cuda_staging_binary),
        "mapped_expected_path": str(expected_torch_cuda_binary),
        "mapped_expected_sha256": _sha256(expected_torch_cuda_binary),
    },
}
active = False
armed = False
execute_index = 0
sample_index = 0

if instrumentation_active:
    result["prepare"] = controller.rcc_prepare(
        700, stage_id, context_id, capacity
    )
else:
    result["prepare"] = {"phase": "skipped-inactive", "capacity": capacity}


def _start() -> None:
    global active
    if not instrumentation_active:
        return
    result["start"] = controller.rcc_arm()
    active = True


def _stop() -> None:
    global active
    result["stop"] = controller.rcc_stop_dump(str(run_dir / "raw.csv"))
    active = False


def _selected_execute_index(current_index: int) -> bool:
    return window_scope == "single-step" and armed and (
        (phase == "graph-prefill" and current_index == 0)
        or (phase == "graph-replay" and current_index == target_step + 1)
    )


def _selected_sample_index(current_index: int) -> bool:
    del current_index
    return False


def _patch_model_runner(model_runner_class: type[Any]) -> None:
    original_execute_model = model_runner_class.execute_model
    original_capture_model = model_runner_class.capture_model
    original_sample = getattr(model_runner_class, "sample", None)

    def measured_execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        global execute_index
        current_index = execute_index
        if armed:
            execute_index += 1
        selected = _selected_execute_index(current_index)
        if selected:
            _start()
        if active:
            if phase == "graph-replay":
                semantic_step = current_index
            else:
                semantic_step = 0
            controller.rcc_set_step(semantic_step)
        try:
            return original_execute_model(self, *args, **kwargs)
        finally:
            if selected and active:
                _stop()

    def measured_capture_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "start" in result:
            raise RuntimeError("GPUModelRunner.capture_model called more than once")
        _start()
        controller.rcc_set_step(0)
        try:
            return original_capture_model(self, *args, **kwargs)
        finally:
            if active:
                _stop()

    model_runner_class.execute_model = measured_execute_model
    if original_sample is not None:

        def measured_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
            global sample_index
            current_index = sample_index
            if armed:
                sample_index += 1
            selected = _selected_sample_index(current_index)
            if selected:
                _start()
                controller.rcc_set_step(target_step)
            try:
                return original_sample(self, *args, **kwargs)
            finally:
                if selected and active:
                    _stop()

        model_runner_class.sample = measured_sample
    if is_capture and phase != "graph-capture-full":
        model_runner_class.capture_model = measured_capture_model


def _patch_capture_mode_runner(model_runner_class: type[Any]) -> None:
    original = getattr(model_runner_class, "_capture_cudagraphs", None)
    if original is None:
        result.setdefault("capture_mode_patch_unavailable", []).append(
            f"{model_runner_class.__module__}.{model_runner_class.__name__}"
        )
        return

    def measured_capture_cudagraphs(
        self: Any, *args: Any, **kwargs: Any
    ) -> Any:
        mode = kwargs.get("cudagraph_runtime_mode")
        if mode is None and len(args) >= 2:
            mode = args[1]
        mode_name = getattr(mode, "name", str(mode))
        result.setdefault("capture_mode_calls_seen", []).append(mode_name)
        selected = phase == "graph-capture-full" and mode == CUDAGraphMode.FULL
        if not selected:
            return original(self, *args, **kwargs)
        if result.get("selected_capture_mode") is not None:
            raise RuntimeError("FULL _capture_cudagraphs called more than once")
        result["selected_capture_mode"] = mode_name
        _start()
        controller.rcc_set_step(0)
        try:
            return original(self, *args, **kwargs)
        finally:
            if active:
                _stop()

    model_runner_class._capture_cudagraphs = measured_capture_cudagraphs


def _patch_v2_capture_manager() -> None:
    original_capture = CudaGraphManager.capture
    capture_lifecycle_stage = 5000 <= stage_id <= 5018

    def measured_manager_capture(
        self: Any,
        create_forward_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        def measured_create_forward_fn(desc: Any, warmup: bool) -> Any:
            mode = desc.cg_mode
            mode_name = getattr(mode, "name", str(mode))
            if warmup:
                result.setdefault("capture_mode_calls_seen", []).append(mode_name)
            if mode != CUDAGraphMode.FULL or warmup:
                return create_forward_fn(desc, warmup)

            if result.get("selected_capture_mode") is not None:
                raise RuntimeError("FULL capture target called more than once")
            result["selected_capture_mode"] = "FULL"
            if capture_lifecycle_stage:
                result["capture_mode_hook_path"] = (
                    "CudaGraphManager.capture->"
                    "create_forward_fn(FULL,warmup=False)->capture-complete"
                )
                _start()
                controller.rcc_set_step(0)
                return create_forward_fn(desc, warmup)

            forward_fn = create_forward_fn(desc, warmup)

            def measured_full_forward(cg_mode: Any) -> Any:
                result["capture_mode_hook_path"] = (
                    "CudaGraphManager.capture->"
                    "create_forward_fn(FULL,warmup=False)->forward_fn"
                )
                _start()
                controller.rcc_set_step(0)
                try:
                    return forward_fn(cg_mode)
                finally:
                    if active:
                        _stop()

            return measured_full_forward

        try:
            return original_capture(
                self, measured_create_forward_fn, *args, **kwargs
            )
        finally:
            if capture_lifecycle_stage and active:
                _stop()

    CudaGraphManager.capture = measured_manager_capture


_patch_model_runner(GPUModelRunnerV1)
_patch_model_runner(GPUModelRunnerV2)
if phase == "graph-capture-full":
    _patch_capture_mode_runner(GPUModelRunnerV1)
    _patch_capture_mode_runner(GPUModelRunnerV2)
    _patch_v2_capture_manager()

try:
    compilation_config = {
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [1],
        "cudagraph_num_of_warmups": 0,
    }
    llm = LLM(
        model="/home/model/Qwen3-8B-Instruct",
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,
        max_num_batched_tokens=163840,
        max_num_seqs=1,
        gpu_memory_utilization=0.8,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        compilation_config=compilation_config,
        seed=0,
    )
    prompt_ids = None
    params = None
    if not is_capture:
        tokenizer = llm.get_tokenizer()
        seed_ids = tokenizer.encode(
            "ReadCoreCycle CPU dispatch measurement.", add_special_tokens=False
        )
        prompt_ids = (
            seed_ids * ((input_tokens + len(seed_ids) - 1) // len(seed_ids))
        )[:input_tokens]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
            seed=0,
        )
        # Complete all lazy Triton/AOT launcher instantiation before collecting
        # provenance or arming a measurement window.  The patched execute/sample
        # hooks remain bypassed because ``armed`` is still false here.
        llm.generate([{"prompt_token_ids": prompt_ids}], params)
    from torch._inductor.runtime import static_triton_launcher

    actual_static_triton_launcher = Path(
        static_triton_launcher.__file__
    ).resolve(strict=True)
    actual_static_launcher_sha = _sha256(actual_static_triton_launcher)
    expected_static_launcher_sha = _sha256(expected_static_triton_launcher)
    if actual_static_launcher_sha != expected_static_launcher_sha:
        raise RuntimeError(
            "static Triton launcher import SHA mismatch: expected "
            f"{expected_static_launcher_sha} from {expected_static_triton_launcher}, "
            f"got {actual_static_launcher_sha} from {actual_static_triton_launcher}"
        )
    static_stage_mapping = dict(
        static_triton_launcher._RCC_QWEN3_GRAPH_STAGE
    )
    static_build_map_path = Path(_required("RCC_TRITON_STATIC_BUILD_MAP"))
    static_runtime_map_path = Path(_required("RCC_TRITON_STATIC_RUNTIME_MAP"))
    static_build_mapping = [
        json.loads(line)
        for line in static_build_map_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if static_build_map_path.is_file() else []
    static_runtime_mapping = [
        json.loads(line)
        for line in static_runtime_map_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if static_runtime_map_path.is_file() else []
    result["triton_static_launcher_provenance"] = {
        "python_path": str(actual_static_triton_launcher),
        "python_sha256": actual_static_launcher_sha,
        "expected_copy_path": str(expected_static_triton_launcher),
        "expected_copy_sha256": expected_static_launcher_sha,
        "implementation": "torch._C._StaticCudaLauncher._launch_kernel",
        "stage_base_mapping": static_stage_mapping,
        "build_mapping": static_build_mapping,
        "runtime_mapping": static_runtime_mapping,
    }
    aot_model_paths = sorted(
        path.resolve(strict=True)
        for path in Path(_required("VLLM_CACHE_ROOT")).glob(
            "torch_compile_cache/torch_aot_compile/**/rank_0_0/model"
        )
    )
    if not aot_model_paths:
        raise RuntimeError("Graph mode produced no traceable AOT model artifact")
    result["aot_model_provenance"] = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in aot_model_paths
    ]
    mapped_paths = _mapped_realpaths()
    expected_measurement_binaries = {
        expected_torch_cuda_binary,
        expected_torch_python_binary,
        expected_custom_binary,
        expected_flash_attention_binary,
    }
    mapped_measurement_binaries = {
        path for path in mapped_paths if path.name in {
            "libtorch_cuda.so",
            "libtorch_python.so",
            "_C_stable_libtorch.abi3.so",
            "_vllm_fa2_C.abi3.so",
        }
    }
    if mapped_measurement_binaries != expected_measurement_binaries:
        raise RuntimeError(
            "measurement binary map mismatch: expected "
            f"{sorted(map(str, expected_measurement_binaries))}, got "
            f"{sorted(map(str, mapped_measurement_binaries))}"
        )
    result["measurement_binary_provenance"] = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in sorted(expected_measurement_binaries)
    ]
    result["probe_source_manifest"] = {
        "path": str(probe_source_manifest),
        "sha256": _sha256(probe_source_manifest),
    }
    if 9000 <= stage_id <= 10009:
        expected_stage_base = stage_id - (stage_id % 100)
        matching_stage_bases = {
            int(record["stage_base"])
            for record in static_runtime_mapping
            if record.get("stage_base") is not None
        }
        if expected_stage_base not in matching_stage_bases:
            raise RuntimeError(
                "selected Triton stage base was not executed through its runtime launcher: "
                f"expected {expected_stage_base}, got {sorted(matching_stage_bases)}"
            )
    if is_capture:
        if "stop" not in result:
            raise RuntimeError("selected capture_model window was not observed")
    else:
        assert prompt_ids is not None
        assert params is not None
        armed = True
        execute_index = 0
        sample_index = 0
        if window_scope == "full-request":
            _start()
            controller.rcc_set_step(0)
        try:
            outputs = llm.generate([{"prompt_token_ids": prompt_ids}], params)
        finally:
            if window_scope == "full-request" and active:
                _stop()
        armed = False
        result["generated_tokens"] = len(outputs[0].outputs[0].token_ids)
        if instrumentation_active and "stop" not in result:
            raise RuntimeError("selected measurement window was not observed")
        if not instrumentation_active:
            result["stop"] = {
                "phase": "inactive",
                "selected_count": 0,
                "event_count": 0,
                "lost_count": 0,
            }

    stop = result["stop"]
    if expected_count is not None and int(stop["event_count"]) != expected_count:
        raise RuntimeError(
            f"event_count {stop['event_count']} does not match expected "
            f"{expected_count}"
        )
    if instrumentation_active:
        if int(stop["event_count"]) == 0 and expected_count is None:
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "sample_count": 0,
                        "selected_count": int(stop["selected_count"]),
                        "event_count": 0,
                        "lost_count": int(stop["lost_count"]),
                        "phase": phase,
                        "step": target_step,
                        "stage_id": stage_id,
                        "pilot_zero_hit": True,
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
        else:
            collector_module.summarize_raw(
                run_dir / "raw.csv",
                run_dir / "summary.json",
                run_dir / "derived.csv",
                cpu_frequency_hz=cpu_frequency_hz,
                pmcr_divider=1,
                selected_count=int(stop["selected_count"]),
                event_count=int(stop["event_count"]),
                lost_count=int(stop["lost_count"]),
                phase=phase,
                step=target_step,
            )
    result["status"] = "ok"
except BaseException as exc:
    result["status"] = "error"
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)
    result["traceback"] = traceback.format_exc()
    if active:
        try:
            result["emergency_stop"] = controller.rcc_stop_dump(
                str(run_dir / "error_raw.csv")
            )
        except BaseException as stop_exc:
            result["emergency_stop_error"] = repr(stop_exc)
    raise
finally:
    (run_dir / "controller.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
