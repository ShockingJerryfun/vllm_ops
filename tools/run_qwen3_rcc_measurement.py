#!/usr/bin/env python3
"""Run one exact Qwen3 CPU-dispatch stage inside one model execution window.

This driver deliberately starts the recorder for exactly one stage and never
synchronizes CUDA.  It is meant to run inside the vLLM runtime container; the
shell wrapper owns PMU setup, CPU pinning, and shared-server preflight checks.
"""

from __future__ import annotations

import importlib.util
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


repo = Path(_required("RCC_REPO"))
run_dir = Path(_required("RCC_RUN_DIR"))
phase = _required("RCC_PHASE")
stage_id = int(_required("RCC_STAGE_ID"))
context_id = int(os.environ.get("RCC_CONTEXT_ID", "700101"))
target_step = int(os.environ.get("RCC_TARGET_STEP", "0"))
expected_count_text = os.environ.get("RCC_EXPECTED_COUNT", "")
expected_count = int(expected_count_text) if expected_count_text else None
input_tokens = int(os.environ.get("RCC_INPUT_TOKENS", "128"))
max_tokens = int(os.environ.get("RCC_MAX_TOKENS", "4"))
kv_cache_memory_bytes = int(os.environ.get("RCC_KV_CACHE_MEMORY_BYTES", "1258291200"))
cpu_frequency_hz = int(_required("RCC_CPU_FREQUENCY_HZ"))
is_eager = phase.startswith("eager-")
is_capture = phase.startswith("graph-capture-")

if phase not in {
    "eager-prefill",
    "eager-decode",
    "graph-capture-full",
    "graph-capture-piecewise",
    "graph-prefill",
    "graph-replay",
}:
    raise RuntimeError(f"unsupported RCC_PHASE: {phase}")

collector_path = repo / "tools" / "collect_readcorecycle.py"
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
from triton.backends.nvidia import driver as triton_driver  # noqa: E402
from triton.runtime import build as triton_build  # noqa: E402

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402

result: dict[str, Any] = {
    "site_id": 700,
    "stage_id": stage_id,
    "phase": phase,
    "target_step": target_step,
    "expected_count": expected_count,
    "cpu": 249,
    "model": "/home/model/Qwen3-8B-Instruct",
    "input_tokens": input_tokens,
    "max_tokens": max_tokens,
    "torch_version": torch.__version__,
    "torch_path": torch.__file__,
    "vllm_path": vllm.__file__,
}
active = False
armed = False
execute_index = 0
original_execute_model = GPUModelRunner.execute_model
original_capture_model = GPUModelRunner.capture_model


def _start() -> None:
    global active
    result["start"] = controller.rcc_start(
        700, context_id, 1048576, exact_stage=stage_id
    )
    active = True


def _stop() -> None:
    global active
    result["stop"] = controller.rcc_stop_dump(str(run_dir / "raw.csv"))
    active = False


def measured_execute_model(
    self: GPUModelRunner, scheduler_output: Any, intermediate_tensors: Any = None
) -> Any:
    global execute_index
    current_index = execute_index
    if armed:
        execute_index += 1
    selected = armed and (
        (phase in {"eager-prefill", "graph-prefill"} and current_index == 0)
        or (
            phase in {"eager-decode", "graph-replay"}
            and current_index == target_step + 1
        )
    )
    if selected:
        _start()
    try:
        return original_execute_model(self, scheduler_output, intermediate_tensors)
    finally:
        if selected and active:
            _stop()


def measured_capture_model(self: GPUModelRunner) -> Any:
    if "start" in result:
        raise RuntimeError("GPUModelRunner.capture_model called more than once")
    _start()
    try:
        return original_capture_model(self)
    finally:
        if active:
            _stop()


GPUModelRunner.execute_model = measured_execute_model
if is_capture:
    GPUModelRunner.capture_model = measured_capture_model

try:
    build_source = Path(triton_build.__file__).read_text(encoding="utf-8")
    driver_source = Path(triton_driver.__file__).read_text(encoding="utf-8")
    if "source_language" not in build_source:
        raise RuntimeError(f"unpatched Triton build helper: {triton_build.__file__}")
    if "_RCC_QWEN3_EAGER_STAGE_BASE" not in driver_source:
        raise RuntimeError(f"unpatched Triton CUDA driver: {triton_driver.__file__}")

    compilation_config = None
    if not is_eager:
        graph_mode = "FULL" if "piecewise" not in phase else "PIECEWISE"
        compilation_config = {
            "cudagraph_mode": graph_mode,
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
        enforce_eager=is_eager,
        compilation_config=compilation_config,
        seed=0,
    )
    if is_capture:
        if "stop" not in result:
            raise RuntimeError("selected capture_model window was not observed")
    else:
        tokenizer = llm.get_tokenizer()
        seed_ids = tokenizer.encode(
            "ReadCoreCycle CPU dispatch measurement.", add_special_tokens=False
        )
        prompt_ids = (seed_ids * ((input_tokens + len(seed_ids) - 1) // len(seed_ids)))[
            :input_tokens
        ]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
            seed=0,
        )
        llm.generate([{"prompt_token_ids": prompt_ids}], params)
        armed = True
        execute_index = 0
        outputs = llm.generate([{"prompt_token_ids": prompt_ids}], params)
        armed = False
        result["generated_tokens"] = len(outputs[0].outputs[0].token_ids)
        if "stop" not in result:
            raise RuntimeError("selected execute_model window was not observed")

    stop = result["stop"]
    if expected_count is not None and int(stop["event_count"]) != expected_count:
        raise RuntimeError(
            f"event_count {stop['event_count']} does not match expected "
            f"{expected_count}"
        )
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
