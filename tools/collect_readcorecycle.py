"""Cold-path ReadCoreCycle control, validation, and cycle conversion."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = (
    "sequence",
    "begin_cycle",
    "end_cycle",
    "delta_cycles",
    "context_id",
    "site_id",
    "stage_id",
    "begin_monotonic_tick",
    "end_monotonic_tick",
    "delta_monotonic_ticks",
    "monotonic_frequency_hz",
    "delta_monotonic_ns",
    "step_id",
)

PHASE_CHOICES = (
    "graph-capture-full",
    "graph-capture-piecewise",
    "graph-prefill",
    "graph-replay",
    "graph-outside",
    "graph-logits",
    "eager-prefill",
    "eager-decode",
    "eager-logits",
)


def _stage_identity(stage_id: int, occurrence_index: int) -> dict[str, int | str]:
    role = "unknown"
    layer_index = -1
    stage_group = "unknown"

    fixed_roles = {
        30: "unified_kv_cache_update",
        40: "unified_attention_with_output",
        100: "qkv_proj",
        110: "o_proj",
        120: "gate_up_proj",
        130: "down_proj",
        140: "lm_head_or_other_gemm",
        220: "SiluAndMul",
        230: "rotary_embedding",
        240: "embed_tokens_gather",
        250: "embed_tokens_index_select",
        400: "auxiliary_device_copy",
    }
    base = -1
    for candidate in fixed_roles:
        width = 10 if candidate in {100, 110, 120, 130, 140} else 10
        if candidate <= stage_id < candidate + width:
            base = candidate
            role = fixed_roles[candidate]
            break

    if 200 <= stage_id < 210:
        base = 200
        if occurrence_index == 72:
            role = "final_norm"
        elif occurrence_index % 2 == 0:
            role = "q_norm"
            layer_index = occurrence_index // 2
        else:
            role = "k_norm"
            layer_index = occurrence_index // 2
    elif 210 <= stage_id < 220:
        base = 210
        if occurrence_index % 2 == 0:
            role = "input_layernorm"
            layer_index = occurrence_index // 2
        else:
            role = "post_attention_layernorm"
            layer_index = occurrence_index // 2
    elif base in {30, 100, 110, 120, 130, 220, 230}:
        layer_index = occurrence_index

    if base in {30, 200, 210, 220, 230, 240, 250, 400}:
        offset = stage_id - base
        stage_group = {
            0: "prepare",
            1: "submit",
            2: "total",
            3: "dispatcher",
            4: "return_tail",
            5: "full_dispatch",
        }.get(offset, "unknown")
    elif base in {100, 110, 120, 130, 140}:
        offset = stage_id - base
        stage_group = {
            0: "implementation_prepare",
            1: "backend_dispatch",
            2: "library_prepare",
            3: "submit",
            4: "total",
            5: "dispatcher",
            6: "return_tail",
            7: "full_dispatch",
        }.get(offset, "unknown")
    elif base == 40:
        stage_group = {
            0: "implementation_prepare",
            1: "backend_dispatch",
            2: "launcher_prepare",
            3: "primary_submit",
            4: "combine_submit",
            5: "total",
            6: "dispatcher",
            7: "return_tail",
            8: "full_dispatch",
            9: "submit_group",
        }.get(stage_id - base, "unknown")
    elif stage_id in {50, 51, 52, 53, 54}:
        role = "full_graph_replay"
        stage_group = {
            50: "total",
            51: "prologue",
            52: "stream",
            53: "submit",
            54: "return_tail",
        }[stage_id]
    elif 300 <= stage_id < 600:
        generated_base = stage_id - (stage_id % 10)
        role = {
            300: "embedding_input_norm_fusion",
            310: "layer0_qk_norm_reduction_fusion",
            320: "layer0_k_norm_rope_fusion",
            330: "layer0_q_norm_rope_fusion",
            340: "repeated_qk_norm_reduction_fusion",
            350: "repeated_k_norm_rope_fusion",
            360: "repeated_q_norm_rope_fusion",
            370: "post_attention_residual_norm_fusion",
            380: "silu_slice_fusion",
            390: "down_residual_next_norm_fusion",
            400: "auxiliary_device_copy",
            500: "combine_sampled_and_draft_tokens",
            510: "compute_slot_mappings",
            520: "gather_block_tables",
            530: "prepare_pos_seq_lens",
            540: "prepare_prefill_inputs",
            550: "apply_write",
            560: "gumbel_sample",
            570: "get_num_sampled_and_rejected",
            580: "post_update",
        }.get(generated_base, f"generated_launcher_{generated_base}")
        if generated_base <= 330:
            layer_index = 0
        elif generated_base <= 360:
            layer_index = occurrence_index + 1
        elif generated_base <= 390:
            layer_index = occurrence_index
        stage_group = {
            0: "prepare",
            1: "submit",
            2: "total",
            3: "return_tail",
        }.get(stage_id - generated_base, "unknown")

    return {
        "operator_role": role,
        "layer_idx": layer_index,
        "occurrence_idx": occurrence_index,
        "stage_group": stage_group,
    }


class RccWorkerExtension:
    """Same-thread vLLM worker extension for cold RCC lifecycle calls."""

    def rcc_start(
        self,
        exact_site: int,
        context_id: int,
        capacity: int,
        exact_stage: int | None = None,
    ) -> dict[str, int | str]:
        self._ensure_runtime()
        before = self._owner()
        if exact_stage is None:
            result = int(self._rcc_start(exact_site, context_id, capacity))
        else:
            result = int(
                self._rcc_start_selected(
                    exact_site, exact_stage, context_id, capacity
                )
            )
        after = self._owner()
        if before != after:
            raise RuntimeError(f"RCC owner changed during start: {before} != {after}")
        if result != 0:
            raise RuntimeError(f"rcc_start failed with errno {result}")
        self._rcc_owner_at_start = after
        response: dict[str, int | str] = {
            "phase": "start",
            **after,
            "site_id": exact_site,
            "context_id": context_id,
            "capacity": capacity,
        }
        if exact_stage is not None:
            response["stage_id"] = exact_stage
        return response

    def rcc_prepare(
        self,
        exact_site: int,
        exact_stage: int,
        context_id: int,
        capacity: int,
    ) -> dict[str, int | str]:
        self._ensure_runtime()
        before = self._owner()
        result = int(
            self._rcc_prepare_selected(
                exact_site, exact_stage, context_id, capacity
            )
        )
        after = self._owner()
        if before != after:
            raise RuntimeError(f"RCC owner changed during prepare: {before} != {after}")
        if result != 0:
            raise RuntimeError(f"rcc_prepare_selected failed with errno {result}")
        return {
            "phase": "prepare",
            **after,
            "site_id": exact_site,
            "stage_id": exact_stage,
            "context_id": context_id,
            "capacity": capacity,
        }

    def rcc_arm(self) -> dict[str, int | str]:
        self._ensure_runtime()
        before = self._owner()
        result = int(self._rcc_arm_prepared())
        after = self._owner()
        if before != after:
            raise RuntimeError(f"RCC owner changed during arm: {before} != {after}")
        if result != 0:
            raise RuntimeError(f"rcc_arm_prepared failed with errno {result}")
        self._rcc_owner_at_start = after
        return {"phase": "arm", **after}

    def rcc_stop_dump(self, output_path: str) -> dict[str, int | str]:
        self._ensure_runtime()
        owner_at_start = getattr(self, "_rcc_owner_at_start", None)
        if owner_at_start is None:
            raise RuntimeError("rcc_stop_dump called without a successful start")
        before = self._owner()
        if before != owner_at_start:
            raise RuntimeError(
                f"RCC owner changed before stop: {owner_at_start} != {before}"
            )
        stop_result = int(self._rcc_stop())
        dump_result = int(self._rcc_dump(os.fsencode(os.path.realpath(output_path))))
        after = self._owner()
        if before != after:
            raise RuntimeError(f"RCC owner changed during stop: {before} != {after}")
        if stop_result != 0:
            raise RuntimeError(f"rcc_stop failed with errno {stop_result}")
        if dump_result != 0:
            raise RuntimeError(f"rcc_dump_csv failed with errno {dump_result}")
        self._rcc_owner_at_start = None
        return {
            "phase": "stop_dump",
            **after,
            "selected_count": int(self._rcc_selected()),
            "event_count": int(self._rcc_events()),
            "lost_count": int(self._rcc_lost()),
        }

    def rcc_set_step(self, step_id: int) -> None:
        self._ensure_runtime()
        result = int(self._rcc_set_step(step_id))
        if result != 0:
            raise RuntimeError(f"rcc_set_step failed with errno {result}")

    def _ensure_runtime(self) -> None:
        if hasattr(self, "_rcc_library"):
            return
        runtime_path = os.environ["VLLM_RCC_RUNTIME"]
        self._rcc_library = ctypes.CDLL(
            os.path.realpath(runtime_path), mode=ctypes.RTLD_GLOBAL
        )
        self._rcc_abi_version = self._bind("rcc_tool_abi_version", ctypes.c_uint32)
        self._rcc_start = self._bind(
            "rcc_start",
            ctypes.c_int,
            ctypes.c_uint16,
            ctypes.c_uint32,
            ctypes.c_uint64,
        )
        self._rcc_start_selected = self._bind(
            "rcc_start_selected",
            ctypes.c_int,
            ctypes.c_uint16,
            ctypes.c_uint16,
            ctypes.c_uint32,
            ctypes.c_uint64,
        )
        self._rcc_prepare_selected = self._bind(
            "rcc_prepare_selected",
            ctypes.c_int,
            ctypes.c_uint16,
            ctypes.c_uint16,
            ctypes.c_uint32,
            ctypes.c_uint64,
        )
        self._rcc_arm_prepared = self._bind("rcc_arm_prepared", ctypes.c_int)
        self._rcc_set_step = self._bind(
            "rcc_set_step", ctypes.c_int, ctypes.c_int64
        )
        self._rcc_stop = self._bind("rcc_stop", ctypes.c_int)
        self._rcc_selected = self._bind("rcc_selected_count", ctypes.c_uint64)
        self._rcc_events = self._bind("rcc_event_count", ctypes.c_uint64)
        self._rcc_lost = self._bind("rcc_lost_count", ctypes.c_uint64)
        self._rcc_dump = self._bind("rcc_dump_csv", ctypes.c_int, ctypes.c_char_p)
        if int(self._rcc_abi_version()) != 4:
            raise RuntimeError("unsupported ReadCoreCycle runtime ABI")

    def _bind(self, name: str, result_type: Any, *argument_types: Any) -> Any:
        function = getattr(self._rcc_library, name)
        function.restype = result_type
        function.argtypes = list(argument_types)
        return function

    @staticmethod
    def _owner() -> dict[str, int]:
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) != 1:
            raise RuntimeError(f"RCC requires singleton affinity, got {cpus}")
        return {
            "pid": os.getpid(),
            "native_tid": threading.get_native_id(),
            "cpu": cpus[0],
        }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile without samples")
    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _read_raw_samples(raw_path: Path) -> list[dict[str, int | float]]:
    samples: list[dict[str, int | float]] = []
    with raw_path.open(newline="", encoding="utf-8") as raw_file:
        reader = csv.DictReader(raw_file)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(
                f"unexpected raw columns: {reader.fieldnames}; "
                f"expected {REQUIRED_COLUMNS}"
            )
        for expected_sequence, row in enumerate(reader):
            sample: dict[str, int | float] = {
                column: (
                    float(row[column])
                    if column == "delta_monotonic_ns"
                    else int(row[column])
                )
                for column in REQUIRED_COLUMNS
            }
            if sample["sequence"] != expected_sequence:
                raise ValueError(f"sequence gap at row {expected_sequence}: {sample}")
            if sample["end_cycle"] < sample["begin_cycle"]:
                raise ValueError(f"cycle counter moved backwards: {sample}")
            if sample["end_cycle"] - sample["begin_cycle"] != sample["delta_cycles"]:
                raise ValueError(f"delta cycle mismatch: {sample}")
            if sample["delta_cycles"] == 0:
                raise ValueError(f"zero cycle sample: {sample}")
            if sample["end_monotonic_tick"] < sample["begin_monotonic_tick"]:
                raise ValueError(f"monotonic counter moved backwards: {sample}")
            if (
                sample["end_monotonic_tick"] - sample["begin_monotonic_tick"]
                != sample["delta_monotonic_ticks"]
            ):
                raise ValueError(f"monotonic tick delta mismatch: {sample}")
            if sample["delta_monotonic_ticks"] == 0:
                raise ValueError(f"zero monotonic tick sample: {sample}")
            if sample["step_id"] < 0:
                raise ValueError(f"missing event step identity: {sample}")
            expected_ns = (
                sample["delta_monotonic_ticks"]
                * 1_000_000_000.0
                / sample["monotonic_frequency_hz"]
            )
            if abs(sample["delta_monotonic_ns"] - expected_ns) > 0.001:
                raise ValueError(f"monotonic time conversion mismatch: {sample}")
            samples.append(sample)
    if not samples:
        raise ValueError("raw evidence contains no samples")
    return samples


def _write_derived_csv(
    output_path: Path,
    samples: Sequence[dict[str, int | float]],
    cycles_per_microsecond: float,
    phase: str,
) -> None:
    with output_path.open("x", newline="", encoding="utf-8") as output_file:
        fieldnames = (
            *REQUIRED_COLUMNS,
            "phase",
            "step",
            "operator_role",
            "layer_idx",
            "occurrence_idx",
            "stage_group",
            "latency_us",
            "cpufreq_latency_us",
            "paired_cycle_hz",
        )
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        occurrence_by_step: dict[int, int] = {}
        for sample in samples:
            step_id = int(sample["step_id"])
            occurrence_index = occurrence_by_step.get(step_id, 0)
            occurrence_by_step[step_id] = occurrence_index + 1
            writer.writerow(
                {
                    **sample,
                    "phase": phase,
                    "step": step_id,
                    **_stage_identity(sample["stage_id"], occurrence_index),
                    "latency_us": sample["delta_monotonic_ns"] / 1_000.0,
                    "cpufreq_latency_us": (
                        sample["delta_cycles"] / cycles_per_microsecond
                    ),
                    "paired_cycle_hz": (
                        sample["delta_cycles"]
                        * 1_000_000_000.0
                        / sample["delta_monotonic_ns"]
                    ),
                }
            )


def summarize_raw(
    raw_path: Path,
    summary_path: Path,
    derived_path: Path,
    cpu_frequency_hz: int,
    pmcr_divider: int,
    selected_count: int,
    event_count: int,
    lost_count: int,
    phase: str = "unspecified",
    step: int = -1,
) -> None:
    if cpu_frequency_hz <= 0:
        raise ValueError("CPU frequency must be positive")
    if lost_count != 0:
        raise ValueError(f"lost_count must be zero, got {lost_count}")
    if pmcr_divider != 1:
        raise ValueError("PMCR_EL0.D must be zero so PMCCNTR advances every cycle")
    samples = _read_raw_samples(raw_path)
    if event_count != len(samples):
        raise ValueError(
            f"event_count {event_count} does not match {len(samples)} rows"
        )
    if selected_count != event_count + lost_count:
        raise ValueError("selected_count must equal event_count plus lost_count")
    site_ids = sorted({sample["site_id"] for sample in samples})
    stage_ids = sorted({sample["stage_id"] for sample in samples})
    context_ids = sorted({sample["context_id"] for sample in samples})
    if len(site_ids) != 1 or len(stage_ids) != 1 or len(context_ids) != 1:
        raise ValueError(
            "an exact-stage run must contain one site_id, stage_id, and context_id"
        )
    cycles_per_microsecond = cpu_frequency_hz / 1_000_000.0
    sorted_cycles = sorted(sample["delta_cycles"] for sample in samples)
    monotonic_frequencies = sorted(
        {int(sample["monotonic_frequency_hz"]) for sample in samples}
    )
    if len(monotonic_frequencies) != 1:
        raise ValueError(
            f"one run must use one monotonic frequency, got {monotonic_frequencies}"
        )
    sorted_microseconds = sorted(
        float(sample["delta_monotonic_ns"]) / 1_000.0 for sample in samples
    )
    sorted_paired_cycle_hz = sorted(
        float(sample["delta_cycles"])
        * 1_000_000_000.0
        / float(sample["delta_monotonic_ns"])
        for sample in samples
    )
    summary = {
        "sample_count": len(samples),
        "selected_count": selected_count,
        "event_count": event_count,
        "lost_count": lost_count,
        "cpu_frequency_hz": cpu_frequency_hz,
        "monotonic_frequency_hz": monotonic_frequencies[0],
        "pmcr_divider": pmcr_divider,
        "conversion": (
            "latency_us = paired delta_monotonic_ticks / "
            "monotonic_frequency_hz * 1e6"
        ),
        "cpufreq_conversion_is_reference_only": True,
        "site_id": site_ids[0],
        "stage_id": stage_ids[0],
        "context_id": context_ids[0],
        "phase": phase,
        "step": step,
        "cycles": {
            "min": sorted_cycles[0],
            "p50": _percentile(sorted_cycles, 0.50),
            "p95": _percentile(sorted_cycles, 0.95),
            "max": sorted_cycles[-1],
        },
        "latency_us": {
            "min": sorted_microseconds[0],
            "p50": _percentile(sorted_microseconds, 0.50),
            "p95": _percentile(sorted_microseconds, 0.95),
            "max": sorted_microseconds[-1],
        },
        "paired_cycle_hz": {
            "min": sorted_paired_cycle_hz[0],
            "p50": _percentile(sorted_paired_cycle_hz, 0.50),
            "p95": _percentile(sorted_paired_cycle_hz, 0.95),
            "max": sorted_paired_cycle_hz[-1],
        },
    }
    with summary_path.open("x", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    _write_derived_csv(derived_path, samples, cycles_per_microsecond, phase)


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw ReadCoreCycle evidence and derive microseconds."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--cpu-frequency-hz", type=int, required=True)
    parser.add_argument("--pmcr-divider", type=int, required=True)
    parser.add_argument("--selected-count", type=int, required=True)
    parser.add_argument("--event-count", type=int, required=True)
    parser.add_argument("--lost-count", type=int, required=True)
    parser.add_argument("--phase", choices=PHASE_CHOICES, required=True)
    parser.add_argument("--step", type=int, default=-1)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)
    summarize_raw(
        options.raw,
        options.summary,
        options.derived,
        options.cpu_frequency_hz,
        options.pmcr_divider,
        options.selected_count,
        options.event_count,
        options.lost_count,
        options.phase,
        options.step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
