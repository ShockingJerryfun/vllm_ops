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
)


class RccWorkerExtension:
    """Same-thread vLLM worker extension for cold RCC lifecycle calls."""

    def rcc_start(
        self, exact_site: int, context_id: int, capacity: int
    ) -> dict[str, int | str]:
        self._ensure_runtime()
        before = self._owner()
        result = int(self._rcc_start(exact_site, context_id, capacity))
        after = self._owner()
        if before != after:
            raise RuntimeError(f"RCC owner changed during start: {before} != {after}")
        if result != 0:
            raise RuntimeError(f"rcc_start failed with errno {result}")
        self._rcc_owner_at_start = after
        return {
            "phase": "start",
            **after,
            "site_id": exact_site,
            "context_id": context_id,
            "capacity": capacity,
        }

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
        self._rcc_stop = self._bind("rcc_stop", ctypes.c_int)
        self._rcc_selected = self._bind("rcc_selected_count", ctypes.c_uint64)
        self._rcc_events = self._bind("rcc_event_count", ctypes.c_uint64)
        self._rcc_lost = self._bind("rcc_lost_count", ctypes.c_uint64)
        self._rcc_dump = self._bind("rcc_dump_csv", ctypes.c_int, ctypes.c_char_p)
        if int(self._rcc_abi_version()) != 1:
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


def _read_raw_samples(raw_path: Path) -> list[dict[str, int]]:
    samples: list[dict[str, int]] = []
    with raw_path.open(newline="", encoding="utf-8") as raw_file:
        reader = csv.DictReader(raw_file)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(
                f"unexpected raw columns: {reader.fieldnames}; "
                f"expected {REQUIRED_COLUMNS}"
            )
        for expected_sequence, row in enumerate(reader):
            sample = {column: int(row[column]) for column in REQUIRED_COLUMNS}
            if sample["sequence"] != expected_sequence:
                raise ValueError(f"sequence gap at row {expected_sequence}: {sample}")
            if sample["end_cycle"] < sample["begin_cycle"]:
                raise ValueError(f"cycle counter moved backwards: {sample}")
            if sample["end_cycle"] - sample["begin_cycle"] != sample["delta_cycles"]:
                raise ValueError(f"delta cycle mismatch: {sample}")
            if sample["delta_cycles"] == 0:
                raise ValueError(f"zero cycle sample: {sample}")
            samples.append(sample)
    if not samples:
        raise ValueError("raw evidence contains no samples")
    return samples


def _write_derived_csv(
    output_path: Path,
    samples: Sequence[dict[str, int]],
    cycles_per_microsecond: float,
) -> None:
    with output_path.open("x", newline="", encoding="utf-8") as output_file:
        fieldnames = (*REQUIRED_COLUMNS, "latency_us")
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    **sample,
                    "latency_us": (sample["delta_cycles"] / cycles_per_microsecond),
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
    context_ids = sorted({sample["context_id"] for sample in samples})
    if len(site_ids) != 1 or len(context_ids) != 1:
        raise ValueError(
            "an exact-site run must contain one site_id and one context_id"
        )
    cycles_per_microsecond = cpu_frequency_hz / 1_000_000.0
    sorted_cycles = sorted(sample["delta_cycles"] for sample in samples)
    sorted_microseconds = [cycles / cycles_per_microsecond for cycles in sorted_cycles]
    summary = {
        "sample_count": len(samples),
        "selected_count": selected_count,
        "event_count": event_count,
        "lost_count": lost_count,
        "cpu_frequency_hz": cpu_frequency_hz,
        "pmcr_divider": pmcr_divider,
        "conversion": "latency_us = delta_cycles / (cpu_frequency_hz / 1e6)",
        "site_id": site_ids[0],
        "context_id": context_ids[0],
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
    }
    with summary_path.open("x", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    _write_derived_csv(derived_path, samples, cycles_per_microsecond)


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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
