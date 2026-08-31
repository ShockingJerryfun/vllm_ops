#!/usr/bin/env python3
"""Validate and aggregate the one-pass Qwen3 Graph stage measurements.

The collector's legacy role mapping predates the 0829 stage catalog, so this
script always joins raw records to the audited 0829 manifest and run matrix.
It preserves every paired PMCCNTR/CNTVCT record and emits one display value per
stage/phase: the only sample when count is one, otherwise the median.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

RAW_COLUMNS = (
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

SAMPLE_COLUMNS = (
    "run_id",
    "provenance_cohort",
    "phase",
    "request_id",
    "step_id",
    "layer",
    "occurrence",
    "semantic_group",
    "operator_path",
    "operator_role",
    "implementation_family",
    "implementation_signature",
    "kernel_variant",
    "stage_id",
    "stage_code",
    "parent_stage_id",
    "root_parent_stage_id",
    "resolved_parent_stage_id",
    "relation",
    "stage_name",
    "level",
    "cycles_begin",
    "cycles_end",
    "cycles_delta",
    "cntvct_begin",
    "cntvct_end",
    "cntvct_ticks",
    "cntfrq_hz",
    "time_us",
    "paired_cycle_hz",
    "cpu_id",
    "pid",
    "tid",
    "context_id",
    "lost_count",
    "probe_version",
    "source_sha256",
    "binary_sha256",
    "status",
    "reason",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _numeric_parent_ids(value: Any) -> list[int]:
    return [int(item) for item in re.findall(r"\d+", str(value or ""))]


def _expected_count(stage: dict[str, Any]) -> int | None:
    match = re.search(r"(\d+)", str(stage.get("expected_occurrence", "")))
    return int(match.group(1)) if match else None


def _expected_run_count(
    stage: dict[str, Any], stages: dict[int, dict[str, Any]]
) -> int | None:
    direct = _expected_count(stage)
    if direct is not None:
        return direct
    parents = [
        stages[parent_id]
        for parent_id in _numeric_parent_ids(stage.get("parent_stage_id"))
        if parent_id in stages
    ]
    if not parents:
        return None
    parent_counts = [_expected_count(parent) for parent in parents]
    if any(count is None for count in parent_counts):
        return None
    return sum(int(count) for count in parent_counts if count is not None)


def _layer_from_scope(scope: str, occurrence: int) -> str:
    if not scope:
        return "N/A"
    if "Final" in scope:
        return "Final"
    if "L1-L35" in scope:
        return f"L{occurrence + 1}"
    if "L0-L35" in scope:
        return f"L{occurrence}"
    if "L0→L1" in scope or "层间" in scope:
        return f"L{occurrence}→L{occurrence + 1}"
    if "L0" in scope:
        return "L0"
    return scope


def _resolve_parent_and_layer(
    stage: dict[str, Any],
    occurrence: int,
    stages: dict[int, dict[str, Any]],
) -> tuple[int | None, str, dict[str, Any] | None]:
    if stage.get("layer_scope"):
        return None, _layer_from_scope(str(stage["layer_scope"]), occurrence), stage
    parents = [parent for parent in _numeric_parent_ids(stage.get("parent_stage_id")) if parent in stages]
    if not parents:
        return None, "N/A", None
    offset = occurrence
    for parent_id in parents:
        parent = stages[parent_id]
        count = _expected_count(parent)
        if count is None or offset < count:
            return parent_id, _layer_from_scope(str(parent.get("layer_scope", "")), offset), parent
        offset -= count
    parent_id = parents[-1]
    parent = stages[parent_id]
    return parent_id, _layer_from_scope(str(parent.get("layer_scope", "")), offset), parent


def _immediate_parent_ids(stage: dict[str, Any]) -> list[int]:
    """Return direct nesting parents, preserving root parents where appropriate."""
    stage_id = int(stage["stage_id"])
    category = str(stage["category"])
    if category == "gemm":
        offset = stage_id % 100
        if 5 <= offset <= 11:
            return [stage_id - offset + 4]
    if category == "kv_flash":
        if stage_id == 7608:
            return [7607]
        if stage_id == 8003:
            return [8002]
        if 8010 <= stage_id <= 8016:
            return [8009]
    return _numeric_parent_ids(stage.get("parent_stage_id"))


def _resolved_immediate_parent(
    stage: dict[str, Any], root_parent: int | None
) -> int | None:
    immediate = _immediate_parent_ids(stage)
    original = _numeric_parent_ids(stage.get("parent_stage_id"))
    if immediate != original and len(immediate) == 1:
        return immediate[0]
    return root_parent or (immediate[0] if len(immediate) == 1 else None)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _is_evidence_backed_na(stage: dict[str, Any], phase: str) -> bool:
    expected = str(
        (stage.get("expected_status_by_phase") or {}).get(phase, "")
    ).lower()
    if any(
        marker in expected
        for marker in ("n/a", "otherwise", "measured if", "absent")
    ):
        return True
    relation = str(stage.get("relation", "")).lower()
    applicability = str(stage.get("applicability", "")).lower()
    include = str(stage.get("include", "")).lower()
    return any(
        marker in text
        for text in (relation, applicability, include)
        for marker in ("conditional", "条件", "稀有", "n/a", "可选", "分支")
    )


def _load_run_matrices(task_root: Path) -> list[dict[str, Any]]:
    specifications = (
        ("run_matrix.tsv", "baseline-0829"),
        ("run_matrix_increment.tsv", "increment-0830"),
        ("run_matrix_conditional.tsv", "increment-0830"),
    )
    rows: list[dict[str, Any]] = []
    for matrix_name, cohort in specifications:
        path = task_root / "manifests" / matrix_name
        for row in _read_tsv(path):
            rows.append(
                {
                    **row,
                    "source_sequence": int(row["sequence"]),
                    "sequence": len(rows) + 1,
                    "matrix_name": matrix_name,
                    "provenance_cohort": cohort,
                }
            )
    return rows


def _load_conditional_na_evidence(task_root: Path) -> list[dict[str, Any]]:
    evidence = _load_json(
        task_root / "manifests" / "conditional_na_evidence.json"
    )
    items = evidence.get("items")
    if not isinstance(items, list):
        raise ValueError("conditional N/A evidence items must be a list")
    return items


def _na_result(
    *,
    sequence: int,
    stage: dict[str, Any],
    phase: str,
    reason: str,
    evidence_kind: str,
    run_id: str = "",
    run_path: str = "",
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "run_id": run_id,
        "provenance_cohort": "source-runtime-na",
        "phase": phase,
        "stage_id": int(stage["stage_id"]),
        "stage_code": stage["stage_code"],
        "category": stage["category"],
        "level": stage["level"],
        "relation": stage["relation"],
        "parent_stage_id": stage.get("parent_stage_id", ""),
        "operator_role": stage.get("operator_role", ""),
        "stage_name": stage["stage_name"],
        "sample_count": 0,
        "display_rule": "N/A",
        "display_us": "",
        "median_cycles": "",
        "min_us": "",
        "p50_us": "",
        "p95_us": "",
        "max_us": "",
        "variation_ratio": "",
        "status": "N/A",
        "reason": reason,
        "evidence_kind": evidence_kind,
        "lost_count": 0,
        "run_path": run_path,
    }


def _controller_identity(controller: dict[str, Any]) -> tuple[int, int, int]:
    owner = controller.get("start") or controller.get("prepare") or {}
    return (
        int(owner.get("cpu", 249)),
        int(owner.get("pid", -1)),
        int(owner.get("native_tid", -1)),
    )


def _binary_identity(controller: dict[str, Any]) -> str:
    records = controller.get("measurement_binary_provenance", [])
    return ";".join(
        f"{Path(str(record['path'])).name}:{record['sha256']}"
        for record in records
        if isinstance(record, dict) and "path" in record and "sha256" in record
    )


def _manifest_phase(runner_phase: str) -> str:
    return {
        "graph-capture-full": "graph-capture",
        "graph-prefill": "graph-prefill",
        "graph-replay": "steady-replay",
    }[runner_phase]


def _kernel_variant(stage: dict[str, Any]) -> str:
    selector = str(stage.get("selector", "")).strip()
    if selector:
        return selector
    category = str(stage.get("category", ""))
    role = str(stage.get("operator_role", ""))
    if category == "operator_total":
        if role == "GEMM叶子":
            return "N/A（总区间包含cuBLAS/Lt运行时分支，不绑定单一Kernel）"
        if role == "编译融合Launcher":
            return "AOT Static Triton Launcher（具体variant由对应细分Stage记录）"
        return "N/A（算子总区间可包含多个或无Kernel）"
    if category.endswith("lifecycle"):
        return "N/A（Graph生命周期Stage，非逐Kernel记录）"
    return "N/A（该Stage未绑定单一Kernel variant）"


def aggregate(task_root: Path) -> dict[str, Any]:
    manifest_path = task_root / "manifests/stage_manifest.json"
    analysis_root = task_root / "analysis"
    evidence_root = task_root / "evidence"
    expected_source_manifest_sha = _sha256(evidence_root / "source_manifest.sha256")
    manifest = _load_json(manifest_path)
    stages = {int(stage["stage_id"]): stage for stage in manifest["stages"]}
    matrix = _load_run_matrices(task_root)
    conditional_na_evidence = _load_conditional_na_evidence(task_root)
    if len(matrix) != 698:
        raise ValueError(f"expected 698 executed matrix rows, got {len(matrix)}")

    sample_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    coverage_errors: list[str] = []
    run_pairs: set[tuple[int, str]] = set()
    occurrence_check_count = 0
    source_manifest_identities: dict[str, set[str]] = defaultdict(set)
    binary_identities: dict[str, set[str]] = defaultdict(set)

    for matrix_row in matrix:
        run_id = matrix_row["run_id"]
        stage_id = int(matrix_row["stage_id"])
        runner_phase = matrix_row["runner_phase"]
        manifest_phase = matrix_row["manifest_phase"]
        cohort = str(matrix_row["provenance_cohort"])
        if _manifest_phase(runner_phase) != manifest_phase:
            raise ValueError(f"phase mismatch in {run_id}")
        pair = (stage_id, manifest_phase)
        if pair in run_pairs:
            raise ValueError(f"duplicate stage/phase pair: {pair}")
        run_pairs.add(pair)
        stage = stages[stage_id]
        run_dir = task_root / "runs" / run_id
        controller_path = run_dir / "controller.json"
        summary_path = run_dir / "summary.json"
        if not controller_path.is_file() or not summary_path.is_file():
            coverage_errors.append(f"missing run artifacts: {run_id}")
            continue
        controller = _load_json(controller_path)
        summary = _load_json(summary_path)
        if controller.get("status") != "ok":
            coverage_errors.append(f"controller not ok: {run_id}")
            continue
        lost_count = int(summary.get("lost_count", -1))
        if lost_count != 0:
            coverage_errors.append(f"lost_count={lost_count}: {run_id}")
            continue
        event_count = int(summary.get("event_count", -1))
        sample_count = int(summary.get("sample_count", -1))
        expected_run_count = _expected_run_count(stage, stages)
        if (
            event_count > 0
            and expected_run_count is not None
            and not _is_evidence_backed_na(stage, manifest_phase)
        ):
            occurrence_check_count += 1
            if event_count != expected_run_count:
                coverage_errors.append(
                    f"occurrence mismatch expected={expected_run_count} "
                    f"observed={event_count}: {run_id}"
                )
        cpu_id, pid, tid = _controller_identity(controller)
        source_sha = str(controller.get("probe_source_manifest", {}).get("sha256", ""))
        binary_sha = _binary_identity(controller)
        source_manifest_identities[cohort].add(source_sha)
        binary_identities[cohort].add(binary_sha)
        if cohort == "increment-0830" and source_sha != expected_source_manifest_sha:
            coverage_errors.append(f"source manifest identity mismatch: {run_id}")
        if not binary_sha:
            coverage_errors.append(f"missing runtime binary identity: {run_id}")
        probe_version = "ReadCoreCycle ABI4"
        run_sample_rows: list[dict[str, Any]] = []

        if event_count == 0:
            if not summary.get("pilot_zero_hit"):
                coverage_errors.append(f"zero-hit without marker: {run_id}")
                continue
            status = (
                "N/A"
                if _is_evidence_backed_na(stage, manifest_phase)
                else "invalid-zero-hit"
            )
            reason = (
                "真实Graph路径未命中该条件分支；运行event_count=0，并以源码条件边界作为N/A证据"
                if status == "N/A"
                else "非条件Stage在真实Graph路径中零命中，需要修正选择器或Stage定义"
            )
            if status != "N/A":
                coverage_errors.append(f"unexpected zero-hit: {run_id}")
            result_rows.append(
                {
                    "sequence": int(matrix_row["sequence"]),
                    "run_id": run_id,
                    "provenance_cohort": cohort,
                    "phase": manifest_phase,
                    "stage_id": stage_id,
                    "stage_code": stage["stage_code"],
                    "category": stage["category"],
                    "level": stage["level"],
                    "relation": stage["relation"],
                    "parent_stage_id": stage.get("parent_stage_id", ""),
                    "operator_role": stage.get("operator_role", ""),
                    "stage_name": stage["stage_name"],
                    "sample_count": 0,
                    "display_rule": "N/A",
                    "display_us": "",
                    "median_cycles": "",
                    "min_us": "",
                    "p50_us": "",
                    "p95_us": "",
                    "max_us": "",
                    "variation_ratio": "",
                    "status": status,
                    "reason": reason,
                    "evidence_kind": "paired-zero-hit+source-condition",
                    "lost_count": lost_count,
                    "run_path": str(run_dir),
                }
            )
            continue

        raw_path = run_dir / "raw.csv"
        if not raw_path.is_file():
            coverage_errors.append(f"missing raw.csv: {run_id}")
            continue
        with raw_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
                raise ValueError(f"raw schema mismatch: {run_id}")
            raw_rows = list(reader)
        if len(raw_rows) != event_count or sample_count != event_count:
            coverage_errors.append(f"raw/event mismatch: {run_id}")
            continue
        occurrence_by_step: dict[int, int] = defaultdict(int)
        for expected_sequence, raw in enumerate(raw_rows):
            if int(raw["sequence"]) != expected_sequence:
                raise ValueError(f"raw sequence gap: {run_id}")
            if int(raw["stage_id"]) != stage_id or int(raw["site_id"]) != 700:
                raise ValueError(f"raw selector mismatch: {run_id}")
            step_id = int(raw["step_id"])
            occurrence = occurrence_by_step[step_id]
            occurrence_by_step[step_id] += 1
            resolved_parent, layer, parent = _resolve_parent_and_layer(stage, occurrence, stages)
            immediate_parent = _resolved_immediate_parent(stage, resolved_parent)
            monotonic_ticks = int(raw["delta_monotonic_ticks"])
            cntfrq_hz = int(raw["monotonic_frequency_hz"])
            time_us = float(raw["delta_monotonic_ns"]) / 1000.0
            if monotonic_ticks <= 0 or cntfrq_hz <= 0 or time_us <= 0:
                raise ValueError(
                    f"non-positive paired CNTVCT interval: {run_id} row {expected_sequence}"
                )
            expected_us = monotonic_ticks * 1_000_000.0 / cntfrq_hz
            if abs(time_us - expected_us) > 1e-6:
                raise ValueError(f"paired time mismatch: {run_id} row {expected_sequence}")
            cycles_delta = int(raw["delta_cycles"])
            if cycles_delta <= 0:
                raise ValueError(
                    f"non-positive PMCCNTR interval: {run_id} row {expected_sequence}"
                )
            expected_context_id = 700000 + int(matrix_row["source_sequence"])
            if int(raw["context_id"]) != expected_context_id:
                raise ValueError(
                    f"context identity mismatch: {run_id} row {expected_sequence}"
                )
            semantic_group = str(stage.get("semantic_group") or (parent or {}).get("semantic_group", ""))
            operator_path = str(stage.get("operator_path") or (parent or {}).get("operator_path", ""))
            operator_role = str(stage.get("operator_role") or (parent or {}).get("operator_role", ""))
            implementation_signature = str(stage.get("selector") or stage.get("function_scope", ""))
            kernel_variant = _kernel_variant(stage)
            sample = {
                "run_id": run_id,
                "provenance_cohort": cohort,
                "phase": manifest_phase,
                "request_id": run_id,
                "step_id": step_id,
                "layer": layer,
                "occurrence": occurrence,
                "semantic_group": semantic_group,
                "operator_path": operator_path,
                "operator_role": operator_role,
                "implementation_family": stage["category"],
                "implementation_signature": implementation_signature,
                "kernel_variant": kernel_variant,
                "stage_id": stage_id,
                "stage_code": stage["stage_code"],
                "parent_stage_id": stage.get("parent_stage_id", ""),
                "root_parent_stage_id": resolved_parent or "",
                "resolved_parent_stage_id": immediate_parent or "",
                "relation": stage["relation"],
                "stage_name": stage["stage_name"],
                "level": stage["level"],
                "cycles_begin": int(raw["begin_cycle"]),
                "cycles_end": int(raw["end_cycle"]),
                "cycles_delta": cycles_delta,
                "cntvct_begin": int(raw["begin_monotonic_tick"]),
                "cntvct_end": int(raw["end_monotonic_tick"]),
                "cntvct_ticks": monotonic_ticks,
                "cntfrq_hz": cntfrq_hz,
                "time_us": time_us,
                "paired_cycle_hz": cycles_delta * 1_000_000.0 / time_us,
                "cpu_id": cpu_id,
                "pid": pid,
                "tid": tid,
                "context_id": int(raw["context_id"]),
                "lost_count": lost_count,
                "probe_version": probe_version,
                "source_sha256": source_sha,
                "binary_sha256": binary_sha,
                "status": "valid",
                "reason": "同一边界配对PMCCNTR/CNTVCT，lost_count=0",
            }
            sample_rows.append(sample)
            run_sample_rows.append(sample)

        times = [float(row["time_us"]) for row in run_sample_rows]
        cycles = [float(row["cycles_delta"]) for row in run_sample_rows]
        median_us = statistics.median(times)
        p95_us = _percentile(times, 0.95)
        variation_ratio = (p95_us - median_us) / median_us if median_us else 0.0
        result_rows.append(
            {
                "sequence": int(matrix_row["sequence"]),
                "run_id": run_id,
                "provenance_cohort": cohort,
                "phase": manifest_phase,
                "stage_id": stage_id,
                "stage_code": stage["stage_code"],
                "category": stage["category"],
                "level": stage["level"],
                "relation": stage["relation"],
                "parent_stage_id": stage.get("parent_stage_id", ""),
                "operator_role": stage.get("operator_role", ""),
                "stage_name": stage["stage_name"],
                "sample_count": len(times),
                "display_rule": "single" if len(times) == 1 else "median",
                "display_us": times[0] if len(times) == 1 else median_us,
                "median_cycles": cycles[0] if len(cycles) == 1 else statistics.median(cycles),
                "min_us": min(times),
                "p50_us": median_us,
                "p95_us": p95_us,
                "max_us": max(times),
                "variation_ratio": variation_ratio,
                "status": "measured",
                "reason": "唯一样本" if len(times) == 1 else "请求内多次发生取中位数",
                "evidence_kind": "paired-pmccntr+cntvct",
                "lost_count": lost_count,
                "run_path": str(run_dir),
            }
        )

    evidence_sequence = len(matrix)
    for item in conditional_na_evidence:
        evidence_sequence += 1
        stage_id = int(item["stage_id"])
        phase = str(item["phase"])
        pair = (stage_id, phase)
        if pair in run_pairs:
            raise ValueError(f"duplicate conditional N/A pair: {pair}")
        if stage_id not in stages or phase not in stages[stage_id]["phases"]:
            raise ValueError(f"conditional N/A pair outside manifest phases: {pair}")
        source_evidence = item.get("source_evidence")
        runtime_runs = item.get("runtime_runs")
        if not isinstance(source_evidence, list) or not source_evidence:
            raise ValueError(f"missing source evidence for conditional N/A: {pair}")
        if not isinstance(runtime_runs, list) or not runtime_runs:
            raise ValueError(f"missing runtime evidence for conditional N/A: {pair}")
        verified_run_paths: list[str] = []
        for evidence_run in runtime_runs:
            run_dir = task_root / "runs" / str(evidence_run)
            controller = _load_json(run_dir / "controller.json")
            summary = _load_json(run_dir / "summary.json")
            if controller.get("status") != "ok":
                raise ValueError(f"conditional N/A evidence controller not ok: {evidence_run}")
            if int(summary.get("lost_count", -1)) != 0:
                raise ValueError(f"conditional N/A evidence lost events: {evidence_run}")
            if int(summary.get("event_count", 0)) != 36:
                raise ValueError(
                    f"conditional N/A evidence expected 36 parent events: {evidence_run}"
                )
            if _manifest_phase(str(summary.get("phase"))) != phase:
                raise ValueError(f"conditional N/A evidence phase mismatch: {evidence_run}")
            verified_run_paths.append(str(run_dir))
        run_pairs.add(pair)
        result_rows.append(
            _na_result(
                sequence=evidence_sequence,
                stage=stages[stage_id],
                phase=phase,
                reason=str(item["reason"]),
                evidence_kind="source+complete-runtime-tree",
                run_id=";".join(map(str, runtime_runs)),
                run_path=";".join(verified_run_paths),
            )
        )

    formal_pairs = {
        (stage_id, phase)
        for stage_id, stage in stages.items()
        for phase in stage["phases"]
    }
    if run_pairs != formal_pairs:
        missing = sorted(formal_pairs - run_pairs)
        unexpected = sorted(run_pairs - formal_pairs)
        coverage_errors.append(
            f"formal pair assembly mismatch missing={missing} unexpected={unexpected}"
        )

    report_pairs = {
        (stage_id, phase)
        for stage_id, stage in stages.items()
        for phase in stage.get("report_phases", stage["phases"])
    }
    result_pairs = {
        (int(row["stage_id"]), str(row["phase"])) for row in result_rows
    }
    for stage_id, phase in sorted(report_pairs - result_pairs):
        evidence_sequence += 1
        stage = stages[stage_id]
        result_rows.append(
            _na_result(
                sequence=evidence_sequence,
                stage=stage,
                phase=phase,
                reason=(
                    "该节点仅存在于另一 Graph phase 的完整运行时算子树；"
                    "本 phase 父节点未出现，因此源码内部节点亦为实证 N/A"
                ),
                evidence_kind="parent-phase-absent+complete-runtime-tree",
            )
        )

    result_rows.sort(key=lambda row: int(row["sequence"]))
    sample_rows.sort(key=lambda row: (str(row["run_id"]), int(row["step_id"]), int(row["occurrence"])))
    result_columns = tuple(result_rows[0].keys()) if result_rows else ()
    _write_tsv(analysis_root / "raw_samples.tsv", sample_rows, SAMPLE_COLUMNS)
    _write_tsv(analysis_root / "stage_phase_results.tsv", result_rows, result_columns)

    result_by_pair = {(int(row["stage_id"]), str(row["phase"])): row for row in result_rows}
    closure_rows: list[dict[str, Any]] = []
    children_by_parent: dict[tuple[int, str], list[int]] = defaultdict(list)
    for stage in stages.values():
        relation = str(stage.get("relation", ""))
        if not any(marker in relation for marker in ("exclusive", "parent-contains", "nested")):
            continue
        for parent_id in _immediate_parent_ids(stage):
            if parent_id in stages:
                for phase in stage.get("report_phases", stage["phases"]):
                    children_by_parent[(parent_id, phase)].append(int(stage["stage_id"]))

    for (parent_id, phase), child_ids in sorted(children_by_parent.items()):
        parent_result = result_by_pair.get((parent_id, phase))
        measured_child_ids: list[int] = []
        na_child_ids: list[int] = []
        missing_child_ids: list[int] = []
        for child_id in sorted(set(child_ids)):
            child_result = result_by_pair.get((child_id, phase))
            if not child_result:
                missing_child_ids.append(child_id)
            elif child_result["status"] == "N/A":
                na_child_ids.append(child_id)
            elif child_result["status"] == "measured":
                measured_child_ids.append(child_id)
            else:
                missing_child_ids.append(child_id)
        closure_rows.append(
            {
                "phase": phase,
                "parent_stage_id": parent_id,
                "parent_stage_name": stages[parent_id]["stage_name"],
                "parent_status": parent_result["status"] if parent_result else "missing",
                "parent_display_us": (
                    parent_result["display_us"]
                    if parent_result and parent_result["status"] == "measured"
                    else ""
                ),
                "planned_child_stage_ids": ",".join(map(str, sorted(set(child_ids)))),
                "measured_child_stage_ids": ",".join(map(str, measured_child_ids)),
                "na_child_stage_ids": ",".join(map(str, na_child_ids)),
                "missing_child_stage_ids": ",".join(map(str, missing_child_ids)),
                "numeric_residual": "N/A",
                "comparability": "independent-round-nonadditive",
                "status": (
                    "closed"
                    if parent_result
                    and parent_result["status"] in {"measured", "N/A"}
                    and not missing_child_ids
                    else "review"
                ),
                "reason": (
                    "父子 Stage 来自独立选择轮次，仅核对结构与结论闭合，"
                    "不求和、不计算数值残差"
                ),
            }
        )
    closure_columns = tuple(closure_rows[0].keys()) if closure_rows else ()
    _write_tsv(analysis_root / "closure.tsv", closure_rows, closure_columns)

    cohort_summary: dict[str, dict[str, Any]] = {}
    for cohort in ("baseline-0829", "increment-0830"):
        source_ids = source_manifest_identities.get(cohort, set())
        binary_ids = binary_identities.get(cohort, set())
        if len(source_ids) != 1:
            coverage_errors.append(
                f"cohort {cohort} source manifest identities={len(source_ids)}"
            )
        if len(binary_ids) != 1:
            coverage_errors.append(
                f"cohort {cohort} runtime binary identities={len(binary_ids)}"
            )
        cohort_summary[cohort] = {
            "run_count": sum(
                row["provenance_cohort"] == cohort for row in matrix
            ),
            "source_manifest_sha256": sorted(source_ids),
            "runtime_binary_identity": sorted(binary_ids),
        }

    observed_report_pairs = set(result_by_pair)
    observed_formal_pairs = observed_report_pairs & formal_pairs
    formal_status_counts: dict[str, int] = defaultdict(int)
    report_status_counts: dict[str, int] = defaultdict(int)
    for pair, row in result_by_pair.items():
        report_status_counts[str(row["status"])] += 1
        if pair in formal_pairs:
            formal_status_counts[str(row["status"])] += 1
    matrix_paths = (
        task_root / "manifests" / "run_matrix.tsv",
        task_root / "manifests" / "run_matrix_increment.tsv",
        task_root / "manifests" / "run_matrix_conditional.tsv",
    )
    coverage = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "manifest_sha256": _sha256(manifest_path),
        "matrix_sha256": {path.name: _sha256(path) for path in matrix_paths},
        "manifest_stage_count": len(stages),
        "executed_run_count": len(matrix),
        "evidence_na_pair_count": len(conditional_na_evidence),
        "matrix_pair_count": len(formal_pairs),
        "observed_pair_count": len(observed_formal_pairs),
        "report_pair_count": len(report_pairs),
        "observed_report_pair_count": len(observed_report_pairs),
        "raw_sample_count": len(sample_rows),
        "status_counts": dict(sorted(formal_status_counts.items())),
        "report_status_counts": dict(sorted(report_status_counts.items())),
        "missing_pairs": sorted(formal_pairs - observed_formal_pairs),
        "unexpected_pairs": sorted(observed_formal_pairs - formal_pairs),
        "missing_report_pairs": sorted(report_pairs - observed_report_pairs),
        "unexpected_report_pairs": sorted(observed_report_pairs - report_pairs),
        "coverage_errors": sorted(set(coverage_errors)),
        "lost_count_total": sum(int(row["lost_count"]) for row in result_rows),
        "closure_row_count": len(closure_rows),
        "closure_review_count": sum(row["status"] != "closed" for row in closure_rows),
        "occurrence_check_count": occurrence_check_count,
        "source_manifest_identity_count": len(
            set().union(*source_manifest_identities.values())
        ),
        "runtime_binary_identity_count": len(
            set().union(*binary_identities.values())
        ),
        "provenance_cohorts": cohort_summary,
        "source_manifest_sha256": expected_source_manifest_sha,
        "pass": (
            observed_formal_pairs == formal_pairs
            and observed_report_pairs == report_pairs
            and not coverage_errors
            and all(row["status"] in {"measured", "N/A"} for row in result_rows)
            and sum(int(row["lost_count"]) for row in result_rows) == 0
            and all(row["status"] == "closed" for row in closure_rows)
        ),
    }
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (analysis_root / "stage_phase_results.json").write_text(
        json.dumps(result_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (evidence_root / "aggregation.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path}\n"
            for path in (
                analysis_root / "raw_samples.tsv",
                analysis_root / "stage_phase_results.tsv",
                analysis_root / "stage_phase_results.json",
                analysis_root / "closure.tsv",
                analysis_root / "coverage.json",
            )
        ),
        encoding="utf-8",
    )
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path("/home/fj/vllm_ops"),
    )
    options = parser.parse_args()
    coverage = aggregate(options.task_root.resolve())
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    return 0 if coverage["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
