#!/usr/bin/env python3
"""Expand the immutable stage manifest into one minimal run per phase."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PHASE_MAP = {
    "graph-capture": "graph-capture-full",
    "graph-prefill": "graph-prefill",
    "steady-replay": "graph-replay",
}
PHASE_ORDER = {
    "graph-replay": 0,
    "graph-prefill": 1,
    "graph-capture-full": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Include only this category; repeat for multiple categories.",
    )
    parser.add_argument("--stage-id-min", type=int)
    parser.add_argument("--stage-id-max", type=int)
    parser.add_argument("--run-prefix", default="stage0829")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, str | int]] = []
    for stage in manifest["stages"]:
        stage_id = int(stage["stage_id"])
        if args.categories and stage["category"] not in args.categories:
            continue
        if args.stage_id_min is not None and stage_id < args.stage_id_min:
            continue
        if args.stage_id_max is not None and stage_id > args.stage_id_max:
            continue
        for manifest_phase in stage["phases"]:
            runner_phase = PHASE_MAP[manifest_phase]
            rows.append(
                {
                    "stage_id": stage_id,
                    "stage_code": stage["stage_code"],
                    "category": stage["category"],
                    "manifest_phase": manifest_phase,
                    "runner_phase": runner_phase,
                    "stage_name": stage["stage_name"],
                }
            )
    rows.sort(
        key=lambda row: (
            PHASE_ORDER[str(row["runner_phase"])],
            int(row["stage_id"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        runner_phase = str(row["runner_phase"])
        stage_id = int(row["stage_id"])
        row["sequence"] = index
        row["run_id"] = (
            f"{args.run_prefix}_{index:04d}_{runner_phase}_stage{stage_id}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "run_id",
        "stage_id",
        "stage_code",
        "category",
        "manifest_phase",
        "runner_phase",
        "stage_name",
    ]
    with args.output.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} phase-stage runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
