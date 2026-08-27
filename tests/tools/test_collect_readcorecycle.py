import csv
import json

from tools.collect_readcorecycle import _stage_identity, summarize_raw


def test_qwen3_norm_occurrences_map_to_semantic_roles() -> None:
    assert _stage_identity(202, 0) == {
        "operator_role": "q_norm",
        "layer_idx": 0,
        "occurrence_idx": 0,
        "stage_group": "total",
    }
    assert _stage_identity(202, 1)["operator_role"] == "k_norm"
    assert _stage_identity(202, 70)["layer_idx"] == 35
    assert _stage_identity(202, 72)["operator_role"] == "final_norm"
    assert _stage_identity(212, 0)["operator_role"] == "input_layernorm"
    assert _stage_identity(212, 1)["operator_role"] == "post_attention_layernorm"
    assert _stage_identity(212, 70)["layer_idx"] == 35
    assert _stage_identity(212, 71)["operator_role"] == "post_attention_layernorm"


def test_auxiliary_copy_and_generated_total_have_fixed_identity() -> None:
    assert _stage_identity(402, 4) == {
        "operator_role": "auxiliary_device_copy",
        "layer_idx": -1,
        "occurrence_idx": 4,
        "stage_group": "total",
    }
    assert _stage_identity(582, 0)["stage_group"] == "total"


def test_public_operator_stage_is_full_dispatch() -> None:
    assert _stage_identity(205, 0)["stage_group"] == "full_dispatch"
    assert _stage_identity(107, 0)["stage_group"] == "full_dispatch"
    assert _stage_identity(48, 0)["stage_group"] == "full_dispatch"


def test_summarize_raw_keeps_each_occurrence_and_adds_identity(tmp_path) -> None:
    raw_path = tmp_path / "raw.csv"
    summary_path = tmp_path / "summary.json"
    derived_path = tmp_path / "derived.csv"
    raw_path.write_text(
        "sequence,begin_cycle,end_cycle,delta_cycles,context_id,site_id,stage_id\n"
        "0,100,140,40,7,700,202\n"
        "1,200,260,60,7,700,202\n",
        encoding="utf-8",
    )

    summarize_raw(
        raw_path,
        summary_path,
        derived_path,
        cpu_frequency_hz=2_000_000_000,
        pmcr_divider=1,
        selected_count=2,
        event_count=2,
        lost_count=0,
        phase="eager-decode",
        step=3,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["phase"] == "eager-decode"
    assert summary["stage_id"] == 202
    with derived_path.open(newline="", encoding="utf-8") as derived_file:
        rows = list(csv.DictReader(derived_file))
    assert len(rows) == 2
    assert rows[0]["operator_role"] == "q_norm"
    assert rows[1]["operator_role"] == "k_norm"
    assert rows[1]["layer_idx"] == "0"
