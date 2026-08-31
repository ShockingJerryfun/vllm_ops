#!/usr/bin/env bash
set -euo pipefail

task_root=/home/fj/vllm_ops
matrix=${RCC_RUN_MATRIX:-$task_root/manifests/run_matrix.tsv}
runner=$task_root/tools/run_stage.sh
status_file=${RCC_STATUS_FILE:-$task_root/evidence/matrix_status.tsv}
task_log_root=${RCC_TASK_LOG_ROOT:-$task_root/logs/tasks}
start_sequence=${MATRIX_START_SEQUENCE:-1}
end_sequence=${MATRIX_END_SEQUENCE:-999999}

validate_run() {
  local stage_id=$1
  local summary=$2
  local manifest_phase=$3
  local lost_count event_count summary_stage summary_phase canonical_summary_phase
  local expected_count conditional
  lost_count=$(jq -r '.lost_count' "$summary")
  event_count=$(jq -r '.event_count' "$summary")
  summary_stage=$(jq -r '.stage_id' "$summary")
  summary_phase=$(jq -r '.phase' "$summary")
  canonical_summary_phase=$summary_phase
  if [ "$summary_phase" = graph-capture-full ]; then
    canonical_summary_phase=graph-capture
  fi
  if [ "$summary_stage" -ne "$stage_id" ] || [ "$canonical_summary_phase" != "$manifest_phase" ]; then
    echo "matrix validation failed: expected stage/phase $stage_id/$manifest_phase, got $summary_stage/$summary_phase" >&2
    return 41
  fi
  if [ "$lost_count" -ne 0 ]; then
    echo "matrix validation failed: stage $stage_id lost_count=$lost_count" >&2
    return 42
  fi
  expected_count=$(jq -r --argjson id "$stage_id" '
      first(.stages[] | select(.stage_id == $id)).expected_occurrence |
      tostring | capture("(?<count>[0-9]+)").count
    ' "$task_root/manifests/stage_manifest.json")
  conditional=$(jq -r --argjson id "$stage_id" --arg phase "$manifest_phase" '
      first(.stages[] | select(.stage_id == $id)) as $stage |
      (($stage.expected_status_by_phase[$phase] // "") |
       tostring | ascii_downcase |
       test("n/a|otherwise|measured if|absent"))
    ' "$task_root/manifests/stage_manifest.json")
  if [ "$event_count" -eq 0 ] && [ "$conditional" = true ]; then
    return 0
  fi
  if [ "$event_count" -ne "$expected_count" ]; then
    echo "matrix validation failed: stage $stage_id expected $expected_count events, got $event_count" >&2
    return 43
  fi
  return 0
}

mkdir -p "$task_log_root" "$(dirname "$status_file")"
if [ ! -f "$status_file" ]; then
  printf 'sequence\trun_id\tstage_id\tphase\tstatus\tduration_s\n' > "$status_file"
fi

tail -n +2 "$matrix" | while IFS=$'\t' read -r \
  sequence run_id stage_id stage_code category manifest_phase runner_phase stage_name; do
  : "$stage_code" "$category" "$manifest_phase" "$stage_name"
  if [ "$sequence" -lt "$start_sequence" ] || [ "$sequence" -gt "$end_sequence" ]; then
    continue
  fi
  run_dir=$task_root/runs/$run_id
  controller=$run_dir/controller.json
  summary=$run_dir/summary.json
  if [ -s "$controller" ] && [ -s "$summary" ] && \
      jq -e '.status == "ok"' "$controller" >/dev/null; then
    validate_run "$stage_id" "$summary" "$manifest_phase"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$sequence" "$run_id" "$stage_id" "$runner_phase" skipped-complete 0 \
      >> "$status_file"
    continue
  fi

  begin_seconds=$(date +%s)
  set +e
  RCC_PHASE=$runner_phase \
  RCC_STAGE_ID=$stage_id \
  RCC_CAPACITY=4096 \
  RCC_CONTEXT_ID=$((700000 + sequence)) \
  RCC_CACHE_TAG=graph_stage \
  RUN_ID=$run_id \
  "$runner" > "$task_log_root/$run_id.log" 2>&1
  run_status=$?
  set -e
  duration_seconds=$(( $(date +%s) - begin_seconds ))

  failure_status=$run_status
  if [ "$run_status" -eq 0 ]; then
    if [ ! -s "$controller" ] || [ ! -s "$summary" ]; then
      failure_status=44
    elif ! jq -e '.status == "ok"' "$controller" >/dev/null; then
      failure_status=45
    else
      set +e
      validate_run "$stage_id" "$summary" "$manifest_phase"
      validation_status=$?
      set -e
      if [ "$validation_status" -eq 0 ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$sequence" "$run_id" "$stage_id" "$runner_phase" complete \
          "$duration_seconds" >> "$status_file"
        continue
      fi
      failure_status=$validation_status
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$sequence" "$run_id" "$stage_id" "$runner_phase" "failed-$failure_status" \
    "$duration_seconds" >> "$status_file"
  echo "matrix stopped at $run_id with status $failure_status" >&2
  exit "$failure_status"
done
