#!/usr/bin/env bash
# Google Robot / Fractal (SimplerEnv) evaluation, variant-aggregation (VA) protocol:
# coke_can 33 + move_near 10 + open_close 42 scenarios = 825 + 600 + 378 = 1803
# episodes per checkpoint, gripper threshold 0.5, zero proprio. The GPU holds
# the policy server and the SAPIEN workers of one task at a time;
# --render-gpu moves rendering to a second card.
#
#   bash evaluation/fractal/run_eval.sh --ckpt runs/fractal/ckpt-200000
#   bash evaluation/fractal/run_eval.sh --ckpt runs/fractal/ckpt-200000 --gpu 0 --render-gpu 1
#
# Shared options and the result layout are described in evaluation/eval_common.sh
# and printed by --help.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../eval_common.sh
source "$SCRIPT_DIR/../eval_common.sh"
# shellcheck source=../simpler_common.sh
source "$SCRIPT_DIR/../simpler_common.sh"

BENCH="Fractal"
CLIENT_ENV="simvla2-simpler-eval"
SERVER_SCRIPT="$SCRIPT_DIR/serve_simvla2_fractal.py"
DEFAULT_NORM_STATS="$REPO_ROOT/norm_stats/fractal.json"
DEFAULT_PORT=8400
DEFAULT_DTYPE=bf16
DEFAULT_FLOW_STEPS=30

PROTOCOL_TASKS="coke_can,move_near,open_close"
declare -A SCENARIOS=([coke_can]=33 [move_near]=10 [open_close]=42)
declare -A EXPECTED_EPISODES=([coke_can]=825 [move_near]=600 [open_close]=378)
TASKS_STR="$PROTOCOL_TASKS"
WORKERS=4
EPISODE_IDS=""
GRIPPER_THRESHOLD=0.5
RENDER_GPU=""

bench_usage() {
    cat <<EOF
  --tasks LIST        comma-separated tasks (default: $PROTOCOL_TASKS)
  --workers N         simulator processes per task (default: 4)
  --episode-ids LIST  only these episode ids within each move_near scenario
                      (smoke runs; the other tasks ignore it)
  --gripper-threshold F
                      gripper open/close threshold (default: 0.5)
  --render-gpu ID     render GPU (default: the --gpu GPU)
  SIMPLER_DIR (env)   patched SimplerEnv checkout (default: third_party/SimplerEnv)
EOF
}

bench_parse_arg() {
    case "$1" in
        --tasks)             TASKS_STR="${2-}"; return 2 ;;
        --workers)           WORKERS="${2-}"; return 2 ;;
        --episode-ids)       EPISODE_IDS="${2-}"; return 2 ;;
        --gripper-threshold) GRIPPER_THRESHOLD="${2-}"; return 2 ;;
        --render-gpu)        RENDER_GPU="${2-}"; return 2 ;;
    esac
    return 0
}

bench_prepare() {
    [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || _eval_die "--workers must be a positive integer"
    IFS=',' read -r -a TASK_LIST <<<"$TASKS_STR"
    local task
    for task in "${TASK_LIST[@]}"; do
        [[ -n "${SCENARIOS[$task]:-}" ]] || _eval_die "unknown task: $task"
    done
    if [[ "$TASKS_STR" != "$PROTOCOL_TASKS" || -n "$EPISODE_IDS" || "$GRIPPER_THRESHOLD" != 0.5 ]]; then
        FULL_PROTOCOL=false
    fi
    RENDER_GPU="${RENDER_GPU:-$GPU}"
    [[ "$RENDER_GPU" != *,* ]] || _eval_die "--render-gpu takes a single GPU id"
    simpler_prepare "$RENDER_GPU"
}

bench_describe() {
    echo "  tasks       : ${TASK_LIST[*]} (VA), $WORKERS workers, gripper threshold $GRIPPER_THRESHOLD"
    echo "  render gpu  : $RENDER_GPU"
    echo "  simpler     : $SIMPLER_DIR"
}

bench_eval() {
    local ckpt="$1" out="$2" gpu="$3" port="$4"
    eval_start_server "$ckpt" "$gpu" "$port" "$out/logs/server.log" || return 1

    local episode_args=()
    [[ -n "$EPISODE_IDS" ]] && episode_args=(--episode_ids "$EPISODE_IDS")
    local name task count workers per_worker w start end pid failed=0 pids
    name="$(basename "$ckpt")"
    # One task at a time. Running the three tasks against one server at once
    # scores the same, but the SAPIEN workers already saturate the GPU, so it
    # saves little time.
    for task in "${TASK_LIST[@]}"; do
        count="${SCENARIOS[$task]}"
        workers=$(( WORKERS < count ? WORKERS : count ))
        per_worker=$(( (count + workers - 1) / workers ))
        pids=()
        for ((w = 0; w < workers; w++)); do
            start=$((w * per_worker))
            end=$((start + per_worker))
            (( end > count )) && end=$count
            (( start >= end )) && continue
            simpler_run_client "$RENDER_GPU" "fractal-$name-$task-w$w" "$out/logs/$task-worker$w.log" \
                fractal_client.py --host 127.0.0.1 --port "$port" \
                --task "$task" --output_dir "$out" \
                --scenario_start "$start" --scenario_end "$end" \
                --gripper_threshold "$GRIPPER_THRESHOLD" --resume \
                ${episode_args[@]+"${episode_args[@]}"} &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || failed=1
        done
    done
    eval_stop_server "$port"

    local expected_json
    expected_json="{$(for task in "${TASK_LIST[@]}"; do printf '"%s": %s,' "$task" "${EXPECTED_EPISODES[$task]}"; done | sed 's/,$//')}"
    # An --episode-ids subset cannot be checked against the scenario totals.
    python3 - "$out" "$expected_json" "${EPISODE_IDS:+subset}" <<'PY' || return 1
import json
import sys
from pathlib import Path

out, expected, subset = Path(sys.argv[1]), json.loads(sys.argv[2]), bool(sys.argv[3])
groups = {}
for task in expected:
    seen = {}
    path = out / "VA" / task / "google_results.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen[(r.get("scenario"), r.get("episode_id"))] = bool(r.get("done"))
    success = sum(seen.values())
    groups[task] = {"success": success, "total": len(seen),
                    "rate": success / len(seen) if seen else 0.0}
success = sum(g["success"] for g in groups.values())
total = sum(g["total"] for g in groups.values())
# The tasks have different episode counts, so the pooled rate and the mean of
# the per-task rates differ; mean_rate is the reported number.
summary = {"checkpoint": out.name, "success": success, "total": total,
           "rate": success / total if total else 0.0,
           "mean_rate": sum(g["rate"] for g in groups.values()) / len(groups),
           "groups": groups}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
sys.exit(0 if subset or all(groups[t]["total"] == n for t, n in expected.items()) else 3)
PY
    return "$failed"
}

eval_main "$@"
