#!/usr/bin/env bash
# WidowX / Bridge (SimplerEnv) evaluation: 4 tasks x 24 episodes per checkpoint.
# The GPU holds the policy server plus --workers simulator processes that
# render on the same card; SAPIEN is CPU-bound at that size, so stacking them is
# a strict win. Keep --workers at 8 or below.
#
#   bash evaluation/bridge/run_eval.sh --ckpt runs/bridge/ckpt-100000
#   bash evaluation/bridge/run_eval.sh --ckpt runs/bridge/ckpt-100000 --gpu 1
#
# Shared options and the result layout are described in evaluation/eval_common.sh
# and printed by --help.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../eval_common.sh
source "$SCRIPT_DIR/../eval_common.sh"
# shellcheck source=../simpler_common.sh
source "$SCRIPT_DIR/../simpler_common.sh"

BENCH="Bridge"
CLIENT_ENV="simvla2-simpler-eval"
SERVER_SCRIPT="$SCRIPT_DIR/serve_simvla2_bridge.py"
DEFAULT_NORM_STATS="$REPO_ROOT/norm_stats/bridge.json"
DEFAULT_PORT=8200
DEFAULT_DTYPE=bf16
DEFAULT_FLOW_STEPS=30

PROTOCOL_TASKS="widowx_stack_cube,widowx_put_eggplant_in_basket,widowx_carrot_on_plate,widowx_spoon_on_towel"
PROTOCOL_EPISODES=24
TASKS_STR="$PROTOCOL_TASKS"
EPISODES="$PROTOCOL_EPISODES"
WORKERS=8

bench_usage() {
    cat <<EOF
  --tasks LIST        comma-separated tasks (default: all four)
  --episodes N        episodes per task (default: $PROTOCOL_EPISODES)
  --workers N         simulator processes per task (default: 8)
  SIMPLER_DIR (env)   patched SimplerEnv checkout (default: third_party/SimplerEnv)
EOF
}

bench_parse_arg() {
    case "$1" in
        --tasks)    TASKS_STR="${2-}"; return 2 ;;
        --episodes) EPISODES="${2-}"; return 2 ;;
        --workers)  WORKERS="${2-}"; return 2 ;;
    esac
    return 0
}

bench_prepare() {
    [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || _eval_die "--episodes must be a positive integer"
    [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || _eval_die "--workers must be a positive integer"
    IFS=',' read -r -a TASK_LIST <<<"$TASKS_STR"
    if [[ "$TASKS_STR" != "$PROTOCOL_TASKS" || "$EPISODES" != "$PROTOCOL_EPISODES" ]]; then
        FULL_PROTOCOL=false
    fi
    simpler_prepare "$GPU"
}

bench_describe() {
    echo "  tasks       : ${TASK_LIST[*]} x $EPISODES episodes, $WORKERS workers"
    echo "  simpler     : $SIMPLER_DIR"
}

bench_eval() {
    local ckpt="$1" out="$2" gpu="$3" port="$4"
    eval_start_server "$ckpt" "$gpu" "$port" "$out/logs/server.log" || return 1

    local per_worker=$(( (EPISODES + WORKERS - 1) / WORKERS ))
    local name task w start end pid failed=0
    name="$(basename "$ckpt")"
    for task in "${TASK_LIST[@]}"; do
        mkdir -p "$out/$task"
        local pids=()
        for ((w = 0; w < WORKERS; w++)); do
            start=$((w * per_worker))
            end=$((start + per_worker))
            (( end > EPISODES )) && end=$EPISODES
            (( start >= end )) && break
            simpler_run_client "$gpu" "bridge-$name-$task-w$w" "$out/logs/$task-worker$w.log" \
                bridge_client.py --host 127.0.0.1 --port "$port" \
                --task "$task" --output_dir "$out/$task" \
                --episode_start "$start" --episode_end "$end" \
                --num_episodes "$EPISODES" --cuda_device 0 --resume &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || failed=1
        done
    done
    eval_stop_server "$port"

    python3 - "$out" "$EPISODES" "${TASK_LIST[@]}" <<'PY' || return 1
import json
import sys
from pathlib import Path

out, episodes, tasks = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
groups = {}
for task in tasks:
    seen = {}
    path = out / task / f"{task}_results.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen[r["episode"]] = bool(r["success"])
    success = sum(seen.values())
    groups[task] = {"success": success, "total": len(seen),
                    "rate": success / len(seen) if seen else 0.0}
success = sum(g["success"] for g in groups.values())
total = sum(g["total"] for g in groups.values())
summary = {"checkpoint": out.name, "success": success, "total": total,
           "rate": success / total if total else 0.0, "groups": groups}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
sys.exit(0 if total == episodes * len(tasks) else 3)
PY
    return "$failed"
}

eval_main "$@"
