#!/usr/bin/env bash
# LIBERO evaluation: 4 suites x 10 tasks x 50 trials = 2000 episodes per
# checkpoint. The GPU holds the policy server and the LIBERO client; the
# suites run one after another against that server (--parallel-suites runs
# them at once).
#
#   bash evaluation/libero/run_eval.sh --ckpt runs/libero/ckpt-200000
#   bash evaluation/libero/run_eval.sh --ckpt runs/libero/ckpt-200000 --gpu 1
#
# Shared options and the result layout are described in evaluation/eval_common.sh
# and printed by --help.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../eval_common.sh
source "$SCRIPT_DIR/../eval_common.sh"

BENCH="LIBERO"
CLIENT_ENV="simvla2-libero-eval"
SERVER_SCRIPT="$SCRIPT_DIR/serve_simvla2_libero.py"
DEFAULT_NORM_STATS="$REPO_ROOT/norm_stats/libero.json"
DEFAULT_PORT=8100
DEFAULT_DTYPE=bf16
DEFAULT_FLOW_STEPS=10

PROTOCOL_SUITES="libero_spatial,libero_object,libero_goal,libero_10"
PROTOCOL_TRIALS=50
SUITES_STR="$PROTOCOL_SUITES"
NUM_TRIALS="$PROTOCOL_TRIALS"
PARALLEL_SUITES=false
SAVE_VIDEO=true
LIBERO_DIR="${LIBERO_DIR:-$REPO_ROOT/third_party/LIBERO}"

bench_usage() {
    cat <<EOF
  --suites LIST       comma-separated suites (default: $PROTOCOL_SUITES)
  --num-trials N      trials per task (default: $PROTOCOL_TRIALS)
  --parallel-suites   run the suites at once against the server
  --no-video          do not save rollout videos
  LIBERO_DIR (env)    LIBERO checkout (default: third_party/LIBERO, see install_env.sh)
EOF
}

bench_parse_arg() {
    case "$1" in
        --suites)          SUITES_STR="${2-}"; return 2 ;;
        --num-trials)      NUM_TRIALS="${2-}"; return 2 ;;
        --parallel-suites) PARALLEL_SUITES=true; return 1 ;;
        --no-video)        SAVE_VIDEO=false; return 1 ;;
    esac
    return 0
}

bench_prepare() {
    [[ "$NUM_TRIALS" =~ ^[1-9][0-9]*$ ]] || _eval_die "--num-trials must be a positive integer"
    IFS=',' read -r -a SUITES <<<"$SUITES_STR"
    if [[ "$SUITES_STR" != "$PROTOCOL_SUITES" || "$NUM_TRIALS" != "$PROTOCOL_TRIALS" ]]; then
        FULL_PROTOCOL=false
    fi
    [[ -d "$LIBERO_DIR/libero" ]] || \
        _eval_die "LIBERO not found at $LIBERO_DIR; run evaluation/libero/install_env.sh or set LIBERO_DIR"
    LIBERO_DIR="$(cd "$LIBERO_DIR" && pwd)"

    # LIBERO resolves BDDL files and init states through
    # $LIBERO_CONFIG_PATH/config.yaml and otherwise falls back to ~/.libero,
    # which in a fresh env points nowhere. Seed one next to the results.
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$OUTPUT_BASE/.libero}"
    mkdir -p "$LIBERO_CONFIG_PATH"
    if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
        cat >"$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_DIR/libero/libero
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
init_states: $LIBERO_DIR/libero/libero/init_files
datasets: $LIBERO_DIR/libero/libero/datasets
assets: $LIBERO_DIR/libero/libero/assets
YAML
    fi
}

bench_describe() {
    echo "  suites      : ${SUITES[*]} x $NUM_TRIALS trials (parallel: $PARALLEL_SUITES)"
    echo "  libero      : $LIBERO_DIR (config $LIBERO_CONFIG_PATH/config.yaml)"
}

bench_eval() {
    local ckpt="$1" out="$2" gpu="$3" port="$4"
    eval_start_server "$ckpt" "$gpu" "$port" "$out/logs/server.log" || return 1

    local video_args=()
    $SAVE_VIDEO || video_args=(--no_video)
    local suite pid failed=0 pids=()
    for suite in "${SUITES[@]}"; do
        eval_run_client "$SCRIPT_DIR" "$gpu" "$out/logs/$suite.log" \
            PYTHONPATH="$LIBERO_DIR${PYTHONPATH:+:$PYTHONPATH}" -- \
            libero_client.py --host 127.0.0.1 --port "$port" \
            --task_suite "$suite" --num_trials "$NUM_TRIALS" \
            --output_dir "$out" ${video_args[@]+"${video_args[@]}"} &
        if $PARALLEL_SUITES; then
            pids+=("$!")
        else
            wait "$!" || failed=1
        fi
    done
    for pid in ${pids[@]+"${pids[@]}"}; do
        wait "$pid" || failed=1
    done
    eval_stop_server "$port"

    python3 - "$out" "${SUITES[@]}" <<'PY' || return 1
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
groups, complete = {}, True
for suite in sys.argv[2:]:
    path = out / suite / "results.json"
    if not path.exists():
        complete = False
        continue
    r = json.loads(path.read_text())
    groups[suite] = {k: r[k] for k in ("success", "total", "rate")}
    complete = complete and r["total"] == r["expected"]
success = sum(g["success"] for g in groups.values())
total = sum(g["total"] for g in groups.values())
summary = {"checkpoint": out.name, "success": success, "total": total,
           "rate": success / total if total else 0.0, "groups": groups}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
sys.exit(0 if complete and groups else 3)
PY
    return "$failed"
}

eval_main "$@"
