#!/usr/bin/env bash
# CALVIN ABC->D evaluation: chains of 5 subtasks on the D validation split,
# scored as the mean completed chain length. The GPU holds the policy server
# and PyBullet's EGL renderer; --num-workers is CPU-side env parallelism.
#
#   bash evaluation/calvin/run_eval.sh --ckpt runs/calvin/ckpt-200000
#   bash evaluation/calvin/run_eval.sh --ckpt runs/calvin/ckpt-200000 --gpu 1
#
# Shared options and the result layout are described in evaluation/eval_common.sh
# and printed by --help.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../eval_common.sh
source "$SCRIPT_DIR/../eval_common.sh"

BENCH="CALVIN"
CLIENT_ENV="simvla2-calvin-eval"
SERVER_SCRIPT="$SCRIPT_DIR/serve_simvla2_calvin.py"
DEFAULT_NORM_STATS="$REPO_ROOT/norm_stats/calvin.json"
DEFAULT_PORT=8300
DEFAULT_DTYPE=bf16
DEFAULT_FLOW_STEPS=20

CALVIN_DIR="${CALVIN_DIR:-$REPO_ROOT/third_party/calvin}"
DATASET_PATH=""
NUM_SEQUENCES=1000
NUM_WORKERS=5
PLAN_LEN=30
GRIPPER_THRESHOLD=0.0
SAVE_VIDEO=false

bench_usage() {
    cat <<'EOF'
  --dataset-path DIR  task_ABC_D validation split
                      (default: $CALVIN_DIR/dataset/task_ABC_D/validation)
  --num-sequences N   evaluation chains (default: 1000; fewer is a partial run)
  --num-workers N     CALVIN env processes (default: 5)
  --plan-len N        actions executed per server call (default: 30)
  --gripper-threshold F
                      gripper open/close threshold (default: 0.0)
  --video             save rollout videos
  CALVIN_DIR (env)    CALVIN checkout (default: third_party/calvin, see install_env.sh)
EOF
}

bench_parse_arg() {
    case "$1" in
        --dataset-path)      DATASET_PATH="${2-}"; return 2 ;;
        --num-sequences)     NUM_SEQUENCES="${2-}"; return 2 ;;
        --num-workers)       NUM_WORKERS="${2-}"; return 2 ;;
        --plan-len)          PLAN_LEN="${2-}"; return 2 ;;
        --gripper-threshold) GRIPPER_THRESHOLD="${2-}"; return 2 ;;
        --video)             SAVE_VIDEO=true; return 1 ;;
    esac
    return 0
}

bench_prepare() {
    [[ -d "$CALVIN_DIR/calvin_env" ]] || \
        _eval_die "CALVIN not found at $CALVIN_DIR; run evaluation/calvin/install_env.sh or set CALVIN_DIR"
    CALVIN_DIR="$(cd "$CALVIN_DIR" && pwd)"
    DATASET_PATH="${DATASET_PATH:-$CALVIN_DIR/dataset/task_ABC_D/validation}"
    [[ -d "$DATASET_PATH" ]] || _eval_die "CALVIN validation split not found: $DATASET_PATH (see evaluation/calvin/README.md)"
    if [[ "$NUM_SEQUENCES" != 1000 || "$PLAN_LEN" != 30 || "$GRIPPER_THRESHOLD" != 0.0 ]]; then
        FULL_PROTOCOL=false
    fi
}

bench_describe() {
    echo "  chains      : $NUM_SEQUENCES, plan_len $PLAN_LEN, $NUM_WORKERS workers"
    echo "  dataset     : $DATASET_PATH"
}

bench_eval() {
    local ckpt="$1" out="$2" gpu="$3" port="$4"
    eval_start_server "$ckpt" "$gpu" "$port" "$out/logs/server.log" || return 1

    local video_args=()
    $SAVE_VIDEO || video_args=(--no_video)
    eval_run_client "$SCRIPT_DIR" "$gpu" "$out/logs/client.log" \
        EGL_VISIBLE_DEVICES="$gpu" \
        PYOPENGL_PLATFORM=egl \
        PYTHONPATH="$CALVIN_DIR/calvin_models:$CALVIN_DIR/calvin_env:$REPO_ROOT" -- \
        calvin_client.py --host 127.0.0.1 --port "$port" \
        --num_workers "$NUM_WORKERS" \
        --num_sequences "$NUM_SEQUENCES" \
        --plan_len "$PLAN_LEN" \
        --gripper_threshold "$GRIPPER_THRESHOLD" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$out" \
        ${video_args[@]+"${video_args[@]}"}
}

eval_main "$@"
