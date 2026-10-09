#!/usr/bin/env bash
# Shared launcher behind every evaluation/<bench>/run_eval.sh.
#
# Every benchmark runs the same way: a policy server (the `simvla2` env) holds
# the checkpoint on one GPU, and the benchmark's simulator client (its own
# conda env) talks to it over localhost. This file owns everything that is not
# benchmark-specific: the command line, starting and stopping the server, and
# the _COMPLETE / _FAILED bookkeeping. One invocation evaluates one checkpoint;
# once it finished the full protocol it is marked _COMPLETE, so re-running the
# same command skips it.
#
# A run_eval.sh sources this file, sets the variables and functions below, then
# calls `eval_main "$@"`:
#
#   BENCH               benchmark name, used in logs and the help text
#   SERVER_ENV          conda env of the policy server (default simvla2)
#   CLIENT_ENV          conda env of the simulator client
#   SERVER_SCRIPT       policy server script
#   DEFAULT_NORM_STATS  shipped norm stats, for checkpoints without their own
#   DEFAULT_PORT, DEFAULT_DTYPE, DEFAULT_FLOW_STEPS
#   bench_usage         print the benchmark-specific options
#   bench_parse_arg     consume one benchmark option from "$@"; return the
#                       number of words used, or 0 if it is not one
#   bench_prepare       optional; runs once after parsing
#   bench_describe      optional; prints the benchmark settings
#   bench_eval CKPT OUT GPU PORT
#                       start the server with eval_start_server, run the
#                       simulator (eval_run_client), write OUT/summary.json;
#                       return 0 only if the run finished
#
# Set FULL_PROTOCOL=false (in bench_prepare) when the options narrow the
# reported protocol: results are still written but the checkpoint is never
# marked _COMPLETE, so a later full run redoes it.
#
# summary.json carries "success", "total", "rate" and optional per-suite/task
# "groups"; chained-task benchmarks report "score" and "sr" instead.

EVAL_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_COMMON_DIR}/.." && pwd)"

SERVER_ENV="simvla2"
FULL_PROTOCOL=true
EVAL_SERVER_PID=""
EVAL_RUN_PID=""

# Environment every simulator client runs with; benchmarks add their own.
EVAL_CLIENT_ENV=(PYTHONNOUSERSITE=1 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost)
# Per-user scratch for shader caches and XDG runtime dirs.
EVAL_TMP="${TMPDIR:-/tmp}/simvla2-eval-$(id -u)"

eval_now() { date '+%F %T'; }

# eval_log MSG...: timestamped line on stderr.
eval_log() {
    printf '[%s] %s\n' "$(eval_now)" "$*" >&2
}

_eval_die() {
    echo "run_eval.sh: $*" >&2
    exit 2
}

# -----------------------------------------------------------------------------
# Result markers
# -----------------------------------------------------------------------------

eval_is_done() { [[ -f "$1/_COMPLETE" ]]; }

eval_mark_complete() {
    touch "$1/_COMPLETE"
    rm -f "$1/_FAILED"
}

# eval_mark_failed DIR REASON: append the reason, so post-hoc scans can tell
# "never started" from "tried and failed".
eval_mark_failed() {
    mkdir -p "$1"
    printf '%s  %s\n' "$(eval_now)" "${2:-unspecified}" >> "$1/_FAILED"
}

# eval_summary_line SUMMARY_JSON: one-line digest for the logs.
eval_summary_line() {
    python3 - "$1" <<'PY'
import json
import sys

s = json.load(open(sys.argv[1]))
if "score" in s:
    sr = " ".join(f"{r * 100:.1f}" for r in s.get("sr", []))
    print(f"score={s['score']:.3f}  SR@1-5: {sr}")
else:
    line = f"{s.get('success', 0)}/{s.get('total', 0)} = {s.get('rate', 0.0) * 100:.2f}%"
    if "mean_rate" in s:
        line += f" (task mean {s['mean_rate'] * 100:.2f}%)"
    groups = s.get("groups") or {}
    if groups:
        line += "  " + " ".join(
            f"{name}={g.get('rate', 0.0) * 100:.1f}%" for name, g in sorted(groups.items()))
    print(line)
PY
}

# -----------------------------------------------------------------------------
# Ports and processes
# -----------------------------------------------------------------------------

eval_port_free() { ! ss -ltn 2>/dev/null | grep -q ":${1}\b"; }

# eval_wait_port_free PORT [MAX_WAIT]: 0 once PORT is free, 1 on timeout.
eval_wait_port_free() {
    local port="$1" max_wait="${2:-30}" waited=0
    while (( waited < max_wait )); do
        eval_port_free "$port" && return 0
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

# eval_kill_pid PID: TERM, then KILL after 20 s.
eval_kill_pid() {
    local pid="$1" i
    [[ -z "$pid" ]] && return 0
    kill -0 "$pid" 2>/dev/null || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for ((i = 0; i < 20; i++)); do
        kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null; return 0; }
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# eval_wait_health PID PORT MAX_WAIT: 0 once the server accepts a connection
# on PORT (it listens only after the model is loaded), 1 on timeout, 2 if the
# process exited first.
eval_wait_health() {
    PID="$1" PORT="$2" MAX_WAIT="${3:-600}" python3 - <<'PY'
import os
import socket
import time

pid = int(os.environ["PID"])
port = int(os.environ["PORT"])
deadline = time.monotonic() + int(os.environ["MAX_WAIT"])
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit(2)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            raise SystemExit(0)
    except OSError:
        pass
    time.sleep(2)
raise SystemExit(1)
PY
}

# Norm stats to pass for CKPT: --norm-stats if given, else the copy training
# saved into the checkpoint, else the benchmark's shipped file. The path is
# always passed explicitly: some servers treat "--norm_stats given" and "stats
# loaded with the checkpoint" differently (quantile default, stats dtype).
eval_norm_stats() {
    if [[ -n "$NORM_STATS" ]]; then
        echo "$NORM_STATS"
    elif [[ -f "$1/norm_stats.json" ]]; then
        echo "$1/norm_stats.json"
    else
        echo "${DEFAULT_NORM_STATS:-}"
    fi
}

# eval_start_server CKPT GPU PORT LOG [EXTRA_SERVER_ARGS...]
#   Serve CKPT on GPU/PORT with the shared server flags and wait until it
#   accepts connections. Sets EVAL_SERVER_PID; the caller stops it with
#   eval_stop_server (the launcher also does once bench_eval returns).
eval_start_server() {
    local ckpt="$1" gpu="$2" port="$3" log="$4"
    shift 4
    local args=(--checkpoint "$ckpt" --host 127.0.0.1 --port "$port"
                --dtype "$DTYPE" --flow_steps "$FLOW_STEPS")
    local norm
    norm="$(eval_norm_stats "$ckpt")"
    [[ -n "$norm" ]] && args+=(--norm_stats "$norm")
    (
        cd "$REPO_ROOT" || exit 1
        # shellcheck disable=SC2086  # SERVER_PY may be a command with arguments
        exec env CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONNOUSERSITE=1 \
            PYTHONPATH="$REPO_ROOT" \
            TOKENIZERS_PARALLELISM=false \
            NO_PROXY=127.0.0.1,localhost \
            no_proxy=127.0.0.1,localhost \
            $SERVER_PY -u "$SERVER_SCRIPT" "${args[@]}" "$@"
    ) >"$log" 2>&1 &
    EVAL_SERVER_PID=$!
    local rc=0
    eval_wait_health "$EVAL_SERVER_PID" "$port" "${SERVER_START_TIMEOUT:-600}" || rc=$?
    if (( rc != 0 )); then
        echo "server did not come up (rc=$rc); tail of $log:" >&2
        tail -n 20 "$log" >&2
    fi
    return "$rc"
}

# eval_stop_server [PORT]: stop EVAL_SERVER_PID; with PORT, wait until it is free.
eval_stop_server() {
    [[ -n "$EVAL_SERVER_PID" ]] || return 0
    eval_kill_pid "$EVAL_SERVER_PID"
    EVAL_SERVER_PID=""
    if [[ -n "${1:-}" ]]; then
        eval_wait_port_free "$1" 30 || true
    fi
}

# eval_run_client DIR GPU LOG [VAR=VALUE...] -- SCRIPT [ARGS...]
#   Run the simulator client SCRIPT from DIR with CLIENT_PY, EVAL_CLIENT_ENV,
#   CUDA_VISIBLE_DEVICES=GPU and the extra VAR=VALUE pairs; output goes to LOG.
#   Runs in the foreground; append & for a background worker.
eval_run_client() {
    local dir="$1" gpu="$2" log="$3"
    shift 3
    local vars=()
    while (( $# )) && [[ "$1" != -- ]]; do
        vars+=("$1")
        shift
    done
    shift
    (
        cd "$dir" || exit 1
        # shellcheck disable=SC2086  # CLIENT_PY may be a command with arguments
        exec env "${EVAL_CLIENT_ENV[@]}" CUDA_VISIBLE_DEVICES="$gpu" ${vars[@]+"${vars[@]}"} \
            $CLIENT_PY -u "$@"
    ) >"$log" 2>&1
}

# Every pid in the process tree rooted at PID.
_eval_tree() {
    local child
    echo "$1"
    for child in $(pgrep -P "$1" 2>/dev/null); do
        _eval_tree "$child"
    done
}

_eval_cleanup() {
    [[ -n "$EVAL_RUN_PID" ]] || return 0
    local pids
    pids="$(_eval_tree "$EVAL_RUN_PID")"
    # shellcheck disable=SC2086
    [[ -n "${pids// /}" ]] && kill -TERM $pids 2>/dev/null
    return 0
}

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

# Run bench_eval for the checkpoint and mark the result complete or failed.
_eval_run() {
    local name
    name="$(basename "$OUT")"
    mkdir -p "$OUT/logs"
    rm -f "$OUT/_FAILED" "$OUT/summary.json"
    # Another evaluation may own the port, so fail instead of killing it.
    if ! eval_port_free "$PORT"; then
        eval_mark_failed "$OUT" "port $PORT busy"
        eval_log "FAILED $name: port $PORT busy; pass another --port or stop the process holding it"
        return 1
    fi
    eval_log "start $name gpu=$GPU port=$PORT"

    local rc=0
    bench_eval "$CKPT" "$OUT" "$GPU" "$PORT" || rc=$?
    eval_stop_server "$PORT"

    if (( rc != 0 )) || [[ ! -f "$OUT/summary.json" ]]; then
        eval_mark_failed "$OUT" "rc=$rc"
        eval_log "FAILED $name (rc=$rc), logs in $OUT/logs"
        return 1
    fi
    if $FULL_PROTOCOL; then
        eval_mark_complete "$OUT"
        eval_log "done $name: $(eval_summary_line "$OUT/summary.json")"
    else
        eval_log "done $name (partial, not marked _COMPLETE): $(eval_summary_line "$OUT/summary.json")"
    fi
}

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

_eval_usage() {
    cat <<EOF
Usage: $(basename "$0") --ckpt DIR [options]

  --ckpt DIR          checkpoint directory to evaluate (required)

Common options:
  --gpu ID            GPU for the server and the simulator (default: 0)
  --port N            server port (default: ${DEFAULT_PORT}); runs that share
                      a host at the same time need distinct ports
  --output-base DIR   results root; checkpoint X writes DIR/X/summary.json
                      (default: eval-<name> next to the directory holding the
                      checkpoint, e.g. runs/libero/X -> runs/eval-libero/X)
  --norm-stats PATH   norm stats override (default: the checkpoint's own
                      norm_stats.json, else ${DEFAULT_NORM_STATS#"$REPO_ROOT"/})
  --dtype DT          server weight dtype, bf16 | fp16 | fp32 (default: ${DEFAULT_DTYPE})
  --flow-steps N      flow-matching Euler steps (default: ${DEFAULT_FLOW_STEPS})
  -h, --help          show this help

${BENCH} options:
$(bench_usage)

Environment:
  SERVER_PY           python of the ${SERVER_ENV} env
                      (default: <conda>/envs/${SERVER_ENV}/bin/python, else python)
  CLIENT_PY           python of the simulator env
                      (default: <conda>/envs/${CLIENT_ENV}/bin/python, else python)
  CONDA_BASE          conda root used for the defaults above
EOF
}

# Python of a conda env, if that env exists.
_eval_env_python() {
    if [[ -z "${CONDA_BASE:-}" ]] && command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
    fi
    if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/envs/$1/bin/python" ]]; then
        echo "${CONDA_BASE}/envs/$1/bin/python"
    fi
}

eval_main() {
    CKPT="" GPU="0" OUTPUT_BASE="" PORT="$DEFAULT_PORT"
    NORM_STATS="" DTYPE="$DEFAULT_DTYPE" FLOW_STEPS="$DEFAULT_FLOW_STEPS"
    local n
    while (( $# )); do
        case "$1" in
            -h|--help)
                _eval_usage
                exit 0 ;;
            --ckpt|--gpu|--output-base|--port|--norm-stats|--dtype|--flow-steps)
                (( $# >= 2 )) || _eval_die "$1 needs a value"
                case "$1" in
                    --ckpt)        CKPT="$2" ;;
                    --gpu)         GPU="$2" ;;
                    --output-base) OUTPUT_BASE="$2" ;;
                    --port)        PORT="$2" ;;
                    --norm-stats)  NORM_STATS="$2" ;;
                    --dtype)       DTYPE="$2" ;;
                    --flow-steps)  FLOW_STEPS="$2" ;;
                esac
                shift 2 ;;
            *)
                n=0
                bench_parse_arg "$@" || n=$?
                (( n > 0 )) || _eval_die "unknown option: $1 (see --help)"
                (( n <= $# )) || _eval_die "$1 needs a value"
                shift "$n" ;;
        esac
    done

    [[ -n "$CKPT" ]] || _eval_die "--ckpt is required (see --help)"
    [[ -d "$CKPT" ]] || _eval_die "checkpoint not found: $CKPT"
    CKPT="$(cd "$CKPT" && pwd)"
    local root
    root="$(dirname "$CKPT")"
    OUTPUT_BASE="${OUTPUT_BASE:-$(dirname "$root")/eval-$(basename "$root")}"
    mkdir -p "$OUTPUT_BASE" || _eval_die "cannot create $OUTPUT_BASE"
    OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"
    OUT="$OUTPUT_BASE/$(basename "$CKPT")"

    if [[ -n "$NORM_STATS" ]]; then
        [[ -f "$NORM_STATS" ]] || _eval_die "norm stats not found: $NORM_STATS"
        NORM_STATS="$(cd "$(dirname "$NORM_STATS")" && pwd)/$(basename "$NORM_STATS")"
    fi
    [[ -n "$GPU" && "$GPU" != *,* ]] || _eval_die "--gpu takes a single GPU id"
    case "$DTYPE" in bf16|fp16|fp32) ;; *) _eval_die "--dtype must be bf16, fp16 or fp32" ;; esac
    [[ "$FLOW_STEPS" =~ ^[1-9][0-9]*$ ]] || _eval_die "--flow-steps must be a positive integer"
    [[ "$PORT" =~ ^[0-9]+$ ]] || _eval_die "--port must be an integer"
    [[ -f "$SERVER_SCRIPT" ]] || _eval_die "server script not found: $SERVER_SCRIPT"

    SERVER_PY="${SERVER_PY:-$(_eval_env_python "$SERVER_ENV")}"
    SERVER_PY="${SERVER_PY:-python}"
    CLIENT_PY="${CLIENT_PY:-$(_eval_env_python "$CLIENT_ENV")}"
    CLIENT_PY="${CLIENT_PY:-python}"

    if declare -F bench_prepare >/dev/null; then
        bench_prepare || exit $?
    fi

    local norm
    norm="$(eval_norm_stats "$CKPT")"
    echo "[$(eval_now)] ${BENCH} evaluation"
    echo "  checkpoint  : $CKPT"
    echo "  output      : $OUT"
    echo "  gpu / port  : $GPU / $PORT"
    echo "  server      : dtype=$DTYPE flow_steps=$FLOW_STEPS"
    echo "  norm stats  : ${norm:-<none>}"
    echo "  python      : server=$SERVER_PY client=$CLIENT_PY"
    $FULL_PROTOCOL || echo "  protocol    : partial (results are not marked complete)"
    if declare -F bench_describe >/dev/null; then
        bench_describe
    fi
    if eval_is_done "$OUT"; then
        echo "already complete: $OUT (remove $OUT/_COMPLETE to re-run)"
        return 0
    fi

    # _eval_run runs in the background so a signal reaches the trap at once
    # instead of after the current foreground client exits.
    trap _eval_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    _eval_run &
    EVAL_RUN_PID=$!
    local rc=0
    wait "$EVAL_RUN_PID" || rc=$?
    trap - EXIT INT TERM
    return "$rc"
}
