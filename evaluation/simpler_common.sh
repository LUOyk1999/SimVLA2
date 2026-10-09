#!/usr/bin/env bash
# SimplerEnv pieces shared by evaluation/bridge/run_eval.sh and
# evaluation/fractal/run_eval.sh; sourced after eval_common.sh.

# The patched SimplerEnv checkout; stock SimplerEnv lacks the absolute-pose
# controller and silently produces near-zero success.
SIMPLER_DIR="${SIMPLER_DIR:-$REPO_ROOT/third_party/SimplerEnv}"

# simpler_prepare RENDER_GPU...: resolve SIMPLER_DIR and create each render
# GPU's shader cache and XDG runtime dirs.
simpler_prepare() {
    [[ -d "$SIMPLER_DIR/ManiSkill2_real2sim" ]] || \
        _eval_die "SimplerEnv not found at $SIMPLER_DIR; run evaluation/$(basename "$SCRIPT_DIR")/install_env.sh or set SIMPLER_DIR"
    SIMPLER_DIR="$(cd "$SIMPLER_DIR" && pwd)"
    local gpu
    for gpu in "$@"; do
        mkdir -p "$EVAL_TMP/glcache/gpu$gpu" "$EVAL_TMP/xdg-runtime-$gpu"
        chmod 700 "$EVAL_TMP/xdg-runtime-$gpu"
    done
}

# simpler_run_client RENDER_GPU CACHE_NAME LOG SCRIPT [ARGS...]
#   One SimplerEnv client rendering on RENDER_GPU, with its own shader cache.
simpler_run_client() {
    local gpu="$1" cache="$EVAL_TMP/glcache/gpu$1/$2" log="$3"
    shift 3
    mkdir -p "$cache"
    eval_run_client "$SCRIPT_DIR" "$gpu" "$log" \
        PYTHONPATH="$SIMPLER_DIR:$SIMPLER_DIR/ManiSkill2_real2sim${PYTHONPATH:+:$PYTHONPATH}" \
        VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}" \
        __GL_SHADER_DISK_CACHE=1 \
        __GL_SHADER_DISK_CACHE_PATH="$cache" \
        XDG_RUNTIME_DIR="$EVAL_TMP/xdg-runtime-$gpu" \
        -- "$@"
}
