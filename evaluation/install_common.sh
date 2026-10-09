#!/usr/bin/env bash
# Shared helpers behind every evaluation/<bench>/install_env.sh.
#
# Each benchmark's simulator gets its own conda env (simvla2-<bench>-eval) and
# its upstream repo is cloned from the official remote at a pinned commit into
# third_party/<Repo>, with this repo's patch (evaluation/patches/) applied on
# top. The SimVLA2 policy itself runs in the main `simvla2` env, so none of
# these envs needs the model's dependencies.
#
# An install_env.sh sets the variables below, sources this file, calls
# `install_parse_args "$@"`, and then runs its steps with the helpers:
#
#   INSTALL_TITLE       one-line description for --help
#   DEFAULT_ENV_NAME    e.g. simvla2-libero-eval
#   SRC_DIR_VAR         the variable run_eval.sh reads the checkout from,
#                       e.g. LIBERO_DIR; it also sets the default here
#   DEFAULT_SRC_DIR     e.g. $REPO_ROOT/third_party/LIBERO
#   UPSTREAM_URL        official git remote
#   UPSTREAM_COMMIT     pinned commit
#   SRC_DIR_NOTE        optional --help line on how the checkout is used
#
# Every install_env.sh accepts the same options:
#   --env-name NAME     conda env to create or reuse (env var ENV_NAME)
#   --src-dir DIR       upstream checkout (env var SRC_DIR or $SRC_DIR_VAR)
#   CONDA_BASE (env)    conda root (default: `conda info --base`)
#
# Re-running is safe: an existing env, checkout or applied patch is detected
# and reused, and packages that already import are not reinstalled.

INSTALL_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${INSTALL_COMMON_DIR}/.." && pwd)"
PATCH_DIR="${INSTALL_COMMON_DIR}/patches"

step() { printf '\n=== %s ===\n' "$*"; }

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# have MODULE: the env's python can import MODULE.
have() { "${ENV_PREFIX}/bin/python" -c "import $1" 2>/dev/null; }

install_usage() {
    cat <<EOF
Usage: $(basename "$0") [--env-name NAME] [--src-dir DIR]

${INSTALL_TITLE}

  --env-name NAME   conda env to create or reuse (default: ${DEFAULT_ENV_NAME})
  --src-dir DIR     checkout of ${UPSTREAM_URL}
                    (default: \$${SRC_DIR_VAR}, else ${DEFAULT_SRC_DIR#"$REPO_ROOT"/})
  -h, --help        show this help

Upstream pin: ${UPSTREAM_COMMIT}
Environment: ENV_NAME, SRC_DIR / ${SRC_DIR_VAR} as above; CONDA_BASE (conda root).
${SRC_DIR_NOTE:-run_eval.sh finds a checkout outside the default through ${SRC_DIR_VAR}.}
EOF
    if declare -F install_extra_usage >/dev/null; then
        install_extra_usage
    fi
}

install_parse_args() {
    ENV_NAME="${ENV_NAME:-$DEFAULT_ENV_NAME}"
    SRC_DIR="${SRC_DIR:-${!SRC_DIR_VAR:-$DEFAULT_SRC_DIR}}"
    while (( $# )); do
        case "$1" in
            --env-name) ENV_NAME="${2:?--env-name needs a value}"; shift 2 ;;
            --src-dir)  SRC_DIR="${2:?--src-dir needs a value}"; shift 2 ;;
            -h|--help)  install_usage; exit 0 ;;
            *) echo "unknown option: $1" >&2; install_usage >&2; exit 2 ;;
        esac
    done
    mkdir -p "$(dirname "$SRC_DIR")" || die "cannot create $(dirname "$SRC_DIR")"
    SRC_DIR="$(cd "$(dirname "$SRC_DIR")" && pwd)/$(basename "$SRC_DIR")"

    if [[ -z "${CONDA_BASE:-}" ]]; then
        command -v conda >/dev/null 2>&1 || die "conda not on PATH; set CONDA_BASE to the conda root"
        CONDA_BASE="$(conda info --base)"
    fi
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh" || die "no conda.sh under CONDA_BASE=${CONDA_BASE}"
    ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
}

# install_conda_env PYTHON_VERSION: create the env if needed and activate it.
install_conda_env() {
    step "conda env ${ENV_NAME} (python $1)"
    if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
        echo "exists: $("${ENV_PREFIX}/bin/python" -V 2>&1)"
    else
        conda create -y -n "${ENV_NAME}" "python=$1" || die "conda create failed"
    fi
    conda activate "${ENV_NAME}" || die "conda activate ${ENV_NAME} failed"
    python -V
}

# install_clone: check out UPSTREAM_URL at UPSTREAM_COMMIT (with submodules)
# into SRC_DIR, reusing an existing checkout.
install_clone() {
    step "clone ${UPSTREAM_URL} @ ${UPSTREAM_COMMIT:0:12} -> ${SRC_DIR}"
    if [[ -d "${SRC_DIR}/.git" ]]; then
        echo "checkout exists"
    elif [[ -e "${SRC_DIR}" && -n "$(ls -A "${SRC_DIR}" 2>/dev/null)" ]]; then
        die "${SRC_DIR} exists but is not a git checkout; move it away or pass --src-dir"
    else
        git clone "${UPSTREAM_URL}" "${SRC_DIR}" || die "git clone failed"
    fi
    if [[ "$(git -C "${SRC_DIR}" rev-parse HEAD)" != "${UPSTREAM_COMMIT}" ]]; then
        git -C "${SRC_DIR}" cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null \
            || git -C "${SRC_DIR}" fetch origin || die "git fetch failed"
        git -C "${SRC_DIR}" checkout -q "${UPSTREAM_COMMIT}" \
            || die "cannot check out ${UPSTREAM_COMMIT} in ${SRC_DIR} (local changes?)"
    fi
    git -C "${SRC_DIR}" submodule update --init --recursive || die "git submodule update failed"
    git -C "${SRC_DIR}" log --oneline -1
}

# install_patch NAME: apply evaluation/patches/NAME.patch to SRC_DIR once.
install_patch() {
    local patch="${PATCH_DIR}/$1.patch"
    step "patch ${SRC_DIR##*/} with ${patch#"$REPO_ROOT"/}"
    [[ -f "$patch" ]] || die "missing $patch"
    if git -C "${SRC_DIR}" apply --reverse --check "$patch" 2>/dev/null; then
        echo "already applied"
    elif git -C "${SRC_DIR}" apply --check "$patch" 2>/dev/null; then
        git -C "${SRC_DIR}" apply --whitespace=nowarn "$patch" || die "git apply failed"
        echo "applied"
    else
        die "$patch does not apply to ${SRC_DIR}; is it at ${UPSTREAM_COMMIT}?"
    fi
}

# install_verify MODULE...: import every module in the env and fail if any is
# broken, so a bad install shows up here instead of minutes into an eval.
install_verify() {
    step "verify imports"
    python - "$@" <<'PY'
import importlib
import sys

failed = []
for name in sys.argv[1:]:
    try:
        mod = importlib.import_module(name)
        print(f"  ok   {name} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__} {str(e)[:160]}")
        failed.append(name)
try:
    import torch
    print(f"  torch cuda {torch.version.cuda} available={torch.cuda.is_available()}")
except ImportError:
    pass
if failed:
    sys.exit(f"{len(failed)} import(s) failed: {failed}")
PY
}

# install_done LINE...: closing message with the next steps.
install_done() {
    echo
    if [[ -d "${SRC_DIR}" ]]; then
        echo "Done: env '${ENV_NAME}', source at ${SRC_DIR}"
    else
        echo "Done: env '${ENV_NAME}'"
    fi
    local line
    for line in "$@"; do
        echo "  $line"
    done
}
