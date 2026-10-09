#!/usr/bin/env bash
# Build the LIBERO client env (default: simvla2-libero-eval).
#
# Follows LIBERO's own install instructions at the pinned commit (python
# 3.8.13, torch 1.11.0+cu113, its requirements.txt, editable install), then
# adds what libero_client.py needs on top to talk to the policy server.
# LIBERO is used unmodified; there is no patch.
#
#   bash evaluation/libero/install_env.sh
#
# Prereqs: conda, git, network access to PyPI, download.pytorch.org and
# GitHub; an EGL-capable GPU driver at runtime (robosuite renders offscreen).
set -uo pipefail

# shellcheck source=../install_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../install_common.sh"

INSTALL_TITLE="Build the LIBERO client env and check out LIBERO."
DEFAULT_ENV_NAME="simvla2-libero-eval"
SRC_DIR_VAR="LIBERO_DIR"
DEFAULT_SRC_DIR="${REPO_ROOT}/third_party/LIBERO"
UPSTREAM_URL="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
UPSTREAM_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"

install_parse_args "$@"
install_conda_env 3.8.13
install_clone

# Importing libero.libero without a config.yaml drops into an interactive
# prompt, so every import check below runs against a throwaway config.
# run_eval.sh seeds its own.
LIBERO_CONFIG_PATH="$(mktemp -d)"
export LIBERO_CONFIG_PATH
trap 'rm -rf "${LIBERO_CONFIG_PATH}"' EXIT
cat >"${LIBERO_CONFIG_PATH}/config.yaml" <<YAML
benchmark_root: ${SRC_DIR}/libero/libero
bddl_files: ${SRC_DIR}/libero/libero/bddl_files
init_states: ${SRC_DIR}/libero/libero/init_files
datasets: ${SRC_DIR}/libero/libero/datasets
assets: ${SRC_DIR}/libero/libero/assets
YAML

step "torch 1.11.0 + cu113 (LIBERO's documented build)"
if have torch && python -c 'import sys, torch; sys.exit(0 if torch.__version__.startswith("1.11.0") else 1)'; then
    echo "torch $(python -c 'import torch; print(torch.__version__)') already installed"
else
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113 || die "torch install failed"
fi

step "LIBERO requirements.txt + editable install"
if have libero.libero && have robosuite; then
    echo "libero / robosuite already installed"
else
    pip install -r "${SRC_DIR}/requirements.txt" || die "LIBERO requirements failed"
    pip install -e "${SRC_DIR}" || die "LIBERO editable install failed"
fi

# LIBERO already pins robosuite, gym, numpy and friends; only add the wire
# deps here and do not re-pin anything it owns.
step "client deps (msgpack-numpy, websockets, imageio-ffmpeg, tqdm)"
pip install msgpack-numpy websockets imageio imageio-ffmpeg tqdm \
    || die "client deps failed"

cd "${SRC_DIR}" || die "cannot cd to ${SRC_DIR}"
install_verify torch robosuite bddl gym imageio tqdm PIL msgpack_numpy websockets \
    libero.libero libero.libero.benchmark libero.libero.envs || exit 1

install_done \
    "Evaluate: bash evaluation/libero/run_eval.sh --ckpt <checkpoint>" \
    "A checkout elsewhere: LIBERO_DIR=<dir> bash evaluation/libero/run_eval.sh ..."
