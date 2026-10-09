#!/usr/bin/env bash
# Build the CALVIN client env (default: simvla2-calvin-eval).
#
# This env only runs CALVIN's PyBullet simulator and calvin_client.py, which
# talks to the policy server over a websocket. Keeping it apart from the
# `simvla2` env is what lets CALVIN keep its torch 1.13.1 / pytorch-lightning
# 1.8.6 / gym 0.21 pins. CALVIN is used unmodified (submodules calvin_env @
# 1431a46, tacto @ dd53360 come with the pinned commit); there is no patch.
#
#   bash evaluation/calvin/install_env.sh
#
# Prereqs: conda, git, gcc/g++ (calvin_env builds C extensions), network
# access to PyPI, download.pytorch.org and GitHub; EGL at runtime.
#
# Not done here, because they depend on the dataset copy on each machine:
# downloading task_ABC_D and rewriting the absolute paths baked into its
# .hydra/merged_config.yaml (evaluation/calvin/README.md, "Data").
set -uo pipefail

# shellcheck source=../install_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../install_common.sh"

INSTALL_TITLE="Build the CALVIN client env and check out CALVIN."
DEFAULT_ENV_NAME="simvla2-calvin-eval"
SRC_DIR_VAR="CALVIN_DIR"
DEFAULT_SRC_DIR="${REPO_ROOT}/third_party/calvin"
UPSTREAM_URL="https://github.com/mees/calvin.git"
UPSTREAM_COMMIT="fa03f01f19c65920e18cf37398a9ce859274af76"

install_parse_args "$@"
# python 3.8: required by pytorch-lightning 1.8.x / gym 0.21.
install_conda_env 3.8
install_clone

# gym 0.21's setup.py uses the `use_2to3` kwarg that setuptools 58 removed,
# and pip 24 rejects its metadata, so both are pinned below those. wheel 0.41.3
# rejects its `opencv-python>=3.` requirement while building, hence 0.38.4.
step "pin pip / setuptools / wheel / cmake"
python -m pip install --upgrade 'pip==23.3.2' 'setuptools==57.5.0' 'wheel==0.38.4' 'cmake==3.27.9' \
    || die "tooling pins failed"

step "torch 1.13.1 + cu117"
if have torch && python -c 'import sys, torch; sys.exit(0 if torch.__version__.startswith("1.13.1") else 1)'; then
    echo "torch $(python -c 'import torch; print(torch.__version__)') already installed"
else
    pip install --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 torchvision==0.14.1 \
        || die "torch install failed"
fi

# Installed together so pip resolves them against the torch already in place;
# pytorch-lightning on its own happily upgrades torch.
step "pytorch-lightning 1.8.6 + gym 0.21"
if have pytorch_lightning && have gym; then
    echo "already installed"
else
    pip install 'pytorch-lightning==1.8.6' 'gym==0.21.0' || die "pytorch-lightning / gym install failed"
fi

# tacto has to be installed before calvin_env, whose setup.py imports it.
step "tacto, calvin_env (editable)"
have tacto || pip install -e "${SRC_DIR}/calvin_env/tacto" || die "tacto install failed"
have calvin_env || pip install -e "${SRC_DIR}/calvin_env" || die "calvin_env install failed"

# --no-deps: calvin_models' setup.cfg pulls unpinned torch / PL / gym that
# would replace the pins above. Only its source tree has to be importable.
step "calvin_models (editable, --no-deps)"
have calvin_agent || pip install --no-deps -e "${SRC_DIR}/calvin_models" || die "calvin_models install failed"

# What calvin_env / calvin_agent import at env-reset and sequence-eval time,
# skipped by the --no-deps install above.
step "runtime deps (hydra, omegaconf, opencv, ...)"
pip install 'hydra-core==1.1.1' 'hydra-colorlog==1.1.0' 'omegaconf==2.1.2' \
    opencv-python-headless termcolor easydict imageio imageio-ffmpeg scipy tqdm \
    'numpy<1.24' pyyaml pyhash || die "runtime deps failed"

step "client deps (msgpack, msgpack-numpy, websockets)"
pip install msgpack msgpack-numpy websockets 'numpy<1.24' || die "client deps failed"

cd "${SRC_DIR}" || die "cannot cd to ${SRC_DIR}"
install_verify torch torchvision pytorch_lightning gym hydra omegaconf cv2 termcolor \
    easydict imageio scipy tqdm msgpack_numpy websockets tacto calvin_env \
    calvin_env.envs.play_table_env calvin_agent calvin_agent.evaluation.utils \
    calvin_agent.evaluation.multistep_sequences calvin_agent.models.calvin_base_model || exit 1

install_done \
    "Next: download task_ABC_D and fix its merged_config.yaml (evaluation/calvin/README.md, \"Data\")" \
    "Evaluate: bash evaluation/calvin/run_eval.sh --ckpt <checkpoint>" \
    "A checkout elsewhere: CALVIN_DIR=<dir> bash evaluation/calvin/run_eval.sh ..."
