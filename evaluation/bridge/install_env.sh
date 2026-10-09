#!/usr/bin/env bash
# Build the SimplerEnv client env (default: simvla2-simpler-eval), shared by
# the Bridge (WidowX) and Fractal (Google robot) evaluations;
# evaluation/fractal/install_env.sh runs this same script.
#
# Clones SimplerEnv at the pinned commit (ManiSkill2_real2sim comes in as a
# submodule at ef7a4d4), applies evaluation/patches/simplerenv.patch, and
# follows SimplerEnv's own install steps. The patch registers the absolute
# end-effector pose controller (arm_pd_ee_target_base_pose) that SimVLA2's
# absolute-pose actions need and makes it the WidowX default; without it the
# pose is integrated as a delta every tick and success drops to ~0. The rest
# of the patch is listed in evaluation/bridge/README.md ("Why the patch").
#
#   bash evaluation/bridge/install_env.sh
#
# Prereqs: conda, git, network access to PyPI and GitHub; a Vulkan-capable
# NVIDIA driver at runtime (see evaluation/bridge/README.md, "Vulkan").
set -uo pipefail

# shellcheck source=../install_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../install_common.sh"

INSTALL_TITLE="Build the SimplerEnv client env (Bridge + Fractal) and check out SimplerEnv."
DEFAULT_ENV_NAME="simvla2-simpler-eval"
SRC_DIR_VAR="SIMPLER_DIR"
DEFAULT_SRC_DIR="${REPO_ROOT}/third_party/SimplerEnv"
UPSTREAM_URL="https://github.com/simpler-env/SimplerEnv.git"
UPSTREAM_COMMIT="06accaca93535902d408da4855f21cece12bceb7"

install_parse_args "$@"
install_conda_env 3.10
install_clone
install_patch simplerenv

# SAPIEN 2.x and mani_skill2_real2sim import pkg_resources (gone in
# setuptools 70); SimplerEnv asks for numpy<2 because pinocchio's IK breaks on
# 2.x (segfault in the first env.step). The numpy pin is repeated on every later
# install: mani_skill2_real2sim's unpinned opencv-python otherwise resolves to
# 5.x, which requires numpy>=2 and silently replaces 1.24.4.
step "setuptools<70, numpy 1.24.4"
pip install "setuptools<70" numpy==1.24.4 || die "base pins failed"

step "ManiSkill2_real2sim + SimplerEnv (editable)"
if have mani_skill2_real2sim && have simpler_env; then
    echo "already installed"
else
    pip install -e "${SRC_DIR}/ManiSkill2_real2sim" numpy==1.24.4 || die "ManiSkill2_real2sim install failed"
    pip install -e "${SRC_DIR}" numpy==1.24.4 || die "SimplerEnv install failed"
fi

step "client deps (scipy, msgpack, msgpack-numpy, websockets, mediapy)"
pip install scipy msgpack msgpack-numpy websockets mediapy numpy==1.24.4 \
    || die "client deps failed"

cd "${SRC_DIR}" || die "cannot cd to ${SRC_DIR}"
install_verify numpy sapien gymnasium scipy transforms3d msgpack_numpy websockets \
    mediapy mani_skill2_real2sim simpler_env simpler_env.utils.env.env_builder || exit 1

install_done \
    "Evaluate: bash evaluation/bridge/run_eval.sh --ckpt <checkpoint>" \
    "          bash evaluation/fractal/run_eval.sh --ckpt <checkpoint>" \
    "A checkout elsewhere: SIMPLER_DIR=<dir> bash evaluation/<bridge|fractal>/run_eval.sh ..."
