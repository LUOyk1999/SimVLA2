#!/usr/bin/env bash
# Fractal and Bridge share one SimplerEnv client env (simvla2-simpler-eval);
# this runs evaluation/bridge/install_env.sh with the same options.
#
#   bash evaluation/fractal/install_env.sh [--env-name NAME] [--src-dir DIR]
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../bridge/install_env.sh" "$@"
