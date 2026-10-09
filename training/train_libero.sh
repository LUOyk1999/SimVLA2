#!/usr/bin/env bash
# SimVLA2 LIBERO training entry.
#
# Benchmark : LIBERO (evaluated on 4 suites: spatial / object / goal / 10;
#             the training meta also includes libero_90)
#
# Usage: bash training/train_libero.sh [NUM_GPUS] [GPU_IDS]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS="${1:-4}"
GPU_IDS="${2:-0,1,2,3}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

OUT="${OUT:-$REPO_ROOT/runs/libero}"
PORT="${PORT:-29841}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip2-large-patch16-256}"
TOKENIZER="${TOKENIZER:-bert-base-uncased}"

mkdir -p "$OUT"
exec "${SIMVLA2_PY:-python}" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" --main_process_port "$PORT" --mixed_precision bf16 \
  "$REPO_ROOT/train_simvla2.py" \
  --output_dir "$OUT" \
  --train_metas_path "$REPO_ROOT/datasets/metas/libero.json" \
  --siglip_model_path "$SIGLIP_MODEL" \
  --tokenizer_name "$TOKENIZER" \
  --action_mode libero --num_actions 10 --batch_size 64 \
  --learning_rate 5e-4 --learning_coef_vision 0.1 --learning_coef_text 1.0 \
  --iters 200000 --warmup_steps 2000 --freeze_steps 1000 \
  --hidden_size 1024 --depth 24 --num_heads 16 \
  --max_len_seq 1024 \
  --num_workers 4 --save_interval 10000 --log_interval 20 --image_size 256 \
  --max_grad_norm 1.0 --weight_decay 0.0 --seed 1 --shuffle_buffer_size 2000 \
  --norm_stats_path "$REPO_ROOT/norm_stats/libero.json" \
  --use_warmup
