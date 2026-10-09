#!/usr/bin/env bash
# SimVLA2 CALVIN training entry.
#
# Benchmark : CALVIN task_ABC_D (long-horizon lang-conditioned tabletop)
#
# Usage: bash training/train_calvin.sh [NUM_GPUS] [GPU_IDS]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS="${1:-4}"
GPU_IDS="${2:-0,1,2,3}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

OUT="${OUT:-$REPO_ROOT/runs/calvin}"
PORT="${PORT:-29851}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip2-large-patch16-256}"
TOKENIZER="${TOKENIZER:-bert-base-uncased}"

mkdir -p "$OUT"
exec "${SIMVLA2_PY:-python}" -m accelerate.commands.launch \
  --num_processes "$NUM_GPUS" --main_process_port "$PORT" --mixed_precision bf16 \
  "$REPO_ROOT/train_simvla2.py" \
  --output_dir "$OUT" \
  --train_metas_path "$REPO_ROOT/datasets/metas/calvin.json" \
  --siglip_model_path "$SIGLIP_MODEL" \
  --tokenizer_name "$TOKENIZER" \
  --action_mode calvin --num_actions 30 --batch_size 32 \
  --learning_rate 1e-4 --learning_coef_vision 0.1 --learning_coef_text 1.0 \
  --use_cosine_decay \
  --iters 200000 --warmup_steps 2000 --freeze_steps 1000 \
  --hidden_size 1024 --depth 24 --num_heads 16 --max_len_seq 1024 \
  --num_workers 8 --save_interval 5000 --log_interval 50 --image_size 256 \
  --max_grad_norm 1.0 --weight_decay 0.0 --seed 0 --shuffle_buffer_size 2000 \
  --norm_stats_path "$REPO_ROOT/norm_stats/calvin.json"
