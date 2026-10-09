#!/usr/bin/env python3
"""Train SimVLA2: SigLIP2 vision, a fresh text embedding and a flow-matching head.

The per-benchmark recipes are ``training/train_<bench>.sh``. A minimal run:

    python -m accelerate.commands.launch --mixed_precision bf16 train_simvla2.py \
        --output_dir runs/bridge \
        --train_metas_path datasets/metas/bridge.json \
        --norm_stats_path norm_stats/bridge.json \
        --action_mode bridge
"""

import os
import math
import time
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import create_dataloader
from models.modeling_simvla2 import NORM_STATS_FILE, SimVLA2
from models.processing_simvla2 import SimVLA2Processor

import logging
import sys

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None



# -----------------------------------------------------------------------------
# EMA (exponential moving average)
# -----------------------------------------------------------------------------
class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights, updated each step via:
        shadow = decay * shadow + (1 - decay) * model_param

    The EMA model is saved alongside the regular checkpoint for evaluation.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        """Replace model params with EMA shadow (for saving/eval)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore original model params after apply_shadow."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser("SimVLA2 Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, default=None,
                        help="Path to pretrained SimVLA2 checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="runs",
                        help="Directory to save checkpoints")

    # Vision encoder path (SigLIP2)
    parser.add_argument("--siglip_model_path", type=str,
                        default="google/siglip2-large-patch16-256",
                        help="Path or HF repo for the SigLIP2 vision encoder weights.")

    # Text embedding
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                        help="Tokenizer for text encoding")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True,
                        help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for SigLIP2 (default: 256)")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef_vision", type=float, default=0.05,
                        help="LR multiplier for SigLIP2 vision encoder")
    parser.add_argument("--learning_coef_text", type=float, default=None,
                        help="LR multiplier for the text embedding (default: 1.0)")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--use_warmup", action="store_true", default=False,
                        help="Apply linear warmup without cosine decay. Ignored when "
                             "--use_cosine_decay is set, which already warms up.")

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    # Action mode
    parser.add_argument("--action_mode", type=str, required=True,
                        help="Action mode: bridge, libero, calvin or fractal")

    # Data loading
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer_size", type=int, default=2000)

    # Normalization
    parser.add_argument("--norm_stats_path", type=str, default=None)
    parser.add_argument("--use_quantile_norm", action="store_true", default=False,
                        help="Normalize with (x - q01) / (q99 - q01) * 2 - 1 instead of "
                             "z-score. Requires q01/q99 in the norm stats. Not every "
                             "action space accepts it (joint modes do not).")

    # Action horizon
    parser.add_argument("--num_actions", type=int, default=30)

    # Language augmentation
    parser.add_argument("--lang_aug", type=str, default=None,
                        help="Path to language augmentation map JSON")
    parser.add_argument("--lang_dropout_prob", type=float, default=0.0,
                        help="Language dropout probability. >0 enables null conditioning "
                             "and includes unlabeled episodes (default: 0 = skip them)")

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--max_len_seq", type=int, default=512,
                        help="Max positional embedding length for action transformer. "
                             "Must be >= num_actions + num_views*num_patches + language_max_length. "
                             "E.g. 30 actions + 2 views x 256 patches + 50 tokens = 592, so use >= 1024")

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay rate. 0 = disabled. Typical: 0.9999")

    # Gradient accumulation
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (default: 1 = no accumulation)")

    return parser


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: SimVLA2, lr: float, weight_decay: float, betas=(0.9, 0.95)):
    """AdamW with groups vision, text_embedding, transformer_core and action_heads.

    Every group starts at lr 0 except the action heads; ``update_group_lrs``
    sets the real values each step.
    """
    vision_params = list(model.vision_model.parameters())
    text_embed_params = list(model.text_embedding.parameters())

    action_params = (list(model.transformer.action_decoder.parameters())
                     + list(model.transformer.action_encoder.parameters()))

    exclude = set(map(id, vision_params + text_embed_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]

    param_groups = [
        {"name": "vision", "params": vision_params,
         "lr": 0.0, "weight_decay": weight_decay},
        {"name": "text_embedding", "params": text_embed_params,
         "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params,
         "lr": 0.0, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params,
         "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups:
        if g["name"] == name:
            g["lr"] = lr


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio=0.1):
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Set every group's lr for ``step``.

    For the first ``freeze_steps`` only the action heads train, at the full
    base lr. After that each group follows the schedule (cosine, linear warmup
    or constant), scaled by its ``learning_coef_*``.
    """
    lr_coef_text = args.learning_coef_text
    if lr_coef_text is None:
        lr_coef_text = 1.0

    base = {
        "vision": args.learning_rate * args.learning_coef_vision,
        "text_embedding": args.learning_rate * lr_coef_text,
        "transformer_core": args.learning_rate,
        "action_heads": args.learning_rate,
    }

    def schedule(step, base_lr):
        if args.use_cosine_decay:
            return linear_warmup_cosine(
                step, args.freeze_steps, args.warmup_steps, args.iters, base_lr
            )
        if args.use_warmup:
            progress = step - args.freeze_steps
            if progress < args.warmup_steps:
                return base_lr * (progress / max(1, args.warmup_steps))
        return base_lr

    if step < args.freeze_steps:
        set_group_lr(optim, "vision", 0.0)
        set_group_lr(optim, "text_embedding", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            set_group_lr(optim, name, schedule(step, base_lr))


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def main(args):
    output_dir = Path(args.output_dir)

    # wandb is opt-in through the WANDB_API_KEY / WANDB_PROJECT env vars.
    wandb_project = os.environ.get("WANDB_PROJECT")
    use_wandb = WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY")

    log_with = ["tensorboard"]
    if use_wandb:
        log_with.append("wandb")

    # Every parameter gets a gradient on every step, so DDP can skip the
    # unused-parameter scan.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with=log_with,
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    tracker_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "siglip_model_path": args.siglip_model_path,
        "tokenizer_name": args.tokenizer_name,
        "freeze_steps": args.freeze_steps,
        "warmup_steps": args.warmup_steps,
        "use_warmup": args.use_warmup,
        "use_cosine_decay": args.use_cosine_decay,
        "action_mode": args.action_mode,
        "num_actions": args.num_actions,
        "image_size": args.image_size,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
    }

    if use_wandb:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"simvla2-{time.strftime('%Y%m%d-%H%M%S')}"}}
        )
    else:
        accelerator.init_trackers("SimVLA2-Training", config=tracker_config)

    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {vars(args)}")
    logger.info(f"SigLIP2 vision: {args.siglip_model_path}")
    logger.info(f"Text embedding: random init, tokenizer={args.tokenizer_name}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")

    from models.configuration_simvla2 import SimVLA2Config
    from models.action_hub import build_action_space

    action_space_kwargs = {}
    if args.norm_stats_path:
        action_space_kwargs["norm_stats_path"] = args.norm_stats_path
        logger.info(f"Using normalization stats from: {args.norm_stats_path}")
    # Only forwarded when set, so the action spaces that take no kwargs keep working.
    if args.use_quantile_norm:
        action_space_kwargs["use_quantile_norm"] = True
        logger.info("Normalization mode: quantile (q01/q99)")

    load_path = args.models

    if load_path and os.path.isdir(load_path) and os.path.exists(os.path.join(load_path, "model.safetensors")):
        logger.info(f"Loading SimVLA2 from checkpoint: {load_path}")
        model = SimVLA2.from_pretrained(load_path)

        if args.action_mode != model.action_mode:
            logger.warning(f"Overriding action_mode from '{model.action_mode}' to '{args.action_mode}'")
            model.action_mode = args.action_mode
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
        elif action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)

        if args.num_actions != model.num_actions:
            logger.warning(f"Overriding num_actions from {model.num_actions} to {args.num_actions}")
            model.config.num_actions = args.num_actions
            model.num_actions = args.num_actions
    else:
        logger.info("Initializing SimVLA2 from config")
        config = SimVLA2Config(
            siglip_model_path=args.siglip_model_path,
            tokenizer_name=args.tokenizer_name,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            max_len_seq=args.max_len_seq,
            action_mode=args.action_mode,
            num_actions=args.num_actions,
            image_size=args.image_size,
        )
        model = SimVLA2(config)

        if action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)

    # Saved into every checkpoint so inference normalizes exactly as training did.
    model.config.use_quantile_norm = bool(model.action_space.use_quantile_norm)
    norm_stats_src = args.norm_stats_path
    if norm_stats_src is None and load_path and os.path.exists(os.path.join(load_path, NORM_STATS_FILE)):
        norm_stats_src = os.path.join(load_path, NORM_STATS_FILE)
    if norm_stats_src is None:
        logger.warning("No norm stats: checkpoints will not carry a norm_stats.json")

    # Make parameters contiguous for DDP efficiency
    for p in model.parameters():
        p.data = p.data.contiguous()

    vision_p = sum(p.numel() for p in model.vision_model.parameters()) / 1e6
    text_p = sum(p.numel() for p in model.text_embedding.parameters()) / 1e6
    action_p = sum(p.numel() for p in model.transformer.parameters()) / 1e6
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: vision={vision_p:.1f}M, text_embed={text_p:.1f}M, "
                f"action_transformer={action_p:.1f}M, total={total_p:.1f}M")

    processor = SimVLA2Processor(
        tokenizer_name=args.tokenizer_name,
        image_size=args.image_size,
    )

    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        training=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        lang_aug=getattr(args, 'lang_aug', None),
        lang_dropout_prob=getattr(args, 'lang_dropout_prob', 0.0),
    )
    logger.info(f"DataLoader: shuffle_buffer={args.shuffle_buffer_size}")

    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # bf16 trainable parameters can stop moving without any error: AdamW's
    # step is about lr, and a step below half a bf16 ULP rounds back to the
    # original value. Mixed precision is unaffected -- autocast casts
    # activations, not parameters. SIGLIP_VISION_BF16=1 (see
    # models/modeling_simvla2.py) loads the tower in bf16 and skips this guard.
    _vision_bf16 = os.environ.get("SIGLIP_VISION_BF16", "0") == "1"
    if not _vision_bf16:
        wrong_dtype = [
            (name, param.dtype)
            for name, param in model.named_parameters()
            if param.requires_grad and param.dtype != torch.float32
        ]
        if wrong_dtype:
            raise RuntimeError(
                f"trainable parameters must be fp32, found {len(wrong_dtype)} that are not, "
                f"e.g. {wrong_dtype[:3]}"
            )
    else:
        bf16_trainable = sum(
            1 for _, p in model.named_parameters()
            if p.requires_grad and p.dtype == torch.bfloat16
        )
        logger.warning(
            f"SIGLIP_VISION_BF16=1: skipping fp32-trainable guard; "
            f"{bf16_trainable} trainable params are bf16"
        )

    model, optim = accelerator.prepare(model, optim)

    ema = None
    if args.ema_decay > 0:
        ema = EMAModel(accelerator.unwrap_model(model), decay=args.ema_decay)
        logger.info(f"EMA enabled with decay={args.ema_decay}")

    model.train()

    global_step, t0 = 0, time.time()
    logger.info(f"Start SimVLA2 training for {args.iters} iterations")
    logger.info(f"  world_size={accelerator.num_processes}")

    for batch in train_dataloader:
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        update_group_lrs(optim, global_step, args)

        with accelerator.accumulate(model):
            loss_dict: Dict[str, torch.Tensor] = model(**inputs)
            # Only keys ending in _loss enter the objective; anything else the
            # model returns is for logging.
            loss = sum(v for k, v in loss_dict.items() if k.endswith("_loss"))

            accelerator.backward(loss)
            # accelerate guards step() and zero_grad() against partial accumulation
            # but not clip_grad_norm_, which would otherwise rescale a half-built
            # gradient and leave the accumulated one no longer equal to the
            # single-batch gradient it is supposed to reproduce. sync_gradients is
            # True on every step when accumulation is off, so this is a no-op there.
            if args.max_grad_norm and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            optim.zero_grad()

        # EMA update, once per optimizer step: updating on micro-batches would
        # average the same unchanged weights repeatedly and shorten the effective
        # decay horizon by the accumulation factor.
        if ema is not None and accelerator.sync_gradients:
            ema.update(accelerator.unwrap_model(model))

        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"lr_vision={logs['lr_vision']:.2e} "
                    f"lr_text={logs['lr_text_embedding']:.2e} "
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_action={logs['lr_action_heads']:.2e} ({dt:.2f}s/it)"
                )

        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)
                if norm_stats_src:
                    shutil.copyfile(norm_stats_src, os.path.join(save_dir, NORM_STATS_FILE))

                if ema is not None:
                    ema_dir = os.path.join(output_dir, f"ckpt-{global_step}-ema")
                    accelerator.print(f"Saving EMA model to {ema_dir}")
                    unwrapped = accelerator.unwrap_model(model)
                    ema.apply_shadow(unwrapped)
                    unwrapped.save_pretrained(ema_dir, safe_serialization=True)
                    with open(os.path.join(ema_dir, "state.json"), "w") as f:
                        json.dump({"global_step": global_step, "ema_decay": args.ema_decay}, f)
                    if norm_stats_src:
                        shutil.copyfile(norm_stats_src, os.path.join(ema_dir, NORM_STATS_FILE))
                    ema.restore(unwrapped)

        if global_step >= args.iters:
            break

    accelerator.end_training()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SimVLA2 training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
