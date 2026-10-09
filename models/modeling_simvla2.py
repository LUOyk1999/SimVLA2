"""SimVLA2 Model.

Vision-Language-Action policy with a fixed recipe:
  * vision : SigLIP2 encoder (standalone, fp32 by default,
             bf16 + FlashAttention2 when ``SIGLIP_VISION_BF16=1``)
  * text   : random-init ``nn.Embedding`` at the vision hidden size
  * head   : flow-matching, ``concat`` conditioning

Pipeline:
  pixel_values -> SigLIP2       -> [B, num_patches, vision_dim]
  input_ids    -> Embedding     -> [B, L, vision_dim]
                    |
                    concat -> [B, num_patches + L, vision_dim]
                    |
              ActionTransformer -> predicted velocity
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from transformers import AutoConfig, PreTrainedModel, SiglipVisionModel
from .transformer import ActionTransformer
from .action_hub import build_action_space
from .configuration_simvla2 import SimVLA2Config

# Copy of the training norm stats, written next to model.safetensors.
NORM_STATS_FILE = "norm_stats.json"


class SimVLA2(PreTrainedModel):
    """Lightweight Vision-Language-Action policy."""

    config_class = SimVLA2Config
    base_model_prefix = "simvla2"
    supports_gradient_checkpointing = True

    def __init__(self, config: SimVLA2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.num_actions: int = config.num_actions
        self.action_mode: str = config.action_mode.lower()

        # Action space
        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # Vision encoder: SigLIP2. fp32 by default; SIGLIP_VISION_BF16=1
        # loads bf16 weights + FlashAttention-2. The fp32 path uses SDPA,
        # since FA2 does not accept fp32 inputs outside autocast.
        logging.info(f"Loading SigLIP2 vision from: {config.siglip_model_path}")
        _vision_bf16 = os.environ.get("SIGLIP_VISION_BF16", "0") == "1"
        if _vision_bf16:
            try:
                self.vision_model = SiglipVisionModel.from_pretrained(
                    config.siglip_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                logging.warning("SIGLIP_VISION_BF16=1: SigLIP vision bf16 + flash_attention_2")
            except (RuntimeError, ImportError, ValueError):
                self.vision_model = SiglipVisionModel.from_pretrained(
                    config.siglip_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                logging.warning("SIGLIP_VISION_BF16=1: FA2 unavailable, bf16 + sdpa")
        else:
            try:
                self.vision_model = SiglipVisionModel.from_pretrained(
                    config.siglip_model_path,
                    torch_dtype=torch.float32,
                    attn_implementation="sdpa",
                )
            except (OSError, EnvironmentError):
                # Only reachable when the base weights are absent, e.g. when a
                # checkpoint load is about to overwrite them anyway.
                vision_cfg = AutoConfig.from_pretrained(config.siglip_model_path)
                self.vision_model = SiglipVisionModel(vision_cfg)
                logging.warning("SigLIP2 weights not loaded; vision tower is randomly initialized.")
        if hasattr(self.vision_model.vision_model, "head"):
            self.vision_model.vision_model.use_head = False
            del self.vision_model.vision_model.head

        # Activation checkpointing on the SigLIP2 vision tower is required for
        # the shipped bs=64 recipes to fit in 80 GB of GPU memory.
        if hasattr(self.vision_model, "gradient_checkpointing_enable"):
            self.vision_model.gradient_checkpointing_enable()

        vision_hidden_size = self.vision_model.config.hidden_size
        vision_native_size = getattr(self.vision_model.config, "image_size", None)
        logging.info(
            f"SigLIP2 vision hidden_size={vision_hidden_size}, "
            f"native_image_size={vision_native_size}"
        )

        # Text: fresh random-init nn.Embedding, sized to concat with the image tokens.
        self.text_embedding = nn.Embedding(config.vocab_size, vision_hidden_size)
        logging.info(f"Fresh text embedding: vocab={config.vocab_size}, dim={vision_hidden_size}")

        self.transformer = ActionTransformer(
            hidden_size=config.hidden_size,
            vlm_hidden_size=vision_hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            num_actions=config.num_actions,
        )

    # -------------------------------------------------------------------------
    # from_pretrained override
    # -------------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        **kwargs,
    ):
        """Load a SimVLA2 checkpoint, bypassing meta-device issues."""
        import os
        from safetensors.torch import load_file

        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)

        dtype = kwargs.pop("torch_dtype", None)
        model = cls(config)

        sf = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        state_dict = load_file(sf)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            logging.warning("SimVLA2: unexpected keys: %s", unexpected[:5])
        non_head_missing = [k for k in missing if "head" not in k]
        if non_head_missing:
            logging.warning("SimVLA2: missing keys: %s", non_head_missing[:5])

        norm_stats_path = os.path.join(pretrained_model_name_or_path, NORM_STATS_FILE)
        if os.path.exists(norm_stats_path):
            model.action_space = build_action_space(
                model.action_mode,
                norm_stats_path=norm_stats_path,
                use_quantile_norm=bool(config.use_quantile_norm),
            )

        if dtype is not None:
            model = model.to(dtype=dtype)
            # With SIGLIP_VISION_BF16=1 vision is built with flash_attention_2 in
            # __init__, but FlashAttention only supports fp16/bf16. After casting the
            # full model to float32, use SDPA or inference fails with:
            # "FlashAttention only support fp16 and bf16 data type".
            if dtype == torch.float32 and getattr(
                model.vision_model.config, "_attn_implementation", None
            ) == "flash_attention_2":
                model.vision_model.config._attn_implementation = "sdpa"
        model.eval()
        return model

    # -------------------------------------------------------------------------
    # VLM forward
    # -------------------------------------------------------------------------
    def forward_vlm(
        self,
        pixel_values: torch.FloatTensor,                     # [B, V, C, H, W]
        image_mask: torch.Tensor,                            # [B, V]
        input_ids: torch.LongTensor,                         # [B, L]
        text_attention_mask: Optional[torch.Tensor] = None,  # [B, L], BERT-style 1/0
    ) -> Dict[str, torch.Tensor]:
        """Encode images with SigLIP2 and text with embedding, then concatenate.

        Pipeline (D = SigLIP2 hidden size, 1024 for siglip2-large):
          pixel_values -> SigLIP2 -> image_features [N_valid, num_patches, D]
          input_ids    -> Embedding -> text_features [B, L, D]
          concat per sample -> vlm_features [B, T_total, D]

        Returns:
          {"vlm_features": [B, T_total, D]}
          plus "vlm_attention_mask": [B, T_total] (1=attend, 0=block) when
          text_attention_mask is provided. Masks out text padding and the
          per-sample right-pad added when views vary within a batch.
        """
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype

        # Only the valid views go through the vision tower.
        flat_images = pixel_values.flatten(0, 1)        # [B*V, C, H, W]
        flat_mask = image_mask.view(-1).bool()
        valid_images = flat_images[flat_mask]            # [N_valid, C, H, W]

        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")

        # SigLIP2's position table is sized to the checkpoint's native resolution
        # (256 here), so any other image_size would fail on the shape of the add.
        # transformers short-circuits back to the un-interpolated table when the
        # patch count already matches, so this is a no-op at 256.
        vision_out = self.vision_model(
            pixel_values=valid_images,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        image_features = vision_out.last_hidden_state    # [N_valid, num_patches, D]

        text_features = self.text_embedding(input_ids)

        # Put the image features back in their [B, V] slots, then concatenate each
        # sample's valid views with its text.
        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]

        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)

        valid_per_sample = image_mask.sum(dim=1).int()   # [B]

        batch_embeds = []
        batch_lens = []
        max_seq_len = 0

        for b in range(B):
            n_valid = valid_per_sample[b].item()
            sample_img = full_image_features[b, :n_valid].reshape(-1, hidden_size)
            sample_txt = text_features[b]
            combined = torch.cat([sample_img, sample_txt], dim=0)
            batch_embeds.append(combined)
            batch_lens.append((n_valid * num_patches, sample_txt.shape[0]))
            max_seq_len = max(max_seq_len, combined.shape[0])

        out_dtype = batch_embeds[0].dtype if len(batch_embeds) > 0 else dtype
        vlm_features = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=out_dtype)
        for b, embeds in enumerate(batch_embeds):
            vlm_features[b, :embeds.shape[0]] = embeds

        out = {"vlm_features": vlm_features}
        if text_attention_mask is not None:
            # Build [B, max_seq_len] where 1=attend, 0=block. Image slots are
            # always real (image_mask handled view-level, invalid views are
            # dropped rather than zero-padded here); text slots take the BERT
            # attention_mask; the tail past (n_img + L) is right-pad from the
            # ragged concat above and must be masked.
            attn = torch.zeros(B, max_seq_len, device=device, dtype=torch.bool)
            text_am = text_attention_mask.to(device=device, dtype=torch.bool)
            for b, (n_img_tokens, n_txt_tokens) in enumerate(batch_lens):
                if n_img_tokens > 0:
                    attn[b, :n_img_tokens] = True
                attn[b, n_img_tokens : n_img_tokens + n_txt_tokens] = text_am[b, :n_txt_tokens]
            out["vlm_attention_mask"] = attn
        return out

    def _normalize_inputs(self, action, proprio):
        return self.action_space.normalize_action(action), self.action_space.normalize_state(proprio)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        image_input: torch.FloatTensor,      # [B, V, C, H, W]
        image_mask: torch.Tensor,            # [B, V]
        proprio: torch.Tensor,               # [B, dim_proprio]
        action: torch.Tensor,                # [B, T, dim_action]
        action_valid: Optional[torch.Tensor] = None,   # [B] valid action rows
        attention_mask: Optional[torch.Tensor] = None,   # [B, L] BERT text mask
    ) -> Dict[str, torch.Tensor]:
        """Flow-matching training step.

        ``action_valid`` is the per-sample count of real (non-padded) action
        rows. A chunk that runs past the end of an episode is padded with the
        final pose, and the loss drops those padded rows. When
        ``action_valid`` is None, the loss is a plain global mean.
        """
        enc = self.forward_vlm(image_input, image_mask, input_ids, attention_mask)
        action_norm, proprio_norm = self._normalize_inputs(action, proprio)
        return self._forward_flow_matching(enc, action_norm, proprio_norm, action_valid)

    @staticmethod
    def _masked_mean(
        sq_err: torch.Tensor,
        action_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mean over valid action rows.

        ``action_valid[i]`` is the number of non-padded rows for sample ``i``.
        When ``None``, this is a plain global mean. Otherwise the sum is
        divided by the number of valid elements in the batch.
        """
        if action_valid is None:
            return torch.mean(sq_err)
        H = sq_err.shape[1]
        steps = torch.arange(H, device=sq_err.device).view(1, -1)
        keep = (steps < action_valid.to(sq_err.device).view(-1, 1)).unsqueeze(-1)
        keep = keep.expand_as(sq_err)
        denom = keep.sum().clamp(min=1)
        return (sq_err * keep).sum() / denom

    def _forward_flow_matching(self, enc, action_norm, proprio_norm, action_valid=None) -> Dict[str, torch.Tensor]:
        """1) t ~ Beta(1.5, 1)  2) x_t = t*noise + (1-t)*x0  3) target noise - x0."""
        B = action_norm.shape[0]
        device = action_norm.device

        beta_dist = torch.distributions.Beta(
            torch.tensor(1.5, device=device),
            torch.tensor(1.0, device=device),
        )
        t = beta_dist.sample((B,)) * 0.999 + 0.001

        noise = torch.randn_like(action_norm)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * noise + (1 - t_expanded) * action_norm
        u_t = noise - action_norm

        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            vlm_attention_mask=enc.get("vlm_attention_mask"),
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
        )
        return {"velocity_loss": self._masked_mean(
            torch.square(v_t - u_t), action_valid,
        )}

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flow-matching inference."""
        self.eval()
        enc = self.forward_vlm(image_input, image_mask, input_ids, attention_mask)

        B = input_ids.shape[0]
        D = self.action_space.dim_action
        proprio_norm = self.action_space.normalize_state(proprio)

        x_0 = self._sample_flow_matching(
            enc, proprio_norm, B, D, proprio.device, proprio.dtype, steps,
        )
        return self.action_space.postprocess(x_0)

    def _sample_flow_matching(self, enc, proprio_norm, B, D, device, dtype, steps):
        """Euler integration from t=1 down to t=0."""
        steps = max(1, int(steps))
        dt = -1.0 / steps

        x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        t = 1.0

        while t > -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)
            v_t = self.transformer(
                vlm_features=enc["vlm_features"],
                vlm_attention_mask=enc.get("vlm_attention_mask"),
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t_tensor,
            )
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t
