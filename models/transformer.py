"""Action transformer for SimVLA2.

Fixed recipe:
  * concat conditioning: VLM features projected into ``hidden_size`` and
    concatenated with the (action + proprio + time) tokens
  * flow-matching head: action tokens are the noisy inputs, the model
    predicts the velocity ``noise - x0``

  x_action = encode([action_noise, proprio, time_emb])   # [B, num_actions, H]
  x_vlm    = vlm_proj(vlm_features)                      # [B, T_vlm,  H]
  x        = cat([x_action, x_vlm]) + pos_emb
  y        = decode(norm(blocks(x)))[:, :num_actions]    # velocity
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def basic_init(module: nn.Module) -> None:
    """Xavier-uniform for ``nn.Linear`` weights, zero bias."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """Sinusoidal timestep embedding (same convention as DiT / DDPM)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Mlp(nn.Module):
    """MLP used in ViT-style blocks."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention on PyTorch's fused SDPA."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x.device, dtype=torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() != 4:
                raise ValueError(
                    f"attn_mask must have rank 2, 3, or 4, got {attn_mask.dim()}."
                )

        # SDPA's boolean mask means True = attend, so invert our True = block.
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~attn_mask if attn_mask is not None else None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block (pre-LN)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ActionTransformer(nn.Module):
    """Concat-conditioned Transformer for SimVLA2's flow-matching action head.

    Layout per step:
      x = cat([action_encoder([noise, proprio, time_emb]),  vlm_proj(vlm)])
      x = x + pos_emb
      y = action_decoder(norm(blocks(x)))[:, :num_actions]
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        vlm_hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dim_action: int = 26,
        dim_propio: int = 21,
        dim_time: int = 32,
        max_len_seq: int = 1024,
        num_actions: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.num_actions = num_actions

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)

        action_input_dim = dim_action + dim_time + dim_propio
        self.action_encoder = nn.Linear(action_input_dim, hidden_size)
        self.action_decoder = nn.Linear(hidden_size, dim_action)

        self.apply(basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,                      # [B, T_vlm, D_vlm]
        vlm_attention_mask: torch.Tensor | None = None,  # [B, T_vlm], True = attend
        action_with_noise: torch.Tensor | None = None,   # [B, num_actions, dim_action]
        proprio: torch.Tensor | None = None,             # [B, dim_proprio]
        t: torch.Tensor | None = None,                   # [B]
    ) -> torch.Tensor:
        """Predict velocity of the flow-matching action chunk.

        Returns
        -------
        Tensor: predicted velocity, shape ``[B, num_actions, dim_action]``
        """
        B, num_actions = action_with_noise.shape[:2]

        # Encode (action + proprio + time) into per-step tokens.
        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])

        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens)  # [B, N, H]

        # Concat VLM features and add positional embedding.
        x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} (action={num_actions}, "
                f"vlm={vlm_features.shape[1]}) exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        x = x + self.pos_emb[:, :seq_len, :]

        # Build key-padding mask over the concatenated stream, if provided.
        attn_mask = None
        if vlm_attention_mask is not None:
            if vlm_attention_mask.shape != vlm_features.shape[:2]:
                raise ValueError(
                    "vlm_attention_mask shape must match vlm_features[:2], "
                    f"got {tuple(vlm_attention_mask.shape)} and {tuple(vlm_features.shape[:2])}."
                )
            token_valid = torch.cat(
                [
                    torch.ones(B, num_actions, device=x.device, dtype=torch.bool),
                    vlm_attention_mask.to(device=x.device, dtype=torch.bool),
                ],
                dim=1,
            )
            # Convention (see Attention.forward): True = block.
            attn_mask = (~token_valid)[:, None, None, :]

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        return self.action_decoder(self.norm(x[:, :num_actions]))
