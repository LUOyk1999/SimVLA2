"""SimVLA2 configuration.

Fixed model recipe:
  * vision: SigLIP2 encoder (loaded via ``siglip_model_path``)
  * text:   random-init embedding at the vision hidden size
  * head:   flow-matching, ``concat`` conditioning
"""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class SimVLA2Config(PretrainedConfig):
    """Configuration for the SimVLA2 model."""

    model_type = "simvla2"

    def __init__(
        self,
        # SigLIP2 vision encoder
        siglip_model_path: str = "google/siglip2-large-patch16-256",

        # Text embedding
        tokenizer_name: str = "bert-base-uncased",
        vocab_size: int = 30522,

        # Action transformer head
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dim_time: int = 32,
        max_len_seq: int = 512,

        # Action and proprio
        num_actions: int = 30,
        action_mode: str = "bridge",
        # None: checkpoint predates this field, so the flag is unknown.
        use_quantile_norm: Optional[bool] = None,

        # Images
        image_size: int = 256,
        num_views: int = 3,

        **kwargs,
    ):
        self.siglip_model_path = siglip_model_path

        self.tokenizer_name = tokenizer_name
        self.vocab_size = vocab_size

        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq

        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_quantile_norm = use_quantile_norm

        self.image_size = image_size
        self.num_views = num_views

        super().__init__(**kwargs)
