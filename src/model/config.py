"""Aurelius 1.3B model configuration."""

from dataclasses import dataclass


@dataclass
class AureliusConfig:
    """Hyperparameters for the Aurelius 1.3B transformer.

    Architecture: decoder-only transformer with GQA, SwiGLU FFN, RoPE, RMSNorm.
    Target parameter count: ~1.3B.
    """

    # Model dimensions
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16       # query heads
    n_kv_heads: int = 8     # key/value heads (GQA ratio 2:1)
    head_dim: int = 128      # d_model // n_heads
    d_ff: int = 5632         # SwiGLU intermediate dim ≈ 2/3 × 4 × d_model

    # Vocabulary and sequence
    vocab_size: int = 128_000
    max_seq_len: int = 8192

    # RoPE
    rope_theta: float = 500_000.0

    # RoPE scaling (YaRN / NTK / linear context extension)
    rope_scaling_type: str = "none"          # "none", "linear", "ntk", "yarn"
    rope_scaling_factor: float = 1.0         # e.g. 16.0 for 8K → 128K
    yarn_original_max_seq_len: int = 8192    # original trained context length
    yarn_beta_fast: float = 32.0             # high-freq boundary
    yarn_beta_slow: float = 1.0              # low-freq boundary

    # Normalization
    rms_norm_eps: float = 1e-6

    # Regularization
    dropout: float = 0.0

    # Embedding
    tie_embeddings: bool = True

    # NoPE: remove RoPE from every N-th layer (SmolLM3 technique)
    # 0 = disabled, 4 = every 4th layer (25% NoPE, matching SmolLM3's 3:1 ratio)
    nope_every_n_layers: int = 0

    # Differential Attention (ICLR 2025, Microsoft Research)
    # Computes two attention maps and takes their difference to cancel noise
    use_diff_attn: bool = False
    diff_attn_lambda_init: float = 0.8  # initial λ value

    # Mixture of Depths (DeepMind, 2024)
    # Routes only the top-k tokens through each layer; the rest skip via residual
    use_mod: bool = False
    mod_capacity_factor: float = 0.5  # fraction of tokens routed per layer

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) "
            f"* head_dim ({self.head_dim})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
