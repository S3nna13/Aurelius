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

    # RoPE scaling for context extension
    rope_scaling_type: str = "none"   # "none" or "yarn"
    rope_scaling_factor: float = 1.0  # scale factor for YaRN (e.g., 4.0 for 4x extension)
    rope_original_max_seq_len: int = 8192  # original training context length

    # Normalization
    rms_norm_eps: float = 1e-6

    # Regularization
    dropout: float = 0.0

    # Embedding
    tie_embeddings: bool = True

    # Training efficiency
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) "
            f"* head_dim ({self.head_dim})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
