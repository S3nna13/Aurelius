"""Universal Transformers: weight-shared transformer blocks applied repeatedly.

Implements the Universal Transformer from:
    Dehghani et al. 2018 — "Universal Transformers"
    https://arxiv.org/abs/1807.03819

Unlike standard transformers where each layer has its own independent weights,
Universal Transformers apply a SINGLE shared block N times (steps). This
reduces parameter count roughly N-fold while maintaining representational depth.

Each step is augmented with a learned step embedding so the block knows which
iteration it is performing ("I am on step 3 of 6"), enabling the model to
exhibit different behavior at each depth without separate parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention, precompute_rope_frequencies
from .config import AureliusConfig
from .ffn import SwiGLUFFN
from .rms_norm import RMSNorm


class UniversalTransformerBlock(nn.Module):
    """Single shared transformer block applied repeatedly.

    Unlike standard transformers where each layer has its own weights,
    Universal Transformers apply the SAME block multiple times.
    This reduces parameter count while maintaining depth.

    The hidden state is augmented with both token position (via RoPE) and a
    step (depth) embedding so the block can behave differently at each step
    without separate per-step parameters.

    Args:
        config: AureliusConfig containing model hyperparameters.
    """

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)

        # Step embedding: learned vector per depth step (up to 32 steps).
        # Added to x before each application so the block knows its iteration.
        self.step_embedding = nn.Embedding(32, config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        step: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one step of the Universal Transformer.

        Args:
            x: Input tensor of shape ``(B, S, d_model)``.
            freqs_cis: RoPE frequency tensor of shape ``(S, head_dim // 2)``.
            step: Which step number (0-indexed). Used to select a step embedding
                  that is broadcast-added to x before the residual blocks.
            mask: Optional attention mask.

        Returns:
            Output tensor of shape ``(B, S, d_model)``.
        """
        # Add step embedding so the block knows its iteration index.
        # step_emb: (d_model,) — broadcast over batch and sequence dimensions.
        step_idx = torch.tensor(step, device=x.device)
        step_emb = self.step_embedding(step_idx)  # (d_model,)
        x = x + step_emb  # broadcast: (B, S, d_model)

        # Pre-norm attention residual
        attn_out, _kv = self.attn(self.norm1(x), freqs_cis, mask=mask, past_kv=None)
        x = x + attn_out

        # Pre-norm FFN residual
        x = x + self.ffn(self.norm2(x))

        return x


class UniversalTransformer(nn.Module):
    """Universal Transformer: applies one shared block N times.

    A single ``UniversalTransformerBlock`` (NOT a ModuleList) is applied
    ``n_steps`` times to the hidden states.  Because weights are shared across
    steps, the parameter count is roughly ``n_steps``-fold smaller than a
    standard transformer of equivalent depth.

    Args:
        config: AureliusConfig containing model hyperparameters.
        n_steps: How many times to apply the shared block.
                 Defaults to ``config.n_layers``.
        use_act: Reserved for Adaptive Computation Time halting (not yet
                 implemented). Must be ``False``.
    """

    def __init__(
        self,
        config: AureliusConfig,
        n_steps: int | None = None,
        use_act: bool = False,
    ) -> None:
        super().__init__()
        if use_act:
            raise NotImplementedError("ACT halting is not yet implemented.")

        self.config = config
        self.n_steps = n_steps if n_steps is not None else config.n_layers

        # Single shared block — NOT a ModuleList; same weights reused every step.
        self.shared_block = UniversalTransformerBlock(config)

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Final layer norm
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

        # Output projection (no bias, following Aurelius convention)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute RoPE frequencies as a non-persistent buffer
        self._register_rope(config)

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register_rope(self, config: AureliusConfig) -> None:
        freqs = precompute_rope_frequencies(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, None]:
        """Run Universal Transformer forward pass.

        Args:
            input_ids: Token indices of shape ``(B, S)``.
            labels: Optional target token ids of shape ``(B, S)`` for computing
                    cross-entropy loss (next-token prediction with 1-token shift).

        Returns:
            Tuple of ``(loss, logits, None)``:
                - ``loss``: Scalar cross-entropy loss, or ``None`` if no labels.
                - ``logits``: ``(B, S, vocab_size)`` unnormalised token scores.
                - Third element is always ``None`` (no KV cache support).
        """
        B, S = input_ids.shape
        assert S <= self.config.max_seq_len, (  # noqa: S101
            f"Sequence length {S} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.embed(input_ids)  # (B, S, d_model)
        freqs_cis = self.freqs_cis[:S]  # (S, head_dim // 2)

        # Apply shared block n_steps times, each time with a unique step index.
        for step in range(self.n_steps):
            x = self.shared_block(x, freqs_cis, step=step)

        x = self.norm(x)  # (B, S, d_model)
        logits = self.lm_head(x)  # (B, S, vocab_size)

        loss: torch.Tensor | None = None
        if labels is not None:
            # Standard next-token prediction: shift by 1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return loss, logits, None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        """Return the number of unique (non-shared) trainable parameters.

        Because ``shared_block`` is a single module, ``parameters()`` already
        yields each tensor exactly once — no de-duplication needed.
        """
        return sum(p.numel() for p in self.parameters())

    def effective_depth(self) -> int:
        """Return the conceptual depth (number of times the block is applied)."""
        return self.n_steps
