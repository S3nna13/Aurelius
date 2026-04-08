"""Multi-Token Prediction (MTP) heads for parallel decoding speedup.

Reference: Gloeckle et al. 2024 — "Better & Faster Large Language Models via
Multi-Token Prediction".

Key idea: Train k additional prediction heads to predict tokens at positions
t+2, t+3, ..., t+k (not just t+1). At inference, use these heads in parallel
to speculate multiple tokens ahead. Unlike speculative decoding, all heads are
part of the same model (no separate draft model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig
from .rms_norm import RMSNorm
from .transformer import TransformerBlock


# ---------------------------------------------------------------------------
# MTPConfig
# ---------------------------------------------------------------------------

@dataclass
class MTPConfig:
    """Configuration for Multi-Token Prediction heads."""

    n_heads: int = 4            # number of extra prediction steps
    shared_head: bool = False   # if True, all heads share weights
    head_type: str = "linear"   # "linear" | "transformer_layer"
    detach_hidden: bool = True  # detach hidden states for head inputs


# ---------------------------------------------------------------------------
# MTPHead
# ---------------------------------------------------------------------------

class MTPHead(nn.Module):
    """Single multi-token prediction head.

    Predicts token at t+step from hidden state at position t.

    Architecture options:
    - "linear": RMSNorm(d_model) → Linear(d_model, vocab_size)
    - "transformer_layer": 1 TransformerBlock + Linear

    Args:
        config: AureliusConfig
        step: int (which future token this head predicts, 1=next, 2=two ahead)
        head_type: str
    """

    def __init__(
        self,
        config: AureliusConfig,
        step: int,
        head_type: str = "linear",
    ) -> None:
        super().__init__()
        self.step = step
        self.head_type = head_type
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        if head_type == "linear":
            self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        elif head_type == "transformer_layer":
            self.block = TransformerBlock(config)
            self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
            # Precompute RoPE frequencies for transformer_layer heads
            from .attention import precompute_rope_frequencies
            freqs = precompute_rope_frequencies(
                config.head_dim, config.max_seq_len, config.rope_theta
            )
            self.register_buffer("freqs_cis", freqs, persistent=False)
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}. Use 'linear' or 'transformer_layer'.")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D)

        Returns:
            logits: (B, T, V)
        """
        if self.head_type == "linear":
            return self.proj(self.norm(hidden_states))
        else:
            # transformer_layer: pass through one block then project
            B, T, D = hidden_states.shape
            freqs_cis = self.freqs_cis[:T]
            x, _ = self.block(hidden_states, freqs_cis, mask=None, past_kv=None)
            return self.proj(x)


# ---------------------------------------------------------------------------
# MultiTokenPredictionModel
# ---------------------------------------------------------------------------

class MultiTokenPredictionModel(nn.Module):
    """Wraps a base transformer and adds k MTP heads.

    During training: returns losses from all k+1 heads (main + k auxiliary).
    During inference: uses heads for speculative multi-token decoding.

    Args:
        base_model: AureliusTransformer
        mtp_config: MTPConfig
    """

    def __init__(
        self,
        base_model: nn.Module,
        mtp_config: MTPConfig | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.mtp_config = mtp_config or MTPConfig()

        cfg: AureliusConfig = base_model.config

        # Build MTPHead instances
        if self.mtp_config.shared_head:
            # One canonical head; all steps share its parameters
            _canonical = MTPHead(cfg, step=1, head_type=self.mtp_config.head_type)
            self.mtp_heads = nn.ModuleList(
                [_canonical] * self.mtp_config.n_heads
            )
        else:
            self.mtp_heads = nn.ModuleList(
                [
                    MTPHead(cfg, step=k + 1, head_type=self.mtp_config.head_type)
                    for k in range(self.mtp_config.n_heads)
                ]
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hidden_states(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Run base model, return (hidden_before_norm, logits, pkv)."""
        # Capture last-layer output via a forward hook
        _hidden_cache: dict[str, torch.Tensor] = {}

        def _hook(module: nn.Module, inp: tuple, output: object) -> None:
            if isinstance(output, tuple):
                _hidden_cache["last"] = output[0]
            else:
                _hidden_cache["last"] = output

        handle = self.base_model.layers[-1].register_forward_hook(_hook)
        try:
            loss_base, logits, pkv = self.base_model(input_ids)
        finally:
            handle.remove()

        raw_hidden = _hidden_cache["last"]  # (B, T, D) — pre final-norm
        # Apply base model's final norm to match "base_model.norm(hidden)"
        hidden = self.base_model.norm(raw_hidden)
        return hidden, logits, pkv

    def _get_hidden_states_with_labels(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Run base model with labels, return (hidden, main_loss, logits, pkv)."""
        _hidden_cache: dict[str, torch.Tensor] = {}

        def _hook(module: nn.Module, inp: tuple, output: object) -> None:
            if isinstance(output, tuple):
                _hidden_cache["last"] = output[0]
            else:
                _hidden_cache["last"] = output

        handle = self.base_model.layers[-1].register_forward_hook(_hook)
        try:
            main_loss, logits, pkv = self.base_model(input_ids, labels=labels)
        finally:
            handle.remove()

        raw_hidden = _hidden_cache["last"]
        hidden = self.base_model.norm(raw_hidden)
        return hidden, main_loss, logits, pkv

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_all_logits: bool = False,
    ) -> tuple:
        """
        Returns (loss, logits, past_key_values) — same API as AureliusTransformer.

        If labels provided:
          - Run base model, capture hidden states from last layer
          - Each MTP head predicts shifted targets: head_k predicts labels[:, k+1:]
          - total_loss = (main_loss + sum(head_losses)) / (n_heads + 1)

        If return_all_logits=True: return list of logits from all heads.
        """
        if labels is not None:
            hidden, main_loss, logits, pkv = self._get_hidden_states_with_labels(
                input_ids, labels
            )

            if self.mtp_config.detach_hidden:
                hidden = hidden.detach()

            # Compute auxiliary head losses
            aux_losses: list[torch.Tensor] = []
            all_logits = [logits]

            for k, head in enumerate(self.mtp_heads):
                step = k + 1  # head k predicts token at t+step
                # Hidden: positions 0..T-step-1 → predict labels[:, step+1:]
                # (labels are already shifted by 1 relative to input in base model,
                #  so head_k targets are labels[:, step:] — the tokens at t+step+1
                #  relative to original input)
                # The base model's loss is: logits[:,:-1] vs labels[:,1:]
                # → head_k should predict: labels[:, step:] using hidden[:, :-step]
                if step >= hidden.shape[1]:
                    # Sequence too short for this head; skip
                    continue

                h_slice = hidden[:, :-step, :]    # (B, T-step, D)
                target = labels[:, step:]          # (B, T-step)

                head_logits = head(h_slice)        # (B, T-step, V)
                B, Ts, V = head_logits.shape

                head_loss = F.cross_entropy(
                    head_logits.reshape(B * Ts, V),
                    target.reshape(B * Ts),
                )
                aux_losses.append(head_loss)
                all_logits.append(head_logits)

            if aux_losses:
                total_loss = (main_loss + sum(aux_losses)) / (len(aux_losses) + 1)
            else:
                total_loss = main_loss

            if return_all_logits:
                return total_loss, all_logits, pkv
            return total_loss, logits, pkv

        else:
            # Inference path
            hidden, logits, pkv = self._get_hidden_states(input_ids)

            if return_all_logits:
                if self.mtp_config.detach_hidden:
                    hidden = hidden.detach()
                all_logits = [logits]
                for head in self.mtp_heads:
                    head_logits = head(hidden)
                    all_logits.append(head_logits)
                return None, all_logits, pkv

            return None, logits, pkv

    # ------------------------------------------------------------------
    # Parallel decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def parallel_decode(
        self,
        input_ids: torch.Tensor,
        n_tokens: int = 4,
    ) -> tuple[list[int], dict]:
        """Multi-token parallel decoding using MTP heads.

        One forward pass → k predictions simultaneously.
        Accept greedily: take argmax from main head as first token,
        argmax from head_k as (k+1)th token.

        n_tokens may be > mtp_config.n_heads — loop as needed.

        Returns (generated_token_ids, {'n_forward_passes': int, 'tokens_per_pass': float})
        """
        n_mtp = self.mtp_config.n_heads  # number of speculative steps per pass
        tokens_per_pass = 1 + n_mtp      # main head + k MTP heads

        generated: list[int] = []
        n_forward_passes = 0
        cur_ids = input_ids  # (1, T)

        while len(generated) < n_tokens:
            # Run base model + capture hidden states
            hidden, logits, _pkv = self._get_hidden_states(cur_ids)

            if self.mtp_config.detach_hidden:
                hidden = hidden.detach()

            n_forward_passes += 1

            # Main head: argmax of last position
            main_next = logits[:, -1, :].argmax(dim=-1)  # (B,)
            generated.append(main_next[0].item())

            if len(generated) >= n_tokens:
                break

            # MTP heads: each predicts one step ahead from last-position hidden
            h_last = hidden[:, -1:, :]  # (1, 1, D)
            for head in self.mtp_heads:
                if len(generated) >= n_tokens:
                    break
                head_logits = head(h_last)          # (1, 1, V)
                next_tok = head_logits[:, -1, :].argmax(dim=-1)
                generated.append(next_tok[0].item())

            if len(generated) >= n_tokens:
                break

            # Build new context for next pass: append all generated tokens so far
            new_tokens = torch.tensor(
                generated, dtype=input_ids.dtype, device=input_ids.device
            ).unsqueeze(0)  # (1, len(generated))
            cur_ids = torch.cat([input_ids, new_tokens], dim=1)

        # Trim to exactly n_tokens
        generated = generated[:n_tokens]

        stats = {
            "n_forward_passes": n_forward_passes,
            "tokens_per_pass": n_tokens / max(n_forward_passes, 1),
        }
        return generated, stats


# ---------------------------------------------------------------------------
# MTPTrainer
# ---------------------------------------------------------------------------

class MTPTrainer:
    """Trainer for multi-token prediction.

    Wraps a base AureliusTransformer with MTP heads and trains jointly.
    """

    def __init__(
        self,
        base_model: nn.Module,
        mtp_config: MTPConfig,
        optimizer: torch.optim.Optimizer,
        tokenizer_encode=None,
        max_seq_len: int = 512,
    ) -> None:
        self.mtp_model = MultiTokenPredictionModel(base_model, mtp_config)
        self.optimizer = optimizer
        self.tokenizer_encode = tokenizer_encode
        self.max_seq_len = max_seq_len

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Forward all MTP heads, compute weighted loss, backward, step.

        Returns: {'loss': float, 'main_loss': float, 'aux_loss': float}
        """
        self.mtp_model.train()
        self.optimizer.zero_grad()

        # Use input_ids as labels (standard causal LM: predict next token)
        labels = input_ids

        # Get hidden states and compute losses for each head individually
        hidden, main_loss, logits, pkv = self.mtp_model._get_hidden_states_with_labels(
            input_ids, labels
        )

        if self.mtp_model.mtp_config.detach_hidden:
            hidden = hidden.detach()

        # Compute auxiliary head losses
        aux_losses: list[torch.Tensor] = []
        for k, head in enumerate(self.mtp_model.mtp_heads):
            step = k + 1
            if step >= hidden.shape[1]:
                continue
            h_slice = hidden[:, :-step, :]
            target = labels[:, step:]
            head_logits = head(h_slice)
            B, Ts, V = head_logits.shape
            head_loss = F.cross_entropy(
                head_logits.reshape(B * Ts, V),
                target.reshape(B * Ts),
            )
            aux_losses.append(head_loss)

        if aux_losses:
            aux_loss_tensor = sum(aux_losses) / len(aux_losses)
            total_loss = (main_loss + sum(aux_losses)) / (len(aux_losses) + 1)
        else:
            aux_loss_tensor = torch.zeros_like(main_loss)
            total_loss = main_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "main_loss": main_loss.item(),
            "aux_loss": aux_loss_tensor.item(),
        }


# ---------------------------------------------------------------------------
# acceptance_rate_stats
# ---------------------------------------------------------------------------

def acceptance_rate_stats(
    base_logits: torch.Tensor,        # (T, V) main head predictions
    head_logits: list[torch.Tensor],  # k tensors of (T, V)
    true_tokens: torch.Tensor,        # (T,) ground truth
) -> dict:
    """For evaluation: compute how often each MTP head correctly predicts future tokens.

    Returns: {'head_{k}_accuracy': float for k in 1..n_heads}
    """
    stats: dict[str, float] = {}

    for k, h_logits in enumerate(head_logits, start=1):
        step = k
        T = true_tokens.shape[0]

        # head k predicts tokens at position t+step; valid for t in [0, T-step)
        if step >= T:
            stats[f"head_{k}_accuracy"] = 0.0
            continue

        # Predicted tokens from argmax
        preds = h_logits.argmax(dim=-1)  # (T,) or (T-step,)

        # Align: head logits at position t → ground truth at t+step
        # If head_logits has T entries (full sequence), slice accordingly
        if preds.shape[0] == T:
            preds_aligned = preds[: T - step]
            targets = true_tokens[step:]
        else:
            # Already sliced — just use what we have
            n = min(preds.shape[0], T - step)
            preds_aligned = preds[:n]
            targets = true_tokens[step : step + n]

        if targets.shape[0] == 0:
            stats[f"head_{k}_accuracy"] = 0.0
        else:
            correct = (preds_aligned == targets).float().mean().item()
            stats[f"head_{k}_accuracy"] = correct

    return stats
