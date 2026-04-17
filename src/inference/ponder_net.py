"""PonderNet — adaptive computation via halting probability (Banino et al. 2021).

At each pondering step:
1. Run the base model to get hidden states and logits.
2. Compute a halting probability h_t from the last-token hidden state.
3. Accumulate the halt probability; stop when it exceeds the threshold.
4. The final output is a weighted combination of per-step logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PonderConfig:
    d_model: int = 64
    max_steps: int = 8          # maximum pondering steps
    halt_threshold: float = 0.9  # cumulative halt probability threshold
    lambda_p: float = 0.01      # geometric prior regularization weight
    p_geometric: float = 0.2    # geometric prior success probability


# ---------------------------------------------------------------------------
# Halting Unit
# ---------------------------------------------------------------------------

class HaltingUnit(nn.Module):
    """Computes halting probability at each step: h_t = sigmoid(linear(hidden))."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: (B, T, d_model) -> halt_prob: (B,) using last token hidden state."""
        # Take the last token hidden state: (B, d_model)
        last = hidden[:, -1, :]
        # Sigmoid to get probability in (0, 1): (B,)
        return torch.sigmoid(self.linear(last)).squeeze(-1)


# ---------------------------------------------------------------------------
# PonderNet
# ---------------------------------------------------------------------------

class PonderNet(nn.Module):
    """Wraps a base model with adaptive computation (pondering).

    At each step:
    1. Run base model to get hidden states
    2. Compute halt probability h_t
    3. If cumulative halt prob > threshold, stop
    4. Output is weighted sum of per-step predictions
    """

    def __init__(self, base_model: nn.Module, config: PonderConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.halting_unit = HaltingUnit(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)  # optional per-step projection

    # ------------------------------------------------------------------
    # Internal helper: run base model, return (hidden, logits)
    # ------------------------------------------------------------------
    def _run_base(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Run the base model and return (hidden_states, logits).

        Handles both plain Tensor outputs and the
        (loss, logits, present_key_values) tuple returned by AureliusTransformer.
        """
        out = self.base_model(input_ids)
        if isinstance(out, tuple):
            # AureliusTransformer: (loss, logits, present_key_values)
            logits = out[1]
        else:
            logits = out

        # We need hidden states as well. Since AureliusTransformer doesn't expose
        # hidden states directly, we derive a proxy by mapping logits back through
        # the embedding weight (tied) or use a fixed projection. In practice, the
        # halting unit receives the embedding of the last input token multiplied by
        # the base model embedding table — but the simplest robust approach is to
        # use the logits themselves as a d_vocab proxy and instead pass the
        # per-token embedding from the base model.
        #
        # However, to keep the API clean and not modify the base model, we treat
        # `logits` as our "hidden" proxy for the halting unit by projecting them
        # back to d_model via the output_proj weight. We reshape logits: (B, T, V)
        # -> need (B, T, d_model). We'll use a lazy linear if needed, but actually
        # the cleanest solution is to expose hidden states through a hook or simply
        # use the embedding lookup from the base model.
        #
        # Pragmatic solution: use the base model's embedding table to re-embed the
        # greedy-decoded tokens, giving a (B, T, d_model) tensor that represents
        # the "thought" at this step.
        greedy_ids = logits.argmax(dim=-1)  # (B, T)
        embed = self.base_model.embed if hasattr(self.base_model, "embed") else None
        if embed is not None:
            hidden = embed(greedy_ids)  # (B, T, d_model)
        else:
            # Fallback: slice logits to d_model dimensions
            hidden = logits[:, :, : self.config.d_model]

        return hidden, logits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_ids: Tensor) -> Tuple[Tensor, dict]:
        """
        Returns: (weighted_logits, info_dict)
        weighted_logits: (B, T, vocab) weighted combination of per-step logits
        info_dict: {'n_steps': int, 'halt_probs': (B, n_steps), 'step_weights': (B, n_steps)}
        """
        B = input_ids.shape[0]

        all_halt_probs: List[Tensor] = []   # each: (B,)
        all_logits: List[Tensor] = []       # each: (B, T, vocab)

        # Cumulative NOT-halted probability — starts at 1.0 for each batch item
        cum_not_halted = torch.ones(B, device=input_ids.device, dtype=torch.float32)
        weighted_logits: Tensor | None = None

        for step in range(self.config.max_steps):
            hidden, logits = self._run_base(input_ids)  # (B,T,d_model), (B,T,V)

            h_t = self.halting_unit(hidden)  # (B,) in (0,1)
            all_halt_probs.append(h_t)
            all_logits.append(logits)

            # Weight for this step: p(halt at step t) = cum_not_halted * h_t
            # On the last step we assign all remaining weight to ensure sum == 1.
            if step == self.config.max_steps - 1:
                # Remainder: assign all leftover probability
                w_t = cum_not_halted
            else:
                w_t = cum_not_halted * h_t

            # Accumulate weighted logits: (B, T, V)
            w_t_expanded = w_t.view(B, 1, 1)
            if weighted_logits is None:
                weighted_logits = w_t_expanded * logits
            else:
                weighted_logits = weighted_logits + w_t_expanded * logits

            # Update remaining probability
            cum_not_halted = cum_not_halted * (1.0 - h_t)

            # Stop if all batch items have exceeded the threshold
            cum_halted = 1.0 - cum_not_halted  # (B,)
            if (cum_halted >= self.config.halt_threshold).all():
                break

        n_steps = len(all_halt_probs)

        # Stack per-step info: (B, n_steps)
        halt_probs = torch.stack(all_halt_probs, dim=1)

        # Recompute step weights for the info dict (mirrors the loop logic)
        step_weights_list: List[Tensor] = []
        cnot = torch.ones(B, device=input_ids.device, dtype=torch.float32)
        for i in range(n_steps):
            h = halt_probs[:, i]
            if i == n_steps - 1:
                step_weights_list.append(cnot)
            else:
                step_weights_list.append(cnot * h)
            cnot = cnot * (1.0 - h)
        step_weights = torch.stack(step_weights_list, dim=1)  # (B, n_steps)

        info: dict = {
            "n_steps": n_steps,
            "halt_probs": halt_probs,
            "step_weights": step_weights,
        }

        assert weighted_logits is not None
        return weighted_logits, info

    # ------------------------------------------------------------------
    # Adaptive generate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def adaptive_generate(
        self,
        input_ids: Tensor,   # (T,)
        max_new_tokens: int = 8,
    ) -> Tuple[Tensor, List[int]]:
        """Generate tokens with adaptive computation.
        Returns (generated_ids, steps_per_token) where steps_per_token is list of int."""
        # Ensure 2-D input: (1, T)
        if input_ids.dim() == 1:
            ids = input_ids.unsqueeze(0)
        else:
            ids = input_ids

        generated_ids: List[int] = []
        steps_per_token: List[int] = []

        for _ in range(max_new_tokens):
            weighted_logits, info = self.forward(ids)
            # Greedy decode from last position
            next_token_id = weighted_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)
            generated_ids.append(next_token_id.item())
            steps_per_token.append(info["n_steps"])
            ids = torch.cat([ids, next_token_id], dim=1)

        return torch.tensor(generated_ids, dtype=torch.long, device=input_ids.device), steps_per_token


# ---------------------------------------------------------------------------
# Geometric prior helper
# ---------------------------------------------------------------------------

def geometric_prior(n_steps: int, p: float = 0.2) -> Tensor:
    """Geometric distribution PMF: P(halt at step t) = (1-p)^(t-1) * p.
    Returns (n_steps,) tensor normalized to sum to 1."""
    steps = torch.arange(1, n_steps + 1, dtype=torch.float32)
    probs = (1.0 - p) ** (steps - 1.0) * p
    # Normalize to sum to 1 (handles truncation)
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ponder_loss(
    weighted_logits: Tensor,    # (B, T, vocab) from PonderNet.forward
    target_ids: Tensor,         # (B, T)
    halt_probs: Tensor,         # (B, n_steps) halting probabilities
    lambda_p: float = 0.01,
    p_geometric: float = 0.2,
) -> Tuple[Tensor, dict]:
    """Total loss = NLL + lambda_p * KL(halting || geometric_prior).

    KL = sum_t p(halt_t) * log(p(halt_t) / geom(t))
    Returns (loss, {'nll': float, 'kl_reg': float, 'total': float})
    """
    # --- NLL (cross-entropy on shifted predictions) ---
    B, T, V = weighted_logits.shape
    # Shift: predict token t+1 from position t
    shift_logits = weighted_logits[:, :-1, :].contiguous().view(-1, V)
    shift_labels = target_ids[:, 1:].contiguous().view(-1)
    nll = F.cross_entropy(shift_logits, shift_labels)

    # --- KL regularization ---
    n_steps = halt_probs.shape[1]
    prior = geometric_prior(n_steps, p=p_geometric).to(halt_probs.device)  # (n_steps,)

    # Recompute per-step halting weights from halt_probs
    # p(halt at step t) = prod_{i<t}(1-h_i) * h_t
    cnot = torch.ones(B, 1, device=halt_probs.device)
    step_weights_list: List[Tensor] = []
    for i in range(n_steps):
        h = halt_probs[:, i : i + 1]
        if i == n_steps - 1:
            step_weights_list.append(cnot)
        else:
            step_weights_list.append(cnot * h)
        cnot = cnot * (1.0 - h)
    # (B, n_steps)
    q = torch.cat(step_weights_list, dim=1)
    q = q.clamp(min=1e-8)

    # KL divergence: sum_t q_t * log(q_t / prior_t), averaged over batch
    prior_expanded = prior.unsqueeze(0)  # (1, n_steps)
    kl = (q * (q.log() - prior_expanded.log())).sum(dim=1).mean()

    total = nll + lambda_p * kl

    return total, {
        "nll": nll.item(),
        "kl_reg": kl.item(),
        "total": total.item(),
    }
