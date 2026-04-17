"""Speculative Streaming: fast LLM inference without auxiliary models.

Reference: Bhendawade et al., "Speculative Streaming: Fast LLM Inference without
Auxiliary Models", arXiv:2402.11131 (2024).

Key insight (Section 3): use the *target model's own intermediate hidden states*
at layer l_draft to speculatively predict future tokens via a lightweight draft
head, then verify with the full target forward pass — no separate draft model
required.

Algorithm (Algorithm 1 of the paper):
  1. At step t: run forward pass; at layer l_draft extract h[l_draft].
     Use draft_head(h[l_draft]) to sample γ draft tokens d_{t+1}, ..., d_{t+γ}.
  2. At step t+1: run full forward pass on [x_t, d_{t+1}, ..., d_{t+γ}] in parallel.
  3. Verify: for each k=1..γ accept d_{t+k} if
       U ~ Uniform(0,1) < min(1, p_target(d_{t+k}) / p_draft(d_{t+k})).
     On rejection, resample from the adjusted distribution (p_target - p_draft)+.
  4. Always emit at least 1 token (the target sample at the rejection point).

Public API
----------
DraftHead                   — lightweight MLP (d_model → vocab_size)
SpeculativeStreamingDecoder — wraps target model_fn with speculative streaming
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Draft head (Section 3 — "draft head" MLP)
# ---------------------------------------------------------------------------

class DraftHead(nn.Module):
    """Lightweight MLP that maps hidden states to draft vocabulary logits.

    Architecture: Linear(d_model, d_model) → ReLU → Linear(d_model, vocab_size).

    Parameters
    ----------
    d_model:
        Hidden-state dimensionality of the target model.
    vocab_size:
        Target vocabulary size.

    Shapes
    ------
    Input:  (B, T, d_model)
    Output: (B, T, vocab_size)
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, vocab_size)

    def forward(self, h: Tensor) -> Tensor:
        """Map hidden states → draft logits.

        Parameters
        ----------
        h:
            Hidden states, shape ``(B, T, d_model)``.

        Returns
        -------
        Tensor of shape ``(B, T, vocab_size)``.
        """
        return self.fc2(F.relu(self.fc1(h)))


# ---------------------------------------------------------------------------
# Speculative Streaming Decoder (Section 3, Algorithm 1)
# ---------------------------------------------------------------------------

class SpeculativeStreamingDecoder:
    """Speculative Streaming decoder using the target model's own hidden states.

    The target model is queried via *model_fn*, which must support two calling
    modes controlled by ``return_hidden``:

    * ``model_fn(input_ids, return_hidden=False)``
      → ``logits``  — shape ``(B, T, V)``

    * ``model_fn(input_ids, return_hidden=True)``
      → ``(logits, h_draft)`` — logits ``(B, T, V)`` and intermediate hidden
        states at layer ``draft_layer_idx``, shape ``(B, T, d_model)``.

    Parameters
    ----------
    model_fn:
        Callable for the target model (see above).
    draft_head:
        Trained :class:`DraftHead` instance.
    gamma:
        Speculation length γ (number of draft tokens per step).
    temperature:
        Sampling temperature. Values ≤ 0 are treated as greedy (argmax).
    """

    def __init__(
        self,
        model_fn: Callable[..., Tensor | Tuple[Tensor, Tensor]],
        draft_head: DraftHead,
        gamma: int = 4,
        temperature: float = 1.0,
    ) -> None:
        if gamma < 1:
            raise ValueError(f"gamma must be ≥ 1, got {gamma}")
        self.model_fn = model_fn
        self.draft_head = draft_head
        self.gamma = gamma
        self.temperature = temperature

        # History: list of (n_accepted, gamma) pairs used to estimate α.
        self._acceptance_history: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample(self, logits: Tensor) -> Tensor:
        """Sample or argmax from a ``(V,)`` or ``(B, V)`` logit tensor."""
        if self.temperature <= 0.0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / self.temperature, dim=-1)
        # multinomial requires 2-D input
        if probs.dim() == 1:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _probs(self, logits: Tensor) -> Tensor:
        """Convert logits to probabilities, respecting temperature."""
        if self.temperature <= 0.0:
            # Hard distribution: 1 on argmax, 0 elsewhere.
            probs = torch.zeros_like(logits)
            probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
            return probs
        return F.softmax(logits / self.temperature, dim=-1)

    # ------------------------------------------------------------------
    # Core generation step
    # ------------------------------------------------------------------

    def generate_step(
        self,
        input_ids: Tensor,
        past_key_values: Optional[object] = None,
    ) -> Tuple[List[int], int]:
        """Run one speculative streaming step.

        Implements Algorithm 1 of Bhendawade et al. (2024):

        1. Forward pass on ``input_ids`` with ``return_hidden=True`` to get
           both logits and the draft-layer hidden state ``h_l``.
        2. Draft head predicts draft logits from ``h_l[:, -1, :]``; sample
           γ draft tokens ``d_{t+1}, ..., d_{t+γ}``.
        3. Run full forward pass on ``[x_t, d_{t+1}, ..., d_{t+γ}]`` to get
           target logits for positions t+1 … t+γ.
        4. Accept/reject each draft token (rejection sampling); always return
           ≥ 1 token.

        Parameters
        ----------
        input_ids:
            ``(B, T)`` int64 tensor with the current context.
        past_key_values:
            Unused; kept for API compatibility.

        Returns
        -------
        accepted_tokens:
            List of accepted token IDs (ints), length ∈ [1, γ+1].
        n_accepted:
            Number of accepted tokens (= ``len(accepted_tokens)``).
        """
        B, T = input_ids.shape
        if B != 1:
            raise ValueError(
                "SpeculativeStreamingDecoder currently supports batch_size=1 only."
            )

        # ------------------------------------------------------------------ #
        # Step 1 — forward pass on current context; extract h_l (draft layer)
        # ------------------------------------------------------------------ #
        result = self.model_fn(input_ids, return_hidden=True)
        if not (isinstance(result, (tuple, list)) and len(result) == 2):
            raise RuntimeError(
                "model_fn(input_ids, return_hidden=True) must return (logits, h_draft). "
                f"Got type {type(result)}."
            )
        logits_ctx, h_draft = result
        # logits_ctx: (B, T, V);  h_draft: (B, T, d_model)

        # ------------------------------------------------------------------ #
        # Step 2 — draft γ tokens from h_l of the *last* position
        # ------------------------------------------------------------------ #
        h_last = h_draft[:, -1:, :]                   # (B, 1, d_model)
        draft_logits_single = self.draft_head(h_last)  # (B, 1, V)
        V = draft_logits_single.shape[-1]

        draft_token_ids: List[int] = []
        draft_probs_list: List[Tensor] = []

        # We autoregressively sample γ draft tokens using the draft head.
        # Because the draft head operates only on the hidden state of the last
        # position (a local MLP with no recurrence), we re-use the same hidden
        # state for all γ drafts; this matches the "streaming" variant where
        # draft tokens are proposed in a single MLP pass.
        draft_logits_all = self.draft_head(h_draft)  # (B, T, V) — draft logits for all positions

        # For speculative generation, we need γ draft tokens starting from position T.
        # We use the last position's draft logits repeatedly (no recurrence in draft head).
        last_draft_logits = draft_logits_all[0, -1, :]  # (V,)
        p_draft_last = self._probs(last_draft_logits)   # (V,)

        for k in range(self.gamma):
            tok = self._sample(last_draft_logits)       # scalar tensor
            draft_token_ids.append(tok.item())
            draft_probs_list.append(p_draft_last)

        # ------------------------------------------------------------------ #
        # Step 3 — full forward pass on [context, d_{t+1}, ..., d_{t+γ}]
        # ------------------------------------------------------------------ #
        draft_tensor = torch.tensor(draft_token_ids, dtype=torch.long, device=input_ids.device)
        draft_tensor = draft_tensor.unsqueeze(0)        # (1, γ)
        extended_ids = torch.cat([input_ids, draft_tensor], dim=1)  # (1, T+γ)

        target_out = self.model_fn(extended_ids, return_hidden=False)
        if isinstance(target_out, (tuple, list)):
            target_logits = target_out[0]
        else:
            target_logits = target_out
        # target_logits: (1, T+γ, V) — logits[T..T+γ-1] correspond to draft positions

        # ------------------------------------------------------------------ #
        # Step 4 — Algorithm 1 accept/reject loop
        # ------------------------------------------------------------------ #
        accepted_tokens: List[int] = []

        for k in range(self.gamma):
            d_k = draft_token_ids[k]
            # Target probability at verification position T+k (predicts token at T+k+1,
            # but by standard speculative decoding indexing, position T+k predicts d_{t+k+1}).
            # The logit at position T-1+k predicts the token at position T+k (= d_k).
            tgt_logit_k = target_logits[0, T - 1 + k, :]  # (V,)
            p_target_k = self._probs(tgt_logit_k)          # (V,)
            p_draft_k = draft_probs_list[k]                # (V,)

            p_t_dk = p_target_k[d_k].clamp(min=0.0)
            p_d_dk = p_draft_k[d_k].clamp(min=1e-9)

            ratio = (p_t_dk / p_d_dk).clamp(max=1.0)

            u = torch.rand(1, device=input_ids.device).item()
            if u < ratio.item():
                # Accept d_k
                accepted_tokens.append(d_k)
            else:
                # Reject: resample from adjusted distribution (p_target - p_draft)+
                adjusted = (p_target_k - p_draft_k).clamp(min=0.0)
                mass = adjusted.sum()
                if mass < 1e-9:
                    # Fallback: sample from target
                    adjusted = p_target_k
                    mass = adjusted.sum()
                adjusted = adjusted / mass
                resampled = torch.multinomial(adjusted, num_samples=1).item()
                accepted_tokens.append(int(resampled))
                # Stop after first rejection
                n_accepted = len(accepted_tokens)
                self._acceptance_history.append((n_accepted, self.gamma))
                return accepted_tokens, n_accepted

        # All γ draft tokens accepted — also emit the target's own next token
        # (standard speculative decoding bonus token at position T+γ).
        bonus_logit = target_logits[0, T - 1 + self.gamma, :]  # (V,)
        bonus_tok = self._sample(bonus_logit).item()
        accepted_tokens.append(int(bonus_tok))

        n_accepted = len(accepted_tokens)  # = γ + 1
        self._acceptance_history.append((n_accepted, self.gamma))
        return accepted_tokens, n_accepted

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def acceptance_rate(self, history: Optional[List[Tuple[int, int]]] = None) -> float:
        """Compute mean acceptance rate α = E[n_accepted / gamma].

        Parameters
        ----------
        history:
            List of ``(n_accepted, gamma)`` tuples.  If ``None``, uses the
            decoder's internal ``_acceptance_history``.

        Returns
        -------
        Float in ``[0, 1]`` representing the mean per-step acceptance fraction.
        Raises ``ValueError`` if history is empty.
        """
        h = history if history is not None else self._acceptance_history
        if not h:
            raise ValueError("acceptance_rate requires at least one step in history.")
        # n_accepted can be gamma+1 (all accepted + bonus); cap at gamma for rate.
        rates = [min(n, g) / g for n, g in h]
        return float(sum(rates) / len(rates))
