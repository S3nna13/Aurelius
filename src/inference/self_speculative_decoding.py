"""Self-Speculative Decoding via Early Exit.

A single model serves as both drafter (early-exit) and verifier (full model).
Draft tokens are generated using intermediate layer representations,
then accepted/rejected using the full model's distribution.

Reference: Zhang et al. 2024 (SELF-SD) — https://arxiv.org/abs/2401.17448
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# EarlyExitHead
# ---------------------------------------------------------------------------


class EarlyExitHead(nn.Module):
    """Lightweight output head attached to an intermediate transformer layer.

    Maps intermediate hidden states directly to vocabulary logits, enabling
    early-exit draft token generation without running all layers.

    Args:
        d_model: Hidden dimension of the model.
        vocab_size: Vocabulary size.
        draft_layer: Index of the layer this head is attached to.
    """

    def __init__(self, d_model: int, vocab_size: int, draft_layer: int) -> None:
        super().__init__()
        self.draft_layer = draft_layer
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        """Map intermediate hidden states to vocabulary logits.

        Args:
            hidden: (B, T, d_model) intermediate hidden states from draft_layer.

        Returns:
            logits: (B, T, vocab_size) draft logits.
        """
        return self.head(self.norm(hidden))


# ---------------------------------------------------------------------------
# LayeredModel
# ---------------------------------------------------------------------------


class LayeredModel(nn.Module):
    """Simulated multi-layer transformer with early-exit capabilities.

    Each layer is a simple Linear + GELU block for testing purposes.
    Supports splitting execution at any layer boundary for early-exit drafting.

    Args:
        n_layers: Total number of transformer layers.
        d_model: Hidden dimension.
        vocab_size: Vocabulary size.
    """

    def __init__(self, n_layers: int, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model), nn.GELU()) for _ in range(n_layers)]
        )
        self.final_head = nn.Linear(d_model, vocab_size)

    def forward_to_layer(self, x: Tensor, up_to_layer: int) -> tuple[Tensor, Tensor]:
        """Run input through layers 0..up_to_layer (inclusive).

        Args:
            x: (B, T, d_model) input hidden states.
            up_to_layer: Last layer index to execute (inclusive).

        Returns:
            Tuple of:
                hidden_at_layer: (B, T, d_model) hidden states after up_to_layer.
                final_logits: (B, T, vocab_size) logits from final_head applied to
                              the output of the last executed layer.
        """
        h = x
        for i in range(up_to_layer + 1):
            h = self.layers[i](h)
        logits = self.final_head(h)
        return h, logits

    def forward_remaining(self, hidden: Tensor, from_layer: int) -> Tensor:
        """Run layers from_layer+1 through n_layers-1, then apply final_head.

        Args:
            hidden: (B, T, d_model) hidden states at from_layer.
            from_layer: Index of the layer whose output is provided in hidden.

        Returns:
            logits: (B, T, vocab_size) full-model logits.
        """
        h = hidden
        for i in range(from_layer + 1, self.n_layers):
            h = self.layers[i](h)
        return self.final_head(h)


# ---------------------------------------------------------------------------
# SelfSpeculativeDecoder
# ---------------------------------------------------------------------------


class SelfSpeculativeDecoder:
    """Decodes using early-exit drafting and full-model verification.

    The same model acts as both drafter and verifier:
      - Drafter: exits at draft_layer, uses EarlyExitHead for cheap draft tokens.
      - Verifier: runs all remaining layers to produce full-model logits.

    Args:
        model: LayeredModel serving as the backbone.
        draft_head: EarlyExitHead attached at draft_layer.
        draft_layer: Layer index where early exit occurs.
        n_draft_tokens: Number of draft tokens to propose per step.
    """

    def __init__(
        self,
        model: LayeredModel,
        draft_head: EarlyExitHead,
        draft_layer: int,
        n_draft_tokens: int = 4,
    ) -> None:
        self.model = model
        self.draft_head = draft_head
        self.draft_layer = draft_layer
        self.n_draft_tokens = n_draft_tokens

    def _draft(self, hidden_at_draft: Tensor) -> tuple[Tensor, Tensor]:
        """Generate draft tokens greedily from intermediate hidden states.

        Args:
            hidden_at_draft: (B, T, d_model) hidden states at draft_layer.

        Returns:
            Tuple of:
                draft_tokens: (n_draft_tokens,) LongTensor of proposed token ids.
                draft_probs: (n_draft_tokens, vocab_size) softmax probabilities.
        """
        # Use the last position for autoregressive drafting
        logits = self.draft_head(hidden_at_draft)  # (B, T, vocab_size)
        last_logits = logits[0, -1, :]  # (vocab_size,)

        draft_tokens: list[int] = []
        draft_probs_list: list[Tensor] = []

        current_logits = last_logits
        for _ in range(self.n_draft_tokens):
            probs = F.softmax(current_logits, dim=-1)
            token = torch.argmax(probs, dim=-1)
            draft_tokens.append(token.item())
            draft_probs_list.append(probs)
            # For simplicity, reuse the same logits (stateless draft head)
            current_logits = last_logits

        tokens_tensor = torch.tensor(draft_tokens, dtype=torch.long)
        probs_tensor = torch.stack(draft_probs_list, dim=0)  # (n_draft_tokens, vocab_size)
        return tokens_tensor, probs_tensor

    def _verify(self, input_ids: Tensor, draft_tokens: Tensor) -> tuple[Tensor, int]:
        """Verify draft tokens using the full model.

        Runs the full model on [input_ids | draft_tokens] and accepts the greedy
        prefix — stopping at the first position where the full model disagrees.

        Args:
            input_ids: (T,) LongTensor of current context token ids.
            draft_tokens: (n_draft_tokens,) LongTensor of proposed tokens.

        Returns:
            Tuple of:
                accepted_tokens: (n_accepted,) LongTensor of accepted token ids.
                n_accepted: int, number of accepted draft tokens.
        """
        n_draft = draft_tokens.shape[0]
        d_model = self.model.d_model

        # Build combined sequence: [input_ids | draft_tokens]
        combined = torch.cat([input_ids, draft_tokens], dim=0)  # (T + n_draft,)

        # Create a simple embedding: one-hot style via lookup in identity-ish space
        # We use a fixed random embedding for the token ids (same each call for consistency)
        T_total = combined.shape[0]
        # Use a deterministic embedding based on token id modulo d_model
        hidden = torch.zeros(1, T_total, d_model)
        for pos, tok_id in enumerate(combined.tolist()):
            hidden[0, pos, tok_id % d_model] = 1.0

        # Run full model (all layers)
        full_logits = self.model.forward_remaining(
            *self.model.forward_to_layer(hidden, self.model.n_layers - 1)[:1],
            from_layer=self.model.n_layers - 1,
        )
        # full_logits shape: (1, T_total, vocab_size) — but forward_remaining
        # receives hidden after all layers already, so we call forward directly
        h = hidden
        for layer in self.model.layers:
            h = layer(h)
        full_logits = self.model.final_head(h)  # (1, T_total, vocab_size)

        # Verify each draft position
        # Position i in draft corresponds to index (len(input_ids) + i - 1) in full_logits
        # because full_logits[t] predicts token at position t+1
        context_len = input_ids.shape[0]
        accepted: list[int] = []

        for i in range(n_draft):
            # The full model's prediction at position (context_len - 1 + i) predicts
            # what should come next (the draft token at index i)
            pred_pos = context_len - 1 + i
            full_pred = torch.argmax(full_logits[0, pred_pos, :]).item()
            draft_tok = draft_tokens[i].item()
            if full_pred == draft_tok:
                accepted.append(draft_tok)
            else:
                break

        n_accepted = len(accepted)
        if n_accepted == 0:
            accepted_tokens = torch.empty(0, dtype=torch.long)
        else:
            accepted_tokens = torch.tensor(accepted, dtype=torch.long)
        return accepted_tokens, n_accepted

    def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Generate tokens via self-speculative decoding.

        At each step:
          1. Run model to draft_layer to get intermediate hidden states.
          2. Generate n_draft_tokens using the early-exit head.
          3. Verify with the full model and accept the greedy prefix.
          4. Advance context by accepted tokens (at least 1 via fallback).

        Args:
            prompt_ids: (T,) LongTensor of prompt token ids.
            max_new_tokens: Total number of new tokens to generate.

        Returns:
            generated: (max_new_tokens,) LongTensor of generated token ids.
        """
        d_model = self.model.d_model
        generated: list[int] = []
        context = prompt_ids.clone()

        while len(generated) < max_new_tokens:
            # Build hidden from context
            T = context.shape[0]
            hidden = torch.zeros(1, T, d_model)
            for pos, tok_id in enumerate(context.tolist()):
                hidden[0, pos, tok_id % d_model] = 1.0

            # Run to draft layer
            hidden_at_draft, _ = self.model.forward_to_layer(hidden, self.draft_layer)

            # Generate draft tokens
            draft_tokens, _ = self._draft(hidden_at_draft)

            # Verify with full model
            accepted_tokens, n_accepted = self._verify(context, draft_tokens)

            if n_accepted > 0:
                # Accept the greedy prefix
                tokens_to_add = accepted_tokens.tolist()
            else:
                # Fallback: run full model on context, take greedy next token
                h = hidden
                for layer in self.model.layers:
                    h = layer(h)
                full_logits = self.model.final_head(h)  # (1, T, vocab_size)
                next_token = torch.argmax(full_logits[0, -1, :]).item()
                tokens_to_add = [next_token]

            for tok in tokens_to_add:
                generated.append(tok)
                if len(generated) >= max_new_tokens:
                    break

            torch.tensor(
                tokens_to_add[: len(generated) - (len(generated) - len(tokens_to_add))],
                dtype=torch.long,
            )
            # Extend context
            added = min(len(tokens_to_add), max_new_tokens - (len(generated) - len(tokens_to_add)))
            context = torch.cat(
                [context, torch.tensor(tokens_to_add[:added], dtype=torch.long)], dim=0
            )

        return torch.tensor(generated[:max_new_tokens], dtype=torch.long)


# ---------------------------------------------------------------------------
# DraftQualityMonitor
# ---------------------------------------------------------------------------


class DraftQualityMonitor:
    """Tracks and reports draft token acceptance statistics.

    Maintains running totals and per-step history to estimate the effective
    speedup achieved by self-speculative decoding.
    """

    def __init__(self) -> None:
        self.total_drafted: int = 0
        self.total_accepted: int = 0
        self.acceptance_history: list[float] = []

    def record(self, n_drafted: int, n_accepted: int) -> None:
        """Record a single draft-verify step.

        Args:
            n_drafted: Number of tokens proposed in this step.
            n_accepted: Number of tokens accepted in this step.
        """
        self.total_drafted += n_drafted
        self.total_accepted += n_accepted
        step_rate = n_accepted / max(1, n_drafted)
        self.acceptance_history.append(step_rate)

    def acceptance_rate(self) -> float:
        """Compute overall acceptance rate.

        Returns:
            Float in [0, 1]: total_accepted / total_drafted.
        """
        return self.total_accepted / max(1, self.total_drafted)

    def speedup_estimate(self) -> float:
        """Estimate theoretical speedup from acceptance rate.

        Each draft step proposes n_draft_tokens, and we always get at least
        one token (the base token from the verifier). The simplified estimate is:

            speedup ≈ acceptance_rate * n_draft_tokens + 1

        Returns:
            Float >= 1.0 representing estimated tokens-per-verifier-call.
        """
        # Derive n_draft_tokens from history if available, else use total_drafted
        if self.total_drafted == 0:
            return 1.0
        n_steps = len(self.acceptance_history)
        if n_steps == 0:
            return 1.0
        avg_drafted_per_step = self.total_drafted / n_steps
        return self.acceptance_rate() * avg_drafted_per_step + 1.0
