"""Eagle3 Speculative Decoding — confidence-gated variable-length drafting (2025).

Eagle3 extends Eagle2 by adding a *confidence head* that predicts the
acceptance probability of each drafted token BEFORE running the verifier.
This enables dynamic draft length: draft more tokens when confidence is high,
fewer when confidence is low, reducing unnecessary target-model forward passes.

Algorithm overview
------------------
1. Draft head generates a candidate token + hidden state at each step.
2. Confidence head estimates the probability that the verifier will accept it.
3. If confidence < threshold, drafting stops early (dynamic length).
4. Target model verifies the drafted sequence via rejection sampling.
5. Accepted tokens are returned along with statistics.

Public API
----------
Eagle3Config      — configuration dataclass
ConfidenceHead    — MLP that maps hidden state → acceptance probability
Eagle3Drafter     — draft model with confidence-gated variable-length drafting
Eagle3Verifier    — rejection-sampling verifier (stateless helper)
Eagle3Decoder     — orchestrates drafting + verification; registered in DECODER_REGISTRY
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Eagle3Config:
    """Configuration for Eagle3 speculative decoding."""

    d_model: int = 2048
    vocab_size: int = 128000
    max_draft_len: int = 8  # maximum tokens to draft per step
    min_draft_len: int = 1  # minimum tokens to draft per step
    confidence_threshold: float = 0.8  # stop drafting if confidence falls below this
    confidence_head_hidden: int | None = None  # defaults to d_model // 4
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Confidence Head
# ---------------------------------------------------------------------------


class ConfidenceHead(nn.Module):
    """Predicts acceptance probability for a draft token from a hidden state.

    Accepts 2-D input [B, d_model] or 3-D input [B, T, d_model] and returns
    a sigmoid-activated scalar in (0, 1) for each (batch, [time]) position.
    """

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        """Compute acceptance probability.

        Args:
            hidden: [B, d_model] or [B, T, d_model]

        Returns:
            confidence: [B] or [B, T] — values in (0, 1)
        """
        out = self.fc2(self.act(self.fc1(hidden)))
        # Remove the trailing size-1 dimension
        return torch.sigmoid(out.squeeze(-1))


# ---------------------------------------------------------------------------
# Eagle3 Drafter
# ---------------------------------------------------------------------------


class Eagle3Drafter(nn.Module):
    """Draft model with confidence-gated variable-length drafting.

    The drafter is intentionally lightweight (embed → linear → lm_head) so
    that it can be tested without a GPU.  The confidence head attaches to the
    draft hidden state and predicts whether the target model will accept the
    proposed token.
    """

    def __init__(self, config: Eagle3Config) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.fc = nn.Linear(config.d_model, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        conf_hidden = config.confidence_head_hidden or (config.d_model // 4)
        self.confidence_head = ConfidenceHead(config.d_model, conf_hidden)

    # ------------------------------------------------------------------
    # Single draft step
    # ------------------------------------------------------------------

    def draft_step(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run one draft step from the current hidden state.

        Args:
            hidden: [B, d_model] — current draft hidden state

        Returns:
            logits:      [B, vocab_size] — vocabulary logits
            next_hidden: [B, d_model]   — updated hidden state
            confidence:  [B]            — predicted acceptance probability in (0, 1)
        """
        next_hidden = F.gelu(self.fc(hidden))  # [B, d_model]
        logits = self.lm_head(next_hidden)  # [B, vocab_size]
        confidence = self.confidence_head(next_hidden)  # [B]
        return logits, next_hidden, confidence

    # ------------------------------------------------------------------
    # Multi-step drafting with confidence gating
    # ------------------------------------------------------------------

    def draft(
        self,
        initial_hidden: Tensor,
        n_tokens: int | None = None,
    ) -> dict:
        """Draft up to *n_tokens* (or max_draft_len) tokens with early stopping.

        Drafting stops early when the mean confidence across the batch falls
        below ``config.confidence_threshold``.  At minimum, ``min_draft_len``
        tokens are always drafted.

        Args:
            initial_hidden: [B, d_model] — hidden state from the target model
            n_tokens: maximum tokens to draft; defaults to ``config.max_draft_len``

        Returns:
            dict with keys:
                ``draft_tokens`` — list of [B] token-id tensors (length = n_drafted)
                ``draft_logits`` — list of [B, vocab_size] logit tensors
                ``confidences``  — list of float mean-confidence values per step
                ``n_drafted``    — int, number of tokens drafted
        """
        max_steps = n_tokens if n_tokens is not None else self.config.max_draft_len
        max_steps = max(max_steps, self.config.min_draft_len)

        draft_tokens: list[Tensor] = []
        draft_logits: list[Tensor] = []
        confidences: list[float] = []

        hidden = initial_hidden  # [B, d_model]

        for step in range(max_steps):
            logits, hidden, conf = self.draft_step(hidden)  # conf: [B]

            # Sample next token
            if self.config.temperature == 0.0:
                token = logits.argmax(dim=-1)  # [B]
            else:
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

            # Advance hidden state by incorporating the new token embedding
            token_emb = self.embed(token)  # [B, d_model]
            hidden = hidden + token_emb  # residual-style update

            draft_tokens.append(token)
            draft_logits.append(logits)
            mean_conf = conf.mean().item()
            confidences.append(mean_conf)

            # Early stopping: always draft at least min_draft_len tokens
            if step + 1 >= self.config.min_draft_len:
                if mean_conf < self.config.confidence_threshold:
                    break

        return {
            "draft_tokens": draft_tokens,
            "draft_logits": draft_logits,
            "confidences": confidences,
            "n_drafted": len(draft_tokens),
        }


# ---------------------------------------------------------------------------
# Eagle3 Verifier
# ---------------------------------------------------------------------------


class Eagle3Verifier:
    """Simulates the target-model verifier using speculative rejection sampling.

    This is a stateless helper class.  The acceptance criterion follows the
    standard speculative decoding rule (Leviathan et al., 2023):

        accept if  target_prob[t] / draft_prob[t]  >=  Uniform(0, 1)
    """

    @staticmethod
    def verify(
        draft_tokens: list[Tensor],
        target_probs: list[Tensor],
        draft_logits: list[Tensor] | None = None,
    ) -> tuple[list[bool], int]:
        """Verify drafted tokens against target model probabilities.

        Args:
            draft_tokens: list of [B] tensors — drafted token IDs (B=1 typical)
            target_probs: list of [B, vocab_size] tensors — target model probs
            draft_logits: optional list of [B, vocab_size] logits from drafter;
                          if None, the draft distribution is assumed uniform so
                          every token is tested purely against a uniform random.

        Returns:
            accepted_mask: list[bool] — True for each accepted token
            n_accepted:    int        — number of accepted tokens
        """
        accepted_mask: list[bool] = []
        n_accepted = 0

        for i, (token, tgt_prob) in enumerate(zip(draft_tokens, target_probs)):
            # token: [B], tgt_prob: [B, vocab_size]
            # Gather target probability for the drafted token
            tgt_p = tgt_prob.gather(dim=-1, index=token.unsqueeze(-1)).squeeze(-1)  # [B]

            if draft_logits is not None:
                draft_p = F.softmax(draft_logits[i], dim=-1)
                draft_p_tok = draft_p.gather(dim=-1, index=token.unsqueeze(-1)).squeeze(-1)
                # Clamp to avoid division-by-zero
                ratio = tgt_p / draft_p_tok.clamp(min=1e-9)
            else:
                # Treat draft probability as uniform: ratio = tgt_p * vocab_size
                # Accept with probability min(1, ratio)
                ratio = tgt_p

            threshold = torch.rand_like(ratio)
            accept = bool((ratio >= threshold).all().item())
            accepted_mask.append(accept)
            if accept:
                n_accepted += 1
            else:
                # Stop at first rejection (standard speculative decoding)
                break

        return accepted_mask, n_accepted


# ---------------------------------------------------------------------------
# Eagle3 Decoder
# ---------------------------------------------------------------------------


class Eagle3Decoder:
    """Orchestrates Eagle3 drafting + verification.

    Usage::

        config = Eagle3Config(d_model=64, vocab_size=256)
        decoder = Eagle3Decoder(config)

        def target_model_fn(tokens):
            # Returns list of [1, vocab_size] probability tensors
            ...

        result = decoder.decode_step(initial_hidden, target_model_fn)
    """

    def __init__(self, config: Eagle3Config) -> None:
        self.config = config
        self.drafter = Eagle3Drafter(config)

    def decode_step(
        self,
        initial_hidden: Tensor,
        target_model_fn: Callable,
    ) -> dict:
        """Run one speculative decoding step.

        1. Draft tokens using Eagle3Drafter (confidence-gated).
        2. Query the target model for probabilities.
        3. Verify via rejection sampling.
        4. Return accepted tokens and statistics.

        Args:
            initial_hidden: [B, d_model] — hidden state from the target model's
                            last forward pass.
            target_model_fn: Callable[[list[Tensor]], list[Tensor]]
                             Receives the list of draft token tensors and returns
                             a list of [B, vocab_size] probability tensors (one
                             per draft position).

        Returns:
            dict with keys:
                ``accepted_tokens`` — list[int] of accepted token IDs
                ``n_accepted``      — int
                ``n_drafted``       — int
                ``acceptance_rate`` — float in [0, 1]
        """
        # Step 1: Draft
        draft_out = self.drafter.draft(initial_hidden)
        draft_tokens: list[Tensor] = draft_out["draft_tokens"]
        draft_logits: list[Tensor] = draft_out["draft_logits"]
        n_drafted: int = draft_out["n_drafted"]

        # Step 2: Get target model probabilities
        target_probs: list[Tensor] = target_model_fn(draft_tokens)

        # Step 3: Verify
        accepted_mask, n_accepted = Eagle3Verifier.verify(
            draft_tokens=draft_tokens,
            target_probs=target_probs,
            draft_logits=draft_logits,
        )

        # Collect accepted token IDs (works for B=1 or B>1)
        accepted_tokens: list[int] = []
        for i, accept in enumerate(accepted_mask):
            if accept:
                tok = draft_tokens[i]
                if tok.numel() == 1:
                    accepted_tokens.append(int(tok.item()))
                else:
                    accepted_tokens.extend(tok.tolist())

        acceptance_rate = n_accepted / n_drafted if n_drafted > 0 else 0.0

        return {
            "accepted_tokens": accepted_tokens,
            "n_accepted": n_accepted,
            "n_drafted": n_drafted,
            "acceptance_rate": acceptance_rate,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

try:
    from src.inference import DECODER_REGISTRY  # type: ignore[attr-defined]

    DECODER_REGISTRY["eagle3"] = Eagle3Decoder
except Exception:  # noqa: S110
    # Registry not yet initialised; will be wired in __init__.py
    pass

__all__ = [
    "Eagle3Config",
    "ConfidenceHead",
    "Eagle3Drafter",
    "Eagle3Verifier",
    "Eagle3Decoder",
]
