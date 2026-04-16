"""Speculative Consistency Checking for Aurelius.

Verifies that speculative decoding is lossless — the output distribution
should be identical to standard autoregressive decoding. Provides tools for
statistical testing and acceptance threshold calibration.

Public API
----------
SpeculativeConsistencyConfig  — configuration dataclass
ConsistencyReport             — result dataclass
greedy_decode                 — standard greedy autoregressive decoding
speculative_decode_simple     — simple draft-then-verify speculative decoding
SpeculativeConsistencyChecker — main checker class with verify / stats / calibrate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration / Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeConsistencyConfig:
    """Configuration for speculative consistency checking."""

    n_test_samples: int = 100
    max_new_tokens: int = 20
    acceptance_threshold: float = 0.0
    target_acceptance_rate: float = 0.8


@dataclass
class ConsistencyReport:
    """Result of a consistency check run."""

    n_tests: int
    n_consistent: int
    consistency_rate: float
    mean_acceptance_rate: float
    kl_divergence: float


# ---------------------------------------------------------------------------
# Standalone decoding helpers
# ---------------------------------------------------------------------------

def greedy_decode(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Standard greedy autoregressive decoding.

    Parameters
    ----------
    model:
        Language model; ``forward(input_ids)`` must return a tuple whose
        second element is ``(B, T, V)`` logits.
    input_ids:
        ``(B, T)`` int64 prompt tensor.
    max_new_tokens:
        Number of tokens to generate.

    Returns
    -------
    generated : ``(B, max_new_tokens)`` int64 tensor of generated token ids.
    """
    generated: list[Tensor] = []
    current = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(current)
            logits = output[1]               # (B, T, V)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            generated.append(next_tok)
            current = torch.cat([current, next_tok], dim=1)

    return torch.cat(generated, dim=1)       # (B, max_new_tokens)


def speculative_decode_simple(
    target: nn.Module,
    draft: nn.Module,
    input_ids: Tensor,
    n_draft: int = 3,
    max_new: int = 20,
    acceptance_threshold: float = 0.0,
) -> Tensor:
    """Simple speculative decoding: draft *n_draft* tokens, verify with target.

    For each speculation window the draft model proposes ``n_draft`` tokens
    greedily; the target model verifies them via rejection sampling.  Any
    rejected suffix is replaced by the target's own greedy choice.

    Parameters
    ----------
    target / draft:
        Language models with the same ``forward`` signature as ``greedy_decode``.
    input_ids:
        ``(B, T)`` int64 prompt tensor.
    n_draft:
        Number of tokens to speculate per window.
    max_new:
        Maximum total new tokens to generate.
    acceptance_threshold:
        Minimum probability ratio required to accept a draft token.

    Returns
    -------
    generated : ``(B, max_new)`` int64 tensor of generated token ids.
    """
    generated: list[Tensor] = []
    current = input_ids.clone()

    with torch.no_grad():
        while len(generated) < max_new:
            remaining = max_new - len(generated)
            k = min(n_draft, remaining)

            # --- Draft phase: greedily propose k tokens ---
            draft_ctx = current.clone()
            draft_ids: list[Tensor] = []
            for _ in range(k):
                d_out = draft(draft_ctx)
                d_logits = d_out[1][:, -1, :]          # (B, V)
                d_tok = d_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                draft_ids.append(d_tok)
                draft_ctx = torch.cat([draft_ctx, d_tok], dim=1)

            # --- Verify phase: run target once over prefix + draft tokens ---
            verify_input = torch.cat([current, *draft_ids], dim=1)  # (B, T+k)
            t_out = target(verify_input)
            t_logits = t_out[1]  # (B, T+k, V)

            # Acceptance loop
            accepted = 0
            for i in range(k):
                # Target probability at position T+i-1 predicts token at T+i
                t_pos_logits = t_logits[:, current.shape[1] + i - 1, :]  # (B, V)
                t_probs = F.softmax(t_pos_logits, dim=-1)

                d_tok = draft_ids[i]                                  # (B, 1)
                t_prob = t_probs.gather(1, d_tok).squeeze(1)          # (B,)

                d_pos_logits = draft(torch.cat([current, *draft_ids[:i+1]], dim=1))[1][:, -2, :]
                d_probs = F.softmax(d_pos_logits, dim=-1)
                d_prob = d_probs.gather(1, d_tok).squeeze(1)          # (B,)

                # Rejection sampling ratio
                ratio = (t_prob / (d_prob + 1e-10)).clamp(max=1.0)
                accept = (ratio > acceptance_threshold).all().item()

                if accept:
                    generated.append(d_tok)
                    accepted += 1
                else:
                    break

            # Advance context by accepted tokens
            if accepted > 0:
                current = torch.cat([current, *draft_ids[:accepted]], dim=1)

            # If we didn't fill the window, sample one from target
            if accepted < k and len(generated) < max_new:
                t_next_logits = t_logits[:, current.shape[1] - 1, :]  # (B, V)
                t_tok = t_next_logits.argmax(dim=-1, keepdim=True)    # (B, 1)
                generated.append(t_tok)
                current = torch.cat([current, t_tok], dim=1)

    # Trim to max_new
    result = torch.cat(generated[:max_new], dim=1)  # (B, max_new)
    return result


# ---------------------------------------------------------------------------
# Main checker class
# ---------------------------------------------------------------------------

class SpeculativeConsistencyChecker:
    """Verify that speculative decoding is statistically consistent with
    standard greedy decoding."""

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        n_test_samples: int = 100,
        acceptance_threshold: float = 0.0,
    ) -> None:
        self.target = target_model
        self.draft = draft_model
        self.n_test_samples = n_test_samples
        self.acceptance_threshold = acceptance_threshold

    # ------------------------------------------------------------------
    # Single sample verification
    # ------------------------------------------------------------------

    def verify_single(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
    ) -> bool:
        """Return True if greedy-target and speculative outputs are identical."""
        ref = greedy_decode(self.target, input_ids, max_new_tokens)
        spec = speculative_decode_simple(
            self.target,
            self.draft,
            input_ids,
            n_draft=3,
            max_new=max_new_tokens,
            acceptance_threshold=self.acceptance_threshold,
        )
        return bool((ref == spec).all().item())

    # ------------------------------------------------------------------
    # Acceptance statistics
    # ------------------------------------------------------------------

    def compute_acceptance_stats(
        self,
        input_ids: Tensor,
        n_samples: int = 50,
    ) -> dict:
        """Run speculative decoding multiple times and measure acceptance rates.

        Returns
        -------
        dict with keys: mean_acceptance_rate, std_acceptance_rate,
                        min_acceptance, max_acceptance
        """
        rates: list[float] = []

        with torch.no_grad():
            for _ in range(n_samples):
                accepted = 0
                total = 0
                current = input_ids.clone()
                n_draft = 3
                max_new = 10

                while total < max_new:
                    remaining = max_new - total
                    k = min(n_draft, remaining)

                    # Draft
                    draft_ctx = current.clone()
                    draft_ids: list[Tensor] = []
                    for _ in range(k):
                        d_out = self.draft(draft_ctx)
                        d_tok = d_out[1][:, -1, :].argmax(dim=-1, keepdim=True)
                        draft_ids.append(d_tok)
                        draft_ctx = torch.cat([draft_ctx, d_tok], dim=1)

                    # Verify
                    verify_input = torch.cat([current, *draft_ids], dim=1)
                    t_out = self.target(verify_input)
                    t_logits = t_out[1]

                    window_accepted = 0
                    for i in range(k):
                        t_pos_logits = t_logits[:, current.shape[1] + i - 1, :]
                        t_probs = F.softmax(t_pos_logits, dim=-1)
                        d_tok = draft_ids[i]
                        t_prob = t_probs.gather(1, d_tok).squeeze(1)

                        d_pos_logits = self.draft(
                            torch.cat([current, *draft_ids[:i+1]], dim=1)
                        )[1][:, -2, :]
                        d_probs = F.softmax(d_pos_logits, dim=-1)
                        d_prob = d_probs.gather(1, d_tok).squeeze(1)

                        ratio = (t_prob / (d_prob + 1e-10)).clamp(max=1.0)
                        accept = (ratio > self.acceptance_threshold).all().item()
                        if accept:
                            window_accepted += 1
                        else:
                            break

                    accepted += window_accepted
                    total += k

                    if window_accepted > 0:
                        current = torch.cat([current, *draft_ids[:window_accepted]], dim=1)
                    if window_accepted < k:
                        t_next_logits = t_logits[:, current.shape[1] - 1, :]
                        t_tok = t_next_logits.argmax(dim=-1, keepdim=True)
                        current = torch.cat([current, t_tok], dim=1)
                        total = min(total + 1 - window_accepted, max_new)

                rate = accepted / max(total, 1)
                rates.append(rate)

        rates_t = torch.tensor(rates)
        return {
            "mean_acceptance_rate": float(rates_t.mean()),
            "std_acceptance_rate": float(rates_t.std()) if len(rates) > 1 else 0.0,
            "min_acceptance": float(rates_t.min()),
            "max_acceptance": float(rates_t.max()),
        }

    # ------------------------------------------------------------------
    # Distribution test
    # ------------------------------------------------------------------

    def distribution_test(
        self,
        input_ids: Tensor,
        n_samples: int = 100,
        position: int = 0,
    ) -> dict:
        """Compare token probability distributions at *position*.

        Computes KL divergence D_KL(target || draft) and total variation
        distance (TVD) between the softmax distributions at the given position.

        Returns
        -------
        dict with keys: kl_divergence, tvd, is_consistent
        """
        with torch.no_grad():
            # Use a fixed context up to 'position' (or full input if short)
            ctx_len = min(position + 1, input_ids.shape[1])
            ctx = input_ids[:, :ctx_len]

            t_out = self.target(ctx)
            t_logits = t_out[1][:, -1, :]         # (B, V)
            t_probs = F.softmax(t_logits, dim=-1)  # (B, V)

            d_out = self.draft(ctx)
            d_logits = d_out[1][:, -1, :]          # (B, V)
            d_probs = F.softmax(d_logits, dim=-1)  # (B, V)

            # KL divergence: D_KL(target || draft) — mean over batch
            eps = 1e-10
            kl = (t_probs * ((t_probs + eps).log() - (d_probs + eps).log())).sum(dim=-1)
            kl_mean = float(kl.mean())

            # Total variation distance: 0.5 * sum |p - q|
            tvd = 0.5 * (t_probs - d_probs).abs().sum(dim=-1)
            tvd_mean = float(tvd.mean())

            is_consistent = kl_mean < 0.1 and tvd_mean < 0.1

        return {
            "kl_divergence": kl_mean,
            "tvd": tvd_mean,
            "is_consistent": is_consistent,
        }

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def calibrate_threshold(
        self,
        input_ids_list: List[Tensor],
        target_acceptance: float = 0.8,
    ) -> float:
        """Binary search for the acceptance threshold that achieves
        *target_acceptance* rate.

        Parameters
        ----------
        input_ids_list:
            List of input tensors to average acceptance rates over.
        target_acceptance:
            Desired mean acceptance rate.

        Returns
        -------
        threshold : float in [0, 1]
        """

        def _mean_rate(threshold: float) -> float:
            orig_thresh = self.acceptance_threshold
            self.acceptance_threshold = threshold
            rates = []
            for ids in input_ids_list:
                stats = self.compute_acceptance_stats(ids, n_samples=10)
                rates.append(stats["mean_acceptance_rate"])
            self.acceptance_threshold = orig_thresh
            return sum(rates) / len(rates) if rates else 0.0

        lo, hi = 0.0, 1.0
        best = lo

        for _ in range(20):  # 20 iterations → ~1e-6 precision
            mid = (lo + hi) / 2.0
            rate = _mean_rate(mid)
            if rate >= target_acceptance:
                best = mid
                lo = mid
            else:
                hi = mid

        return float(best)
