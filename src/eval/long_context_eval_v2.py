"""Long-context assessment v2: Needle-in-Haystack, position bias, context utilization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# NeedleInHaystack
# ---------------------------------------------------------------------------


class NeedleInHaystack:
    """Check ability to retrieve a 'needle' (key fact) from a long context."""

    def __init__(self, n_positions: int = 10, n_depths: int = 5) -> None:
        self.n_positions = n_positions
        self.n_depths = n_depths

    def create_test(
        self,
        haystack_ids: Tensor,
        needle_ids: Tensor,
        depth_fraction: float,
    ) -> tuple[Tensor, int]:
        """Insert needle into haystack at the given depth fraction.

        depth_fraction in [0, 1]: 0 = start, 1 = end.
        Returns (context_ids, needle_position).
        """
        depth_fraction = max(0.0, min(1.0, float(depth_fraction)))
        insert_at = int(depth_fraction * len(haystack_ids))
        context_ids = torch.cat([haystack_ids[:insert_at], needle_ids, haystack_ids[insert_at:]])
        return context_ids, insert_at

    def score_retrieval(
        self,
        model: nn.Module,
        context_ids: Tensor,
        needle_ids: Tensor,
        needle_position: int,
    ) -> float:
        """Compute normalised log-probability of needle_ids given context.

        For each needle token at index i, computes
        log p(needle[i] | context[:needle_position + i]).
        Returns mean log-prob (<=0).
        """
        needle_len = len(needle_ids)
        if needle_len == 0:
            return 0.0

        model.train(False)
        device = next(model.parameters()).device
        context_ids = context_ids.to(device)
        needle_ids = needle_ids.to(device)

        log_probs: list[float] = []
        with torch.no_grad():
            for i in range(needle_len):
                # Context up to (but not including) needle token i
                prefix_end = needle_position + i
                prefix_end = max(prefix_end, 1)
                input_seq = context_ids[:prefix_end].unsqueeze(0)  # (1, T)
                out = model(input_seq)
                if isinstance(out, (tuple, list)):
                    logits = out[1]
                else:
                    logits = out
                last_logits = logits[0, -1]  # (V,)
                lp = F.log_softmax(last_logits, dim=-1)
                target_token = needle_ids[i].long()
                log_probs.append(lp[target_token].item())

        return sum(log_probs) / needle_len


# ---------------------------------------------------------------------------
# PositionBiasAnalyzer
# ---------------------------------------------------------------------------


class PositionBiasAnalyzer:
    """Measure how model attention varies with token position."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def compute_position_attention(self, attn_weights: Tensor) -> Tensor:
        """Compute mean attention received by each key position.

        attn_weights: (B, H, T, T).
        Returns position_scores: (T,) normalised so values sum to 1.
        """
        # Average over batch and heads -> (T, T)
        mean_attn = attn_weights.float().mean(dim=0).mean(dim=0)  # (T, T)
        # Mean over query dim -> (T,) attention received per key position
        position_scores = mean_attn.mean(dim=0)
        total = position_scores.sum()
        if total > 0:
            position_scores = position_scores / total
        return position_scores

    def recency_bias(self, position_scores: Tensor) -> float:
        """Spearman correlation between position index and attention score.

        Positive value => recency bias. Returns value in [-1, 1].
        Tied values receive the average of their ranks (standard correction).
        """
        n = len(position_scores)
        if n < 2:
            return 0.0

        scores_cpu = position_scores.float().cpu().tolist()

        # Assign average ranks with tie handling for attention scores
        score_ranks = _average_ranks(scores_cpu)

        # Position ranks: 0, 1, ..., n-1 (no ties, already distinct)
        pos_ranks_list = list(range(n))

        # If all attention scores are identical (all-tie), correlation is 0
        score_set = set(scores_cpu)
        if len(score_set) == 1:
            return 0.0

        d_sq_sum = 0.0
        for pr, sr in zip(pos_ranks_list, score_ranks):
            d = pr - sr
            d_sq_sum += d * d

        denom = n * (n * n - 1)
        if denom == 0:
            return 0.0
        spearman = 1.0 - 6.0 * d_sq_sum / denom
        return float(max(-1.0, min(1.0, spearman)))

    def primacy_bias(self, position_scores: Tensor) -> float:
        """Ratio: first-10% attention mean / middle-80% attention mean. >= 0."""
        n = len(position_scores)
        if n < 2:
            return 0.0

        first_end = max(1, int(math.ceil(0.10 * n)))
        last_start = max(first_end, int(math.floor(0.90 * n)))

        first_mean = position_scores[:first_end].float().mean().item()
        middle = position_scores[first_end:last_start]
        if len(middle) == 0:
            middle_mean = first_mean if first_mean > 0 else 1.0
        else:
            middle_mean = middle.float().mean().item()

        if middle_mean == 0:
            return 0.0
        return max(0.0, first_mean / middle_mean)


# ---------------------------------------------------------------------------
# LostInTheMiddleScore
# ---------------------------------------------------------------------------


class LostInTheMiddleScore:
    """Measure U-shaped attention (primacy + recency, middle forgotten)."""

    def __init__(self) -> None:
        pass

    def score(self, position_scores: Tensor) -> float:
        """Compute U-shape score.

        (first_quarter_mean + last_quarter_mean) / middle_half_mean.
        Score > 1 means U-shaped. Returns >= 0.
        """
        n = len(position_scores)
        if n < 4:
            return 1.0

        q = n // 4
        first_quarter = position_scores[:q].float()
        last_quarter = position_scores[n - q :].float()
        middle_half = position_scores[q : n - q].float()

        first_mean = first_quarter.mean().item()
        last_mean = last_quarter.mean().item()

        if len(middle_half) == 0:
            middle_mean = (first_mean + last_mean) / 2.0
        else:
            middle_mean = middle_half.mean().item()

        if middle_mean == 0:
            return 0.0

        return max(0.0, (first_mean + last_mean) / middle_mean)

    def optimal_insertion_depth(self, scores_per_depth: list[float]) -> int:
        """Index of the depth position with the highest retrieval score."""
        if not scores_per_depth:
            return 0
        best = 0
        best_score = scores_per_depth[0]
        for i, s in enumerate(scores_per_depth):
            if s > best_score:
                best_score = s
                best = i
        return best


# ---------------------------------------------------------------------------
# ContextUtilizationScore
# ---------------------------------------------------------------------------


class ContextUtilizationScore:
    """How much of the context influences the output (gradient-based)."""

    def __init__(self) -> None:
        pass

    def token_influence(
        self,
        model: nn.Module,
        input_ids: Tensor,
        output_position: int,
    ) -> Tensor:
        """Gradient of output logit w.r.t. each input embedding norm.

        Returns (T,) tensor of non-negative gradient magnitudes.
        """
        device = next(model.parameters()).device
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        T = input_ids.shape[1]

        embed_layer = _find_embedding(model)

        model.train(False)
        # Compute embeddings with grad tracking
        embeds = embed_layer(input_ids)  # (1, T, d_model)
        embeds = embeds.detach().requires_grad_(True)

        out = _forward_from_embeds(model, embed_layer, embeds)
        if isinstance(out, (tuple, list)):
            logits = out[1]
        else:
            logits = out
        target_logit = logits[0, -1, output_position]
        target_logit.backward()

        grad = embeds.grad  # (1, T, d_model)
        if grad is None:
            return torch.zeros(T)
        influence = grad[0].norm(dim=-1)  # (T,)
        return influence.detach()

    def effective_context_fraction(
        self,
        influence_scores: Tensor,
        threshold: float = 0.01,
    ) -> float:
        """Fraction of tokens with influence > threshold * max influence. In [0,1]."""
        if len(influence_scores) == 0:
            return 0.0
        max_inf = influence_scores.max().item()
        if max_inf == 0:
            return 0.0
        cutoff = threshold * max_inf
        fraction = (influence_scores > cutoff).float().mean().item()
        return float(fraction)


# ---------------------------------------------------------------------------
# LongContextBenchmark
# ---------------------------------------------------------------------------


class LongContextBenchmark:
    """Aggregate long-context assessment."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._nih = NeedleInHaystack()
        self._pba = PositionBiasAnalyzer()
        self._litm = LostInTheMiddleScore()

    def run_needle_test(
        self,
        haystack_ids: Tensor,
        needle_ids: Tensor,
    ) -> dict:
        """Test needle retrieval across 5 depth positions (0.1, 0.3, 0.5, 0.7, 0.9).

        Returns dict with 'mean_score' and 'per_depth_scores'.
        """
        depths = [0.1, 0.3, 0.5, 0.7, 0.9]
        per_depth: list[float] = []
        for d in depths:
            ctx, needle_pos = self._nih.create_test(haystack_ids, needle_ids, d)
            s = self._nih.score_retrieval(self.model, ctx, needle_ids, needle_pos)
            per_depth.append(s)
        mean_score = sum(per_depth) / len(per_depth)
        return {"mean_score": mean_score, "per_depth_scores": per_depth}

    def position_bias_report(self, input_ids: Tensor) -> dict:
        """Compute position bias metrics.

        Returns dict with keys 'recency_bias', 'primacy_bias', 'lost_in_middle'.
        If model exposes no attention weights, uniform attention is used.
        """
        device = next(self.model.parameters()).device
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        T = input_ids.shape[1]

        attn_weights = _extract_attention(self.model, input_ids, T)
        pos_scores = self._pba.compute_position_attention(attn_weights)
        return {
            "recency_bias": self._pba.recency_bias(pos_scores),
            "primacy_bias": self._pba.primacy_bias(pos_scores),
            "lost_in_middle": self._litm.score(pos_scores),
        }

    def summarize(self, needle_results: dict, bias_results: dict) -> dict:
        """Produce overall_score combining needle quality and bias penalty.

        Returns dict with key 'overall_score' (finite float).
        """
        mean_needle = needle_results.get("mean_score", 0.0)
        needle_quality = math.exp(mean_needle) if math.isfinite(mean_needle) else 0.0

        recency = bias_results.get("recency_bias", 0.0)
        litm = bias_results.get("lost_in_middle", 1.0)
        bias_penalty = abs(recency) + max(0.0, litm - 1.0)
        overall = needle_quality / (1.0 + bias_penalty)
        if not math.isfinite(overall):
            overall = 0.0
        return {
            "overall_score": overall,
            "needle_quality": needle_quality,
            "bias_penalty": bias_penalty,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_embedding(model: nn.Module) -> nn.Embedding:
    """Return the first nn.Embedding found in the model."""
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return module
    raise ValueError("No nn.Embedding found in model")


def _forward_from_embeds(
    model: nn.Module,
    embed_layer: nn.Embedding,
    embeds: Tensor,
) -> object:
    """Run forward pass with pre-computed embeddings by patching the layer."""
    B, T, _ = embeds.shape
    original_forward = embed_layer.forward

    def patched_forward(x: Tensor) -> Tensor:
        return embeds

    embed_layer.forward = patched_forward  # type: ignore[method-assign]
    device = embeds.device
    dummy_ids = torch.zeros(B, T, dtype=torch.long, device=device)
    try:
        out = model(dummy_ids)
    finally:
        embed_layer.forward = original_forward  # type: ignore[method-assign]
    return out


def _extract_attention(
    model: nn.Module,
    input_ids: Tensor,
    T: int,
) -> Tensor:
    """Extract attention weights (B, H, T, T) from model, or return uniform fallback."""
    captured: list[Tensor] = []

    def hook(module: nn.Module, inp: tuple, out: object) -> None:
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            w = out[1]
            if isinstance(w, Tensor) and w.dim() == 4:
                captured.append(w.detach())

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(hook))

    with torch.no_grad():
        try:
            model(input_ids)
        finally:
            for h in hooks:
                h.remove()

    if captured:
        stacked = torch.stack(captured, dim=0).mean(dim=0)  # (B, H, T, T)
        return stacked

    # Fallback: uniform attention
    B = input_ids.shape[0]
    attn = torch.ones(B, 1, T, T) / T
    return attn


def _average_ranks(values: list[float]) -> list[float]:
    """Assign average ranks to a list of values, with tie handling.

    Tied values receive the average of the ordinal ranks they would occupy.
    Ranks start at 0 (0-based).

    Parameters
    ----------
    values : list[float]
        Input values.

    Returns
    -------
    list[float]
        Average ranks, same length as values.
    """
    n = len(values)
    # Sort by value, keeping original indices
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find the run of tied values
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        # Average rank for the tied group (0-based)
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks
