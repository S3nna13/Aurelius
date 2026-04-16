"""LLM-based data selection for instruction-tuning datasets.

Implements modern data selection approaches inspired by:
- LIMA: 1000 carefully selected examples can match much larger training sets.
- AlpaGasus: LLM-judge scoring to filter low-quality instruction data.
- NUGGETS: keyword-coverage heuristics for data quality estimation.

Core classes: DataScorer, InstructionFollowingScorer, DiversitySelector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataSelectionConfig:
    """Configuration for LLM-based data selection pipeline."""
    top_fraction: float = 0.1       # fraction of data to keep
    diversity_weight: float = 0.5   # weight for diversity vs quality
    min_score: float = 0.0          # minimum score threshold
    batch_size: int = 8             # scoring batch size


# ---------------------------------------------------------------------------
# DataScorer
# ---------------------------------------------------------------------------

class DataScorer:
    """Score a dataset using an arbitrary scoring function and select subsets.

    Usage::

        scorer = DataScorer(score_fn=my_llm_judge)
        scores = scorer.score_dataset(examples)
        selected, sel_scores = scorer.select_top_k(examples, scores, k=100)
    """

    def __init__(
        self,
        score_fn: Callable[[str, str], float],
        batch_size: int = 8,
    ) -> None:
        self.score_fn = score_fn
        self.batch_size = batch_size

    def score_dataset(
        self,
        examples: List[dict],
        instruction_key: str = "instruction",
        response_key: str = "response",
    ) -> List[float]:
        """Score all examples using score_fn.

        Args:
            examples: list of dicts, each with instruction and response fields.
            instruction_key: key for the instruction field.
            response_key: key for the response field.

        Returns:
            List of float scores, one per example.
        """
        scores: List[float] = []
        for ex in examples:
            instruction = ex.get(instruction_key, "")
            response = ex.get(response_key, "")
            score = self.score_fn(instruction, response)
            scores.append(float(score))
        return scores

    def select_top_k(
        self,
        examples: List[dict],
        scores: List[float],
        k: Optional[int] = None,
        fraction: Optional[float] = None,
    ) -> Tuple[List[dict], List[float]]:
        """Select top-k examples by score.

        Exactly one of k or fraction must be specified.

        Args:
            examples: list of dicts.
            scores:   parallel list of float scores.
            k:        absolute number of examples to keep.
            fraction: fraction of total examples to keep (e.g. 0.5 = 50%).

        Returns:
            (selected_examples, selected_scores) sorted best-first.
        """
        n = len(examples)
        if k is None and fraction is None:
            raise ValueError("Exactly one of k or fraction must be specified.")
        if k is None:
            k = max(1, int(math.ceil(n * fraction)))

        k = min(k, n)

        # Sort indices by score descending
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indexed[:k]]

        selected_examples = [examples[i] for i in top_indices]
        selected_scores = [scores[i] for i in top_indices]
        return selected_examples, selected_scores

    def filter_by_threshold(
        self,
        examples: List[dict],
        scores: List[float],
        threshold: float,
    ) -> Tuple[List[dict], List[float]]:
        """Keep only examples with score >= threshold.

        Args:
            examples:  list of dicts.
            scores:    parallel list of float scores.
            threshold: minimum acceptable score.

        Returns:
            (filtered_examples, filtered_scores)
        """
        filtered_examples: List[dict] = []
        filtered_scores: List[float] = []
        for ex, sc in zip(examples, scores):
            if sc >= threshold:
                filtered_examples.append(ex)
                filtered_scores.append(sc)
        return filtered_examples, filtered_scores


# ---------------------------------------------------------------------------
# InstructionFollowingScorer
# ---------------------------------------------------------------------------

class InstructionFollowingScorer:
    """Score how well a response follows an instruction via log-probability.

    Uses a language model to compute the (negative) perplexity of the
    response conditioned on the instruction.  Higher = better.

    Usage::

        scorer = InstructionFollowingScorer(model)
        score = scorer.score(instruction_ids, response_ids)
        scores = scorer.batch_score([inst1, inst2], [resp1, resp2])
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_vocab_size: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer_vocab_size = tokenizer_vocab_size

    @torch.no_grad()
    def score(
        self,
        instruction_ids: torch.Tensor,  # (S_i,)
        response_ids: torch.Tensor,     # (S_r,)
    ) -> float:
        """Return negative perplexity of response given instruction.

        Concatenates [instruction, response], runs the model, and measures
        the average cross-entropy loss only on the response tokens.

        Higher score (less negative) means the model assigns higher probability
        to the response, indicating better instruction-following quality.

        Args:
            instruction_ids: 1-D token-id tensor for the instruction.
            response_ids:    1-D token-id tensor for the response.

        Returns:
            float -- negative mean per-token cross-entropy (higher is better).
        """
        self.model.eval()

        # Concatenate and add batch dimension: (1, S_i + S_r)
        full_ids = torch.cat([instruction_ids, response_ids], dim=0).unsqueeze(0)

        _loss, logits, _extra = self.model(full_ids)

        # We only care about response tokens.
        # logits[:, t-1, :] predicts full_ids[:, t]
        inst_len = instruction_ids.shape[0]
        resp_len = response_ids.shape[0]

        if resp_len == 0:
            return 0.0

        # Slice logits/labels for the response window
        logit_start = max(inst_len - 1, 0)
        logit_end = inst_len + resp_len - 1

        resp_logits = logits[0, logit_start:logit_end, :]  # (resp_len, V)
        resp_targets = response_ids[:resp_logits.shape[0]]  # (resp_len,)

        # Clamp targets to valid vocab range
        resp_targets = resp_targets.clamp(0, resp_logits.shape[-1] - 1)

        loss = F.cross_entropy(resp_logits, resp_targets, reduction="mean")
        return -loss.item()

    @torch.no_grad()
    def batch_score(
        self,
        instruction_ids_list: List[torch.Tensor],
        response_ids_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Score a batch of (instruction, response) pairs.

        Args:
            instruction_ids_list: list of 1-D tensors.
            response_ids_list:    list of 1-D tensors.

        Returns:
            (n,) float tensor of scores (one per pair).
        """
        scores = [
            self.score(inst, resp)
            for inst, resp in zip(instruction_ids_list, response_ids_list)
        ]
        return torch.tensor(scores, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DiversitySelector
# ---------------------------------------------------------------------------

class DiversitySelector:
    """Select a maximally diverse subset from a collection of examples.

    Uses greedy maximum-coverage (farthest-point sampling) to iteratively
    pick the example that is farthest (most different) from the already-
    selected set.

    Usage::

        selector = DiversitySelector(embedding_fn=my_embed, n_select=50)
        embeddings = selector.compute_embeddings(token_id_tensors)
        indices = selector.select_diverse(embeddings, n=50)
        indices = selector.quality_diversity_select(embeddings, scores)
    """

    def __init__(
        self,
        embedding_fn: Callable,
        n_select: int = 100,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.n_select = n_select

    def compute_embeddings(
        self,
        examples: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute embeddings for all examples.

        Calls self.embedding_fn on each example and stacks results.

        Args:
            examples: list of tensors passed to embedding_fn one at a time.

        Returns:
            (n, embedding_dim) float tensor.
        """
        vecs = [self.embedding_fn(ex) for ex in examples]
        return torch.stack(vecs, dim=0)  # (n, D)

    def select_diverse(
        self,
        embeddings: torch.Tensor,  # (n, D)
        n: Optional[int] = None,
    ) -> List[int]:
        """Greedy maximum-coverage selection (farthest-point sampling).

        Iteratively selects the example that maximises the minimum cosine
        distance to the already-selected set.

        Args:
            embeddings: (n, D) float tensor.
            n:          number of examples to select (defaults to self.n_select).

        Returns:
            List of selected indices (length == n), in selection order.
        """
        if n is None:
            n = self.n_select

        total = embeddings.shape[0]
        n = min(n, total)

        if n == 0:
            return []

        # Normalise for cosine distance
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = embeddings / norms  # (N, D)

        selected: List[int] = [0]
        # min cosine distance from each point to the selected set
        min_dist = torch.full((total,), float("inf"), dtype=embeddings.dtype)
        min_dist[0] = -1.0  # already selected

        for _ in range(n - 1):
            last = normed[selected[-1]].unsqueeze(0)  # (1, D)
            cos_sim = (normed @ last.T).squeeze(-1)   # (N,)
            dist = 1.0 - cos_sim                       # cosine distance

            # Update running minimum
            min_dist = torch.minimum(min_dist, dist)
            # Exclude already-selected
            for idx in selected:
                min_dist[idx] = -1.0

            selected.append(int(min_dist.argmax().item()))

        return selected

    def quality_diversity_select(
        self,
        embeddings: torch.Tensor,  # (n, D)
        scores: torch.Tensor,      # (n,)
        alpha: float = 0.5,
    ) -> List[int]:
        """Combined quality + diversity selection.

        Combined score = alpha * normalised_quality + (1 - alpha) * normalised_diversity.

        Diversity is measured as the mean cosine distance to all other examples.

        Args:
            embeddings: (n, D) float tensor.
            scores:     (n,) float quality scores.
            alpha:      weight for quality (1-alpha goes to diversity).

        Returns:
            List of indices sorted by combined score (best first),
            length == self.n_select (capped at n).
        """
        n = embeddings.shape[0]
        k = min(self.n_select, n)

        # Normalise quality scores to [0, 1]
        q_min = scores.min()
        q_max = scores.max()
        if (q_max - q_min).abs() < 1e-8:
            norm_quality = torch.zeros_like(scores)
        else:
            norm_quality = (scores - q_min) / (q_max - q_min)

        # Diversity: mean cosine distance to all others
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = embeddings / norms                        # (n, D)
        cos_sim_matrix = normed @ normed.T                 # (n, n)
        cos_dist_matrix = 1.0 - cos_sim_matrix
        mask = 1.0 - torch.eye(n, device=embeddings.device, dtype=embeddings.dtype)
        if n > 1:
            mean_div = (cos_dist_matrix * mask).sum(dim=-1) / (n - 1)  # (n,)
        else:
            mean_div = torch.zeros(n, device=embeddings.device, dtype=embeddings.dtype)

        # Normalise diversity to [0, 1]
        d_min = mean_div.min()
        d_max = mean_div.max()
        if (d_max - d_min).abs() < 1e-8:
            norm_div = torch.zeros_like(mean_div)
        else:
            norm_div = (mean_div - d_min) / (d_max - d_min)

        combined = alpha * norm_quality + (1.0 - alpha) * norm_div

        # Sort descending by combined score, return top-k indices
        sorted_indices = combined.argsort(descending=True).tolist()
        return sorted_indices[:k]


# ---------------------------------------------------------------------------
# Standalone scoring functions
# ---------------------------------------------------------------------------

def nuggets_score(
    instruction: str,
    response: str,
    keywords: List[str],
) -> float:
    """NUGGETS-style heuristic quality score.

    Combines keyword coverage of the response and a length normalisation
    factor into a single score in [0, 1].

    Args:
        instruction: the instruction text (unused beyond reference).
        response:    the response text to score.
        keywords:    list of keywords that should appear in a high-quality response.

    Returns:
        Float in [0, 1].
    """
    if not response:
        return 0.0

    # Keyword coverage: fraction of keywords present in response (case-insensitive)
    if keywords:
        response_lower = response.lower()
        hits = sum(1 for kw in keywords if kw.lower() in response_lower)
        coverage = hits / len(keywords)
    else:
        coverage = 1.0  # no keywords -> full coverage by convention

    # Length normalisation: ideal length 50-500 chars
    length = len(response)
    ideal_min, ideal_max = 50, 500
    if ideal_min <= length <= ideal_max:
        length_score = 1.0
    elif length < ideal_min:
        length_score = length / ideal_min
    else:  # length > ideal_max
        length_score = max(0.0, 1.0 - (length - ideal_max) / ideal_max)

    score = 0.6 * coverage + 0.4 * length_score
    return float(min(max(score, 0.0), 1.0))


def alpagasus_filter(
    examples: List[dict],
    scores: List[float],
    threshold: float = 4.5,
) -> List[dict]:
    """AlpaGasus-style filter: keep examples with score >= threshold.

    AlpaGasus uses a 5.0-point scale (ChatGPT ratings) and keeps examples
    that score >= 4.5, dramatically improving data efficiency.

    Args:
        examples:  list of example dicts.
        scores:    parallel list of numeric scores (e.g. 1.0 to 5.0).
        threshold: minimum score to keep (default 4.5, matching the paper).

    Returns:
        List of examples whose score meets the threshold.
    """
    return [ex for ex, sc in zip(examples, scores) if sc >= threshold]
