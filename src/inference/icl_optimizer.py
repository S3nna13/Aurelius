"""ICL Optimizer -- selection, ordering, and formatting of few-shot demonstrations.

Pure stdlib + torch only.  No transformers / einops / trl / etc.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# DemonstrationExample
# ---------------------------------------------------------------------------


@dataclass
class DemonstrationExample:
    """A single few-shot demonstration."""

    input_ids: Tensor
    label_ids: Tensor
    embedding: Tensor | None = None
    metadata: dict = field(default_factory=dict)

    def to_sequence(self, separator_id: int = 1) -> Tensor:
        """Concatenate input_ids + [separator_id] + label_ids into a 1-D tensor."""
        sep = torch.tensor([separator_id], dtype=self.input_ids.dtype)
        return torch.cat([self.input_ids, sep, self.label_ids], dim=0)

    def length(self) -> int:
        """Total token count: len(input_ids) + 1 (sep) + len(label_ids)."""
        return int(self.input_ids.shape[0]) + 1 + int(self.label_ids.shape[0])


# ---------------------------------------------------------------------------
# ExemplarSelector
# ---------------------------------------------------------------------------


def _cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity between 1-D vectors a and b."""
    a_norm = F.normalize(a.float().unsqueeze(0), dim=-1)
    b_norm = F.normalize(b.float().unsqueeze(0), dim=-1)
    return (a_norm * b_norm).sum()


class ExemplarSelector:
    """Select k relevant demonstrations from a pool for a given query."""

    def __init__(self, method: str = "random", k: int = 4) -> None:
        if method not in ("random", "similarity", "diverse"):
            raise ValueError(f"Unknown method '{method}'. Choose: random, similarity, diverse.")
        self.method = method
        self.k = k

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def random_select(
        self,
        pool: list[DemonstrationExample],
        query_embedding: Tensor | None = None,
    ) -> list[DemonstrationExample]:
        """Return k randomly sampled examples (without replacement when possible)."""
        k = min(self.k, len(pool))
        return _random.sample(pool, k)

    def similarity_select(
        self,
        pool: list[DemonstrationExample],
        query_embedding: Tensor,
    ) -> list[DemonstrationExample]:
        """Return the top-k examples most similar to query_embedding."""
        scored: list[tuple] = []
        for ex in pool:
            if ex.embedding is None:
                raise ValueError("similarity_select requires every example to have an embedding.")
            sim = float(_cosine_sim(query_embedding, ex.embedding))
            scored.append((sim, ex))
        scored.sort(key=lambda t: t[0], reverse=True)
        k = min(self.k, len(scored))
        return [ex for _, ex in scored[:k]]

    def diverse_select(
        self,
        pool: list[DemonstrationExample],
        query_embedding: Tensor,
    ) -> list[DemonstrationExample]:
        """Greedy diversity: start with most-similar, then maximise min-distance."""
        if not pool:
            return []
        for ex in pool:
            if ex.embedding is None:
                raise ValueError("diverse_select requires every example to have an embedding.")

        k = min(self.k, len(pool))
        if k == 0:
            return []

        sims = [float(_cosine_sim(query_embedding, ex.embedding)) for ex in pool]
        best_idx = int(max(range(len(pool)), key=lambda i: sims[i]))

        selected_indices = [best_idx]
        remaining = list(range(len(pool)))
        remaining.remove(best_idx)

        while len(selected_indices) < k and remaining:
            best_next = -1
            best_score = -1.0
            for idx in remaining:
                emb_idx = pool[idx].embedding.float()
                min_dist = min(
                    float(1.0 - _cosine_sim(emb_idx, pool[s].embedding)) for s in selected_indices
                )
                if min_dist > best_score:
                    best_score = min_dist
                    best_next = idx
            selected_indices.append(best_next)
            remaining.remove(best_next)

        return [pool[i] for i in selected_indices]

    def select(
        self,
        pool: list[DemonstrationExample],
        query_embedding: Tensor | None = None,
    ) -> list[DemonstrationExample]:
        """Dispatch to the configured selection method."""
        if self.method == "random":
            return self.random_select(pool, query_embedding)
        if self.method == "similarity":
            if query_embedding is None:
                raise ValueError("similarity method requires query_embedding.")
            return self.similarity_select(pool, query_embedding)
        if self.method == "diverse":
            if query_embedding is None:
                raise ValueError("diverse method requires query_embedding.")
            return self.diverse_select(pool, query_embedding)
        raise ValueError(f"Unknown method '{self.method}'.")


# ---------------------------------------------------------------------------
# PromptOrdering
# ---------------------------------------------------------------------------


class PromptOrdering:
    """Reorder demonstrations for best ICL performance."""

    def __init__(self) -> None:
        pass

    def random_order(self, examples: list[DemonstrationExample]) -> list[DemonstrationExample]:
        """Return a shuffled copy."""
        shuffled = list(examples)
        _random.shuffle(shuffled)
        return shuffled

    def similarity_order(
        self,
        examples: list[DemonstrationExample],
        query_embedding: Tensor,
    ) -> list[DemonstrationExample]:
        """Sort by cosine similarity to query -- most similar last (recency bias)."""

        def _sim(ex: DemonstrationExample) -> float:
            if ex.embedding is None:
                raise ValueError("similarity_order requires every example to have an embedding.")
            return float(_cosine_sim(query_embedding, ex.embedding))

        return sorted(examples, key=_sim)  # ascending -> most similar last

    def curriculum_order(
        self,
        examples: list[DemonstrationExample],
        difficulty_scores: list[float],
    ) -> list[DemonstrationExample]:
        """Sort easy-to-hard: ascending difficulty_scores."""
        if len(examples) != len(difficulty_scores):
            raise ValueError("examples and difficulty_scores must have the same length.")
        paired = sorted(zip(difficulty_scores, range(len(examples))))
        return [examples[i] for _, i in paired]


# ---------------------------------------------------------------------------
# ICLPromptBuilder
# ---------------------------------------------------------------------------


class ICLPromptBuilder:
    """Construct a full ICL prompt tensor from demonstrations + query."""

    def __init__(self, max_length: int = 512, separator_id: int = 1) -> None:
        self.max_length = max_length
        self.separator_id = separator_id

    def build(
        self,
        demonstrations: list[DemonstrationExample],
        query_ids: Tensor,
    ) -> Tensor:
        """Concatenate demos (each followed by separator) then query; truncate from left if needed."""  # noqa: E501
        query_len = int(query_ids.shape[0])

        # Build individual demo sequences (each ends with separator)
        demo_seqs: list[Tensor] = []
        for demo in demonstrations:
            seq = demo.to_sequence(self.separator_id)
            sep = torch.tensor([self.separator_id], dtype=query_ids.dtype)
            demo_seqs.append(torch.cat([seq, sep], dim=0))

        # Greedy right-to-left packing: always include query, add demos from last
        budget = self.max_length - query_len
        selected_demo_seqs: list[Tensor] = []
        for seq in reversed(demo_seqs):
            if budget >= int(seq.shape[0]):
                selected_demo_seqs.insert(0, seq)
                budget -= int(seq.shape[0])

        parts = selected_demo_seqs + [query_ids]
        result = torch.cat(parts, dim=0)
        # Hard truncation from the left as safety net
        if result.shape[0] > self.max_length:
            result = result[-self.max_length :]
        return result

    def n_shots_that_fit(
        self,
        demonstrations: list[DemonstrationExample],
        query_ids: Tensor,
    ) -> int:
        """Maximum k such that the first k demonstrations + query fit in max_length."""
        query_len = int(query_ids.shape[0])
        budget = self.max_length - query_len
        if budget <= 0:
            return 0
        count = 0
        for demo in demonstrations:
            # to_sequence() length + 1 extra separator added in build()
            needed = demo.length() + 1
            if budget >= needed:
                budget -= needed
                count += 1
            else:
                break
        return count


# ---------------------------------------------------------------------------
# ICLScorer  (model-scoring utilities, avoids name collision with built-ins)
# ---------------------------------------------------------------------------


class ICLScorer:
    """Score ICL prompts using a language model (forward/backward capable)."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def score_sequence(self, prompt_ids: Tensor, label_ids: Tensor) -> float:
        """Return sum of log-probs of label_ids tokens given prompt_ids context."""
        self.model.train(False)
        with torch.no_grad():
            full_ids = torch.cat([prompt_ids, label_ids], dim=0)  # (T,)
            input_ids = full_ids.unsqueeze(0)  # (1, T)

            logits = self.model(input_ids)  # (1, T, V)
            logits = logits.squeeze(0)  # (T, V)

            log_probs = F.log_softmax(logits, dim=-1)  # (T, V)

            prompt_len = int(prompt_ids.shape[0])
            label_len = int(label_ids.shape[0])

            score = 0.0
            for j in range(label_len):
                pred_pos = prompt_len - 1 + j
                tok = int(label_ids[j].item())
                score += float(log_probs[pred_pos, tok].item())
        return score

    def accuracy(
        self,
        prompts: list[Tensor],
        labels: list[Tensor],
        candidates: list[list[Tensor]],
    ) -> float:
        """Predict completion via argmax scoring; return fraction correct."""
        if not prompts:
            return 0.0
        correct = 0
        for prompt, label, cands in zip(prompts, labels, candidates):
            scores = [self.score_sequence(prompt, c) for c in cands]
            pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            pred = cands[pred_idx]
            if pred.shape == label.shape and bool(torch.all(pred == label).item()):
                correct += 1
        return correct / len(prompts)

    def calibration_bias(
        self,
        prompts: list[Tensor],
        labels: list[Tensor],
    ) -> float:
        """Estimate calibration bias relative to a content-free (empty) context."""
        if not prompts:
            return 0.0
        total = 0.0
        for prompt, label in zip(prompts, labels):
            real_score = self.score_sequence(prompt, label)
            empty_prompt = torch.zeros(1, dtype=prompt.dtype)
            baseline_score = self.score_sequence(empty_prompt, label)
            total += real_score - baseline_score
        return total / len(prompts)


# ---------------------------------------------------------------------------
# ICLEvaluator  (alias kept for API compatibility)
# ---------------------------------------------------------------------------


class ICLEvaluator(ICLScorer):
    """Alias for ICLScorer -- evaluates ICL performance via model scoring."""

    pass
