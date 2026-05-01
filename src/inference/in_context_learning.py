"""In-context learning: few-shot example retrieval, prompt construction, and calibration."""

from __future__ import annotations

import random as random_module
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ICLConfig:
    """Configuration for in-context learning."""

    n_shots: int = 4
    selection_strategy: str = "random"  # "random" | "similarity" | "diverse"
    max_context_len: int = 2048
    label_smoothing: float = 0.0  # for calibration
    calibrate: bool = False  # apply contextual calibration
    template: str = "Q: {input}\nA: {output}"  # few-shot template


@dataclass
class FewShotExample:
    """A single few-shot example with optional cached embedding."""

    input: str
    output: str
    embedding: Tensor | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def format_few_shot_prompt(
    examples: list[FewShotExample],
    query: str,
    template: str,
) -> str:
    """Format examples using *template* and append the query with blank output.

    Each example is formatted with both {input} and {output} filled in.
    The query line fills {input} but leaves {output} blank.

    Returns the complete prompt string.
    """
    parts: list[str] = []

    for ex in examples:
        parts.append(template.format(input=ex.input, output=ex.output))

    # Query: fill input, leave output blank
    # Split on "A: " (or the output placeholder) to get the prefix.
    # We replace the output slot with an empty string.
    query_line = template.format(input=query, output="").rstrip()
    parts.append(query_line)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------


def select_random_examples(
    pool: list[FewShotExample],
    n: int,
    rng: random_module.Random,
) -> list[FewShotExample]:
    """Randomly select *n* examples without replacement.

    If n > len(pool), returns all examples (in random order).
    """
    if n >= len(pool):
        result = list(pool)
        rng.shuffle(result)
        return result
    return rng.sample(pool, n)


def compute_embedding(token_ids: Tensor, logits: Tensor) -> Tensor:
    """Compute mean-pooled, L2-normalized representation from logits.

    Args:
        token_ids: (B, T) token indices (unused in computation but kept for API symmetry).
        logits: (B, T, V) raw model logits.

    Returns:
        (B, V) L2-normalized mean-pooled embedding.
    """
    # Mean-pool over the sequence dimension T → (B, V)
    pooled = logits.mean(dim=1)
    # L2-normalize
    normed = F.normalize(pooled, p=2, dim=-1)
    return normed


def select_similar_examples(
    pool: list[FewShotExample],
    query_embedding: Tensor,
    n: int,
) -> list[FewShotExample]:
    """Select the top-*n* most similar examples by cosine similarity.

    Examples without an embedding are skipped.  The returned list is ordered
    from most to least similar.

    Args:
        pool: candidate examples.
        query_embedding: (V,) query representation.
        n: number of examples to return.
    """
    scored: list[tuple[float, FewShotExample]] = []
    for ex in pool:
        if ex.embedding is None:
            continue
        # Both query_embedding and ex.embedding should already be L2-normalised.
        sim = torch.dot(query_embedding.float(), ex.embedding.float()).item()
        scored.append((sim, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:n]]


def select_diverse_examples(
    pool: list[FewShotExample],
    n: int,
    rng: random_module.Random,
) -> list[FewShotExample]:
    """Greedy maximum-marginal-relevance (MMR) selection for diversity.

    Starts with a random example, then iteratively adds the example that is
    *most different* (smallest max cosine similarity) from the already-selected
    set.  Falls back to random selection when no embeddings are available.

    Args:
        pool: candidate examples.
        n: number of examples to select.
        rng: random number generator for the initial seed.
    """
    if n >= len(pool):
        result = list(pool)
        rng.shuffle(result)
        return result

    # Check if any embeddings are available
    with_emb = [ex for ex in pool if ex.embedding is not None]
    if not with_emb:
        return select_random_examples(pool, n, rng)

    # Use only examples that have embeddings
    candidates = list(with_emb)

    # Seed: pick one random example
    seed_idx = rng.randrange(len(candidates))
    selected = [candidates.pop(seed_idx)]

    while len(selected) < n and candidates:
        best_idx = -1
        best_min_sim = float("inf")

        for i, candidate in enumerate(candidates):
            # Maximum similarity to any already-selected example
            max_sim = max(
                torch.dot(
                    candidate.embedding.float(),
                    sel.embedding.float(),
                ).item()
                for sel in selected
                if sel.embedding is not None
            )
            # Choose the candidate with the smallest max-similarity (most diverse)
            if max_sim < best_min_sim:
                best_min_sim = max_sim
                best_idx = i

        if best_idx == -1:
            break
        selected.append(candidates.pop(best_idx))

    return selected


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_logits(
    logits: Tensor,
    content_free_probs: Tensor,
) -> Tensor:
    """Apply contextual calibration (Zhao et al., 2021).

    Adjusts raw class logits by subtracting the log probability estimated
    from a content-free input (e.g., "N/A").

    Args:
        logits: (n_classes,) raw logits for a classification task.
        content_free_probs: (n_classes,) model output probabilities with
            a content-free input such as "N/A".

    Returns:
        (n_classes,) calibrated logits.
    """
    return logits - torch.log(content_free_probs.float() + 1e-9)


# ---------------------------------------------------------------------------
# ICLEvaluator
# ---------------------------------------------------------------------------


class ICLEvaluator:
    """Runs in-context learning inference with a given model and config.

    Args:
        model: A ``torch.nn.Module`` whose forward returns
            ``(loss, logits, past_key_values)``.
        tokenizer_encode: callable ``str -> list[int]``.
        tokenizer_decode: callable ``list[int] -> str``.
        config: :class:`ICLConfig` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode,
        tokenizer_decode,
        config: ICLConfig,
    ) -> None:
        self.model = model
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.config = config
        self._rng = random_module.Random(42)  # noqa: S311

    def select_examples(
        self,
        pool: list[FewShotExample],
        query: str,
    ) -> list[FewShotExample]:
        """Select examples from *pool* according to the configured strategy.

        For "similarity" and "diverse", falls back to random when the pool
        has no embeddings.
        """
        strategy = self.config.selection_strategy
        n = self.config.n_shots

        if strategy == "random":
            return select_random_examples(pool, n, self._rng)

        if strategy == "similarity":
            with_emb = [ex for ex in pool if ex.embedding is not None]
            if not with_emb:
                return select_random_examples(pool, n, self._rng)
            # Build a dummy query embedding from the mean of existing embeddings
            stack = torch.stack([ex.embedding for ex in with_emb])
            query_emb = F.normalize(stack.mean(dim=0), p=2, dim=-1)
            return select_similar_examples(pool, query_emb, n)

        if strategy == "diverse":
            return select_diverse_examples(pool, n, self._rng)

        # Unknown strategy — fall back to random
        return select_random_examples(pool, n, self._rng)

    def build_prompt(
        self,
        examples: list[FewShotExample],
        query: str,
    ) -> str:
        """Format prompt and truncate to max_context_len characters."""
        prompt = format_few_shot_prompt(examples, query, self.config.template)
        return prompt[: self.config.max_context_len]

    def evaluate_sample(self, prompt: str) -> dict[str, float]:
        """Tokenize *prompt*, greedily generate 1 token, and return statistics.

        Returns:
            ``{"top1_prob": float, "entropy": float}``
        """
        token_ids = self.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # (1, S)

        with torch.no_grad():
            _loss, logits, _pkv = self.model(input_ids)

        # Take logits at the last position → (vocab_size,)
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        top1_prob = probs.max().item()
        # Shannon entropy in nats
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()

        return {"top1_prob": top1_prob, "entropy": entropy}
