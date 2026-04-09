"""Open-ended generation evaluation: ROUGE-L, BERTScore proxy, n-gram diversity, and length stats."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GenerationEvalConfig:
    """Configuration for open-ended generation evaluation."""
    max_new_tokens: int = 64
    temperature: float = 1.0
    do_sample: bool = False      # False = greedy
    repetition_penalty: float = 1.0  # 1.0 = disabled
    n_gram_n: int = 4            # for n-gram diversity


# ---------------------------------------------------------------------------
# Repetition penalty
# ---------------------------------------------------------------------------

def apply_repetition_penalty(
    logits: Tensor,
    generated_ids: Tensor,
    penalty: float,
) -> Tensor:
    """Apply repetition penalty to logits for already-generated tokens.

    For each token in generated_ids:
        - if logit > 0: divide by penalty
        - if logit < 0: multiply by penalty

    This is the standard CTRL repetition penalty.

    Args:
        logits: (B, vocab_size) current step logits.
        generated_ids: (B, T) previously generated token ids.
        penalty: Scalar penalty > 1.0 discourages repetition.

    Returns:
        Modified logits of the same shape.
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    B = logits.shape[0]

    for b in range(B):
        for token_id in generated_ids[b]:
            tid = int(token_id.item())
            score = logits[b, tid]
            if score > 0:
                logits[b, tid] = score / penalty
            else:
                logits[b, tid] = score * penalty

    return logits


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
) -> Tensor:
    """Autoregressively generate max_new_tokens tokens using greedy decoding.

    Args:
        model: AureliusTransformer. Forward signature: _, logits, _ = model(input_ids)
        input_ids: (1, S) prompt token ids.
        max_new_tokens: Number of tokens to generate.
        repetition_penalty: > 1.0 penalizes previously seen tokens.

    Returns:
        (1, max_new_tokens) tensor of generated token ids.
    """
    model.train(False)
    device = input_ids.device
    generated: list[int] = []
    current_ids = input_ids  # (1, S)

    for _ in range(max_new_tokens):
        _, logits, _ = model(current_ids)
        # logits: (1, S, vocab_size) — take last position
        next_logits = logits[:, -1, :]  # (1, vocab_size)

        if repetition_penalty != 1.0 and generated:
            gen_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            next_logits = apply_repetition_penalty(next_logits, gen_tensor, repetition_penalty)

        next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
        generated.append(int(next_token.item()))
        current_ids = torch.cat([current_ids, next_token], dim=1)

    return torch.tensor([generated], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between hypothesis and reference.

    Uses LCS-based precision/recall on whitespace-split word tokens:
        precision = LCS / len(hypothesis_tokens)
        recall    = LCS / len(reference_tokens)
        F1        = 2 * P * R / (P + R)

    Args:
        hypothesis: Generated text.
        reference: Reference text.

    Returns:
        Float in [0, 1].
    """
    hyp_tokens = hypothesis.split()
    ref_tokens = reference.split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # LCS via DP (two-row)
    m, n = len(hyp_tokens), len(ref_tokens)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)

    lcs = prev[n]

    if lcs == 0:
        return 0.0

    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Distinct-N
# ---------------------------------------------------------------------------

def compute_distinct_n(token_ids: Tensor, n: int) -> float:
    """Compute Distinct-N: ratio of unique n-grams to total n-grams.

    Args:
        token_ids: Flat or batched token id tensor.
        n: N-gram order.

    Returns:
        Float in [0, 1]. Returns 1.0 if all n-grams are unique.
    """
    ids = token_ids.reshape(-1).tolist()

    if len(ids) < n:
        return 1.0

    ngrams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))

    if total == 0:
        return 1.0

    return unique / total


# ---------------------------------------------------------------------------
# Length stats
# ---------------------------------------------------------------------------

def compute_length_stats(generated_texts: list[str]) -> dict:
    """Compute word-count length statistics over a list of generated texts.

    Args:
        generated_texts: List of decoded strings.

    Returns:
        Dict with keys: mean_len (float), min_len (int), max_len (int), std_len (float).
    """
    if not generated_texts:
        return {"mean_len": 0.0, "min_len": 0, "max_len": 0, "std_len": 0.0}

    lengths = [len(text.split()) for text in generated_texts]
    total = len(lengths)
    mean = sum(lengths) / total
    variance = sum((length - mean) ** 2 for length in lengths) / total
    std = math.sqrt(variance)

    return {
        "mean_len": mean,
        "min_len": min(lengths),
        "max_len": max(lengths),
        "std_len": std,
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GenerationEvaluator:
    """Evaluate open-ended text generation quality.

    Args:
        model: AureliusTransformer.
        encode_fn: Callable that maps str -> list[int].
        decode_fn: Callable that maps list[int] -> str.
        config: GenerationEvalConfig.
    """

    def __init__(
        self,
        model,
        encode_fn: Callable,
        decode_fn: Callable,
        config: GenerationEvalConfig,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.config = config

    def evaluate_sample(self, prompt: str, reference: str) -> dict:
        """Generate a completion for prompt and score it against reference.

        Returns:
            Dict with keys: rouge_l (float), distinct_n (float), length (int).
        """
        device = next(self.model.parameters()).device
        input_ids = torch.tensor(
            [self.encode_fn(prompt)], dtype=torch.long, device=device
        )

        generated_ids = greedy_generate(
            self.model,
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty,
        )  # (1, max_new_tokens)

        hypothesis = self.decode_fn(generated_ids[0].tolist())

        rouge = compute_rouge_l(hypothesis, reference)
        distinct = compute_distinct_n(generated_ids[0], self.config.n_gram_n)
        length = len(hypothesis.split())

        return {
            "rouge_l": rouge,
            "distinct_n": distinct,
            "length": length,
        }

    def evaluate_batch(self, prompts: list[str], references: list[str]) -> dict:
        """Evaluate a batch of prompts and return aggregated metrics.

        Returns:
            Dict with keys: mean_rouge_l, mean_distinct_n, mean_len, min_len, max_len, std_len.
        """
        results = [
            self.evaluate_sample(p, r)
            for p, r in zip(prompts, references)
        ]

        n = len(results)
        if n == 0:
            return {
                "mean_rouge_l": 0.0,
                "mean_distinct_n": 0.0,
                "mean_len": 0.0,
                "min_len": 0,
                "max_len": 0,
                "std_len": 0.0,
            }

        mean_rouge_l = sum(r["rouge_l"] for r in results) / n
        mean_distinct_n = sum(r["distinct_n"] for r in results) / n

        # Build dummy texts to reuse compute_length_stats
        dummy_texts = [" ".join(["x"] * max(r["length"], 0)) for r in results]
        length_stats = compute_length_stats(dummy_texts)

        return {
            "mean_rouge_l": mean_rouge_l,
            "mean_distinct_n": mean_distinct_n,
            **length_stats,
        }
