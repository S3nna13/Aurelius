"""Inference-time compute scaling via repeated sampling and verification.

Strategies:
- BEST_OF_N: Score N samples with verifier, return highest-scoring
- MAJORITY_VOTE: Return most common answer (string matching)
- WEIGHTED_MAJORITY: Weighted vote using verifier scores
- BEAM_RERANK: Generate N beams, rerank with verifier

Reference: Snell et al. 2024 "Scaling LLM Test-Time Compute Optimally"
(arXiv:2408.03314)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class SelectionStrategy(Enum):
    BEST_OF_N = "best_of_n"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_MAJORITY = "weighted_majority"
    BEAM_RERANK = "beam_rerank"


@dataclass
class ComputeOptimalConfig:
    n_samples: int = 8  # number of candidates to generate
    strategy: SelectionStrategy = SelectionStrategy.BEST_OF_N
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    eos_token_id: int | None = None


@dataclass
class ComputeOptimalResult:
    """Result of compute-optimal inference."""

    selected_ids: torch.Tensor  # (S_gen,) — selected response token ids
    selected_score: float  # verifier score for selected response
    all_scores: list[float]  # scores for all N candidates
    n_samples_generated: int  # actual number of candidates tried
    strategy: SelectionStrategy

    @property
    def score_mean(self) -> float:
        return sum(self.all_scores) / len(self.all_scores) if self.all_scores else 0.0

    @property
    def score_std(self) -> float:
        mean = self.score_mean
        return (sum((s - mean) ** 2 for s in self.all_scores) / max(1, len(self.all_scores))) ** 0.5


def score_with_model(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,  # (S_prompt,)
    response_ids: torch.Tensor,  # (S_resp,)
) -> float:
    """Score a response using the model's own log-probability as verifier.

    Uses mean log-prob of response tokens given prompt (self-certainty scoring).
    Higher = more confident = better.
    """
    if response_ids.numel() == 0:
        return float("-inf")

    # Build full sequence: prompt + response, shape (1, S_prompt + S_resp)
    prompt_1d = prompt_ids.view(-1)
    response_1d = response_ids.view(-1)
    full_ids = torch.cat([prompt_1d, response_1d], dim=0).unsqueeze(0)  # (1, S)

    prompt_len = prompt_1d.shape[0]
    resp_len = response_1d.shape[0]

    with torch.no_grad():
        _, logits, _ = model(full_ids)  # (1, S, vocab)

    # Logits at [prompt_len-1 : prompt_len + resp_len - 1] predict response tokens
    gen_logits = logits[0, prompt_len - 1 : prompt_len + resp_len - 1, :]  # (resp_len, vocab)

    log_probs = F.log_softmax(gen_logits, dim=-1)  # (resp_len, vocab)
    token_log_probs = log_probs[torch.arange(resp_len), response_1d]  # (resp_len,)
    return token_log_probs.mean().item()


def generate_n_samples(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,  # (1, S_prompt) or (S_prompt,)
    cfg: ComputeOptimalConfig,
) -> list[torch.Tensor]:
    """Generate n_samples candidate responses.

    Uses model.generate() (which exists on AureliusTransformer).
    Each call generates with cfg.temperature and cfg.top_p.
    Returns list of n_samples tensors, each (S_resp,) 1D.
    """
    # Normalise to (1, S_prompt)
    if prompt_ids.dim() == 1:
        input_ids = prompt_ids.unsqueeze(0)
    else:
        input_ids = prompt_ids

    prompt_len = input_ids.shape[1]
    samples: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(cfg.n_samples):
            completion = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=cfg.eos_token_id,
            )  # (1, prompt_len + generated_len)
            response = completion[0, prompt_len:]  # (generated_len,)
            samples.append(response)

    return samples


def extract_answer(response_ids: torch.Tensor, tokenizer_decode_fn=None) -> str:
    """Extract answer string from response token ids.

    If decode_fn provided, decode to string. Otherwise use str(ids.tolist()).
    Used for majority vote string matching.
    """
    if tokenizer_decode_fn is not None:
        return tokenizer_decode_fn(response_ids)
    return str(response_ids.tolist())


def compute_optimal_generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,  # (1, S) or (S,)
    cfg: ComputeOptimalConfig,
    verifier=None,  # callable(prompt_ids, response_ids) -> float, or None
    decode_fn=None,  # callable(token_ids) -> str, for majority vote
) -> ComputeOptimalResult:
    """Generate multiple candidates and select the best using the strategy.

    If verifier is None, uses score_with_model (self-certainty).

    Strategies:
    - BEST_OF_N: return argmax(scores)
    - MAJORITY_VOTE: return most frequent decoded answer
    - WEIGHTED_MAJORITY: each candidate votes with weight = exp(score), pick winner
    - BEAM_RERANK: generate with beam search params, rerank with verifier
    """
    # Normalise prompt to 1D for scoring
    prompt_1d = prompt_ids.view(-1)

    # Effective scoring function
    def _score(response_ids: torch.Tensor) -> float:
        if verifier is not None:
            return float(verifier(prompt_1d, response_ids))
        return score_with_model(model, prompt_1d, response_ids)

    # Generate candidates
    samples = generate_n_samples(model, prompt_ids, cfg)

    # Score all candidates
    scores = [_score(resp) for resp in samples]

    strategy = cfg.strategy

    if strategy == SelectionStrategy.BEST_OF_N or strategy == SelectionStrategy.BEAM_RERANK:
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        selected = samples[best_idx]
        selected_score = scores[best_idx]

    elif strategy == SelectionStrategy.MAJORITY_VOTE:
        # Decode each candidate, pick the most common answer string
        decoded = [extract_answer(r, decode_fn) for r in samples]
        counts = Counter(decoded)
        winning_answer = counts.most_common(1)[0][0]
        # Among candidates with the winning answer, pick highest score
        winning_indices = [i for i, d in enumerate(decoded) if d == winning_answer]
        best_idx = max(winning_indices, key=lambda i: scores[i])
        selected = samples[best_idx]
        selected_score = scores[best_idx]

    elif strategy == SelectionStrategy.WEIGHTED_MAJORITY:
        # Weighted vote: weight = exp(score), accumulate per decoded answer
        decoded = [extract_answer(r, decode_fn) for r in samples]
        weights: dict[str, float] = {}
        for d, s in zip(decoded, scores):
            weights[d] = weights.get(d, 0.0) + math.exp(s)
        winning_answer = max(weights, key=lambda k: weights[k])
        winning_indices = [i for i, d in enumerate(decoded) if d == winning_answer]
        best_idx = max(winning_indices, key=lambda i: scores[i])
        selected = samples[best_idx]
        selected_score = scores[best_idx]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return ComputeOptimalResult(
        selected_ids=selected,
        selected_score=selected_score,
        all_scores=scores,
        n_samples_generated=len(samples),
        strategy=strategy,
    )
