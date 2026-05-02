"""Chain-of-thought verification and step-level process reward for reasoning models."""

import string
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class VerifierConfig:
    max_steps: int = 10
    step_delimiter: str = "\n"
    answer_marker: str = "Answer:"
    confidence_threshold: float = 0.5


def parse_chain_of_thought(text: str, config: VerifierConfig) -> tuple[list[str], str]:
    """Split text into reasoning steps and extract final answer.

    Returns (steps, answer) where steps is a list of non-empty strings
    before the answer marker, and answer is the text after the last
    occurrence of answer_marker (empty string if not found).
    """
    marker = config.answer_marker
    last_idx = text.rfind(marker)

    if last_idx == -1:
        raw_steps = text.split(config.step_delimiter)
        steps = [s for s in raw_steps if s.strip()]
        steps = steps[: config.max_steps]
        return steps, ""

    chain_text = text[:last_idx]
    answer = text[last_idx + len(marker) :].strip()

    raw_steps = chain_text.split(config.step_delimiter)
    steps = [s for s in raw_steps if s.strip()]
    steps = steps[: config.max_steps]

    return steps, answer


def verify_answer(predicted: str, gold: str, exact_match: bool = False) -> float:
    """Score predicted answer against gold answer.

    If exact_match: 1.0 if strings match exactly (after strip), else 0.0.
    Otherwise: word-overlap F1 (normalized partial match).

    Returns float in [0, 1].
    """
    if exact_match:
        return 1.0 if predicted.strip() == gold.strip() else 0.0

    def _normalize(text: str) -> list[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    pred_tokens = _normalize(predicted)
    gold_tokens = _normalize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


class StepVerifier(nn.Module):
    """Scores each reasoning step for quality."""

    def __init__(self, d_model: int, vocab_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model, nhead=2, dim_feedforward=d_model * 2, batch_first=True
        )
        encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model, nhead=2, dim_feedforward=d_model * 2, batch_first=True
        )
        self.encoder = nn.ModuleList([encoder_layer_1, encoder_layer_2])
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Score reasoning steps.

        Args:
            token_ids: (B, T) tokenized step text

        Returns:
            (B,) step quality scores in (-inf, inf); apply sigmoid for probability
        """
        x = self.embedding(token_ids)  # (B, T, d_model)
        for layer in self.encoder:
            x = layer(x)  # (B, T, d_model)
        pooled = x.mean(dim=1)  # (B, d_model)
        scores = self.scorer(pooled).squeeze(-1)  # (B,)
        return scores


class ChainOfThoughtVerifier:
    """End-to-end CoT verification pipeline."""

    def __init__(
        self,
        step_verifier: StepVerifier,
        tokenize_fn: Callable[[str], list[int]],
        config: VerifierConfig,
    ):
        self.step_verifier = step_verifier
        self.tokenize_fn = tokenize_fn
        self.config = config

    def score_steps(self, steps: list[str]) -> list[float]:
        """Tokenize each step, run step_verifier, return list of sigmoid scores."""
        if not steps:
            return []

        scores: list[float] = []
        self.step_verifier.eval()
        with torch.no_grad():
            for step in steps:
                token_ids = self.tokenize_fn(step)
                if not token_ids:
                    scores.append(0.5)
                    continue
                ids_tensor = torch.tensor([token_ids], dtype=torch.long)  # (1, T)
                raw = self.step_verifier(ids_tensor)  # (1,)
                prob = torch.sigmoid(raw).item()
                scores.append(float(prob))

        return scores

    def verify_chain(self, text: str, gold_answer: str) -> dict[str, float]:
        """Parse CoT, score each step, verify final answer.

        Returns dict with answer_score, mean_step_score, min_step_score, n_steps.
        """
        steps, predicted_answer = parse_chain_of_thought(text, self.config)
        answer_score = verify_answer(predicted_answer, gold_answer)
        step_scores = self.score_steps(steps)

        if step_scores:
            mean_step_score = float(sum(step_scores) / len(step_scores))
            min_step_score = float(min(step_scores))
        else:
            mean_step_score = 0.0
            min_step_score = 0.0

        return {
            "answer_score": float(answer_score),
            "mean_step_score": mean_step_score,
            "min_step_score": min_step_score,
            "n_steps": float(len(steps)),
        }

    def filter_by_quality(self, candidates: list[str], gold: str, top_k: int = 1) -> list[str]:
        """Score all candidates' CoT chains, return top_k by mean_step_score * answer_score."""
        if not candidates:
            return []

        scored = []
        for candidate in candidates:
            result = self.verify_chain(candidate, gold)
            combined = result["mean_step_score"] * result["answer_score"]
            scored.append((combined, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


class BestOfNVerifier:
    """Generate N candidates, pick best by verifier score."""

    def __init__(self, verifier: ChainOfThoughtVerifier, n: int = 8):
        self.verifier = verifier
        self.n = n

    def select_best(
        self, candidates: list[str], gold_answer: str | None = None
    ) -> tuple[str, dict]:
        """Score all candidates, return (best_candidate, scores_dict).

        If gold_answer provided: prefer by answer_score; else by mean_step_score.
        """
        if not candidates:
            raise ValueError("candidates list is empty")

        scores_dict: dict[str, dict] = {}
        for i, candidate in enumerate(candidates):
            gold = gold_answer if gold_answer is not None else ""
            result = self.verifier.verify_chain(candidate, gold)
            scores_dict[str(i)] = result

        if gold_answer is not None:
            key = "answer_score"
        else:
            key = "mean_step_score"

        best_idx = max(range(len(candidates)), key=lambda i: scores_dict[str(i)][key])
        return candidates[best_idx], scores_dict


def compute_process_reward(
    steps: list[str], final_reward: float, discount: float = 0.9
) -> list[float]:
    """Assign per-step process rewards using final reward with discount.

    Earlier steps: final_reward * discount^(n_steps - i - 1) for step i.
    Last step reward equals final_reward.

    Returns list of per-step rewards.
    """
    n = len(steps)
    rewards = []
    for i in range(n):
        exponent = n - i - 1
        rewards.append(final_reward * (discount**exponent))
    return rewards
