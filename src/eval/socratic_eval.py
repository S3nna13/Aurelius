"""Socratic self-evaluation for AureliusTransformer.

The model critiques its own reasoning by answering targeted yes/no questions
about response quality. Scores are derived from logit probabilities over the
yes/no tokens at the first generated position.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SocraticConfig:
    """Configuration for Socratic self-evaluation."""
    n_critique_questions: int = 5   # questions per response
    max_new_tokens: int = 20        # tokens to generate per answer
    score_from_logits: bool = True  # use logit probabilities for scoring
    temperature: float = 0.7


# ---------------------------------------------------------------------------
# Critique question templates and polarities
# ---------------------------------------------------------------------------

CRITIQUE_QUESTIONS: list[str] = [
    "Is this response logically consistent? Answer yes or no:",
    "Does this response answer the question directly? Answer yes or no:",
    "Are there factual errors in this response? Answer yes or no:",
    "Is the reasoning clear and well-structured? Answer yes or no:",
    "Could this response be misunderstood? Answer yes or no:",
]

# True = positive question (yes = good quality), False = negative (no = good quality)
QUESTION_POLARITIES: list[bool] = [True, True, False, True, False]

# Token IDs for a byte-level tokenizer: ord('y') = 121, ord('n') = 110
YES_TOKEN_ID: int = 121  # ord('y')
NO_TOKEN_ID: int = 110   # ord('n')


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CritiqueResult:
    """Result of a single critique question."""
    question: str
    answer: str           # generated answer text
    yes_probability: float
    no_probability: float
    score: float          # 0-1 quality score derived from yes/no probs + polarity


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_from_logits(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    prompt: str,
    yes_token_id: int,
    no_token_id: int,
) -> tuple[float, float]:
    """Get probability of yes/no tokens as the next token after prompt.

    Runs the model on the encoded prompt, extracts logits at the last position,
    applies softmax over the full vocabulary, and returns (p_yes, p_no).

    Args:
        model: AureliusTransformer.
        encode_fn: Maps string to list of token ids.
        prompt: Text prompt to condition on.
        yes_token_id: Vocabulary index for the "yes" token.
        no_token_id: Vocabulary index for the "no" token.

    Returns:
        (p_yes, p_no) - both in (0, 1).
    """
    model.train(False)
    ids = encode_fn(prompt)
    if not ids:
        return 0.5, 0.5

    input_ids = torch.tensor([ids], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    _, logits, _ = model(input_ids)  # (1, T, V)
    last_logits = logits[0, -1, :]   # (V,)
    probs = F.softmax(last_logits.float(), dim=-1)

    p_yes = probs[yes_token_id].item()
    p_no = probs[no_token_id].item()
    return float(p_yes), float(p_no)


def compute_critique_score(
    yes_prob: float,
    no_prob: float,
    positive_question: bool,
) -> float:
    """Convert yes/no probabilities to a 0-1 quality score.

    For positive questions (e.g. "Is it consistent?"):
        score = yes_prob / (yes_prob + no_prob)
    For negative questions (e.g. "Are there errors?"):
        score = no_prob / (yes_prob + no_prob)

    Handles the degenerate case where both probs are zero by returning 0.5.

    Args:
        yes_prob: Probability assigned to "yes" token.
        no_prob: Probability assigned to "no" token.
        positive_question: If True, yes is the good outcome.

    Returns:
        Score in [0, 1].
    """
    total = yes_prob + no_prob
    if total <= 0.0:
        return 0.5
    if positive_question:
        return yes_prob / total
    else:
        return no_prob / total


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Greedy decode up to max_new_tokens additional tokens.

    Args:
        model: AureliusTransformer.
        input_ids: (1, T) tensor of prompt token ids.
        max_new_tokens: Number of tokens to generate.

    Returns:
        (max_new_tokens,) tensor of generated token ids.
    """
    model.train(False)
    device = next(model.parameters()).device
    current_ids = input_ids.to(device)
    generated: list[int] = []

    cfg_obj = getattr(model, "config", None)
    max_seq_len = getattr(cfg_obj, "max_seq_len", 512)

    for _ in range(max_new_tokens):
        # Truncate to max_seq_len if needed
        if current_ids.shape[1] > max_seq_len:
            current_ids = current_ids[:, -max_seq_len:]

        _, logits, _ = model(current_ids)   # (1, T, V)
        next_token = logits[0, -1, :].argmax(dim=-1)  # greedy
        next_token_id = int(next_token.item())
        generated.append(next_token_id)
        current_ids = torch.cat(
            [current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
        )

    return torch.tensor(generated, dtype=torch.long)


@torch.no_grad()
def critique_response(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    decode_fn: Callable[[list[int]], str],
    question: str,
    response: str,
    critique_question: str,
    question_idx: int,
    cfg: SocraticConfig,
) -> CritiqueResult:
    """Ask a single critique question about the response.

    Constructs a prompt of the form:
        "Question: {question}\\nResponse: {response}\\n{critique_question}"
    Gets yes/no probabilities and generates an answer text.

    Args:
        model: AureliusTransformer.
        encode_fn: Tokenizer encode function.
        decode_fn: Tokenizer decode function.
        question: The original question asked to the model.
        response: The model's response to critique.
        critique_question: The specific critique question to ask.
        question_idx: Index into QUESTION_POLARITIES.
        cfg: SocraticConfig controlling generation behaviour.

    Returns:
        CritiqueResult with all fields populated.
    """
    prompt = f"Question: {question}\nResponse: {response}\n{critique_question}"

    # Get yes/no probabilities from logits
    p_yes, p_no = score_from_logits(
        model, encode_fn, prompt, YES_TOKEN_ID, NO_TOKEN_ID
    )

    # Generate answer text
    ids = encode_fn(prompt)
    if ids:
        input_ids = torch.tensor([ids], dtype=torch.long)
        generated_ids = greedy_decode(model, input_ids, cfg.max_new_tokens)
        answer_text = decode_fn(generated_ids.tolist())
    else:
        answer_text = ""

    # Compute score based on polarity
    positive = QUESTION_POLARITIES[question_idx]
    score = compute_critique_score(p_yes, p_no, positive)

    return CritiqueResult(
        question=critique_question,
        answer=answer_text,
        yes_probability=p_yes,
        no_probability=p_no,
        score=score,
    )


def socratic_evaluate(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    decode_fn: Callable[[list[int]], str],
    question: str,
    response: str,
    cfg: SocraticConfig,
) -> dict[str, float | list]:
    """Run all CRITIQUE_QUESTIONS and aggregate scores.

    Args:
        model: AureliusTransformer.
        encode_fn: Tokenizer encode function.
        decode_fn: Tokenizer decode function.
        question: Original question.
        response: Model's response to evaluate.
        cfg: SocraticConfig.

    Returns:
        dict with keys:
            "overall_score": mean of all critique scores (float in [0, 1])
            "critique_results": list of CritiqueResult
            "consistency_score": score from question 0
            "directness_score": score from question 1
            "error_score": score from question 2 (1 = no errors)
            "clarity_score": score from question 3
    """
    n = min(cfg.n_critique_questions, len(CRITIQUE_QUESTIONS))
    results: list[CritiqueResult] = []

    for idx in range(n):
        result = critique_response(
            model=model,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            question=question,
            response=response,
            critique_question=CRITIQUE_QUESTIONS[idx],
            question_idx=idx,
            cfg=cfg,
        )
        results.append(result)

    scores = [r.score for r in results]
    overall = float(sum(scores) / len(scores)) if scores else 0.0

    out: dict = {
        "overall_score": overall,
        "critique_results": results,
        "consistency_score": results[0].score if len(results) > 0 else 0.0,
        "directness_score": results[1].score if len(results) > 1 else 0.0,
        "error_score": results[2].score if len(results) > 2 else 0.0,
        "clarity_score": results[3].score if len(results) > 3 else 0.0,
    }
    return out


# ---------------------------------------------------------------------------
# SocraticEvaluator class
# ---------------------------------------------------------------------------

class SocraticEvaluator:
    """Batch Socratic evaluation of model responses."""

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[list[int]], str],
        cfg: SocraticConfig,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.cfg = cfg

    def evaluate(self, question: str, response: str) -> dict:
        """Run Socratic evaluation for a single (question, response) pair."""
        return socratic_evaluate(
            model=self.model,
            encode_fn=self.encode_fn,
            decode_fn=self.decode_fn,
            question=question,
            response=response,
            cfg=self.cfg,
        )

    def evaluate_batch(
        self,
        qa_pairs: list[tuple[str, str]],
    ) -> dict[str, float]:
        """Evaluate all (question, response) pairs.

        Args:
            qa_pairs: List of (question, response) tuples.

        Returns:
            dict with keys:
                "mean_overall_score": mean overall score across all pairs
                "mean_consistency": mean consistency score
                "mean_clarity": mean clarity score
                "n_evaluated": number of pairs evaluated
        """
        if not qa_pairs:
            return {
                "mean_overall_score": 0.0,
                "mean_consistency": 0.0,
                "mean_clarity": 0.0,
                "n_evaluated": 0,
            }

        overall_scores: list[float] = []
        consistency_scores: list[float] = []
        clarity_scores: list[float] = []

        for q, r in qa_pairs:
            result = self.evaluate(q, r)
            overall_scores.append(float(result["overall_score"]))
            consistency_scores.append(float(result["consistency_score"]))
            clarity_scores.append(float(result["clarity_score"]))

        n = len(qa_pairs)
        return {
            "mean_overall_score": sum(overall_scores) / n,
            "mean_consistency": sum(consistency_scores) / n,
            "mean_clarity": sum(clarity_scores) / n,
            "n_evaluated": float(n),
        }

    def rank_responses(
        self,
        question: str,
        responses: list[str],
    ) -> list[int]:
        """Return indices of responses sorted by overall_score descending (best first).

        Args:
            question: The question that was answered.
            responses: List of candidate responses to rank.

        Returns:
            List of indices into responses, sorted best-first.
        """
        scored: list[tuple[int, float]] = []
        for idx, response in enumerate(responses):
            result = self.evaluate(question, response)
            scored.append((idx, float(result["overall_score"])))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored]
