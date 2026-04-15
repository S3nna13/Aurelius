"""G-Eval: GPT-based NLG evaluation framework (Liu et al. 2023).

Evaluates Natural Language Generation outputs using an LLM as the judge.
Uses logit probability weighting over score tokens for reliable, calibrated scores.
No external APIs, no HuggingFace -- pure PyTorch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GEvalCriteria:
    """A single evaluation criterion for G-Eval."""

    name: str                   # e.g. "Coherence"
    description: str            # criterion description
    scale: int = 5              # 1-N scale
    weight: float = 1.0         # for composite score


@dataclass
class GEvalResult:
    """Result of a G-Eval evaluation run."""

    criteria_scores: Dict[str, float]        # per-criterion score
    composite_score: float
    raw_logprobs: Dict[str, List[float]]     # per-criterion token logprobs


# ---------------------------------------------------------------------------
# Default criteria
# ---------------------------------------------------------------------------


def make_default_criteria() -> List[GEvalCriteria]:
    """Return standard NLG evaluation criteria: Coherence, Fluency, Consistency, Relevance."""
    return [
        GEvalCriteria(
            name="Coherence",
            description=(
                "The summary should be well-structured and well-organized. "
                "It should not just be a heap of related information, but should "
                "build from sentence to sentence to a coherent body of information "
                "about a topic."
            ),
        ),
        GEvalCriteria(
            name="Fluency",
            description=(
                "The quality of the summary in terms of grammar, spelling, "
                "punctuation, word choice, and sentence structure."
            ),
        ),
        GEvalCriteria(
            name="Consistency",
            description=(
                "The factual alignment between the summary and the summarized source. "
                "A factually consistent summary contains only statements that are "
                "entailed by the source document."
            ),
        ),
        GEvalCriteria(
            name="Relevance",
            description=(
                "Selection of important content from the source. "
                "The summary should include only important information from the "
                "source document. Penalize summaries which contain redundancies and "
                "excess information."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# G-Eval judge
# ---------------------------------------------------------------------------


class GEvalJudge:
    """
    LLM-based evaluator following the G-Eval protocol.

    Uses logit probability weighting for more reliable scores than greedy
    decoding or parsing a sampled integer from generated text.

    Args:
        model:             forward(input_ids) -> (loss, logits, pkv) or logits
                           logits shape: (batch, seq_len, vocab_size)
        tokenizer_encode:  text -> list of token ids
        tokenizer_decode:  single token id -> text
        criteria:          list of GEvalCriteria to evaluate
        score_tokens:      vocab tokens representing each score level;
                           defaults to ["1", "2", "3", "4", "5"]
        device:            torch device string
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable[[str], List[int]],
        tokenizer_decode: Callable[[int], str],
        criteria: List[GEvalCriteria],
        score_tokens: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.criteria = criteria
        self.score_tokens = score_tokens if score_tokens is not None else ["1", "2", "3", "4", "5"]
        self.device = device

        # Pre-compute score token ids once (take first token id per score string)
        self._score_token_ids: List[int] = []
        for tok in self.score_tokens:
            ids = self.tokenizer_encode(tok)
            self._score_token_ids.append(ids[0] if ids else 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        task_description: str,
        document: str,
        summary: str,
        criterion: GEvalCriteria,
    ) -> str:
        """Build the G-Eval prompt with chain-of-thought instructions."""
        scale_desc = " | ".join(
            f"{i} = {'poor' if i == 1 else 'excellent' if i == criterion.scale else 'fair'}"
            for i in range(1, criterion.scale + 1)
        )
        prompt = (
            f"{task_description}\n\n"
            f"Evaluation Criteria:\n"
            f"{criterion.name} ({criterion.description})\n\n"
            f"Evaluation Steps:\n"
            f"1. Read the source document carefully.\n"
            f"2. Read the summary and compare it to the source.\n"
            f"3. Assign a score for {criterion.name} on a scale of "
            f"1 to {criterion.scale}, where {scale_desc}.\n\n"
            f"Source Document:\n{document}\n\n"
            f"Summary:\n{summary}\n\n"
            f"Score ({criterion.name}, 1-{criterion.scale}):"
        )
        return prompt

    def _get_score_logprobs(
        self,
        prompt_ids: Tensor,
        criterion: GEvalCriteria,
    ) -> List[float]:
        """Run model on prompt_ids, extract logprobs at score token positions.

        Returns a list of log-probabilities, one per score token (length == scale).
        """
        self.model.eval()
        with torch.no_grad():
            input_ids = prompt_ids.unsqueeze(0).to(self.device)  # (1, T)
            output = self.model(input_ids)

            # Support (loss, logits, pkv) or bare logits tensor
            if isinstance(output, (tuple, list)):
                logits = output[1]
            else:
                logits = output

            # logits: (1, T, vocab_size) -- we want the last position
            last_logits = logits[0, -1, :]  # (vocab_size,)

        # Gather logits for the score tokens relevant to this criterion's scale
        num_scores = criterion.scale
        score_token_ids = self._score_token_ids[:num_scores]

        score_logits = torch.stack(
            [last_logits[tid] for tid in score_token_ids]
        )  # (num_scores,)

        # Convert to log-probabilities via log_softmax over the score subset
        log_probs = torch.log_softmax(score_logits, dim=0)
        return log_probs.tolist()

    def _weighted_score(self, logprobs: List[float], scale: int) -> float:
        """Convert logprobs to weighted score using softmax + expectation.

        score = sum_{i=1}^{scale}  i * P(score_i)

        The log_probs already represent a log-probability distribution over
        the score tokens; exp converts them back to probabilities.
        """
        probs_tensor = torch.exp(torch.tensor(logprobs, dtype=torch.float32))
        # Renormalize for numerical safety
        probs_tensor = probs_tensor / probs_tensor.sum()

        scores = torch.arange(1, scale + 1, dtype=torch.float32)
        weighted = (scores * probs_tensor).sum().item()
        return float(weighted)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        document: str,
        hypothesis: str,
        task_description: str = "Evaluate the quality of the following summary.",
    ) -> GEvalResult:
        """Evaluate hypothesis against document on all criteria."""
        criteria_scores: Dict[str, float] = {}
        raw_logprobs: Dict[str, List[float]] = {}

        for criterion in self.criteria:
            prompt = self._build_prompt(task_description, document, hypothesis, criterion)
            prompt_ids = torch.tensor(
                self.tokenizer_encode(prompt), dtype=torch.long
            )

            logprobs = self._get_score_logprobs(prompt_ids, criterion)
            score = self._weighted_score(logprobs, criterion.scale)

            criteria_scores[criterion.name] = score
            raw_logprobs[criterion.name] = logprobs

        # Weighted composite score
        total_weight = sum(c.weight for c in self.criteria)
        composite = sum(
            criteria_scores[c.name] * c.weight for c in self.criteria
        ) / (total_weight if total_weight > 0 else 1.0)

        return GEvalResult(
            criteria_scores=criteria_scores,
            composite_score=composite,
            raw_logprobs=raw_logprobs,
        )

    def batch_evaluate(
        self,
        documents: List[str],
        hypotheses: List[str],
        task_description: str = "Evaluate the quality of the following summary.",
    ) -> List[GEvalResult]:
        """Evaluate multiple document-hypothesis pairs."""
        if len(documents) != len(hypotheses):
            raise ValueError(
                f"documents and hypotheses must have the same length; "
                f"got {len(documents)} and {len(hypotheses)}"
            )
        return [
            self.evaluate(doc, hyp, task_description)
            for doc, hyp in zip(documents, hypotheses)
        ]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def summarization_eval(
    model: nn.Module,
    tokenizer_encode: Callable,
    tokenizer_decode: Callable,
    document: str,
    summary: str,
) -> GEvalResult:
    """Convenience: evaluate summary with default summarization criteria."""
    criteria = make_default_criteria()
    judge = GEvalJudge(
        model=model,
        tokenizer_encode=tokenizer_encode,
        tokenizer_decode=tokenizer_decode,
        criteria=criteria,
    )
    return judge.evaluate(
        document=document,
        hypothesis=summary,
        task_description="Evaluate the quality of the following summary.",
    )
