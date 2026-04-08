"""Written evaluations: model-generated and model-scored multiple-choice questions.

Implements the framework from Perez et al. 2022 "Red Teaming Language Models with
Language Models" — use the model to generate evaluation examples, then score
responses automatically for diverse test sets without human annotation.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    question_id: str
    question: str
    choices: list[str]        # ["A) ...", "B) ...", "C) ...", "D) ..."]
    correct_choice: int       # 0-indexed
    category: str             # e.g. "reasoning", "factual", "math"
    difficulty: float         # 0.0=easy, 1.0=hard (estimated)


@dataclass
class EvalResult:
    question_id: str
    predicted_choice: int     # 0-indexed, -1 if unparseable
    correct: bool
    confidence: float         # max softmax prob over choice tokens
    raw_output: str


# ---------------------------------------------------------------------------
# Regex for parsing model-generated questions
# ---------------------------------------------------------------------------

_QUESTION_RE = re.compile(
    r"Q:\s*(.+?)\nA\)\s*(.+?)\nB\)\s*(.+?)\nC\)\s*(.+?)\nD\)\s*(.+?)\nAnswer:\s*([ABCD])",
    re.DOTALL,
)

_ANSWER_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

# Token ids for choice letters: A=65, B=66, C=67, D=68 (single-byte ASCII)
_CHOICE_TOKEN_IDS = [65, 66, 67, 68]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class WrittenEvalGenerator:
    """Generate evaluation questions by prompting the model with templates.

    Creates multiple-choice questions in a given category.
    Uses structured prompting to get parseable questions.

    Args:
        model: AureliusTransformer (called as model(input_ids) -> (loss, logits, pkv))
        tokenizer_encode: callable str -> list[int]
        tokenizer_decode: callable list[int] -> str (optional, for debug)
        max_seq_len: int
    """

    def __init__(
        self,
        model: object,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str] | None = None,
        max_seq_len: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.max_seq_len = max_seq_len

    def generate_questions(
        self,
        category: str,
        n: int = 10,
        seed: int = 42,
    ) -> list[EvalQuestion]:
        """Generate n questions in the given category.

        Uses a template prompt, does greedy generation, parses the output.
        Returns whatever valid questions were parsed (may be < n if parsing fails).
        """
        torch.manual_seed(seed)
        questions: list[EvalQuestion] = []

        for idx in range(n):
            prompt = self._build_generation_prompt(category, idx)
            token_ids = self.tokenizer_encode(prompt)

            # Truncate to leave room for generation
            max_prompt = max(1, self.max_seq_len - 128)
            token_ids = token_ids[:max_prompt]

            input_tensor = torch.tensor([token_ids], dtype=torch.long)

            # Generate greedily up to 128 new tokens
            generated: list[int] = []
            with torch.no_grad():
                for _ in range(128):
                    seq = token_ids + generated
                    seq = seq[-(self.max_seq_len):]
                    t = torch.tensor([seq], dtype=torch.long)
                    _loss, logits, _pkv = self.model(t)
                    next_token = int(logits[0, -1, :].argmax().item())
                    generated.append(next_token)
                    # Simple EOS heuristic: stop at newline after Answer line
                    # (byte 10 = '\n')
                    decoded_so_far = (
                        self.tokenizer_decode(generated)
                        if self.tokenizer_decode is not None
                        else "".join(chr(t) if 32 <= t < 127 else "" for t in generated)
                    )
                    if "Answer:" in decoded_so_far and decoded_so_far.count("\n") >= 6:
                        break

            # Decode generated text
            if self.tokenizer_decode is not None:
                gen_text = self.tokenizer_decode(generated)
            else:
                gen_text = "".join(chr(t) if 32 <= t < 127 else "" for t in generated)

            parsed = self._parse_question(gen_text, category, idx)
            if parsed is not None:
                questions.append(parsed)

        return questions

    def _build_generation_prompt(self, category: str, idx: int) -> str:
        """Return a structured prompt for generating a multiple-choice question."""
        return (
            f"Generate a multiple-choice question about {category}.\n"
            "Format:\n"
            "Q: [question]\n"
            "A) [choice]\n"
            "B) [choice]\n"
            "C) [choice]\n"
            "D) [choice]\n"
            "Answer: [A/B/C/D]\n"
            "\n"
            f"Question {idx + 1}:"
        )

    def _parse_question(
        self,
        text: str,
        category: str,
        idx: int,
    ) -> EvalQuestion | None:
        """Parse generated text for Q:/A)/B)/C)/D)/Answer: pattern.

        Returns None if parsing fails.
        """
        match = _QUESTION_RE.search(text)
        if match is None:
            return None

        question_text = match.group(1).strip()
        choice_a = match.group(2).strip()
        choice_b = match.group(3).strip()
        choice_c = match.group(4).strip()
        choice_d = match.group(5).strip()
        answer_letter = match.group(6).strip()

        correct_idx = _ANSWER_LETTER_TO_IDX.get(answer_letter, -1)
        if correct_idx == -1:
            return None

        return EvalQuestion(
            question_id=f"{category}_{idx}_{uuid.uuid4().hex[:8]}",
            question=question_text,
            choices=[
                f"A) {choice_a}",
                f"B) {choice_b}",
                f"C) {choice_c}",
                f"D) {choice_d}",
            ],
            correct_choice=correct_idx,
            category=category,
            difficulty=0.5,  # default estimate; updated by runner if desired
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class WrittenEvalRunner:
    """Run a set of EvalQuestions against a model and compute metrics.

    Scoring: present question + choices, let model generate token for A/B/C/D,
    pick the one with highest logit among ['A','B','C','D'] at the first
    generated token.

    Args:
        model: AureliusTransformer
        tokenizer_encode: callable
        max_seq_len: int
    """

    def __init__(
        self,
        model: object,
        tokenizer_encode: Callable[[str], list[int]],
        max_seq_len: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.max_seq_len = max_seq_len

    def run(self, questions: list[EvalQuestion]) -> list[EvalResult]:
        """Score each question. Returns list of EvalResult."""
        results: list[EvalResult] = []
        for q in questions:
            result = self._score_question(q)
            results.append(result)
        return results

    def _score_question(self, q: EvalQuestion) -> EvalResult:
        """Score a single question by extracting logits for A/B/C/D."""
        prompt = self._build_eval_prompt(q)
        token_ids = self.tokenizer_encode(prompt)
        token_ids = token_ids[: self.max_seq_len]

        input_tensor = torch.tensor([token_ids], dtype=torch.long)

        with torch.no_grad():
            _loss, logits, _pkv = self.model(input_tensor)

        # logits shape: (1, T, vocab_size)
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Extract logits for A(65), B(66), C(67), D(68)
        choice_logits = torch.stack([
            last_logits[tid] if tid < last_logits.shape[0] else torch.tensor(float("-inf"))
            for tid in _CHOICE_TOKEN_IDS
        ])  # shape: (4,)

        # Softmax to get confidence
        choice_probs = F.softmax(choice_logits, dim=0)
        predicted_idx = int(choice_probs.argmax().item())
        confidence = float(choice_probs[predicted_idx].item())

        correct = predicted_idx == q.correct_choice

        return EvalResult(
            question_id=q.question_id,
            predicted_choice=predicted_idx,
            correct=correct,
            confidence=confidence,
            raw_output=f"choice_logits={choice_logits.tolist()}",
        )

    def _build_eval_prompt(self, q: EvalQuestion) -> str:
        """Format question for scoring.

        Format: 'Question: {q.question}\nA) {q.choices[0]}\n...\nAnswer:'
        """
        lines = [f"Question: {q.question}"]
        for choice in q.choices:
            lines.append(choice)
        lines.append("Answer:")
        return "\n".join(lines)

    def aggregate(self, results: list[EvalResult]) -> dict:
        """Compute aggregate metrics over a list of EvalResult.

        Returns:
            {
              'accuracy': float,
              'mean_confidence': float,
              'n_unparseable': int,
              'per_category': dict[str, float]  # category -> accuracy
            }
        """
        if not results:
            return {
                "accuracy": 0.0,
                "mean_confidence": 0.0,
                "n_unparseable": 0,
                "per_category": {},
            }

        n_total = len(results)
        n_correct = sum(1 for r in results if r.correct)
        n_unparseable = sum(1 for r in results if r.predicted_choice == -1)
        mean_confidence = sum(r.confidence for r in results) / n_total
        accuracy = n_correct / n_total

        # per_category requires mapping question_id -> category
        # Since EvalResult doesn't store category, we derive it from question_id prefix
        # Convention: question_id = "{category}_{idx}_{hex}"
        # However, to support externally created EvalResults without this convention,
        # we fall back to grouping by question_id prefix up to first "_".
        per_category_correct: dict[str, int] = {}
        per_category_total: dict[str, int] = {}
        for r in results:
            # Extract category from question_id (first segment before "_")
            cat = r.question_id.split("_")[0]
            per_category_correct[cat] = per_category_correct.get(cat, 0) + (1 if r.correct else 0)
            per_category_total[cat] = per_category_total.get(cat, 0) + 1

        per_category = {
            cat: per_category_correct[cat] / per_category_total[cat]
            for cat in per_category_total
        }

        return {
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "n_unparseable": n_unparseable,
            "per_category": per_category,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def difficulty_score(results: list[EvalResult]) -> float:
    """Estimate difficulty as 1 - accuracy (higher = harder set)."""
    if not results:
        return 0.0
    accuracy = sum(1 for r in results if r.correct) / len(results)
    return 1.0 - accuracy
