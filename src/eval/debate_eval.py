"""Debate-based evaluation following Irving et al. 2018 'AI Safety via Debate'.

Two AI debaters argue for/against a claim; a judge model evaluates their
arguments and picks a winner, providing a scalable evaluation signal beyond
pointwise scoring.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration and data structures
# ---------------------------------------------------------------------------


@dataclass
class DebateConfig:
    n_rounds: int = 2
    max_argument_length: int = 500
    judge_temperature: float = 0.0


@dataclass
class DebateArgument:
    debater_id: str  # "A" or "B"
    position: str  # "for" or "against"
    content: str
    round_number: int


@dataclass
class DebateTranscript:
    question: str
    position_a: str
    position_b: str
    arguments: list[DebateArgument] = field(default_factory=list)
    winner: str | None = None  # "A", "B", or "tie"
    judge_reasoning: str = ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_debater_prompt(
    question: str,
    position: str,
    transcript: DebateTranscript,
    round_num: int,
) -> str:
    """Build a prompt for a debater given the transcript so far."""
    lines: list[str] = [
        f"Question: {question}",
        f"Your position: {position}",
        f"Current round: {round_num}",
        "",
        "Transcript so far:",
    ]

    if transcript.arguments:
        for arg in transcript.arguments:
            lines.append(
                f"  [Round {arg.round_number}] Debater {arg.debater_id} ({arg.position}): {arg.content}"  # noqa: E501
            )
    else:
        lines.append("  (no arguments yet)")

    lines += [
        "",
        "Now provide your argument for this round. Be concise and persuasive.",
        "Argument:",
    ]
    return "\n".join(lines)


def build_judge_prompt(transcript: DebateTranscript) -> str:
    """Build a prompt for the judge showing the full transcript."""
    lines: list[str] = [
        f"Question: {transcript.question}",
        "",
        f"Debater A argues: {transcript.position_a}",
        f"Debater B argues: {transcript.position_b}",
        "",
        "Full debate transcript:",
    ]

    for arg in transcript.arguments:
        lines.append(
            f"  [Round {arg.round_number}] Debater {arg.debater_id} ({arg.position}): {arg.content}"
        )

    lines += [
        "",
        "Based on the debate above, decide who made the stronger argument.",
        "Respond with exactly one of:",
        "  Winner: A",
        "  Winner: B",
        "  Winner: tie",
        "Then provide your reasoning.",
        "",
        "Your verdict:",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Winner extraction
# ---------------------------------------------------------------------------

_WINNER_RE = re.compile(r"winner\s*:\s*(A|B|tie)\b", re.IGNORECASE)


def extract_winner(judge_output: str) -> str | None:
    """Parse 'Winner: A/B/tie' from judge output (case-insensitive).

    Returns "A", "B", "tie", or None if unparseable.
    """
    match = _WINNER_RE.search(judge_output)
    if match is None:
        return None
    raw = match.group(1)
    # Normalise to lowercase for tie, uppercase for A/B
    if raw.lower() == "tie":
        return "tie"
    return raw.upper()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DebateEvaluator:
    def __init__(self, generate_fn: Callable[[str], str], config: DebateConfig) -> None:
        self.generate_fn = generate_fn
        self.config = config

    def run_debate(
        self,
        question: str,
        position_a: str,
        position_b: str,
    ) -> DebateTranscript:
        """Alternate n_rounds between debater A and B, then call the judge."""
        transcript = DebateTranscript(
            question=question,
            position_a=position_a,
            position_b=position_b,
        )

        for round_num in range(1, self.config.n_rounds + 1):
            # Debater A goes first each round
            for debater_id, position in (("A", position_a), ("B", position_b)):
                prompt = build_debater_prompt(question, position, transcript, round_num)
                content = self.generate_fn(prompt)
                # Truncate to max_argument_length
                content = content[: self.config.max_argument_length]
                transcript.arguments.append(
                    DebateArgument(
                        debater_id=debater_id,
                        position=position,
                        content=content,
                        round_number=round_num,
                    )
                )

        # Judge
        judge_prompt = build_judge_prompt(transcript)
        judge_output = self.generate_fn(judge_prompt)
        transcript.winner = extract_winner(judge_output)
        transcript.judge_reasoning = judge_output

        return transcript

    def evaluate_claim(self, claim: str) -> DebateTranscript:
        """Evaluate a claim: A argues for it, B argues against it."""
        position_a = f"This claim is correct: {claim}"
        position_b = f"This claim is incorrect: {claim}"
        return self.run_debate(claim, position_a, position_b)

    def batch_evaluate(self, claims: list[str]) -> list[DebateTranscript]:
        """Evaluate a list of claims and return one transcript per claim."""
        return [self.evaluate_claim(claim) for claim in claims]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_debate_results(transcripts: list[DebateTranscript]) -> dict[str, float]:
    """Aggregate outcomes across a list of transcripts.

    Returns::

        {
            'win_rate_a': float,
            'win_rate_b': float,
            'tie_rate': float,
            'n_debates': int,   # stored as float for uniform dict type
        }
    """
    n = len(transcripts)
    if n == 0:
        return {"win_rate_a": 0.0, "win_rate_b": 0.0, "tie_rate": 0.0, "n_debates": 0}

    wins_a = sum(1 for t in transcripts if t.winner == "A")
    wins_b = sum(1 for t in transcripts if t.winner == "B")
    ties = sum(1 for t in transcripts if t.winner == "tie")

    return {
        "win_rate_a": wins_a / n,
        "win_rate_b": wins_b / n,
        "tie_rate": ties / n,
        "n_debates": n,
    }
