"""Debate-style alignment: multiple agents argue positions, judge selects best argument."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DebateConfig:
    """Configuration for a debate session."""

    n_debaters: int = 2
    n_rounds: int = 3
    max_tokens_per_turn: int = 64
    judge_temperature: float = 0.0
    debater_temperature: float = 0.7
    use_simultaneous: bool = True  # all debaters argue at once per round


@dataclass
class DebateTurn:
    """A single turn in a debate."""

    debater_id: int
    round_num: int
    argument: str
    score: float | None = None


class DebateJudge:
    """Scores arguments using model log-likelihood."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    def score_argument(self, question: str, argument: str) -> float:
        """Return mean log-likelihood of argument tokens given question context."""
        context_ids = self.tokenizer_encode(question)
        argument_ids = self.tokenizer_encode(argument)

        if not argument_ids:
            return 0.0

        all_ids = context_ids + argument_ids
        input_ids = torch.tensor([all_ids], dtype=torch.long)

        # Build labels: mask out context tokens, keep argument tokens
        labels = torch.full_like(input_ids, -100)
        context_len = len(context_ids)
        labels[0, context_len:] = input_ids[0, context_len:]

        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)

        loss = outputs[0]  # cross-entropy loss (mean over non-masked tokens)
        if loss is None or torch.isnan(loss):
            return 0.0
        # Convert cross-entropy to mean log-likelihood (negate)
        return float(-loss.item())

    def select_best(self, question: str, arguments: list[str]) -> int:
        """Return index of highest-scoring argument."""
        if not arguments:
            return 0
        scores = [self.score_argument(question, arg) for arg in arguments]
        return int(scores.index(max(scores)))


class DebateDebater:
    """One debating agent."""

    def __init__(
        self,
        debater_id: int,
        model: nn.Module,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
        config: DebateConfig,
    ) -> None:
        self.debater_id = debater_id
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = config

    def generate_argument(
        self,
        question: str,
        history: list[DebateTurn],
        position: str,
    ) -> str:
        """Generate an argument for the given position."""
        # Build prompt
        parts = [f"Question: {question}"]
        if history:
            parts.append("Debate history:")
            for turn in history:
                parts.append(
                    f"  [Debater {turn.debater_id}, Round {turn.round_num}]: {turn.argument}"
                )
        parts.append(f"Debater {self.debater_id} argues for {position}:")
        prompt = "\n".join(parts)

        prompt_ids = self.tokenizer_encode(prompt)
        if not prompt_ids:
            prompt_ids = [0]

        torch.tensor([prompt_ids], dtype=torch.long)
        max_new = self.config.max_tokens_per_turn

        try:
            with torch.no_grad():
                # Greedy / temperature decode manually
                generated = list(prompt_ids)
                for _ in range(max_new):
                    cur_ids = torch.tensor([generated], dtype=torch.long)
                    outputs = self.model(cur_ids)
                    logits = outputs[1]  # (1, seq_len, vocab_size)
                    next_logits = logits[0, -1, :]  # (vocab_size,)

                    if self.config.debater_temperature > 0:
                        next_logits = next_logits / self.config.debater_temperature
                        probs = torch.softmax(next_logits, dim=-1)
                        next_token = int(torch.multinomial(probs, 1).item())
                    else:
                        next_token = int(next_logits.argmax().item())

                    generated.append(next_token)

                new_ids = generated[len(prompt_ids) :]
                argument = self.tokenizer_decode(new_ids)
        except Exception:
            argument = f"[Debater {self.debater_id} argument placeholder]"

        return argument


class DebateSession:
    """Orchestrates a multi-round debate."""

    def __init__(
        self,
        question: str,
        positions: list[str],
        debaters: list[DebateDebater],
        judge: DebateJudge,
        config: DebateConfig,
    ) -> None:
        self.question = question
        self.positions = positions
        self.debaters = debaters
        self.judge = judge
        self.config = config
        self.history: list[DebateTurn] = []

    def run_round(self, round_num: int) -> list[DebateTurn]:
        """Each debater generates an argument; judge scores each. Returns this round's turns."""
        turns: list[DebateTurn] = []

        for i, debater in enumerate(self.debaters):
            position = self.positions[i % len(self.positions)]
            argument = debater.generate_argument(self.question, self.history, position)
            turn = DebateTurn(
                debater_id=debater.debater_id,
                round_num=round_num,
                argument=argument,
                score=None,
            )
            turns.append(turn)

        # Judge scores all arguments from this round
        [t.argument for t in turns]
        for idx, turn in enumerate(turns):
            score = self.judge.score_argument(self.question, turn.argument)
            turns[idx] = DebateTurn(
                debater_id=turn.debater_id,
                round_num=turn.round_num,
                argument=turn.argument,
                score=score,
            )

        self.history.extend(turns)
        return turns

    def run(self) -> tuple[str, list[DebateTurn]]:
        """Run n_rounds, return (winning_position, all_turns)."""
        for round_num in range(1, self.config.n_rounds + 1):
            self.run_round(round_num)

        # Aggregate scores per debater position
        position_scores: dict[str, float] = {pos: 0.0 for pos in self.positions}
        position_counts: dict[str, int] = {pos: 0 for pos in self.positions}

        for turn in self.history:
            debater_idx = next(
                (i for i, d in enumerate(self.debaters) if d.debater_id == turn.debater_id),
                0,
            )
            pos = self.positions[debater_idx % len(self.positions)]
            if turn.score is not None:
                position_scores[pos] += turn.score
                position_counts[pos] += 1

        # Pick position with highest total score
        winning_position = max(
            self.positions,
            key=lambda p: (
                position_scores[p] / position_counts[p] if position_counts[p] > 0 else float("-inf")
            ),
        )
        return winning_position, self.history

    def transcript(self) -> str:
        """Format full debate as human-readable string."""
        lines = [f"=== Debate: {self.question} ==="]
        for turn in self.history:
            score_str = f"{turn.score:.4f}" if turn.score is not None else "N/A"
            lines.append(
                f"[Round {turn.round_num}] Debater {turn.debater_id} "
                f"(score={score_str}): "
                f"{turn.argument}"
            )
        return "\n".join(lines)


def aggregate_debate_results(
    sessions: list[tuple[str, list[DebateTurn]]],
) -> dict:
    """Aggregate multiple debate results into summary statistics.

    Args:
        sessions: list of (winning_position, turns) tuples

    Returns:
        {
            "win_rates": {position: float},
            "mean_rounds": float,
            "total_sessions": int,
        }
    """
    if not sessions:
        return {"win_rates": {}, "mean_rounds": 0.0, "total_sessions": 0}

    win_counts: dict[str, int] = {}
    total_rounds_list: list[int] = []

    for winning_position, turns in sessions:
        win_counts[winning_position] = win_counts.get(winning_position, 0) + 1
        if turns:
            max_round = max(t.round_num for t in turns)
            total_rounds_list.append(max_round)
        else:
            total_rounds_list.append(0)

    total = len(sessions)
    win_rates = {pos: count / total for pos, count in win_counts.items()}
    mean_rounds = sum(total_rounds_list) / len(total_rounds_list) if total_rounds_list else 0.0

    return {
        "win_rates": win_rates,
        "mean_rounds": mean_rounds,
        "total_sessions": total,
    }


def format_debate_for_training(
    session_result: tuple[str, list[DebateTurn]],
    tokenizer_encode: Callable,
) -> dict:
    """Convert a debate result to DPO training format.

    Returns:
        {"prompt": str, "chosen": str, "rejected": str}
    chosen = winning argument, rejected = losing argument (last round)
    """
    winning_position, turns = session_result

    if not turns:
        return {"prompt": "", "chosen": "", "rejected": ""}

    last_round = max(t.round_num for t in turns)
    last_round_turns = [t for t in turns if t.round_num == last_round]

    if not last_round_turns:
        return {"prompt": "", "chosen": "", "rejected": ""}

    # Sort by score to get best (chosen) and worst (rejected)
    scored = sorted(
        last_round_turns,
        key=lambda t: t.score if t.score is not None else float("-inf"),
        reverse=True,
    )

    chosen = scored[0].argument if scored else ""
    rejected = scored[-1].argument if len(scored) > 1 else ""

    # Build prompt from all turns except the last round
    earlier_turns = [t for t in turns if t.round_num < last_round]
    prompt_parts = []
    for t in earlier_turns:
        prompt_parts.append(f"Debater {t.debater_id} (Round {t.round_num}): {t.argument}")
    prompt = "\n".join(prompt_parts)

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
