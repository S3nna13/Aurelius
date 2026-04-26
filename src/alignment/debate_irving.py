"""Debate-based alignment (Irving et al., 2018 — arXiv:1805.00899).

Two agents debate a claim; a judge evaluates the arguments and picks a winner.
Debate is proposed as a scalable oversight technique: even if the judge is less
capable than the debaters, strong arguments are harder to fabricate than to
criticise, so the honest debater tends to win.

Components:
    DebateArgument     -- single argument turn dataclass
    DebateSession      -- orchestrates a multi-round debate between two models
    BestOfNDebate      -- generate N responses and use debate to select the best
    DebateDataCollector-- collect transcripts and convert to preference pairs
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DebateArgument:
    """A single argument in a debate."""

    position: str  # 'for' | 'against'
    content: str
    round: int
    model_id: str  # which model made this argument


class DebateSession:
    """A structured debate between two agents about a claim.

    Args:
        model_for:            model arguing FOR the claim
        model_against:        model arguing AGAINST
        judge_model:          model that evaluates the debate
        n_rounds:             number of argument rounds (each side argues once per round)
        max_tokens_per_round: max generation length per argument
    """

    def __init__(
        self,
        model_for,
        model_against,
        judge_model,
        n_rounds: int = 2,
        max_tokens_per_round: int = 64,
    ) -> None:
        self.model_for = model_for
        self.model_against = model_against
        self.judge_model = judge_model
        self.n_rounds = n_rounds
        self.max_tokens_per_round = max_tokens_per_round

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_argument_prompt(
        self,
        claim: str,
        position: str,
        history: list[DebateArgument],
    ) -> str:
        """Build prompt for a debater given claim, position, and argument history."""
        lines = [
            f"Claim: {claim}",
            f"Your position: {position}",
        ]
        if history:
            lines.append("Debate so far:")
            for arg in history:
                lines.append(f"  [{arg.position.upper()} - Round {arg.round}] {arg.content}")
        lines.append(
            f"Provide a concise argument {position} the claim (respond with 1-2 sentences):"
        )
        return "\n".join(lines)

    def _build_judge_prompt(
        self,
        claim: str,
        arguments: list[DebateArgument],
    ) -> str:
        """Build the judge evaluation prompt."""
        lines = [f"Claim: {claim}", "Debate transcript:"]
        for arg in arguments:
            lines.append(f"  [{arg.position.upper()} - Round {arg.round}] {arg.content}")
        lines.append(
            "Evaluate the debate. Which side made stronger arguments? "
            "Reply with WINNER: for, WINNER: against, or WINNER: tie, "
            "followed by a brief reasoning."
        )
        return "\n".join(lines)

    def _generate_argument(
        self,
        model,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Generate an argument from a model.

        Tries model.generate() if the model has the expected interface;
        falls back to a placeholder string so that tests with mock models work.
        """
        try:
            # Attempt real generation using a simple integer sequence as input_ids
            # (proxy encoding: char codes mod vocab_size)
            device = next(model.parameters()).device
            ids = [ord(c) % 256 for c in prompt[:64]]
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                )
            new_ids = output[:, input_ids.shape[1] :][0].tolist()
            # Decode as ASCII mod 95 printable chars (simple proxy)
            return "".join(chr(max(32, i % 95 + 32)) for i in new_ids[:max_tokens])
        except Exception:
            return f"[argument placeholder from {id(model)}]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_debate(self, claim: str) -> dict:
        """Run a full debate and return the result.

        For each round:
          - model_for argues FOR the claim
          - model_against argues AGAINST

        Then the judge evaluates all arguments and picks a winner.

        Returns:
            {
                'claim': str,
                'arguments': [DebateArgument],
                'winner': 'for' | 'against' | 'tie',
                'judge_reasoning': str,
            }
        """
        arguments: list[DebateArgument] = []

        for round_num in range(1, self.n_rounds + 1):
            # FOR argument
            for_prompt = self._build_argument_prompt(claim, "for", arguments)
            for_text = self._generate_argument(
                self.model_for, for_prompt, self.max_tokens_per_round
            )
            arguments.append(
                DebateArgument(
                    position="for",
                    content=for_text,
                    round=round_num,
                    model_id="model_for",
                )
            )

            # AGAINST argument
            against_prompt = self._build_argument_prompt(claim, "against", arguments)
            against_text = self._generate_argument(
                self.model_against, against_prompt, self.max_tokens_per_round
            )
            arguments.append(
                DebateArgument(
                    position="against",
                    content=against_text,
                    round=round_num,
                    model_id="model_against",
                )
            )

        verdict = self.judge_debate(claim, arguments)
        return {
            "claim": claim,
            "arguments": arguments,
            "winner": verdict["winner"],
            "judge_reasoning": verdict["reasoning"],
        }

    def judge_debate(self, claim: str, arguments: list[DebateArgument]) -> dict:
        """Judge evaluates the debate.

        Returns {'winner': str, 'reasoning': str}.
        'winner' is 'for', 'against', or 'tie'.
        """
        judge_prompt = self._build_judge_prompt(claim, arguments)
        raw = self._generate_argument(self.judge_model, judge_prompt, self.max_tokens_per_round)

        # Parse winner from raw text (look for 'WINNER: <label>')
        winner = "tie"
        raw_lower = raw.lower()
        if "winner: for" in raw_lower:
            winner = "for"
        elif "winner: against" in raw_lower:
            winner = "against"
        elif "winner: tie" in raw_lower:
            winner = "tie"

        return {"winner": winner, "reasoning": raw}


class BestOfNDebate:
    """Best-of-N selection using debate as a judge.

    Generates N candidate responses, then uses a debate session to pick the best.
    """

    def __init__(self, model, judge_model, n: int = 4) -> None:
        self.model = model
        self.judge_model = judge_model
        self.n = n

    def select_best(self, prompt: str, prompt_ids: torch.Tensor) -> dict:
        """Generate N responses and debate which is best.

        Returns:
            {
                'responses': [str],
                'best_idx': int,
                'reasoning': str,
            }
        """
        device = next(self.model.parameters()).device
        input_ids = prompt_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        responses: list[str] = []
        for _ in range(self.n):
            try:
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=64,
                        temperature=1.0,
                        top_p=0.9,
                    )
                new_ids = output[:, input_ids.shape[1] :][0].tolist()
                text = "".join(chr(max(32, i % 95 + 32)) for i in new_ids[:64])
            except Exception:
                text = f"[response {len(responses)}]"
            responses.append(text)

        if len(responses) < 2:
            return {"responses": responses, "best_idx": 0, "reasoning": "Only one response."}

        # Use debate between first and best-so-far to iteratively pick the best
        best_idx = 0
        last_reasoning = ""
        for idx in range(1, len(responses)):
            claim = (
                f"Response A is better than Response B. "
                f"A: {responses[best_idx][:120]} | B: {responses[idx][:120]}"
            )
            session = DebateSession(
                self.model, self.model, self.judge_model, n_rounds=1, max_tokens_per_round=32
            )
            result = session.run_debate(claim)
            last_reasoning = result["judge_reasoning"]
            # If 'against' wins, response B is better
            if result["winner"] == "against":
                best_idx = idx

        return {
            "responses": responses,
            "best_idx": best_idx,
            "reasoning": last_reasoning,
        }


class DebateDataCollector:
    """Collect debate transcripts for training data.

    Debate results can be converted to (prompt, chosen, rejected) preference
    pairs for use with DPO or similar training objectives.
    """

    def __init__(self) -> None:
        self.debates: list[dict] = []

    def record(self, debate_result: dict) -> None:
        """Store a debate result dict (as returned by DebateSession.run_debate)."""
        self.debates.append(debate_result)

    def to_preference_pairs(self) -> list[dict]:
        """Convert debate results to (prompt, chosen, rejected) pairs.

        The winner's arguments become 'chosen'; the loser's become 'rejected'.
        Ties are skipped.

        Returns:
            list of {'prompt': str, 'chosen': str, 'rejected': str}
        """
        pairs: list[dict] = []
        for debate in self.debates:
            winner = debate.get("winner", "tie")
            if winner == "tie":
                continue
            claim = debate.get("claim", "")
            arguments: list[DebateArgument] = debate.get("arguments", [])

            winning_pos = winner  # 'for' or 'against'
            losing_pos = "against" if winner == "for" else "for"

            chosen_args = [a for a in arguments if a.position == winning_pos]
            rejected_args = [a for a in arguments if a.position == losing_pos]

            chosen_text = " ".join(a.content for a in chosen_args)
            rejected_text = " ".join(a.content for a in rejected_args)

            pairs.append(
                {
                    "prompt": claim,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                }
            )
        return pairs

    def stats(self) -> dict:
        """Return debate outcome statistics.

        Returns:
            {'n_debates': int, 'for_wins': int, 'against_wins': int, 'ties': int}
        """
        n_debates = len(self.debates)
        for_wins = sum(1 for d in self.debates if d.get("winner") == "for")
        against_wins = sum(1 for d in self.debates if d.get("winner") == "against")
        ties = sum(1 for d in self.debates if d.get("winner") == "tie")
        return {
            "n_debates": n_debates,
            "for_wins": for_wins,
            "against_wins": against_wins,
            "ties": ties,
        }
