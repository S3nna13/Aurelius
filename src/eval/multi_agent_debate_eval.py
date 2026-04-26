"""
Multi-Agent Debate Evaluator
============================
Implements debate quality assessment following Du et al. (2023),
"Improving Factuality and Reasoning in Language Models through Multiagent Debate."

Metrics:
  - position_drift        : fraction of agents that changed their stated position
  - final_consensus       : mean pairwise Jaccard of agents' final positions
  - argument_diversity    : mean pairwise diversity of final-round arguments
  - round_convergence     : per-round consensus trajectory
  - overall               : weighted composite (0.3·drift + 0.3·consensus + 0.4·diversity)

Cycle 137-D
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DebateConfig:
    """Configuration for the debate evaluation."""

    n_agents: int = 3
    n_rounds: int = 3
    convergence_threshold: float = 0.8  # Jaccard similarity for "consensus"
    ngram_n: int = 2  # n for n-gram diversity (unused in base impl — word-level used)


@dataclass
class AgentTurn:
    """A single turn by one agent in the debate."""

    agent_id: int
    round_idx: int
    position: str  # agent's stated position / answer
    arguments: list[str]  # key claims made this turn


@dataclass
class DebateEvalResult:
    """Evaluation result for one debate."""

    n_agents: int
    n_rounds: int
    position_drift: float  # fraction of agents that changed position
    final_consensus: float  # Jaccard similarity of final positions
    argument_diversity: float  # mean pairwise diversity of arguments
    round_convergence: list[float]  # per-round consensus scores
    overall: float  # weighted composite


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DebateEvaluator:
    """
    Evaluate multi-agent debate quality on four dimensions:
    position drift, final consensus, argument diversity, and per-round convergence.
    """

    def __init__(self, config: DebateConfig | None = None) -> None:
        self.config = config or DebateConfig()

    # ------------------------------------------------------------------
    # Core metric: word-level Jaccard similarity
    # ------------------------------------------------------------------

    def jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """
        Word-level Jaccard similarity: |A ∩ B| / |A ∪ B|.

        Returns 1.0 if both texts are empty, 0.0 if one is empty and the
        other is not.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a and not words_b:
            return 1.0
        union = words_a | words_b
        if not union:
            return 1.0
        intersection = words_a & words_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Metric 1: Position drift
    # ------------------------------------------------------------------

    def position_drift(self, turns: list[AgentTurn]) -> float:
        """
        For each agent compare their round-0 position to their final-round
        position.  drift = 1 - mean(jaccard(initial, final)) across agents.

        A value of 0.0 means no agent changed position.
        A value of 1.0 means every agent completely changed position.
        """
        # Collect per-agent turns, keyed by agent_id
        by_agent: dict[int, list[AgentTurn]] = {}
        for t in turns:
            by_agent.setdefault(t.agent_id, []).append(t)

        if not by_agent:
            return 0.0

        similarities: list[float] = []
        for agent_turns in by_agent.values():
            sorted_turns = sorted(agent_turns, key=lambda x: x.round_idx)
            initial = sorted_turns[0].position
            final = sorted_turns[-1].position
            similarities.append(self.jaccard_similarity(initial, final))

        mean_sim = sum(similarities) / len(similarities)
        return 1.0 - mean_sim

    # ------------------------------------------------------------------
    # Metric 2: Consensus at a given round
    # ------------------------------------------------------------------

    def consensus(self, turns: list[AgentTurn], round_idx: int) -> float:
        """
        Mean pairwise Jaccard similarity of all agents' positions at round_idx.

        If only one agent has a turn at that round, returns 1.0 (trivial consensus).
        If no agents have a turn at that round, returns 0.0.
        """
        round_turns = [t for t in turns if t.round_idx == round_idx]
        positions = [t.position for t in round_turns]

        if len(positions) == 0:
            return 0.0
        if len(positions) == 1:
            return 1.0

        pairs = list(combinations(positions, 2))
        scores = [self.jaccard_similarity(a, b) for a, b in pairs]
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Metric 3: Argument diversity (final round)
    # ------------------------------------------------------------------

    def argument_diversity(self, turns: list[AgentTurn]) -> float:
        """
        Extract all arguments from the final round across all agents and
        compute mean pairwise diversity = 1 - jaccard(arg_i, arg_j).

        If there is only one argument (or zero), returns 0.0.
        If there are no arguments at all, returns 0.0.
        """
        if not turns:
            return 0.0

        final_round = max(t.round_idx for t in turns)
        final_turns = [t for t in turns if t.round_idx == final_round]

        # Gather all individual argument strings
        all_args: list[str] = []
        for t in final_turns:
            all_args.extend(t.arguments)

        if len(all_args) <= 1:
            return 0.0

        pairs = list(combinations(all_args, 2))
        diversities = [1.0 - self.jaccard_similarity(a, b) for a, b in pairs]
        return sum(diversities) / len(diversities)

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self, turns: list[AgentTurn]) -> DebateEvalResult:
        """
        Compute all metrics and return a DebateEvalResult.

        overall = 0.3·position_drift + 0.3·final_consensus + 0.4·argument_diversity
        """
        if not turns:
            return DebateEvalResult(
                n_agents=self.config.n_agents,
                n_rounds=self.config.n_rounds,
                position_drift=0.0,
                final_consensus=0.0,
                argument_diversity=0.0,
                round_convergence=[0.0] * self.config.n_rounds,
                overall=0.0,
            )

        # Determine actual rounds from data (or fall back to config)
        all_rounds = sorted(set(t.round_idx for t in turns))
        n_rounds = max(all_rounds) + 1  # 0-indexed

        # Per-round convergence
        round_convergence = [self.consensus(turns, r) for r in range(n_rounds)]

        drift = self.position_drift(turns)
        final_round = n_rounds - 1
        final_cons = self.consensus(turns, final_round)
        arg_div = self.argument_diversity(turns)

        overall = 0.3 * drift + 0.3 * final_cons + 0.4 * arg_div

        # Clamp to [0, 1] for safety
        overall = max(0.0, min(1.0, overall))

        return DebateEvalResult(
            n_agents=self.config.n_agents,
            n_rounds=n_rounds,
            position_drift=drift,
            final_consensus=final_cons,
            argument_diversity=arg_div,
            round_convergence=round_convergence,
            overall=overall,
        )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(self, debate_sets: list[list[AgentTurn]]) -> list[DebateEvalResult]:
        """Evaluate multiple debates and return a list of results."""
        return [self.evaluate(turns) for turns in debate_sets]

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate(self, results: list[DebateEvalResult]) -> dict[str, float]:
        """
        Compute mean of each scalar metric across a list of DebateEvalResults.

        Returns a dict with keys:
          position_drift, final_consensus, argument_diversity, overall
        """
        if not results:
            return {
                "position_drift": 0.0,
                "final_consensus": 0.0,
                "argument_diversity": 0.0,
                "overall": 0.0,
            }

        keys = ["position_drift", "final_consensus", "argument_diversity", "overall"]
        return {k: sum(getattr(r, k) for r in results) / len(results) for k in keys}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.eval import BENCHMARK_REGISTRY  # noqa: E402

BENCHMARK_REGISTRY["multi_agent_debate"] = DebateEvaluator
