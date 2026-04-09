"""Multi-agent debate framework for improved reasoning.

Multiple model instances debate to reach consensus, improving answer quality
on complex reasoning tasks through iterative position refinement.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DebateConfig:
    """Configuration for multi-agent debate."""

    n_agents: int = 3
    n_rounds: int = 2
    max_new_tokens: int = 64
    temperature: float = 0.8
    consensus_threshold: float = 0.6  # fraction of agents agreeing = consensus


@dataclass
class AgentState:
    """State for a single agent in the debate."""

    agent_id: int
    position: str
    confidence: float
    history: list[str] = field(default_factory=list)


@dataclass
class DebateResult:
    """Result from a multi-agent debate."""

    question: str
    final_answer: str
    consensus_reached: bool
    n_rounds: int
    agent_positions: list[str]
    confidence_scores: list[float]


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(model, prompt_ids: list[int], max_new_tokens: int, vocab_size: int) -> str:
    """Token-by-token greedy decoding.

    Returns decoded bytes string using bytes([t % 256 for t in gen_ids]).
    """
    gen_ids: list[int] = []
    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # (1, S)
    past_key_values = None
    cur_ids = input_ids

    model.train(False)

    for _ in range(max_new_tokens):
        _, logits, past_key_values = model(cur_ids, past_key_values=past_key_values)
        next_logits = logits[:, -1, :]  # (1, vocab)
        next_token_id = int(next_logits.argmax(dim=-1).item())
        gen_ids.append(next_token_id)

        # Prepare next step: only the newly generated token
        cur_ids = torch.tensor([[next_token_id]], dtype=torch.long)

    return bytes([t % 256 for t in gen_ids]).decode("utf-8", errors="replace")


def compute_agreement(positions: list[str]) -> float:
    """Measure pairwise agreement between agent positions using character trigram overlap.

    Returns average pairwise similarity in [0, 1].
    Single agent always returns 1.0.
    """
    if len(positions) <= 1:
        return 1.0

    def get_trigrams(text: str) -> set[str]:
        if len(text) < 3:
            return {text} if text else set()
        return {text[i:i+3] for i in range(len(text) - 2)}

    def trigram_similarity(a: str, b: str) -> float:
        tgrams_a = get_trigrams(a)
        tgrams_b = get_trigrams(b)
        if not tgrams_a and not tgrams_b:
            return 1.0
        if not tgrams_a or not tgrams_b:
            return 0.0
        intersection = tgrams_a & tgrams_b
        union = tgrams_a | tgrams_b
        return len(intersection) / len(union)

    total_similarity = 0.0
    n_pairs = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total_similarity += trigram_similarity(positions[i], positions[j])
            n_pairs += 1

    return total_similarity / n_pairs if n_pairs > 0 else 1.0


def extract_confidence(text: str) -> float:
    """Extract confidence value from text.

    Looks for 'confidence: X%', 'certainty: X%', or 'sure: X%' patterns.
    Returns X/100, or 0.5 as default if not found.
    """
    pattern = r"(?:confidence|certainty|sure)\s*:\s*(\d+(?:\.\d+)?)%"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0
    return 0.5


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DebateOrchestrator:
    """Orchestrates a multi-agent debate to reach consensus on a question.

    Args:
        model: AureliusTransformer
        config: DebateConfig
        tokenizer_encode: callable str -> list[int]
        tokenizer_decode: callable list[int] -> str
    """

    def __init__(
        self,
        model,
        config: DebateConfig,
        tokenizer_encode,
        tokenizer_decode,
    ) -> None:
        self.model = model
        self.config = config
        self._encode = tokenizer_encode
        self._decode = tokenizer_decode
        self.model.train(False)

    def _build_initial_prompt(self, question: str, agent_id: int) -> str:
        """Build prompt for the initial round (no peer context)."""
        return f"You are agent {agent_id}. Answer: {question}\nAnswer:"

    def _build_debate_prompt(self, question: str, agent_id: int, other_positions: list[str]) -> str:
        """Build prompt for debate rounds (with peer positions)."""
        others = "\n".join(f"- {pos}" for pos in other_positions)
        return (
            f"You are agent {agent_id}. Question: {question}\n"
            f"Other agents said:\n{others}\n"
            f"Your updated answer:"
        )

    def _run_agent(self, agent_id: int, prompt: str) -> AgentState:
        """Generate response for a single agent, extract confidence."""
        prompt_ids = self._encode(prompt)
        response = greedy_generate(
            self.model,
            prompt_ids,
            self.config.max_new_tokens,
            vocab_size=256,  # default; model handles internally
        )
        confidence = extract_confidence(response)
        return AgentState(
            agent_id=agent_id,
            position=response,
            confidence=confidence,
            history=[response],
        )

    def debate(self, question: str) -> DebateResult:
        """Run the full multi-agent debate on a question.

        Round 0: each agent answers independently.
        Rounds 1+: each agent sees others' positions and updates.
        Stops early if agreement >= consensus_threshold.

        Returns DebateResult with the majority position.
        """
        n_agents = self.config.n_agents
        n_rounds = self.config.n_rounds

        # Round 0: independent answers
        agents: list[AgentState] = []
        for agent_id in range(n_agents):
            prompt = self._build_initial_prompt(question, agent_id)
            state = self._run_agent(agent_id, prompt)
            agents.append(state)

        consensus_reached = False
        rounds_run = 1

        # Check agreement after round 0
        positions = [a.position for a in agents]
        agreement = compute_agreement(positions)
        if agreement >= self.config.consensus_threshold:
            consensus_reached = True

        # Rounds 1+
        for round_idx in range(1, n_rounds):
            if consensus_reached:
                break
            rounds_run = round_idx + 1
            new_agents: list[AgentState] = []
            for agent in agents:
                other_positions = [a.position for a in agents if a.agent_id != agent.agent_id]
                prompt = self._build_debate_prompt(question, agent.agent_id, other_positions)
                new_state = self._run_agent(agent.agent_id, prompt)
                new_state.history = agent.history + [new_state.position]
                new_agents.append(new_state)
            agents = new_agents

            positions = [a.position for a in agents]
            agreement = compute_agreement(positions)
            if agreement >= self.config.consensus_threshold:
                consensus_reached = True

        # Determine final answer: majority position (most common, or first if tie)
        position_counts = Counter(positions)
        max_count = max(position_counts.values())
        candidates = [pos for pos, cnt in position_counts.items() if cnt == max_count]
        # Preserve order: pick first among candidates by agent ordering
        final_answer = next(pos for pos in positions if pos in candidates)

        return DebateResult(
            question=question,
            final_answer=final_answer,
            consensus_reached=consensus_reached,
            n_rounds=rounds_run,
            agent_positions=[a.position for a in agents],
            confidence_scores=[a.confidence for a in agents],
        )

    def batch_debate(self, questions: list[str]) -> list[DebateResult]:
        """Debate each question sequentially. Returns list of DebateResult."""
        return [self.debate(q) for q in questions]
