"""Mixture-of-Agents (MoA) inference (Wang et al. 2024).

Multiple proposer models generate independent responses to the same prompt.
An aggregator model synthesises those proposals into a refined final response.
Multiple rounds of refinement are supported: each round the aggregator output
becomes the new context fed to the proposers.

Reference: "Mixture-of-Agents Enhances Large Language Model Capabilities"
           Wang et al., 2024 (arXiv:2406.04692).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoAConfig:
    """Configuration for Mixture-of-Agents inference.

    Attributes:
        n_proposers: Number of independent proposer models.
        n_aggregation_rounds: Rounds of proposal -> aggregation refinement.
        max_new_tokens: Maximum new tokens generated per model call.
        temperature: Sampling temperature for proposers (>0 enables sampling).
        aggregation_prompt: Prefix used by the aggregator to frame proposals.
    """
    n_proposers: int = 3
    n_aggregation_rounds: int = 1
    max_new_tokens: int = 128
    temperature: float = 0.7
    aggregation_prompt: str = "Synthesize these responses into a single best answer:\n"


# ---------------------------------------------------------------------------
# AgentResponse dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """A single response from a proposer agent.

    Attributes:
        agent_id: Index of the proposer that produced this response.
        response: Decoded string response.
        confidence: Log-prob-based confidence proxy (higher is better).
        tokens_generated: Number of new tokens produced.
    """
    agent_id: int
    response: str
    confidence: float = 1.0
    tokens_generated: int = 0


# ---------------------------------------------------------------------------
# Standalone generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate_text(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    tokenizer_decode: Callable,
) -> str:
    """Greedy (argmax) decoding from *model* given a list of prompt token ids.

    Args:
        model: Model whose forward signature returns (loss, logits, past_kv).
        prompt_ids: List of integer token ids representing the prompt.
        max_new_tokens: Number of tokens to generate.
        tokenizer_decode: Callable mapping list[int] -> str.

    Returns:
        Decoded string of the newly generated tokens (not the prompt).
    """
    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # (1, S)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        _, logits, _ = model(input_ids)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]   # (vocab_size,)
        next_token = int(next_logits.argmax(dim=-1).item())
        generated.append(next_token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1
        )

    return tokenizer_decode(generated)


@torch.no_grad()
def sample_generate_text(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    tokenizer_decode: Callable,
) -> str:
    """Temperature-sampling decoding from *model*.

    Args:
        model: Model whose forward returns (loss, logits, past_kv).
        prompt_ids: List of integer token ids.
        max_new_tokens: Number of new tokens to generate.
        temperature: Softmax temperature (>0). Values very close to 0 approach
                     greedy behaviour.
        tokenizer_decode: Callable mapping list[int] -> str.

    Returns:
        Decoded string of the newly generated tokens.
    """
    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # (1, S)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        _, logits, _ = model(input_ids)
        next_logits = logits[0, -1, :]
        temp = max(temperature, 1e-8)
        probs = F.softmax(next_logits / temp, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1
        )

    return tokenizer_decode(generated)


@torch.no_grad()
def compute_response_confidence(
    model,
    prompt_ids: list[int],
    response_ids: list[int],
) -> float:
    """Compute mean log-probability of *response_ids* given *prompt_ids*.

    This serves as a confidence proxy: higher (less negative) values indicate
    that the model assigns higher probability to the response.

    Args:
        model: Model whose forward returns (loss, logits, past_kv).
        prompt_ids: List of prompt token ids.
        response_ids: List of response token ids whose log-prob we evaluate.

    Returns:
        Mean log-probability (float, <= 0). Returns 0.0 if response_ids is empty.
    """
    if not response_ids:
        return 0.0

    # Build full sequence: [prompt | response]
    full_ids = prompt_ids + response_ids
    input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)  # (1, T)

    _, logits, _ = model(input_ids)  # (1, T, vocab_size)

    # Log-probs for positions corresponding to response tokens.
    # Position i in logits predicts token i+1.
    # Response tokens start at index len(prompt_ids) in full_ids.
    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)

    # logits at positions [prompt_len-1 .. prompt_len+resp_len-2] predict response tokens
    log_probs = F.log_softmax(logits[0], dim=-1)  # (T, vocab_size)

    total_log_prob = 0.0
    for i, tok in enumerate(response_ids):
        logit_pos = prompt_len - 1 + i  # position predicting response token i
        if logit_pos < log_probs.shape[0]:
            total_log_prob += log_probs[logit_pos, tok].item()

    return total_log_prob / resp_len


# ---------------------------------------------------------------------------
# Mixture-of-Agents
# ---------------------------------------------------------------------------

class MixtureOfAgents:
    """Mixture-of-Agents inference pipeline.

    Uses multiple LLM instances (proposers) to generate diverse responses, then
    an aggregator (models[0]) to synthesise them into a final answer.

    Args:
        models: List of model instances. models[0] is used as the aggregator.
                The same model may appear multiple times to obtain diversity via
                temperature sampling.
        config: :class:`MoAConfig` controlling the pipeline behaviour.
        tokenizer_encode: Callable str -> list[int].
        tokenizer_decode: Callable list[int] -> str.
    """

    def __init__(
        self,
        models: list,
        config: MoAConfig,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
    ) -> None:
        self.models = models
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    # ------------------------------------------------------------------
    # Core building blocks
    # ------------------------------------------------------------------

    def propose(self, prompt: str) -> list[AgentResponse]:
        """Each proposer independently generates a response to *prompt*.

        Temperature sampling is used so that identical model instances can
        produce diverse outputs.

        Args:
            prompt: Input string prompt.

        Returns:
            List of :class:`AgentResponse`, one per proposer in config.
        """
        prompt_ids = self.tokenizer_encode(prompt)
        # Use as many models as n_proposers, cycling if needed.
        responses: list[AgentResponse] = []

        for agent_id in range(self.config.n_proposers):
            model = self.models[agent_id % len(self.models)]
            response_text = sample_generate_text(
                model,
                prompt_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                tokenizer_decode=self.tokenizer_decode,
            )
            # Encode the response text to get token ids for confidence.
            response_ids = self.tokenizer_encode(response_text) if response_text else []
            confidence = compute_response_confidence(model, prompt_ids, response_ids) if response_ids else 0.0
            responses.append(
                AgentResponse(
                    agent_id=agent_id,
                    response=response_text,
                    confidence=confidence,
                    tokens_generated=len(response_ids) if isinstance(response_ids, list) else self.config.max_new_tokens,
                )
            )

        return responses

    def aggregate(self, prompt: str, responses: list[AgentResponse]) -> str:
        """Aggregate proposer responses into a single answer using models[0].

        Builds a context: aggregation_prompt + numbered responses, then
        generates greedily from models[0].

        Args:
            prompt: Original input prompt string.
            responses: List of :class:`AgentResponse` from proposers.

        Returns:
            Aggregated response string.
        """
        numbered = "\n".join(
            f"{i + 1}. {r.response}" for i, r in enumerate(responses)
        )
        agg_prompt = self.config.aggregation_prompt + numbered
        agg_ids = self.tokenizer_encode(agg_prompt)

        return greedy_generate_text(
            self.models[0],
            agg_ids,
            max_new_tokens=self.config.max_new_tokens,
            tokenizer_decode=self.tokenizer_decode,
        )

    def rank_responses(self, responses: list[AgentResponse]) -> list[AgentResponse]:
        """Sort *responses* by confidence descending (highest first).

        Args:
            responses: List of :class:`AgentResponse`.

        Returns:
            New list sorted by confidence descending.
        """
        return sorted(responses, key=lambda r: r.confidence, reverse=True)

    def run(self, prompt: str) -> dict:
        """Full MoA pipeline: propose -> aggregate (optionally multi-round).

        Args:
            prompt: Input string prompt.

        Returns:
            Dict with keys:
                - "answer": str — final aggregated answer.
                - "n_responses": int — number of proposer responses.
                - "responses": list[str] — individual proposer response strings.
                - "mean_confidence": float — mean confidence across proposers.
        """
        all_responses = self.propose(prompt)
        answer = ""

        for _round in range(self.config.n_aggregation_rounds):
            answer = self.aggregate(prompt, all_responses)
            # For subsequent rounds, treat the aggregated answer as a new proposal.
            if _round < self.config.n_aggregation_rounds - 1:
                agg_ids = self.tokenizer_encode(answer) if answer else []
                agg_confidence = compute_response_confidence(
                    self.models[0],
                    self.tokenizer_encode(prompt),
                    agg_ids,
                ) if agg_ids else 0.0
                all_responses = [
                    AgentResponse(
                        agent_id=-1,
                        response=answer,
                        confidence=agg_confidence,
                        tokens_generated=len(agg_ids) if isinstance(agg_ids, list) else 0,
                    )
                ]

        confidences = [r.confidence for r in self.propose(prompt)]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Re-propose to get fresh responses for reporting
        final_responses = self.propose(prompt)

        return {
            "answer": answer,
            "n_responses": self.config.n_proposers,
            "responses": [r.response for r in final_responses],
            "mean_confidence": mean_confidence,
        }
