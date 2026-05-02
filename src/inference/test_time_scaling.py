"""Test-time compute scaling: majority voting, reward-guided search, and step-level Monte Carlo."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScalingConfig:
    """Configuration for test-time compute scaling."""

    n_samples: int = 8
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    voting_method: str = "majority"  # "majority" | "reward_weighted" | "self_consistency"
    use_step_level: bool = False  # step-level vs answer-level


# ---------------------------------------------------------------------------
# Nucleus sampling
# ---------------------------------------------------------------------------


def nucleus_sample(logits: Tensor, top_p: float = 0.95, temperature: float = 1.0) -> int:
    """Top-p (nucleus) sampling from logits of shape (V,).

    Apply temperature, softmax, sort descending, cumsum, mask, renormalize, sample.
    Returns sampled token id (int).
    """
    # Apply temperature scaling
    scaled = logits / max(temperature, 1e-8)

    # Compute probabilities
    probs = F.softmax(scaled, dim=-1)

    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens whose cumulative prob exceeds top_p (shifted by one to keep the first token)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0

    # Renormalize
    total = sorted_probs.sum()
    if total > 0:
        sorted_probs = sorted_probs / total
    else:
        sorted_probs = torch.ones_like(sorted_probs) / sorted_probs.numel()

    # Sample from the filtered distribution
    sample_idx = torch.multinomial(sorted_probs, num_samples=1).item()
    token_id = sorted_indices[sample_idx].item()
    return int(token_id)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


def generate_samples(
    model: nn.Module,
    prompt_ids: Tensor,
    config: ScalingConfig,
) -> list[list[int]]:
    """Generate config.n_samples complete responses via nucleus sampling.

    Args:
        model: AureliusTransformer. Forward: model(input_ids) -> (loss, logits, past_key_values)
        prompt_ids: (1, T_prompt) — single prompt tensor.
        config: ScalingConfig.

    Returns:
        List of n_samples token id lists (just the new tokens, not the prompt).
    """
    device = prompt_ids.device
    results: list[list[int]] = []

    with torch.no_grad():
        for _ in range(config.n_samples):
            current_ids = prompt_ids.clone()  # (1, T)
            generated: list[int] = []

            for _ in range(config.max_new_tokens):
                _, logits, _ = model(current_ids)  # (1, T, V)
                next_logits = logits[0, -1, :]  # (V,)

                token_id = nucleus_sample(
                    next_logits,
                    top_p=config.top_p,
                    temperature=config.temperature,
                )
                generated.append(token_id)

                # Append new token
                new_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                current_ids = torch.cat([current_ids, new_token], dim=1)

            results.append(generated)

    return results


# ---------------------------------------------------------------------------
# Voting utilities
# ---------------------------------------------------------------------------


def majority_vote(answers: list[str]) -> str:
    """Return the most common answer string.

    Tie-breaking: return the first one encountered.
    """
    if not answers:
        return ""
    counts = Counter(answers)
    max_count = max(counts.values())
    for answer in answers:
        if counts[answer] == max_count:
            return answer
    return answers[0]  # unreachable but satisfies type checker


def self_consistency_vote(answers: list[str], normalize: bool = True) -> dict[str, float]:
    """Count answer frequencies, normalize to probabilities if normalize=True.

    Returns {answer: probability} dict.
    """
    if not answers:
        return {}
    counts = Counter(answers)
    total = len(answers)
    if normalize:
        return {ans: count / total for ans, count in counts.items()}
    return {ans: float(count) for ans, count in counts.items()}


# ---------------------------------------------------------------------------
# Step-level scorer
# ---------------------------------------------------------------------------


class StepLevelScorer:
    """Scores individual reasoning steps (proxy for process reward model)."""

    def __init__(self, model: nn.Module, tokenizer_encode: Callable) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode

    def score_step(self, context: str, step: str) -> float:
        """Mean log-likelihood of step tokens given context.

        Returns float score (higher is better).
        """
        context_ids = self.tokenizer_encode(context)
        step_ids = self.tokenizer_encode(step)

        if not step_ids:
            return 0.0

        context_tensor = torch.tensor([context_ids], dtype=torch.long)
        step_tensor = torch.tensor(step_ids, dtype=torch.long)

        # Build full sequence: context + step
        full_ids = torch.cat(
            [context_tensor, step_tensor.unsqueeze(0)], dim=1
        )  # (1, T_ctx + T_step)

        with torch.no_grad():
            _, logits, _ = self.model(full_ids)  # (1, T, V)

        ctx_len = len(context_ids)
        step_len = len(step_ids)

        # Logits at positions [ctx_len-1 : ctx_len + step_len - 1] predict step tokens
        start = max(ctx_len - 1, 0)
        end = ctx_len + step_len - 1
        step_logits = logits[0, start:end, :]  # (step_len, V)

        log_probs = F.log_softmax(step_logits, dim=-1)  # (step_len, V)
        token_log_probs = log_probs[torch.arange(step_len), step_tensor]  # (step_len,)
        return token_log_probs.mean().item()

    def score_steps(self, context: str, steps: list[str]) -> list[float]:
        """Score each step, return list of floats."""
        return [self.score_step(context, step) for step in steps]


# ---------------------------------------------------------------------------
# Beam MCTS
# ---------------------------------------------------------------------------


class BeamMCTS:
    """Simple beam search with step-level scoring (MCTS-inspired)."""

    def __init__(
        self,
        model: nn.Module,
        scorer: StepLevelScorer,
        config: ScalingConfig,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.config = config

    def expand_step(
        self,
        beam: list[int],
        n_expansions: int = 4,
    ) -> list[tuple[list[int], float]]:
        """Sample n_expansions continuations of up to max_new_tokens // 4 tokens.

        Score each continuation and return list of (token_ids, score).
        """
        step_len = max(1, self.config.max_new_tokens // 4)
        device = torch.device("cpu")
        results: list[tuple[list[int], float]] = []

        with torch.no_grad():
            for _ in range(n_expansions):
                current_ids = torch.tensor([beam], dtype=torch.long, device=device)
                continuation: list[int] = []

                for _ in range(step_len):
                    _, logits, _ = self.model(current_ids)  # (1, T, V)
                    next_logits = logits[0, -1, :]  # (V,)
                    token_id = nucleus_sample(
                        next_logits,
                        top_p=self.config.top_p,
                        temperature=self.config.temperature,
                    )
                    continuation.append(token_id)
                    new_tok = torch.tensor([[token_id]], dtype=torch.long, device=device)
                    current_ids = torch.cat([current_ids, new_tok], dim=1)

                # Score the continuation using log-likelihood directly
                cont_tensor = torch.tensor(continuation, dtype=torch.long)
                beam_tensor = torch.tensor([beam], dtype=torch.long)
                full_ids = torch.cat([beam_tensor, cont_tensor.unsqueeze(0)], dim=1)
                _, logits, _ = self.model(full_ids)
                beam_len = len(beam)
                cont_len = len(continuation)
                start = max(beam_len - 1, 0)
                end = beam_len + cont_len - 1
                cont_logits = logits[0, start:end, :]
                log_probs = F.log_softmax(cont_logits, dim=-1)
                tok_log_probs = log_probs[torch.arange(cont_len), cont_tensor]
                score = tok_log_probs.mean().item()

                results.append((continuation, score))

        return results

    def search(self, prompt_ids: Tensor, n_steps: int = 3) -> list[int]:
        """Multi-step beam expansion, keep top beam at each step.

        Returns best token sequence found (just generated tokens, not the prompt).
        """
        beam = prompt_ids[0].tolist()  # Start from the prompt
        best_sequence: list[int] = []

        for _ in range(n_steps):
            expansions = self.expand_step(beam, n_expansions=4)
            if not expansions:
                break
            # Pick best expansion by score
            best_cont, best_score = max(expansions, key=lambda x: x[1])
            best_sequence.extend(best_cont)
            beam = beam + best_cont

        return best_sequence


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------


class TestTimeScaler:
    """Main interface for test-time compute scaling."""

    # Prevent pytest from attempting to collect this utility class.
    __test__ = False

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
        config: ScalingConfig,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = config

    def generate_and_vote(self, prompt: str) -> tuple[str, dict]:
        """Generate n_samples, decode each, apply voting_method.

        Returns (best_answer, {"samples": list[str], "votes": dict}).
        """
        prompt_ids_list = self.tokenizer_encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids_list], dtype=torch.long)

        raw_samples = generate_samples(self.model, prompt_tensor, self.config)
        decoded = [self.tokenizer_decode(s) for s in raw_samples]

        method = self.config.voting_method

        if method == "majority":
            best = majority_vote(decoded)
            votes = self_consistency_vote(decoded, normalize=False)
        elif method == "self_consistency":
            votes = self_consistency_vote(decoded, normalize=True)
            best = max(votes, key=lambda k: votes[k]) if votes else ""
        elif method == "reward_weighted":
            # Use self-consistency as a proxy (no external reward model)
            votes = self_consistency_vote(decoded, normalize=True)
            best = max(votes, key=lambda k: votes[k]) if votes else ""
        else:
            raise ValueError(f"Unknown voting_method: {method!r}")

        return best, {"samples": decoded, "votes": votes}

    def scale_compute(self, prompt: str, n_samples: int | None = None) -> str:
        """Override n_samples if provided, generate+vote, return best answer."""
        if n_samples is not None:
            original = self.config.n_samples
            self.config.n_samples = n_samples
            try:
                best, _ = self.generate_and_vote(prompt)
            finally:
                self.config.n_samples = original
        else:
            best, _ = self.generate_and_vote(prompt)
        return best
