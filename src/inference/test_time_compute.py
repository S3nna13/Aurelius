"""Test-time compute scaling: best-of-n, iterative refinement, and tree search."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TTCConfig:
    """Configuration for test-time compute scaling."""
    n_samples: int = 8
    budget: int = 512
    strategy: str = "best_of_n"   # "best_of_n" | "tree_search" | "iterative_refinement"
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Sampling utility
# ---------------------------------------------------------------------------

def sample_with_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Sample a token id from logits using temperature scaling.

    Args:
        logits: shape (V,) — raw logits for the next token.
        temperature: scaling factor. Higher → more uniform; lower → more peaked.

    Returns:
        Scalar int tensor containing the sampled token id.
    """
    temperature = max(temperature, 1e-8)
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).squeeze(0)
    return token_id


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------

def greedy_generate(model: nn.Module, input_ids: Tensor, max_tokens: int) -> Tensor:
    """Greedily generate up to max_tokens new tokens appended to input_ids.

    Args:
        model: AureliusTransformer. Forward: model(input_ids) -> (loss, logits, pkv)
        input_ids: (1, T) — input prompt tensor.
        max_tokens: maximum number of new tokens to generate.

    Returns:
        Tensor of shape (1, T + max_tokens) — the full sequence including prompt.
    """
    current_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            _, logits, _ = model(current_ids)    # (1, T, V)
            next_logits = logits[0, -1, :]       # (V,)
            next_token = next_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)  # (1, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)

    return current_ids


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model: nn.Module, input_ids: Tensor) -> float:
    """Compute perplexity of a sequence under the model.

    perplexity = exp(mean NLL over all tokens except the first).

    Args:
        model: AureliusTransformer. Forward: model(input_ids) -> (loss, logits, pkv)
        input_ids: (1, T) — sequence tensor.

    Returns:
        Perplexity as a positive float.
    """
    with torch.no_grad():
        _, logits, _ = model(input_ids)    # (1, T, V)

    # Shift: predict token[t+1] from logits[t]
    shift_logits = logits[0, :-1, :]   # (T-1, V)
    shift_targets = input_ids[0, 1:]   # (T-1,)

    log_probs = F.log_softmax(shift_logits, dim=-1)                         # (T-1, V)
    nll = -log_probs[torch.arange(shift_targets.size(0)), shift_targets]    # (T-1,)
    mean_nll = nll.mean().item()
    return math.exp(mean_nll)


# ---------------------------------------------------------------------------
# Best-of-N generation
# ---------------------------------------------------------------------------

def best_of_n_generate(
    model: nn.Module,
    input_ids: Tensor,
    config: TTCConfig,
    score_fn: Optional[Callable[[nn.Module, Tensor], float]] = None,
) -> Tensor:
    """Generate n_samples completions and return the one with the highest score.

    Args:
        model: AureliusTransformer.
        input_ids: (1, T) — prompt tensor.
        config: TTCConfig controlling n_samples, budget, and temperature.
        score_fn: (model, sequence) -> float. Defaults to negative perplexity
                  (higher score = lower perplexity = better sequence).

    Returns:
        Best completion as a tensor of shape (1, T + generated_len).
    """
    if score_fn is None:
        score_fn = lambda m, seq: -compute_perplexity(m, seq)

    max_new = max(1, config.budget // max(config.n_samples, 1))
    best_seq: Optional[Tensor] = None
    best_score: float = float("-inf")

    with torch.no_grad():
        for _ in range(config.n_samples):
            current_ids = input_ids.clone()

            for _ in range(max_new):
                _, logits, _ = model(current_ids)     # (1, T, V)
                next_logits = logits[0, -1, :]        # (V,)
                next_token = sample_with_temperature(next_logits, config.temperature)
                next_token = next_token.view(1, 1)
                current_ids = torch.cat([current_ids, next_token], dim=1)

            score = score_fn(model, current_ids)
            if score > best_score:
                best_score = score
                best_seq = current_ids

    assert best_seq is not None
    return best_seq


# ---------------------------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------------------------

def iterative_refine(
    model: nn.Module,
    input_ids: Tensor,
    config: TTCConfig,
    n_iterations: int = 3,
) -> Tensor:
    """Iteratively regenerate the suffix of the current best sequence.

    At each iteration, keep the first half of the generated tokens and
    resample the remaining suffix using temperature sampling.

    Args:
        model: AureliusTransformer.
        input_ids: (1, T) — prompt tensor.
        config: TTCConfig.
        n_iterations: number of refinement passes.

    Returns:
        Refined sequence tensor of shape approximately (1, T + suffix_len).
    """
    prompt_len = input_ids.size(1)
    suffix_len = max(4, config.budget // max(config.n_samples, 1))

    # Start with a greedy baseline
    current_best = greedy_generate(model, input_ids, suffix_len)
    best_score = -compute_perplexity(model, current_best)

    keep_len = max(1, suffix_len // 2)

    with torch.no_grad():
        for _ in range(n_iterations):
            # Keep prompt + first keep_len generated tokens, resample the rest
            pivot = prompt_len + keep_len
            prefix = current_best[:, :pivot]
            remaining = suffix_len - keep_len

            candidate = prefix.clone()
            for _ in range(remaining):
                _, logits, _ = model(candidate)
                next_logits = logits[0, -1, :]
                next_token = sample_with_temperature(next_logits, config.temperature)
                candidate = torch.cat([candidate, next_token.view(1, 1)], dim=1)

            score = -compute_perplexity(model, candidate)
            if score > best_score:
                best_score = score
                current_best = candidate

    return current_best


# ---------------------------------------------------------------------------
# TestTimeScaler
# ---------------------------------------------------------------------------

class TestTimeScaler:
    """Unified interface for test-time compute scaling strategies."""

    # Prevent pytest from attempting to collect this utility class.
    __test__ = False

    def __init__(self, model: nn.Module, config: TTCConfig) -> None:
        self.model = model
        self.config = config

    def generate(
        self,
        input_ids: Tensor,
        score_fn: Optional[Callable[[nn.Module, Tensor], float]] = None,
    ) -> tuple[Tensor, float]:
        """Dispatch to the configured strategy and return (best_sequence, best_score).

        Args:
            input_ids: (1, T) — prompt tensor.
            score_fn: optional scorer; defaults to negative perplexity.

        Returns:
            (best_sequence, best_score) tuple.
        """
        if score_fn is None:
            score_fn = lambda m, seq: -compute_perplexity(m, seq)

        strategy = self.config.strategy

        if strategy == "best_of_n":
            best_seq = best_of_n_generate(
                self.model, input_ids, self.config, score_fn
            )
        elif strategy == "iterative_refinement":
            best_seq = iterative_refine(
                self.model, input_ids, self.config
            )
        elif strategy == "tree_search":
            # Simple tree search: run best_of_n with greedy scoring as a baseline
            best_seq = best_of_n_generate(
                self.model, input_ids, self.config, score_fn
            )
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy!r}")

        best_score = score_fn(self.model, best_seq)
        return best_seq, best_score
