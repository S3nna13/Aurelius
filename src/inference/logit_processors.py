"""Composable pipeline of logit transformations applied before sampling."""

from abc import ABC, abstractmethod

import torch


class LogitProcessor(ABC):
    """Base class for logit transformations."""

    @abstractmethod
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            logits: (V,) logit tensor for the next token
            input_ids: (S,) 1D tensor of previously generated token IDs

        Returns:
            (V,) modified logit tensor
        """
        ...


class TemperatureScaling(LogitProcessor):
    """Divide logits by temperature before softmax."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(temperature, 1e-8)

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        return logits / self.temperature


class RepetitionPenalty(LogitProcessor):
    """Divide logits of previously seen tokens by penalty factor.

    For each token t in input_ids:
    - If logits[t] > 0: logits[t] /= penalty
    - If logits[t] < 0: logits[t] *= penalty
    """

    def __init__(self, penalty: float = 1.3):
        self.penalty = penalty

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        logits = logits.clone()
        seen = input_ids.unique()
        score = logits[seen]
        score = torch.where(score > 0, score / self.penalty, score * self.penalty)
        logits[seen] = score
        return logits


class MinPSampling(LogitProcessor):
    """Min-p sampling: keep tokens where p(token) >= min_p * p(max_token).

    Sets logits of excluded tokens to -inf.
    """

    def __init__(self, min_p: float = 0.05):
        self.min_p = min_p

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max()
        threshold = self.min_p * max_prob
        logits = logits.clone()
        logits[probs < threshold] = float("-inf")
        return logits


class TopPLogitsProcessor(LogitProcessor):
    """Nucleus (top-p) filtering: keep tokens comprising the top-p cumulative mass.

    Implementation mirrors the standard HuggingFace `top_p` filter:
    - Sort probabilities descending.
    - Compute cumulative sum.
    - Remove tokens with cumulative - prob >= top_p (i.e., those outside
      the smallest set whose sum >= top_p).
    - To ensure at least one token survives, we *include* the token that
      pushes cumsum over the threshold.
    """

    def __init__(self, top_p: float = 0.9):
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        self.top_p = top_p

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        # Softmax to probabilities
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Determine which tokens to remove: tokens *outside* the top-p set.
        # Condition: (cumulative_probs - sorted_probs) >= top_p
        # This removes tokens where the cumulative *without* the token exceeds top_p.
        remove_mask = (cumulative_probs - sorted_probs) >= self.top_p
        # Invert removal: we want to keep if NOT remove_mask. We'll set logits
        # of removed tokens to -inf.
        # Build new logits: copy then mask
        filtered_logits = logits.clone()
        # Map sorted indices back to original positions
        # For each sorted index position, if remove_mask[i] is True, set that token's logit to -inf
        sorted_remove = remove_mask
        # Use scatter to apply mask: for each position in sorted_indices,
        # set logits[sorted_indices[i]] = -inf if remove
        filtered_logits.scatter_(
            0,
            sorted_indices,
            torch.where(
                sorted_remove,
                torch.full_like(logits, -float("inf")),
                logits.gather(0, sorted_indices),
            ),
        )
        return filtered_logits


class NoRepeatNGram(LogitProcessor):
    """Block tokens that would create an n-gram already seen in input_ids.

    If the last (n-1) tokens of input_ids match a prefix in input_ids,
    block any token that would complete a repeated n-gram.
    """

    def __init__(self, n: int = 3):
        self.n = n

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        S = len(input_ids)
        n = self.n
        if S < n - 1:
            return logits  # not enough context

        # Get the last (n-1) tokens as the prefix we're trying to continue
        prefix = input_ids[-(n - 1) :].tolist()

        # Find all positions where this prefix occurred in input_ids
        banned: set[int] = set()
        for i in range(S - (n - 1)):
            if input_ids[i : i + (n - 1)].tolist() == prefix:
                if i + (n - 1) < S:
                    banned.add(input_ids[i + (n - 1)].item())

        if banned:
            logits = logits.clone()
            for token_id in banned:
                logits[token_id] = float("-inf")
        return logits


class LogitProcessorList(LogitProcessor):
    """Chain multiple processors: apply each in order."""

    def __init__(self, processors: list[LogitProcessor]):
        self.processors = processors

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        for proc in self.processors:
            logits = proc(logits, input_ids)
        return logits
