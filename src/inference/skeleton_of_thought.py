"""Skeleton-of-Thought (SoT) Decoding — Bao et al. 2023.

Reference: "Skeleton-of-Thought: Prompting LLMs for Efficient Parallel
Generation" (arXiv:2307.15337).

SoT generates answers in two phases:

1. **Skeleton phase** — generate a short structured plan (list of N points)
   using a normal autoregressive forward, separating points with a special
   separator token.
2. **Expansion phase** — expand each skeleton point into a full answer segment.
   In production the expansions can run as independent batched requests
   (parallel); here they are executed sequentially over a stub model_fn that
   accepts raw token id lists, faithfully simulating the same control flow.

Key benefits over single-pass generation
-----------------------------------------
* Total wall-time ∝ max_point_expansion_length (not sum), given enough compute.
* Each point expansion is independent → straightforward batching.
* Plan gives structured, coherent output even with a weak model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SoTConfig:
    """Hyper-parameters for Skeleton-of-Thought decoding.

    Attributes
    ----------
    max_skeleton_tokens:
        Maximum tokens to generate during the skeleton (plan) phase.
    max_point_tokens:
        Maximum tokens to generate when expanding each skeleton point.
    max_points:
        Hard cap on the number of skeleton points parsed.  Additional
        separator occurrences beyond this limit are ignored.
    skeleton_sep:
        Human-readable label for the separator concept (not used in
        generation logic, which operates on ``sep_token_id``).
    expand_in_parallel:
        When ``True`` the intent is to expand all points in a single
        batched forward pass.  The current sequential implementation
        sets this flag for documentation / downstream scheduling.
    """

    max_skeleton_tokens: int = 128
    max_point_tokens: int = 256
    max_points: int = 8
    skeleton_sep: str = "||"
    expand_in_parallel: bool = True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SkeletonPoint:
    """One point from the skeleton plan, optionally expanded.

    Attributes
    ----------
    index:
        0-based ordinal of this point within the skeleton.
    text:
        Raw token-id list representing this point's skeleton text.
    expanded:
        Token-id list produced during the expansion phase (empty until
        ``expand_point`` has been called for this point).
    """

    index: int
    text: list[int]
    expanded: list[int] = field(default_factory=list)


@dataclass
class SoTResult:
    """Complete output of a Skeleton-of-Thought decode pass.

    Attributes
    ----------
    skeleton_tokens:
        Full token-id sequence produced during the skeleton phase
        (does NOT include the original prompt).
    points:
        Parsed and expanded :class:`SkeletonPoint` objects, one per
        skeleton point, capped at ``SoTConfig.max_points``.
    expanded_text:
        Convenience string representation of all expansions joined
        by a space.  In a real system this would be detokenised; here
        the token ids are formatted as ``"<tok_id>"`` strings.
    total_tokens:
        ``len(skeleton_tokens) + sum(len(p.expanded) for p in points)``
    """

    skeleton_tokens: list[int]
    points: list[SkeletonPoint]
    expanded_text: str
    total_tokens: int


# ---------------------------------------------------------------------------
# Skeleton parser
# ---------------------------------------------------------------------------


class SkeletonParser:
    """Splits a skeleton token sequence into structured points.

    Parameters
    ----------
    sep_token_id:
        The token id that acts as a point delimiter.
    eos_token_id:
        End-of-sequence token id; treated as a hard stop and never
        included in the parsed output.
    """

    def __init__(self, sep_token_id: int, eos_token_id: int) -> None:
        self.sep_token_id = sep_token_id
        self.eos_token_id = eos_token_id

    # ------------------------------------------------------------------

    def parse(self, token_ids: list[int], vocab_size: int) -> list[int]:
        """Return the start index of each point within *token_ids*.

        Points are delimited by ``sep_token_id``.  The first point
        always starts at index 0 (if the sequence is non-empty).

        Parameters
        ----------
        token_ids:
            Raw token id sequence from the skeleton phase.
        vocab_size:
            Not used in the current implementation; reserved for future
            boundary-checking logic.

        Returns
        -------
        list of int
            Start indices (into *token_ids*) for each point segment.
        """
        if not token_ids:
            return []
        starts: list[int] = [0]
        for i, tok in enumerate(token_ids):
            if tok == self.sep_token_id and i + 1 < len(token_ids):
                starts.append(i + 1)
        return starts

    def count_points(self, token_ids: list[int]) -> int:
        """Count the number of skeleton points in *token_ids*.

        This equals the number of ``sep_token_id`` occurrences plus one
        (for the leading segment), unless the sequence is empty.

        Parameters
        ----------
        token_ids:
            Raw token id sequence.

        Returns
        -------
        int
            Number of points (≥ 0).
        """
        if not token_ids:
            return 0
        return 1 + sum(1 for tok in token_ids if tok == self.sep_token_id)


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------


class SkeletonOfThoughtDecoder:
    """Two-phase skeleton-then-expand decoding controller.

    The decoder does **not** hold a reference to a model.  Instead every
    public method accepts a *model_fn* callable with the signature::

        model_fn(token_ids: list[int]) -> Tensor  # shape: (vocab_size,)

    This keeps the decoder model-agnostic and makes unit testing trivial
    (pass a lambda that returns a fixed logit tensor).

    Parameters
    ----------
    config:
        :class:`SoTConfig` instance controlling generation budgets.
    sep_token_id:
        Token id used as the skeleton point separator.
    eos_token_id:
        End-of-sequence token id.  Generation stops immediately when this
        token is sampled (the EOS token is *not* appended to the output).
    """

    def __init__(
        self,
        config: SoTConfig,
        sep_token_id: int,
        eos_token_id: int = 2,
    ) -> None:
        self.config = config
        self.sep_token_id = sep_token_id
        self.eos_token_id = eos_token_id
        self._parser = SkeletonParser(sep_token_id, eos_token_id)

    # ------------------------------------------------------------------
    # Phase 1: skeleton generation
    # ------------------------------------------------------------------

    def generate_skeleton(
        self,
        prompt_tokens: list[int],
        model_fn: Callable[[list[int]], Tensor],
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate the skeleton (plan) token sequence.

        Autoregressively samples tokens from *model_fn* until either the
        EOS token is produced or ``config.max_skeleton_tokens`` tokens have
        been generated.

        Parameters
        ----------
        prompt_tokens:
            The initial context (e.g. the user's question).
        model_fn:
            Callable that accepts the current full token-id list and
            returns a 1-D logit tensor of shape ``(vocab_size,)``.
        temperature:
            Sampling temperature.  Values ≤ 0 are clamped to a small
            positive value to avoid division by zero.

        Returns
        -------
        list of int
            Newly generated token ids (does NOT include *prompt_tokens*).
            Empty if EOS is the very first token sampled.
        """
        temp = max(temperature, 1e-8)
        context = list(prompt_tokens)
        generated: list[int] = []

        for _ in range(self.config.max_skeleton_tokens):
            logits: Tensor = model_fn(context)  # (vocab,)
            if temperature != 1.0:
                logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            next_tok = int(torch.multinomial(probs, num_samples=1).item())
            if next_tok == self.eos_token_id:
                break
            generated.append(next_tok)
            context.append(next_tok)

        return generated

    # ------------------------------------------------------------------
    # Phase 1 post-processing: parse skeleton into points
    # ------------------------------------------------------------------

    def parse_skeleton(self, skeleton_tokens: list[int]) -> list[list[int]]:
        """Split skeleton token ids into per-point token-id lists.

        Parameters
        ----------
        skeleton_tokens:
            Token ids from :meth:`generate_skeleton`.

        Returns
        -------
        list of list of int
            One inner list per skeleton point, capped at
            ``config.max_points``.  Each inner list contains the token ids
            belonging to that point (separator token excluded).
        """
        if not skeleton_tokens:
            return []

        starts = self._parser.parse(skeleton_tokens, vocab_size=0)
        # Build segments between starts (separator token is excluded)
        segments: list[list[int]] = []
        for idx, start in enumerate(starts):
            end = starts[idx + 1] - 1 if idx + 1 < len(starts) else len(skeleton_tokens)
            segment = [
                tok
                for tok in skeleton_tokens[start:end]
                if tok != self.sep_token_id and tok != self.eos_token_id
            ]
            segments.append(segment)

        # Cap at max_points
        return segments[: self.config.max_points]

    # ------------------------------------------------------------------
    # Phase 2: point expansion
    # ------------------------------------------------------------------

    def expand_point(
        self,
        prompt_tokens: list[int],
        point_tokens: list[int],
        model_fn: Callable[[list[int]], Tensor],
        temperature: float = 1.0,
    ) -> list[int]:
        """Expand a single skeleton point into a full answer segment.

        The expansion context is ``prompt_tokens + point_tokens``; up to
        ``config.max_point_tokens`` new tokens are generated.

        Parameters
        ----------
        prompt_tokens:
            Original question / context token ids.
        point_tokens:
            Token ids for this skeleton point (from :meth:`parse_skeleton`).
        model_fn:
            Callable returning logits for the current token sequence.
        temperature:
            Sampling temperature.

        Returns
        -------
        list of int
            Newly generated expansion token ids (EOS excluded).
        """
        temp = max(temperature, 1e-8)
        context = list(prompt_tokens) + list(point_tokens)
        generated: list[int] = []

        for _ in range(self.config.max_point_tokens):
            logits: Tensor = model_fn(context)
            if temperature != 1.0:
                logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            next_tok = int(torch.multinomial(probs, num_samples=1).item())
            if next_tok == self.eos_token_id:
                break
            generated.append(next_tok)
            context.append(next_tok)

        return generated

    # ------------------------------------------------------------------
    # Combined orchestration
    # ------------------------------------------------------------------

    def expand_all(
        self,
        prompt_tokens: list[int],
        skeleton_tokens: list[int],
        model_fn: Callable[[list[int]], Tensor],
        temperature: float = 1.0,
    ) -> SoTResult:
        """Run full skeleton parsing and sequential point expansion.

        Steps
        -----
        1. Call :meth:`parse_skeleton` to get per-point token groups.
        2. For each point call :meth:`expand_point` and record the result.
        3. Assemble and return a :class:`SoTResult`.

        Parameters
        ----------
        prompt_tokens:
            Original prompt token ids.
        skeleton_tokens:
            Token ids from :meth:`generate_skeleton` (not including prompt).
        model_fn:
            Callable returning logits for the current token sequence.
        temperature:
            Sampling temperature passed through to every expansion call.

        Returns
        -------
        SoTResult
            Fully populated result object.
        """
        point_groups = self.parse_skeleton(skeleton_tokens)

        skel_points: list[SkeletonPoint] = []
        for idx, pt_toks in enumerate(point_groups):
            expansion = self.expand_point(
                prompt_tokens=prompt_tokens,
                point_tokens=pt_toks,
                model_fn=model_fn,
                temperature=temperature,
            )
            skel_points.append(SkeletonPoint(index=idx, text=pt_toks, expanded=expansion))

        # Build a human-readable joined string (token ids as placeholder text)
        all_expanded_ids: list[int] = []
        for sp in skel_points:
            all_expanded_ids.extend(sp.expanded)
        expanded_text = " ".join(f"<{t}>" for t in all_expanded_ids)

        total_tokens = len(skeleton_tokens) + sum(len(sp.expanded) for sp in skel_points)

        return SoTResult(
            skeleton_tokens=list(skeleton_tokens),
            points=skel_points,
            expanded_text=expanded_text,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Utility: theoretical speedup estimate
    # ------------------------------------------------------------------

    def speedup_estimate(self, result: SoTResult) -> float:
        """Theoretical parallel-expansion speedup ratio.

        In a truly parallel system the wall-time of the expansion phase is
        proportional to the *longest* individual expansion, not the sum.
        This method estimates the speedup as::

            speedup = total_tokens / max(max_point_tokens, 1)

        A value ≥ 1 indicates SoT would be faster than sequential full-answer
        generation of the same total length.

        Parameters
        ----------
        result:
            A completed :class:`SoTResult`.

        Returns
        -------
        float
            Speedup ratio (always > 0).
        """
        if result.total_tokens == 0:
            return 1.0
        denominator = max(self.config.max_point_tokens, 1)
        return result.total_tokens / denominator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SoTConfig",
    "SkeletonPoint",
    "SoTResult",
    "SkeletonParser",
    "SkeletonOfThoughtDecoder",
]
