"""Toolformer-style self-supervised tool-learning data generator.

Implements the data generation pipeline from:
    "Toolformer: Language Models Can Teach Themselves to Use Tools"
    (Schick et al., arXiv:2302.04761)

The pipeline:
  1. Given raw text and a set of tools, annotate candidate tool-call
     positions.
  2. Execute each candidate tool call.
  3. Measure utility: cross-entropy reduction when the tool result is
     prepended to the suffix of the text.
  4. Keep only annotations whose utility exceeds a threshold.
  5. Serialise annotated examples using the paper's [API(...)→result]
     notation.

Everything is pure PyTorch — no scipy, sklearn, HuggingFace, einops,
trl, etc.  Inputs are treated as untrusted: bad tool output or a
crashing tool is silently skipped.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Tool",
    "ToolCallAnnotation",
    "ToolformerConfig",
    "ToolformerDataGenerator",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ToolformerConfig:
    """Hyperparameters and token-notation for Toolformer data generation.

    Attributes
    ----------
    api_token:
        Left delimiter for a tool call, e.g. ``"[API("``.
    close_token:
        Right delimiter that also starts the result section.
    result_sep:
        Separator between call arguments and the tool result.
    utility_threshold:
        Minimum cross-entropy reduction (nats) required to keep a
        candidate annotation.
    max_candidates_per_position:
        How many tool-call candidates to sample per text position.
    seed:
        Optional RNG seed for reproducible sampling.
    """

    api_token: str = "[API("
    close_token: str = ")]"
    result_sep: str = " -> "
    utility_threshold: float = 0.1
    max_candidates_per_position: int = 1
    seed: int | None = None


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    """A callable tool that an agent can invoke.

    Attributes
    ----------
    name:
        Short identifier used in the ``[API(name, ...)]`` notation.
    description:
        Human-readable description of what the tool does.
    fn:
        Callable that accepts keyword arguments and returns a string
        result.  Failures are caught and skipped — do not raise from here.
    """

    name: str
    description: str
    fn: Callable[..., str]


# ---------------------------------------------------------------------------
# Annotation dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolCallAnnotation:
    """Record of a single (position, tool, args, result, utility) tuple.

    Attributes
    ----------
    position:
        Character offset in the original text where the tool call was
        inserted.
    tool_name:
        Name of the tool that was called.
    args:
        Keyword arguments passed to the tool (may be empty).
    result:
        String returned by the tool after execution.
    utility_gain:
        Cross-entropy reduction (loss_without - loss_with, in nats).
        Positive values mean the tool output was helpful.
    """

    position: int
    tool_name: str
    args: dict
    result: str
    utility_gain: float


# ---------------------------------------------------------------------------
# Internal: tiny character-level LM for utility scoring
# ---------------------------------------------------------------------------


class _CharLM(nn.Module):
    """Minimal character-level language model used internally for utility scoring.

    This is intentionally tiny so that unit tests run fast without a GPU.
    It is *not* the Aurelius 1.3B model; it is purely an internal scoring
    proxy.  In production you would substitute the live model's ``forward``
    here via the ``scoring_model`` parameter of
    :class:`ToolformerDataGenerator`.

    Architecture: a single learned embedding table that maps each byte to a
    logit distribution over the 256-byte vocabulary.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, vocab_size)
        nn.init.zeros_(self.embed.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(seq_len, vocab_size)``."""
        return self.embed(ids)

    def cross_entropy(self, text_bytes: bytes) -> float:
        """Compute mean token-level cross-entropy (nats) for *text_bytes*.

        Parameters
        ----------
        text_bytes:
            UTF-8 encoded text to score.

        Returns
        -------
        float
            Mean cross-entropy in nats.  Returns 0.0 for texts shorter
            than 2 bytes (no prediction targets available).
        """
        if len(text_bytes) < 2:
            return 0.0
        ids = torch.tensor(list(text_bytes), dtype=torch.long)
        ctx = ids[:-1]
        tgt = ids[1:]
        with torch.no_grad():
            logits = self.forward(ctx)  # (L-1, vocab)
            loss = F.cross_entropy(logits, tgt, reduction="mean")
        return loss.item()


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


class ToolformerDataGenerator:
    """Generate Toolformer-style annotated fine-tuning data.

    Parameters
    ----------
    config:
        :class:`ToolformerConfig` instance.  If *None*, defaults are used.
    scoring_model:
        Optional scoring model used for utility computation.  Must expose a
        ``cross_entropy(text_bytes: bytes) -> float`` method.  If *None*,
        an internal :class:`_CharLM` is instantiated.

    Example
    -------
    >>> def my_tool_fn(input=""):
    ...     return "42"
    >>> t = Tool("calc", "Arithmetic", my_tool_fn)
    >>> gen = ToolformerDataGenerator()
    >>> anns = gen.annotate("2 + 2 equals", [t])
    >>> filtered = gen.filter_by_utility(anns)
    >>> example = gen.format_training_example("2 + 2 equals", filtered)
    """

    def __init__(
        self,
        config: ToolformerConfig | None = None,
        scoring_model: Any | None = None,
    ) -> None:
        self.config = config or ToolformerConfig()
        if scoring_model is None:
            self._lm: Any = _CharLM()
        else:
            self._lm = scoring_model

        # Seed module-level RNG if requested.  Does not touch the global
        # torch seed (tests manage that with torch.manual_seed).
        if self.config.seed is not None:
            random.seed(self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        text: str,
        tools: Sequence[Tool],
    ) -> list[ToolCallAnnotation]:
        """Annotate a single text with candidate tool calls.

        For each candidate position (word boundary) and each tool, a tool
        call is executed and the utility gain is measured.  All annotations
        are returned, including low-utility ones.  Filtering happens in
        :meth:`filter_by_utility`.

        Parameters
        ----------
        text:
            Raw input text to annotate.
        tools:
            Sequence of :class:`Tool` objects available for annotation.

        Returns
        -------
        list[ToolCallAnnotation]
            All candidate annotations (unfiltered).
        """
        if not text or not tools:
            return []

        annotations: list[ToolCallAnnotation] = []
        positions = self._candidate_positions(text)

        for pos in positions:
            prefix = text[:pos]
            suffix = text[pos:]
            for tool in tools:
                for _ in range(self.config.max_candidates_per_position):
                    args = self._sample_args(tool, prefix)
                    result = self._execute_tool(tool, args)
                    if result is None:
                        # Tool execution failed — skip gracefully.
                        continue
                    gain = self._utility_gain(suffix, result)
                    annotations.append(
                        ToolCallAnnotation(
                            position=pos,
                            tool_name=tool.name,
                            args=args,
                            result=result,
                            utility_gain=gain,
                        )
                    )
        return annotations

    def batch_annotate(
        self,
        texts: Sequence[str],
        tools: Sequence[Tool],
    ) -> list[list[ToolCallAnnotation]]:
        """Annotate multiple texts.

        Parameters
        ----------
        texts:
            Sequence of raw text strings.
        tools:
            Tools available for all texts.

        Returns
        -------
        list[list[ToolCallAnnotation]]
            One annotation list per input text, preserving order.
        """
        return [self.annotate(t, tools) for t in texts]

    def filter_by_utility(
        self,
        annotations: Sequence[ToolCallAnnotation],
        threshold: float | None = None,
    ) -> list[ToolCallAnnotation]:
        """Keep only annotations that exceed the utility threshold.

        Parameters
        ----------
        annotations:
            Candidate annotations produced by :meth:`annotate`.
        threshold:
            Minimum utility gain (exclusive) to keep.  Defaults to
            ``config.utility_threshold``.

        Returns
        -------
        list[ToolCallAnnotation]
            Filtered annotations.
        """
        cutoff = threshold if threshold is not None else self.config.utility_threshold
        return [a for a in annotations if a.utility_gain > cutoff]

    def format_training_example(
        self,
        text: str,
        annotations: Sequence[ToolCallAnnotation],
    ) -> str:
        """Serialise text + annotations into the Toolformer paper notation.

        Tool calls are inserted at their annotated positions using the format::

            [API(tool_name, key=value, ...) -> result]

        Multiple annotations are inserted in reverse position order so that
        earlier character offsets remain valid after each insertion.

        Parameters
        ----------
        text:
            Original (un-annotated) text.
        annotations:
            Annotations to embed; typically the output of
            :meth:`filter_by_utility`.

        Returns
        -------
        str
            Annotated string ready for fine-tuning.
        """
        # Sort descending so insertions don't invalidate earlier offsets.
        sorted_anns = sorted(annotations, key=lambda a: a.position, reverse=True)
        result_text = text
        for ann in sorted_anns:
            call_str = self._format_call(ann)
            result_text = result_text[: ann.position] + call_str + result_text[ann.position :]
        return result_text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _candidate_positions(self, text: str) -> list[int]:
        """Return character positions (end-of-word) for candidate insertions."""
        positions: list[int] = []
        i = 0
        while i < len(text):
            # Advance past non-space chars to find end-of-word.
            while i < len(text) and not text[i].isspace():
                i += 1
            if i > 0:
                positions.append(i)
            # Advance past spaces.
            while i < len(text) and text[i].isspace():
                i += 1
        return positions

    def _sample_args(self, tool: Tool, context: str) -> dict:
        """Heuristically sample arguments for *tool* given *context*.

        The heuristic passes the last non-empty word of *context* as the
        argument under key ``"input"``.  In production this would be driven
        by the live LM sampling structured arguments.
        """
        words = context.split()
        last_word = words[-1] if words else ""
        return {"input": last_word}

    def _execute_tool(self, tool: Tool, args: dict) -> str | None:
        """Execute *tool* with *args*, returning its string output or *None*.

        Catches *all* exceptions so a misbehaving tool cannot crash the
        pipeline.  Converts non-string returns to strings and enforces a
        maximum result length.
        """
        _MAX_RESULT_LEN = 1024
        try:
            raw = tool.fn(**args)
            if raw is None:
                return ""
            result = str(raw)
            if len(result) > _MAX_RESULT_LEN:
                result = result[:_MAX_RESULT_LEN]
            return result
        except Exception:
            return None

    def _utility_gain(self, suffix: str, tool_result: str) -> float:
        """Compute utility gain: CE(suffix) - CE(tool_result + suffix).

        A positive value means prepending the tool result reduced the
        model's uncertainty about the suffix text.

        Parameters
        ----------
        suffix:
            The text after the insertion point.
        tool_result:
            The string returned by the tool.

        Returns
        -------
        float
            Cross-entropy difference in nats.  May be negative.
        """
        if not suffix:
            return 0.0

        suffix_bytes = suffix.encode("utf-8", errors="replace")
        augmented_bytes = (tool_result + suffix).encode("utf-8", errors="replace")

        loss_without = self._lm.cross_entropy(suffix_bytes)
        loss_with = self._lm.cross_entropy(augmented_bytes)
        return loss_without - loss_with

    def _format_call(self, ann: ToolCallAnnotation) -> str:
        """Render one annotation as the Toolformer paper token string.

        Format: ``[API(tool_name, key=value) -> result]``
        """
        cfg = self.config
        args_str = ", ".join(f"{k}={v}" for k, v in ann.args.items())
        call_args = f", {args_str}" if args_str else ""
        return (
            f"{cfg.api_token}{ann.tool_name}{call_args}"
            f"{cfg.close_token}"
            f"{cfg.result_sep}{ann.result}]"
        )
