"""Vision Token Mixer — early-fusion 10% vision token interleaving.

Implements the early-fusion strategy from Kimi K2.5 §4 (arXiv:2602.02276):
vision tokens are interleaved with text tokens at a constant ratio throughout
training, targeting 10% vision tokens in every batch.

Registry note: There is no top-level src/registry.py in this project.
The data package maintains its own domain-specific registries
(LOADER_REGISTRY, DEDUP_REGISTRY, etc.) in src/data/__init__.py.
VisionTokenMixer is registered in VISION_MIXER_REGISTRY in that file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class VisionTokenMixerConfig:
    """Configuration for the VisionTokenMixer.

    Attributes:
        vision_ratio:    Target fraction of vision tokens in the output sequence.
                         E.g. 0.1 means 10% of output tokens are vision tokens.
        pad_token_id:    Token id used for padding (default 0).
        vision_token_id: Sentinel token id used to represent vision tokens (default 1).
        max_seq_len:     Maximum allowed output sequence length (default 8192).
    """

    vision_ratio: float = 0.1
    pad_token_id: int = 0
    vision_token_id: int = 1
    max_seq_len: int = 8192


class VisionTokenMixer:
    """Data pipeline utility for early-fusion vision-text interleaving.

    Not an nn.Module — this is a pure Python data-preprocessing class.

    The mixer interleaves vision tokens into a text token sequence so that
    vision tokens constitute approximately ``config.vision_ratio`` of the
    output sequence.  Vision tokens are spread evenly throughout the sequence
    rather than being prepended or appended in bulk.

    Args:
        config: A :class:`VisionTokenMixerConfig` instance.
    """

    def __init__(self, config: VisionTokenMixerConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mix(
        self,
        text_ids: List[int],
        vision_ids: List[int],
    ) -> List[int]:
        """Interleave vision tokens into text_ids at the target vision_ratio.

        The output length is ``len(text_ids) + n_vision_inserted`` where
        ``n_vision_inserted`` satisfies::

            n_vision_inserted / (len(text_ids) + n_vision_inserted) ≈ vision_ratio

        Vision tokens are distributed evenly throughout the sequence using a
        uniform striding scheme.

        Args:
            text_ids:   List of text token ids.
            vision_ids: List of vision token ids to interleave.

        Returns:
            A new list with vision tokens evenly interleaved into text_ids.
            If ``vision_ids`` is empty the original ``text_ids`` list is
            returned unchanged.  If ``vision_ids`` has more tokens than the
            ratio allows, they are silently truncated.
        """
        if not vision_ids:
            return list(text_ids)

        ratio = self.config.vision_ratio
        n_text = len(text_ids)

        if ratio <= 0.0:
            # Zero ratio — no vision tokens inserted
            return list(text_ids)

        if ratio >= 1.0:
            # Full ratio — output is purely the vision tokens
            return list(vision_ids)

        # Compute how many vision tokens to insert:
        #   ratio = n_v / (n_t + n_v)  =>  n_v = ratio * n_t / (1 - ratio)
        if n_text == 0:
            # No text — return vision tokens as-is
            return list(vision_ids)

        n_vision_max = int(round(ratio * n_text / (1.0 - ratio)))
        n_vision_max = max(0, n_vision_max)

        # Truncate vision_ids if they exceed the allowed count
        vision_to_insert = list(vision_ids[:n_vision_max])
        n_v = len(vision_to_insert)

        if n_v == 0:
            return list(text_ids)

        # Evenly distribute vision tokens throughout the text sequence.
        # We insert vision tokens at evenly-spaced positions in the *output*
        # index space so that they are spread throughout rather than clustered.
        #
        # Strategy: place vision tokens at positions
        #   floor((i + 0.5) * (n_text + n_v) / n_v)  for i in range(n_v)
        # clipped to valid output indices.  Then build the output by walking
        # through and inserting at those positions.

        output_len = n_text + n_v
        # Compute insertion positions in the output array (0-based)
        insert_positions: Set[int] = set()
        for i in range(n_v):
            pos = int((i + 0.5) * output_len / n_v)
            # Avoid collisions by nudging forward
            while pos in insert_positions:
                pos += 1
            if pos >= output_len:
                pos = output_len - 1
                while pos in insert_positions:
                    pos -= 1
            insert_positions.add(pos)

        # Build result list
        result: List[int] = []
        text_iter = iter(text_ids)
        vision_iter = iter(vision_to_insert)
        for idx in range(output_len):
            if idx in insert_positions:
                result.append(next(vision_iter))
            else:
                result.append(next(text_iter))

        return result

    def mix_batch(
        self,
        batch: List[Dict],
    ) -> List[Dict]:
        """Mix vision tokens into a batch of text sequences.

        Args:
            batch: List of dicts, each containing:
                   - ``"text_ids"``:   list[int]
                   - ``"vision_ids"``: list[int]

        Returns:
            List of dicts, each containing:
            - ``"mixed_ids"``:   list[int] — interleaved sequence
            - ``"vision_mask"``: list[int] — 1 at vision token positions, 0 elsewhere
        """
        results = []
        for sample in batch:
            text_ids: List[int] = sample["text_ids"]
            vision_ids: List[int] = sample["vision_ids"]
            mixed = self.mix(text_ids, vision_ids)
            mask = self._build_vision_mask(mixed, set(vision_ids))
            results.append({"mixed_ids": mixed, "vision_mask": mask})
        return results

    def compute_ratio(
        self,
        mixed_ids: List[int],
        vision_ids_set: Set[int],
    ) -> float:
        """Compute the actual vision token ratio in a mixed sequence.

        Args:
            mixed_ids:     The interleaved token sequence.
            vision_ids_set: Set of token ids that count as vision tokens.

        Returns:
            Fraction of tokens in ``mixed_ids`` that belong to ``vision_ids_set``.
            Returns 0.0 for an empty sequence.
        """
        if not mixed_ids:
            return 0.0
        n_vision = sum(1 for t in mixed_ids if t in vision_ids_set)
        return n_vision / len(mixed_ids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_vision_mask(
        self,
        mixed_ids: List[int],
        vision_ids_set: Set[int],
    ) -> List[int]:
        """Return a binary mask — 1 where a token is a vision token, 0 otherwise."""
        return [1 if t in vision_ids_set else 0 for t in mixed_ids]
