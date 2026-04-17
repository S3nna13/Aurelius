"""Beam search decoding with length normalization.

Batch-level beam search for autoregressive language models.
Pure PyTorch — no external dependencies beyond torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import LongTensor, FloatTensor


@dataclass
class BeamSearchConfig:
    """Configuration for beam search decoding."""

    beam_size: int = 4
    max_new_tokens: int = 50
    length_penalty: float = 1.0  # exponent alpha: score / (length ** alpha)
    eos_token_id: int = 2
    min_length: int = 1  # don't allow EOS before this many generated tokens


class Beam:
    """Represents a single beam hypothesis."""

    def __init__(self, token_ids: List[int], score: float) -> None:
        self.token_ids: List[int] = list(token_ids)
        self.score: float = score

    @property
    def length(self) -> int:
        """Number of tokens in this beam."""
        return len(self.token_ids)

    def normalized_score(self, alpha: float = 1.0) -> float:
        """Return length-normalized score: score / (length ** alpha).

        If length is 0, return raw score to avoid division by zero.
        """
        if self.length == 0:
            return self.score
        return self.score / (self.length ** alpha)

    def extend(self, token_id: int, log_prob: float) -> "Beam":
        """Return a new Beam with token_id appended and log_prob added to score."""
        return Beam(
            token_ids=self.token_ids + [token_id],
            score=self.score + log_prob,
        )

    def __repr__(self) -> str:
        return (
            f"Beam(length={self.length}, score={self.score:.4f}, "
            f"tokens={self.token_ids})"
        )


class BeamSearch:
    """Batch-level beam search decoder for autoregressive language models.

    Args:
        model_fn: Callable that takes a LongTensor of shape (B, T) and returns
                  logits of shape (B, T, V).
        vocab_size: Size of the vocabulary.
        config: BeamSearchConfig instance.
    """

    def __init__(
        self,
        model_fn: Callable[[LongTensor], FloatTensor],
        vocab_size: int,
        config: BeamSearchConfig,
    ) -> None:
        self.model_fn = model_fn
        self.vocab_size = vocab_size
        self.config = config

    def search(self, prompt_ids: LongTensor) -> List[Beam]:
        """Run beam search starting from prompt_ids.

        Args:
            prompt_ids: LongTensor of shape (1, T) — the prompt token ids.

        Returns:
            List of completed Beam objects sorted by normalized_score descending.
            If no beams completed (hit EOS), returns the active beams instead.
        """
        cfg = self.config
        prompt_tokens: List[int] = prompt_ids[0].tolist()

        # Step 1: Initialize beam_size beams from the prompt.
        active_beams: List[Beam] = [Beam(token_ids=prompt_tokens, score=0.0)]
        finished_beams: List[Beam] = []

        for step in range(cfg.max_new_tokens):
            if not active_beams:
                break

            # Step 2: Batch all active beams and run model_fn.
            # Pad to same length if necessary (all start from same prompt,
            # so lengths grow together — same length per step).
            batch_ids = torch.tensor(
                [b.token_ids for b in active_beams],
                dtype=torch.long,
            )  # (num_active, T)

            with torch.no_grad():
                logits = self.model_fn(batch_ids)  # (num_active, T, V)

            # Take log_softmax at last position.
            last_logits = logits[:, -1, :].float()  # (num_active, V)
            log_probs = F.log_softmax(last_logits, dim=-1)  # (num_active, V)

            # Count how many tokens have been generated so far in this search.
            num_generated = step  # 0-indexed: on step 0, we're generating token 1

            # Step 3: Expand each beam with top-beam_size tokens → beam_size² candidates.
            candidates: List[Beam] = []
            for beam_idx, beam in enumerate(active_beams):
                beam_log_probs = log_probs[beam_idx]  # (V,)

                # Enforce min_length: suppress EOS before min_length tokens generated.
                if num_generated < cfg.min_length - 1:
                    beam_log_probs = beam_log_probs.clone()
                    beam_log_probs[cfg.eos_token_id] = float("-inf")

                top_k = min(cfg.beam_size, self.vocab_size)
                topk_log_probs, topk_indices = torch.topk(beam_log_probs, top_k)

                for i in range(top_k):
                    token_id = int(topk_indices[i].item())
                    lp = float(topk_log_probs[i].item())
                    candidates.append(beam.extend(token_id, lp))

            # Step 4: Prune to top beam_size by normalized score.
            candidates.sort(
                key=lambda b: b.normalized_score(cfg.length_penalty),
                reverse=True,
            )
            candidates = candidates[: cfg.beam_size]

            # Step 5: Move completed beams (EOS token) to finished list.
            new_active: List[Beam] = []
            for beam in candidates:
                if beam.token_ids[-1] == cfg.eos_token_id:
                    finished_beams.append(beam)
                else:
                    new_active.append(beam)

            active_beams = new_active

            # Step 6: Stop when all beams finished or max_new_tokens reached.
            if not active_beams:
                break

        # Step 7: If no completed beams, return active beams.
        result = finished_beams if finished_beams else active_beams
        result.sort(
            key=lambda b: b.normalized_score(cfg.length_penalty),
            reverse=True,
        )
        return result


class BeamSearchDecoder:
    """Convenience wrapper around BeamSearch for easy decoding.

    Args:
        model_fn: Callable that takes a LongTensor of shape (B, T) and returns
                  logits of shape (B, T, V).
        vocab_size: Size of the vocabulary.
        config: BeamSearchConfig instance.
    """

    def __init__(
        self,
        model_fn: Callable[[LongTensor], FloatTensor],
        vocab_size: int,
        config: BeamSearchConfig,
    ) -> None:
        self.model_fn = model_fn
        self.vocab_size = vocab_size
        self.config = config
        self._searcher = BeamSearch(model_fn, vocab_size, config)

    def decode(
        self,
        prompt_ids: LongTensor,
        max_new_tokens: Optional[int] = None,
    ) -> LongTensor:
        """Run beam search and return best beam's generated tokens as a tensor.

        Args:
            prompt_ids: LongTensor of shape (1, T) — the prompt token ids.
            max_new_tokens: Override config's max_new_tokens for this call.

        Returns:
            1-D LongTensor containing only the generated tokens (excluding prompt).
        """
        if max_new_tokens is not None:
            # Create a temporary config with overridden max_new_tokens.
            cfg = BeamSearchConfig(
                beam_size=self.config.beam_size,
                max_new_tokens=max_new_tokens,
                length_penalty=self.config.length_penalty,
                eos_token_id=self.config.eos_token_id,
                min_length=self.config.min_length,
            )
            searcher = BeamSearch(self.model_fn, self.vocab_size, cfg)
        else:
            searcher = self._searcher

        beams = searcher.search(prompt_ids)
        best_beam = beams[0]

        prompt_len = prompt_ids.shape[1]
        generated_tokens = best_beam.token_ids[prompt_len:]
        return torch.tensor(generated_tokens, dtype=torch.long)
