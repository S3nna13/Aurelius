"""Chain-of-thought with process reward model scoring and beam search over reasoning steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoTPRMConfig:
    """Configuration for CoT with PRM beam search."""

    n_samples: int = 4          # parallel CoT chains to sample
    max_steps: int = 8          # max reasoning steps per chain
    step_delimiter: str = "\n"  # separator between steps
    temperature: float = 0.8
    max_tokens_per_step: int = 32
    beam_width: int = 2         # keep top-k partial chains by cumulative PRM score


def split_into_steps(text: str, delimiter: str) -> list[str]:
    """Split text by delimiter, strip each step, filter empty.

    Returns list of step strings.
    """
    parts = text.split(delimiter)
    return [p.strip() for p in parts if p.strip()]


def sample_next_step(
    model: nn.Module,
    context_ids: torch.Tensor,
    config: CoTPRMConfig,
) -> tuple[torch.Tensor, str]:
    """Generate up to max_tokens_per_step tokens with temperature sampling.

    Stops early if newline token (token 10 for ASCII '\\n') is generated.

    Returns:
        (generated_ids, decoded_text) — generated_ids is a 1-D tensor.
    """
    NEWLINE_TOKEN = 10  # ASCII '\n'
    generated: list[int] = []
    current_ids = context_ids  # (1, S)

    with torch.no_grad():
        for _ in range(config.max_tokens_per_step):
            _, logits, _ = model(current_ids)  # (1, T, vocab)
            next_logits = logits[0, -1, :]     # (vocab,)

            # Temperature sampling
            if config.temperature > 0:
                next_logits = next_logits / config.temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            # Stop early on newline
            if next_token == NEWLINE_TOKEN:
                break

            # Extend context
            next_id = torch.tensor([[next_token]], dtype=current_ids.dtype, device=current_ids.device)
            current_ids = torch.cat([current_ids, next_id], dim=1)

    generated_ids = torch.tensor(generated, dtype=torch.long, device=context_ids.device)
    # Decode: convert token ids to text via bytes
    decoded_text = bytes(generated_ids.tolist()).decode("latin-1")
    return generated_ids, decoded_text


def score_step_with_prm(
    prm: nn.Module,
    context_ids: torch.Tensor,
    step_ids: torch.Tensor,
) -> float:
    """Run PRM on concatenated context+step and return a score in [0, 1].

    PRM forward: score_logits = prm(combined_ids) -> shape (1, T, 2).
    Uses last token, class 1 probability.
    """
    combined_ids = torch.cat(
        [context_ids, step_ids.unsqueeze(0)], dim=1
    )  # (1, S + step_len)

    with torch.no_grad():
        score_logits = prm(combined_ids)  # (1, T, 2)

    last_logits = score_logits[0, -1, :]  # (2,)
    prob = F.softmax(last_logits, dim=-1)
    return prob[1].item()


class StepBeam:
    """Represents one beam candidate in step-level beam search."""

    def __init__(
        self,
        step_texts: list[str],
        score: float,
        token_ids: torch.Tensor,
    ) -> None:
        self.step_texts = list(step_texts)
        self.score = score
        self.token_ids = token_ids  # (1, S) — full context including prompt

    def add_step(
        self,
        step_text: str,
        step_score: float,
        new_ids: torch.Tensor,
    ) -> "StepBeam":
        """Return new StepBeam with step appended and score accumulated."""
        new_texts = self.step_texts + [step_text]
        new_score = self.score + step_score
        # Append new token ids to existing context
        new_token_ids = torch.cat(
            [self.token_ids, new_ids.unsqueeze(0)], dim=1
        )
        return StepBeam(new_texts, new_score, new_token_ids)

    def text(self) -> str:
        """Join steps with newline."""
        return "\n".join(self.step_texts)


class CoTPRMDecoder:
    """Beam-search decoder that uses a PRM to score reasoning steps."""

    def __init__(
        self,
        model: nn.Module,
        prm: nn.Module,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[list[int]], str],
        config: CoTPRMConfig,
    ) -> None:
        self.model = model
        self.prm = prm
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.config = config

    def decode(self, prompt: str) -> dict:
        """Run beam search over reasoning steps scored by PRM.

        Returns dict with keys: answer, steps, score, n_beams.
        """
        cfg = self.config
        prompt_ids = self.encode_fn(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        # Initialize beams from prompt (beam_width identical starting beams)
        beams: list[StepBeam] = [
            StepBeam(step_texts=[], score=0.0, token_ids=prompt_tensor)
            for _ in range(cfg.beam_width)
        ]

        for _step_idx in range(cfg.max_steps):
            candidates: list[StepBeam] = []

            for beam in beams:
                # Sample next step from this beam's context
                step_ids, step_text = sample_next_step(
                    self.model, beam.token_ids, cfg
                )

                # Score the step with PRM
                step_score = score_step_with_prm(
                    self.prm, beam.token_ids, step_ids
                )

                # Create a new candidate beam
                new_beam = beam.add_step(step_text, step_score, step_ids)
                candidates.append(new_beam)

            # Keep top beam_width candidates by cumulative score
            candidates.sort(key=lambda b: b.score, reverse=True)
            beams = candidates[: cfg.beam_width]

        # Best beam is the first one (highest score)
        best = beams[0]
        return {
            "answer": best.text(),
            "steps": best.step_texts,
            "score": best.score,
            "n_beams": len(beams),
        }
