"""Constitutional AI: self-critique alignment (Bai et al., 2022 - arXiv:2212.08073).

The model critiques its own responses according to a set of principles
("the constitution"), then revises them. This iterative critique-revision
process produces more aligned outputs without a separate reward model or
human judges.

Pipeline:
  1. Initial response: model answers the user prompt
  2. Critique: model critiques its response against each principle
  3. Revision: model revises based on the critique
  4. Optionally: repeat critique-revision for multiple rounds
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Default constitutional principles (Anthropic-style)
DEFAULT_PRINCIPLES: list[str] = [
    "Is the response harmful, dangerous, or promoting illegal activity? If so, rewrite it to be safe and helpful.",
    "Is the response honest and accurate? If it makes false claims, rewrite it to be truthful or acknowledge uncertainty.",
    "Is the response respectful and free from bias? If not, rewrite it to be respectful to all people.",
    "Is the response clear and helpful? If not, rewrite it to better address the user's actual need.",
]


@dataclass
class ConstitutionalConfig:
    principles: list[str] = field(default_factory=lambda: list(DEFAULT_PRINCIPLES))
    num_rounds: int = 1           # number of critique-revision rounds
    max_critique_tokens: int = 128
    max_revision_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


def _build_critique_prompt(
    original_prompt: str,
    response: str,
    principle: str,
) -> str:
    """Build a prompt asking the model to critique a response."""
    return (
        f"<|user|>\n{original_prompt}<|end|>\n"
        f"<|assistant|>\n{response}<|end|>\n"
        f"<|user|>\nCritique the above response based on this principle: {principle}\n"
        f"Identify specific issues (or say 'No issues' if it looks fine).<|end|>\n"
        f"<|assistant|>\n"
    )


def _build_revision_prompt(
    original_prompt: str,
    response: str,
    critique: str,
    principle: str,
) -> str:
    """Build a prompt asking the model to revise based on a critique."""
    return (
        f"<|user|>\n{original_prompt}<|end|>\n"
        f"<|assistant|>\n{response}<|end|>\n"
        f"<|user|>\nCritique: {critique}\n"
        f"Principle: {principle}\n"
        f"Please revise the response to address the critique.<|end|>\n"
        f"<|assistant|>\n"
    )


class ConstitutionalReviser:
    """Apply constitutional self-critique to improve model responses.

    Args:
        model: AureliusTransformer.
        tokenizer: Tokenizer with encode/decode methods.
        cfg: Constitutional AI configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        cfg: ConstitutionalConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or ConstitutionalConfig()

    def _generate_text(self, prompt: str, max_new_tokens: int) -> str:
        """Generate text from a prompt string."""
        ids = self.tokenizer.encode(prompt)
        if not ids:
            return ""

        input_ids = torch.tensor([ids], dtype=torch.long)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # Truncate to leave room for generation
        max_seq = self.model.config.max_seq_len
        if input_ids.shape[1] >= max_seq - 4:
            input_ids = input_ids[:, -(max_seq - max_new_tokens - 4):]

        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

        new_ids = output[:, input_ids.shape[1]:][0].tolist()
        return self.tokenizer.decode(new_ids)

    def critique(self, prompt: str, response: str, principle: str) -> str:
        """Generate a critique of the response based on a principle."""
        critique_prompt = _build_critique_prompt(prompt, response, principle)
        return self._generate_text(critique_prompt, self.cfg.max_critique_tokens)

    def revise(self, prompt: str, response: str, critique: str, principle: str) -> str:
        """Revise a response given a critique and principle."""
        revision_prompt = _build_revision_prompt(prompt, response, critique, principle)
        return self._generate_text(revision_prompt, self.cfg.max_revision_tokens)

    def apply(self, prompt: str, initial_response: str) -> dict[str, str | list]:
        """Apply all constitutional principles to a response.

        Returns:
            Dict with 'final_response', 'rounds' (list of critique/revision records).
        """
        current_response = initial_response
        rounds = []

        for round_idx in range(self.cfg.num_rounds):
            round_record = {"round": round_idx + 1, "revisions": []}

            for principle in self.cfg.principles:
                critique = self.critique(prompt, current_response, principle)

                # Skip revision if critique indicates no issues
                if "no issues" in critique.lower() or len(critique.strip()) < 5:
                    round_record["revisions"].append({
                        "principle": principle,
                        "critique": critique,
                        "revised": False,
                        "response": current_response,
                    })
                    continue

                revised = self.revise(prompt, current_response, critique, principle)
                if revised.strip():
                    current_response = revised

                round_record["revisions"].append({
                    "principle": principle,
                    "critique": critique,
                    "revised": True,
                    "response": current_response,
                })
                logger.debug("Round %d - Applied principle: %s", round_idx + 1, principle[:50])

            rounds.append(round_record)

        return {
            "initial_response": initial_response,
            "final_response": current_response,
            "rounds": rounds,
        }
