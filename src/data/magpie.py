# src/data/magpie.py
"""Magpie self-instruct data synthesis (arXiv:2406.08464).

Generates instruction-following pairs by exploiting the chat template structure
of an aligned model. Feed only the pre-query prefix -- the model generates the
instruction. Then re-wrap and generate the response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.model.transformer import AureliusTransformer


@dataclass
class MagpieConfig:
    """Configuration for Magpie synthesis."""
    max_instruction_tokens: int = 128
    max_response_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    eos_token_id: int = 2  # <|end|> token id -- update when tokenizer is wired in
    # The pre-query prefix: everything up to (but NOT including) the user message content.
    # For the Aurelius chat template: "<|user|>"
    pre_query_prefix: str = "<|user|>"


@dataclass
class MagpieSample:
    """A single generated instruction-response pair."""
    instruction: str
    response: str
    instruction_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)


class MagpieSynthesizer:
    """Generate instruction-response pairs using Magpie self-instruct.

    Args:
        model: The aligned AureliusTransformer model.
        tokenizer: Any object with encode(str)->list[int] and decode(list[int])->str.
        config: Magpie synthesis configuration.
    """

    def __init__(
        self,
        model: AureliusTransformer,
        tokenizer,
        config: MagpieConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or MagpieConfig()
        self.model.train(False)

    def generate_sample(self) -> MagpieSample:
        """Generate one instruction-response pair.

        Step 1: Encode the pre-query prefix and generate the instruction.
        Step 2: Build the full prompt (prefix + instruction + response prefix) and generate the response.

        Returns:
            MagpieSample with instruction and response text + token ids.
        """
        cfg = self.config

        # Step 1: Generate instruction from pre-query prefix only
        prefix_ids = self.tokenizer.encode(cfg.pre_query_prefix)
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long)

        with torch.no_grad():
            instruction_full = self.model.generate(
                prefix_tensor,
                max_new_tokens=cfg.max_instruction_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=cfg.eos_token_id,
            )

        # Extract only the generated instruction tokens (after the prefix)
        instruction_ids = instruction_full[0, len(prefix_ids):].tolist()
        # Strip trailing eos if present
        if instruction_ids and instruction_ids[-1] == cfg.eos_token_id:
            instruction_ids = instruction_ids[:-1]
        instruction_text = self.tokenizer.decode(instruction_ids)

        # Step 2: Build full prompt and generate response
        # Format: <|user|>{instruction}<|end|>\n<|assistant|>
        full_prompt = (
            cfg.pre_query_prefix
            + instruction_text
            + "<|end|>\n<|assistant|>"
        )
        prompt_ids = self.tokenizer.encode(full_prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        with torch.no_grad():
            response_full = self.model.generate(
                prompt_tensor,
                max_new_tokens=cfg.max_response_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=cfg.eos_token_id,
            )

        # Extract only the response tokens
        response_ids = response_full[0, len(prompt_ids):].tolist()
        if response_ids and response_ids[-1] == cfg.eos_token_id:
            response_ids = response_ids[:-1]
        response_text = self.tokenizer.decode(response_ids)

        return MagpieSample(
            instruction=instruction_text,
            response=response_text,
            instruction_ids=instruction_ids,
            response_ids=response_ids,
        )

    def generate_batch(self, n: int) -> list[MagpieSample]:
        """Generate n instruction-response pairs.

        Args:
            n: Number of pairs to generate.

        Returns:
            List of MagpieSample objects.
        """
        return [self.generate_sample() for _ in range(n)]
