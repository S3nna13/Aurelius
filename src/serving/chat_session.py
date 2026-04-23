"""Native chat session for direct model interaction.

Manages conversation history, ChatML prompt construction, and token
generation without an external server. Provides a simple Python API
for interactive use and batch inference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ChatML special tokens (must match tokenizer_config.json)
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

DEFAULT_SYSTEM = (
    "You are Aurelius, a helpful, harmless, and honest AI assistant. "
    "Respond thoughtfully and accurately."
)


@dataclass
class Message:
    role: str    # "system", "user", or "assistant"
    content: str


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    eos_token_id: int | None = None
    repetition_penalty: float = 1.0   # > 1 penalizes repeated tokens


def format_chatml_messages(messages: list[Message]) -> str:
    """Format a list of messages into a ChatML prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"{SYSTEM_TOKEN}\n{msg.content}{END_TOKEN}\n")
        elif msg.role == "user":
            parts.append(f"{USER_TOKEN}\n{msg.content}{END_TOKEN}\n")
        elif msg.role == "assistant":
            parts.append(f"{ASSISTANT_TOKEN}\n{msg.content}{END_TOKEN}\n")
    # End with assistant turn prompt (open-ended, model fills in)
    parts.append(ASSISTANT_TOKEN + "\n")
    return "".join(parts)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits.

    Divides logits of already-generated tokens by `penalty` (if > 1.0, reduces their probability).

    Args:
        logits: (vocab_size,) current step logits.
        generated_ids: (seq_len,) all generated token ids so far.
        penalty: Values > 1 discourage repetition.

    Returns:
        Modified logits.
    """
    if penalty == 1.0:
        return logits
    unique_ids = generated_ids.unique()
    logits[unique_ids] = logits[unique_ids] / penalty
    return logits


class ChatSession:
    """Manages a multi-turn conversation with AureliusTransformer.

    Maintains message history, formats ChatML prompts, and calls the model
    directly for generation.

    Args:
        model: AureliusTransformer model.
        tokenizer: Tokenizer with encode(str) -> list[int] and decode(list[int]) -> str.
        system_prompt: System prompt for the session.
        gen_cfg: Default generation configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        system_prompt: str = DEFAULT_SYSTEM,
        gen_cfg: GenerationConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gen_cfg = gen_cfg or GenerationConfig()
        self.messages: list[Message] = [Message("system", system_prompt)]
        self._device = next(model.parameters()).device

    def reset(self, system_prompt: str | None = None) -> None:
        """Clear conversation history, keeping only the system prompt."""
        sys_msg = system_prompt or self.messages[0].content
        self.messages = [Message("system", sys_msg)]

    def _encode_prompt(self) -> torch.Tensor:
        """Encode current conversation history as token ids."""
        prompt = format_chatml_messages(self.messages)
        ids = self.tokenizer.encode(prompt)
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def _truncate_to_fit(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Truncate prompt to leave room for generation within max_seq_len."""
        max_seq = self.model.config.max_seq_len
        max_prompt = max_seq - max_new_tokens - 1
        if input_ids.shape[1] > max_prompt:
            # Keep system prompt tokens + most recent tokens
            input_ids = input_ids[:, -max_prompt:]
        return input_ids

    @torch.no_grad()
    def chat(
        self,
        user_message: str,
        gen_cfg: GenerationConfig | None = None,
    ) -> str:
        """Send a user message and get the assistant response.

        Appends user message to history, generates response, appends
        assistant response to history.

        Args:
            user_message: The user's input text.
            gen_cfg: Override generation config for this turn.

        Returns:
            The assistant's response as a string.
        """
        cfg = gen_cfg or self.gen_cfg
        self.messages.append(Message("user", user_message))

        input_ids = self._encode_prompt()
        input_ids = self._truncate_to_fit(input_ids, cfg.max_new_tokens)

        prompt_len = input_ids.shape[1]

        if cfg.repetition_penalty != 1.0:
            # Use custom generate with repetition penalty
            response_ids = self._generate_with_penalty(input_ids, cfg)
        else:
            response_ids = self.model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_token_id=cfg.eos_token_id,
            )

        new_ids = response_ids[0, prompt_len:].tolist()

        # Strip trailing EOS/END token if present
        if cfg.eos_token_id is not None and new_ids and new_ids[-1] == cfg.eos_token_id:
            new_ids = new_ids[:-1]

        response = self.tokenizer.decode(new_ids)
        self.messages.append(Message("assistant", response))

        return response

    @torch.no_grad()
    def _generate_with_penalty(
        self,
        input_ids: torch.Tensor,
        cfg: GenerationConfig,
    ) -> torch.Tensor:
        """Custom generation loop with repetition penalty."""
        import torch.nn.functional as F

        generated = input_ids.clone()

        for _ in range(cfg.max_new_tokens):
            output = self.model(generated)
            logits = output[1] if isinstance(output, (tuple, list)) else output
            next_logits = logits[0, -1].clone()

            # Apply repetition penalty
            next_logits = apply_repetition_penalty(
                next_logits, generated[0], cfg.repetition_penalty
            )

            # Temperature + top-p sampling
            if cfg.temperature != 1.0:
                next_logits = next_logits / cfg.temperature

            probs = F.softmax(next_logits, dim=-1)

            if cfg.top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                sorted_probs[cumsum - sorted_probs > cfg.top_p] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)

            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if cfg.eos_token_id is not None and next_token.item() == cfg.eos_token_id:
                break

            # Truncate if needed
            if generated.shape[1] >= self.model.config.max_seq_len:
                break

        return generated

    def get_history(self) -> list[dict[str, str]]:
        """Return conversation history as a list of role/content dicts."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def __len__(self) -> int:
        """Number of messages in the session (including system)."""
        return len(self.messages)


def load_model_for_chat(
    checkpoint_dir: str,
    device: str = "cpu",
) -> tuple[nn.Module, object]:
    """Load a model and tokenizer from a checkpoint for chat.

    Args:
        checkpoint_dir: Path to a checkpoint directory (contains model.pt, meta.json).
        device: Device to load onto.

    Returns:
        (model, tokenizer) tuple ready for ChatSession.

    Note:
        This requires a tokenizer saved at checkpoints/tokenizer/ or specified
        separately. Adjust the tokenizer path as needed for your setup.
    """
    from src.training.checkpoint import load_checkpoint
    from src.model.config import AureliusConfig
    from src.model.transformer import AureliusTransformer

    # Load config from meta.json
    import json
    from pathlib import Path
    meta_path = Path(checkpoint_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta_dict = json.load(f)
        config_dict = meta_dict.get("config", {})
        # Filter to valid AureliusConfig fields
        valid_fields = AureliusConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = AureliusConfig(**filtered) if filtered else AureliusConfig()
    else:
        config = AureliusConfig()

    model = AureliusTransformer(config)
    load_checkpoint(model, checkpoint_dir, map_location=device)
    model.to(device)
    model.eval()

    return model, None  # tokenizer must be loaded separately
