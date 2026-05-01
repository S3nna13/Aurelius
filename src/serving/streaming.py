"""Token-by-token streaming generation for Aurelius."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class StreamToken:
    """A single token emitted during streaming generation."""

    text: str
    token_id: int
    is_final: bool = False
    finish_reason: str | None = None


@dataclass
class StreamingConfig:
    """Configuration for streaming token generation."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    eos_token_id: int = 2
    chunk_delay_ms: float = 0.0
    decode_fn: Callable[[int], str] | None = None


class TokenStreamer:
    """Streams tokens one at a time from an Aurelius model."""

    def __init__(self, config: StreamingConfig) -> None:
        self.config = config

    def _sample_next_token(self, logits: Tensor) -> int:
        """Apply temperature + top-p nucleus sampling to a (vocab_size,) logit vector.

        Returns:
            Sampled token id as a Python int.
        """
        if logits.dim() != 1:
            raise ValueError(f"Expected 1-D logits, got shape {logits.shape}")

        if self.config.temperature == 0.0:
            return int(logits.argmax(dim=-1).item())

        # Temperature scaling
        if self.config.temperature != 1.0 and self.config.temperature > 0.0:
            logits = logits / self.config.temperature

        # Top-p nucleus sampling
        # Sort ascending so cumsum goes from least to most probable
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Mask tokens that push cumulative probability above (1 - top_p)
        sorted_mask = cumulative_probs <= (1.0 - self.config.top_p)
        # Always keep the highest-probability token
        sorted_mask[..., -1:] = False
        mask = sorted_mask.scatter(0, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

        probs = logits.softmax(dim=-1)
        token_id = int(torch.multinomial(probs, num_samples=1).item())
        return token_id

    @torch.no_grad()
    def stream(self, model, input_ids: Tensor) -> Iterator[StreamToken]:
        """Generate tokens one at a time, yielding a StreamToken per step.

        The model is expected to return (loss, logits, present_key_values).
        KV caching is used when the model provides present_key_values.

        Args:
            model: An AureliusTransformer (or any compatible callable).
            input_ids: (1, seq_len) prompt token ids.

        Yields:
            StreamToken for each generated token. The final token has
            is_final=True and a non-None finish_reason.
        """
        cfg = self.config
        cur_ids = input_ids
        past_key_values = None

        for step in range(cfg.max_new_tokens):
            loss, logits, past_key_values = model(cur_ids, past_key_values=past_key_values)

            # Take the last position's logits: shape (batch, vocab) → (vocab,)
            next_logits = logits[0, -1, :]

            token_id = self._sample_next_token(next_logits)

            # Determine whether this is the final token
            is_eos = token_id == cfg.eos_token_id
            is_last_step = step == cfg.max_new_tokens - 1

            if is_eos:
                finish_reason = "eos"
            elif is_last_step:
                finish_reason = "max_tokens"
            else:
                finish_reason = None

            is_final = is_eos or is_last_step

            # Optional inter-token delay (useful for demo/UI pacing)
            if cfg.chunk_delay_ms > 0.0:
                time.sleep(cfg.chunk_delay_ms / 1000.0)

            decoded = self.config.decode_fn(token_id) if self.config.decode_fn else str(token_id)
            yield StreamToken(
                text=decoded,
                token_id=token_id,
                is_final=is_final,
                finish_reason=finish_reason,
            )

            if is_final:
                return

            # Feed only the new token for the next step (KV cache holds context)
            cur_ids = torch.tensor([[token_id]], dtype=input_ids.dtype, device=input_ids.device)

    def stream_to_callback(
        self,
        model,
        input_ids: Tensor,
        callback: Callable[[StreamToken], None],
    ) -> None:
        """Iterate the stream and invoke callback for every StreamToken.

        Args:
            model: Compatible transformer model.
            input_ids: (1, seq_len) prompt token ids.
            callback: Called once per yielded StreamToken.
        """
        for token in self.stream(model, input_ids):
            callback(token)

    def collect(self, model, input_ids: Tensor) -> str:
        """Consume the full stream and return the concatenated text.

        Only non-final tokens contribute to the returned string; the final
        sentinel token (is_final=True) is excluded so callers receive exactly
        the generated text without a trailing duplicate.

        Args:
            model: Compatible transformer model.
            input_ids: (1, seq_len) prompt token ids.

        Returns:
            Concatenated text of all non-final StreamTokens.
        """
        parts: list[str] = []
        for token in self.stream(model, input_ids):
            if not token.is_final:
                parts.append(token.text)
        return "".join(parts)


class SSEFormatter:
    """Format and parse Server-Sent Events for HTTP streaming responses."""

    def format_token(self, token: StreamToken) -> str:
        """Encode a StreamToken as an SSE data line.

        Returns:
            A string of the form ``"data: {json}\\n\\n"``.
        """
        payload = {
            "text": token.text,
            "token_id": token.token_id,
            "is_final": token.is_final,
            "finish_reason": token.finish_reason,
        }
        return f"data: {json.dumps(payload)}\n\n"

    def format_done(self) -> str:
        """Return the SSE stream-termination sentinel."""
        return "data: [DONE]\n\n"

    def parse_sse_line(self, line: str) -> dict | None:
        """Parse a ``"data: {...}"`` SSE line back to a dict.

        Args:
            line: A single SSE line, e.g. ``"data: {\\\"text\\\": \\\"hi\\\"}"``.

        Returns:
            Parsed dict if the line starts with ``"data: "`` and contains valid
            JSON, otherwise None.
        """
        prefix = "data: "
        if not line.startswith(prefix):
            return None
        payload = line[len(prefix) :]
        try:
            result = json.loads(payload)
            if isinstance(result, dict):
                return result
            return None
        except (json.JSONDecodeError, ValueError):
            return None
