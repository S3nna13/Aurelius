"""Whisper-inspired robust generation with temperature fallback cascade and hallucination detection.

Implements:
- Compression ratio check (zlib) to detect repetitive/hallucinated output
- Mean log-probability check to detect low-confidence generations
- No-repeat n-gram logit processor
- Temperature fallback cascade: try temperatures in order, return first passing result
- RobustDecoder class with generate() and batch_generate()

Reference: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RobustDecodingConfig:
    temperatures: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    compression_ratio_threshold: float = 2.4   # zlib: high = less compressible = not repetitive
    logprob_threshold: float = -1.0            # mean log-prob; below = low confidence
    no_repeat_ngram_size: int = 3              # penalize repeated n-grams
    max_new_tokens: int = 128
    min_new_tokens: int = 1
    beam_size: int = 1                         # 1 = greedy


def compute_compression_ratio(text: str) -> float:
    """Compute zlib compression ratio: len(compressed) / len(original bytes).

    Higher ratio = less compressible = more diverse content = less likely to be hallucination.
    Short, repetitive text compresses very well → low ratio.
    """
    encoded = text.encode("utf-8")
    if len(encoded) == 0:
        return 0.0
    compressed = zlib.compress(encoded)
    return len(compressed) / len(encoded)


def compute_mean_logprob(
    model,
    input_ids: Tensor,
    generated_ids: Tensor,
) -> float:
    """Compute mean log-probability of the generated tokens under the model.

    Args:
        model: language model with forward(input_ids) → logits (B, S, V)
        input_ids: (1, S) prompt token ids
        generated_ids: (1, T_gen) newly generated token ids only

    Returns:
        Mean log-prob per token (negative float).
    """
    T_gen = generated_ids.shape[1]
    if T_gen == 0:
        return 0.0

    # Concatenate prompt + generated for one forward pass
    full_ids = torch.cat([input_ids, generated_ids], dim=1)  # (1, S + T_gen)

    with torch.no_grad():
        _loss, logits, _kv = model(full_ids)  # logits: (1, S + T_gen, V)

    # The logit at position i predicts token i+1.
    # We want log-probs for the generated tokens.
    # Generated tokens start at index S in full_ids, so their predicting logits
    # are at positions [S-1 .. S+T_gen-2] in logits.
    S = input_ids.shape[1]
    # logits for positions that predict generated tokens: [S-1, ..., S+T_gen-2]
    pred_logits = logits[:, S - 1 : S + T_gen - 1, :]  # (1, T_gen, V)
    log_probs = F.log_softmax(pred_logits, dim=-1)      # (1, T_gen, V)

    # Gather log-probs of the actual generated tokens
    token_log_probs = log_probs.gather(
        2, generated_ids.unsqueeze(-1)  # (1, T_gen, 1)
    ).squeeze(-1)  # (1, T_gen)

    mean_lp = token_log_probs.mean().item()
    return float(mean_lp)


def no_repeat_ngram_logit_processor(
    input_ids: Tensor,   # (1, S)
    logits: Tensor,      # (1, V)
    ngram_size: int = 3,
) -> Tensor:
    """Set logits to -inf for tokens that would create a repeated n-gram.

    Args:
        input_ids: (1, S) all token ids so far (prompt + generated)
        logits: (1, V) next token logits
        ngram_size: size of n-grams to check

    Returns:
        Modified logits (same shape).
    """
    S = input_ids.shape[1]
    n = ngram_size

    if S < n - 1:
        return logits  # not enough context

    # The prefix we're trying to extend: last (n-1) tokens
    prefix = input_ids[0, -(n - 1):].tolist()  # list of n-1 ints

    # Find all occurrences of this prefix in input_ids and collect their continuations
    seq = input_ids[0].tolist()
    banned: set[int] = set()
    for i in range(S - (n - 1)):
        if seq[i : i + (n - 1)] == prefix:
            if i + (n - 1) < S:
                banned.add(seq[i + (n - 1)])

    if banned:
        logits = logits.clone()
        for token_id in banned:
            logits[0, token_id] = float("-inf")

    return logits


@dataclass
class GenerationResult:
    token_ids: Tensor           # (1, T_gen) generated tokens
    temperature_used: float
    mean_logprob: float
    compression_ratio: float
    passed_logprob_check: bool
    passed_compression_check: bool
    is_reliable: bool           # both checks passed


def _generate_simple(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float,
    no_repeat_ngram_size: int,
) -> Tensor:
    """Simple autoregressive generation loop.

    Args:
        model: language model, forward(ids) → logits (B, S, V)
        input_ids: (1, S) prompt
        max_new_tokens: maximum tokens to generate
        temperature: 0 → greedy (argmax); > 0 → sample from softmax
        no_repeat_ngram_size: block repeated n-grams (0 to disable)

    Returns:
        (1, T_gen) generated token ids only (prompt NOT included).
    """
    generated = []
    current_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            _loss, logits, _kv = model(current_ids)  # logits: (1, S, V)
            next_logits = logits[:, -1, :]           # (1, V)

            # Apply no-repeat n-gram filtering
            if no_repeat_ngram_size > 0:
                next_logits = no_repeat_ngram_logit_processor(
                    current_ids, next_logits, ngram_size=no_repeat_ngram_size
                )

            if temperature == 0.0:
                # Greedy: argmax
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            else:
                # Sample from temperature-scaled softmax
                scaled = next_logits / temperature
                probs = F.softmax(scaled, dim=-1)  # (1, V)
                next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)

            generated.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=1)

    if not generated:
        return torch.zeros(1, 0, dtype=torch.long)

    return torch.cat(generated, dim=1)  # (1, T_gen)


def generate_with_fallback(
    model,
    input_ids: Tensor,           # (1, S) prompt
    cfg: RobustDecodingConfig,
) -> GenerationResult:
    """Try each temperature in cfg.temperatures; return first result passing both checks.

    Algorithm:
    1. For each temperature T in cfg.temperatures:
       a. Generate cfg.max_new_tokens tokens with temperature T
       b. Decode to string (token ids → int list, joined as space-separated) for compression
       c. Check compression_ratio >= cfg.compression_ratio_threshold
       d. Check mean_logprob >= cfg.logprob_threshold
       e. If both pass: return immediately
    2. After exhausting all temperatures: return result from last temperature anyway.
    """
    last_result: GenerationResult | None = None

    for temperature in cfg.temperatures:
        generated_ids = _generate_simple(
            model=model,
            input_ids=input_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=temperature,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        )

        # Use space-separated token ids as "text" for compression ratio
        token_list = generated_ids[0].tolist()
        text_repr = " ".join(str(t) for t in token_list) if token_list else " "

        compression_ratio = compute_compression_ratio(text_repr)
        mean_lp = compute_mean_logprob(model, input_ids, generated_ids)

        passed_compression = compression_ratio >= cfg.compression_ratio_threshold
        passed_logprob = mean_lp >= cfg.logprob_threshold
        is_reliable = passed_compression and passed_logprob

        last_result = GenerationResult(
            token_ids=generated_ids,
            temperature_used=temperature,
            mean_logprob=mean_lp,
            compression_ratio=compression_ratio,
            passed_logprob_check=passed_logprob,
            passed_compression_check=passed_compression,
            is_reliable=is_reliable,
        )

        if is_reliable:
            return last_result

    # All temperatures exhausted — return last attempt
    return last_result  # type: ignore[return-value]


class RobustDecoder:
    """Robust decoder wrapping generate_with_fallback."""

    def __init__(self, model, cfg: RobustDecodingConfig):
        self.model = model
        self.cfg = cfg

    def generate(self, input_ids: Tensor) -> GenerationResult:
        """Generate from a single prompt, returning a GenerationResult."""
        return generate_with_fallback(self.model, input_ids, self.cfg)

    def batch_generate(self, input_ids_list: list[Tensor]) -> list[GenerationResult]:
        """Generate independently for each prompt in the list."""
        return [self.generate(ids) for ids in input_ids_list]
