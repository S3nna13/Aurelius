"""Best-of-N sampling with self-certainty scoring (arXiv:2502.18581).

Generates N completions and selects the one with the highest self-certainty score.

Self-certainty = -KL(model distribution || uniform)
              = mean(log(vocab_size * p(t|x, y<t)))   [higher = more confident]
              = mean(log_p + log(vocab_size))
              = log(vocab_size) + mean(log_p)

A higher score means the model assigns more peaked probability distributions
to its own outputs -- a proxy for generation quality without external reward models.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.model.transformer import AureliusTransformer


def self_certainty_score(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    """Compute the self-certainty score for a generated completion.

    Self-certainty = mean(log(V * p(t|context))) over generated tokens
                   = mean(log_p + log(V))
                   = log(V) + mean(log_p)

    where p(t|context) is the model's probability for the actual generated token,
    V is vocab size.

    Higher = more confident = better quality proxy.

    Args:
        logits: Model logits for the generated sequence, shape (seq_len, vocab_size).
        token_ids: Generated token ids, shape (seq_len,).

    Returns:
        Self-certainty score (higher = better).
    """
    if len(token_ids) == 0:
        return float("-inf")

    log_probs = F.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)
    vocab_size = logits.shape[-1]

    # Gather log prob of each actual generated token
    token_log_probs = log_probs[torch.arange(len(token_ids)), token_ids]  # (seq_len,)

    # Self-certainty = mean(log(V) + log_p) = log(V) + mean(log_p)
    return math.log(vocab_size) + token_log_probs.mean().item()


class BestOfN:
    """Best-of-N generation with self-certainty scoring.

    Generates N independent completions for the same prompt and returns
    the one with the highest self-certainty score.

    Args:
        model: The AureliusTransformer model.
        n: Number of completions to generate per prompt.
    """

    def __init__(self, model: AureliusTransformer, n: int = 4) -> None:
        self.model = model
        self.n = n
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate N completions and return the best by self-certainty.

        Args:
            input_ids: Prompt token ids, shape (1, prompt_len).
                       Note: batch_size must be 1 (we generate N completions serially).
            max_new_tokens: Max tokens to generate per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            eos_token_id: Optional EOS token to stop generation.

        Returns:
            Best completion token ids, shape (1, prompt_len + generated_len).
        """
        if input_ids.shape[0] != 1:
            raise ValueError("BestOfN requires batch_size=1 (generates N completions serially)")

        prompt_len = input_ids.shape[1]
        best_ids = None
        best_score = float("-inf")

        for _ in range(self.n):
            # Generate one completion
            completion = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )  # (1, prompt_len + generated_len)

            generated_ids = completion[0, prompt_len:]  # (generated_len,)
            generated_len = len(generated_ids)

            if generated_len == 0:
                continue

            # Score: run a forward pass on the completion to get logits for generated tokens.
            # Logits at positions [prompt_len-1 : prompt_len + generated_len - 1]
            # predict tokens at positions [prompt_len : prompt_len + generated_len].
            _, logits, _ = self.model(completion)  # (1, prompt_len + generated_len, vocab)
            gen_logits = logits[0, prompt_len - 1 : prompt_len + generated_len - 1, :]
            # gen_logits[i] is the logit for token at position prompt_len + i

            score = self_certainty_score(gen_logits, generated_ids)

            if score > best_score:
                best_score = score
                best_ids = completion

        if best_ids is None:
            # Fallback: return the prompt unchanged
            return input_ids

        return best_ids
