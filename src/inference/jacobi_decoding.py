"""Jacobi Decoding: parallel non-autoregressive decoding via fixed-point iteration.

Instead of generating tokens one at a time (autoregressive), Jacobi decoding
treats the full output sequence as a system of equations solved in parallel.
In practice, multiple tokens converge simultaneously giving 1.5–3× speedup.

Algorithm:
1. Initialize output sequence y^(0) = [random uniform tokens] of length max_new_tokens
2. At each Jacobi iteration:
   a. Concatenate [input_ids, y^(t)] -> run one forward pass (single batch)
   b. For each position i in y^(t), read argmax of logits[len(input_ids)-1+i]
   c. y^(t+1) = newly predicted tokens
   d. Check convergence: positions where y^(t+1)[i] == y^(t)[i]
   e. If all converged (or max_iterations reached), stop
3. Return the converged sequence
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JacobiDecoder:
    """Parallel non-autoregressive decoding via Jacobi fixed-point iteration."""

    def __init__(
        self,
        model: nn.Module,
        max_iterations: int = 10,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
    ) -> None:
        """
        Args:
            model: Language model. forward(input_ids) must return either:
                   (loss, logits, pkv) or just logits of shape (B, T, vocab).
            max_iterations: Maximum number of Jacobi iterations.
            temperature: Sampling temperature. 0.0 or < 1e-6 uses greedy argmax.
            top_k: If > 0, restrict sampling to top-k tokens.
            eos_token_id: If all new tokens converge to eos, stop early.
        """
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.top_k = top_k
        self.eos_token_id = eos_token_id

    def _forward_logits(self, input_ids: Tensor) -> Tensor:
        """Run model forward pass and return logits of shape (B, T, vocab).

        Supports models returning (loss, logits, pkv) or just logits.
        """
        with torch.no_grad():
            out = self.model(input_ids)
        # Handle tuple/list return (loss, logits, pkv) or (logits,)
        if isinstance(out, (tuple, list)):
            # Find the tensor that looks like logits: shape (B, T, vocab)
            # Typically the second element for (loss, logits, pkv)
            for item in out:
                if isinstance(item, Tensor) and item.dim() == 3:
                    return item
            # Fallback: return first tensor
            for item in out:
                if isinstance(item, Tensor):
                    return item
            raise ValueError("Model output contains no tensors.")
        elif isinstance(out, Tensor):
            return out
        else:
            raise ValueError(f"Unexpected model output type: {type(out)}")

    def _sample_token(self, logits: Tensor) -> Tensor:
        """Sample or argmax from logits (1D: vocab_size) -> scalar token.

        Args:
            logits: 1D tensor of shape (vocab_size,).

        Returns:
            Scalar int64 tensor (token id).
        """
        if self.temperature < 1e-6:
            return logits.argmax(dim=-1)

        scaled = logits / self.temperature

        if self.top_k > 0:
            k = min(self.top_k, scaled.size(-1))
            topk_vals, _ = scaled.topk(k)
            threshold = topk_vals[-1]
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _initialize_draft(self, batch_size: int, n_new: int, vocab_size: int) -> Tensor:
        """Initialize draft tokens (B, n_new) with random uniform integers.

        When temperature < 1e-6 (greedy mode), uses zeros for determinism.

        Args:
            batch_size: Number of sequences in the batch.
            n_new: Number of new tokens to generate.
            vocab_size: Vocabulary size for random sampling range.

        Returns:
            Integer tensor of shape (B, n_new).
        """
        if self.temperature < 1e-6:
            # Greedy mode: deterministic initialization avoids randomness
            return torch.zeros((batch_size, n_new), dtype=torch.long)
        return torch.randint(0, vocab_size, (batch_size, n_new))

    def decode(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
    ) -> tuple[Tensor, int]:
        """Run Jacobi decoding.

        Args:
            input_ids: (B, T_prompt) prompt token ids.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Tuple of:
                output_ids: (B, T_prompt + max_new_tokens) token ids.
                n_iterations: Number of Jacobi iterations performed.
        """
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        # Detect vocab size via a quick forward pass on the prompt alone
        with torch.no_grad():
            init_logits = self._forward_logits(input_ids)
        vocab_size = init_logits.shape[-1]

        # Initialize draft tokens randomly
        draft = self._initialize_draft(batch_size, max_new_tokens, vocab_size).to(device)

        n_iterations = 0
        for iteration in range(self.max_iterations):
            n_iterations += 1

            # Concatenate prompt + current draft
            full_ids = torch.cat([input_ids, draft], dim=1)  # (B, T_prompt + n_new)

            # Single forward pass
            logits = self._forward_logits(full_ids)  # (B, T_prompt + n_new, vocab)

            # Extract predictions for draft positions
            # Position i in draft corresponds to logit at index (prompt_len - 1 + i)
            # because logit at position t predicts token at position t+1
            new_draft = torch.zeros_like(draft)
            for i in range(max_new_tokens):
                logit_pos = prompt_len - 1 + i  # predict position i in draft
                for b in range(batch_size):
                    new_draft[b, i] = self._sample_token(logits[b, logit_pos])

            # Check convergence: which positions have stabilized
            converged = new_draft == draft  # (B, n_new) bool

            draft = new_draft

            # Stop if all positions across all batch elements have converged
            if converged.all():
                break

            # Stop if eos_token_id is set and all draft positions are eos
            if self.eos_token_id is not None and (draft == self.eos_token_id).all():
                break

        output_ids = torch.cat([input_ids, draft], dim=1)
        return output_ids, n_iterations

    def decode_with_stats(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
    ) -> dict:
        """Run Jacobi decoding with detailed statistics.

        Args:
            input_ids: (B, T_prompt) prompt token ids.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Dict with keys:
                - output_ids: (B, T_prompt + max_new_tokens) Tensor
                - n_iterations: int
                - tokens_per_iteration: List[int] (converged positions per step)
                - convergence_mask: bool Tensor of shape (max_new_tokens,)
        """
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        # Detect vocab size via a quick forward pass on the prompt alone
        with torch.no_grad():
            init_logits = self._forward_logits(input_ids)
        vocab_size = init_logits.shape[-1]

        # Initialize draft tokens randomly
        draft = self._initialize_draft(batch_size, max_new_tokens, vocab_size).to(device)

        n_iterations = 0
        tokens_per_iteration: list[int] = []
        # Track which positions have ever converged (across all batch elems, use mean)
        # We track convergence mask based on last batch element [0] for simplicity
        convergence_mask = torch.zeros(max_new_tokens, dtype=torch.bool, device=device)

        for iteration in range(self.max_iterations):
            n_iterations += 1

            full_ids = torch.cat([input_ids, draft], dim=1)
            logits = self._forward_logits(full_ids)

            new_draft = torch.zeros_like(draft)
            for i in range(max_new_tokens):
                logit_pos = prompt_len - 1 + i
                for b in range(batch_size):
                    new_draft[b, i] = self._sample_token(logits[b, logit_pos])

            # Convergence per position: all batch elements agree
            pos_converged = (new_draft == draft).all(dim=0)  # (n_new,) bool
            n_converged_this_iter = int(pos_converged.sum().item())
            tokens_per_iteration.append(n_converged_this_iter)

            # Update cumulative convergence mask
            convergence_mask = convergence_mask | pos_converged

            draft = new_draft

            if pos_converged.all():
                break

            if self.eos_token_id is not None and (draft == self.eos_token_id).all():
                break

        output_ids = torch.cat([input_ids, draft], dim=1)

        return {
            "output_ids": output_ids,
            "n_iterations": n_iterations,
            "tokens_per_iteration": tokens_per_iteration,
            "convergence_mask": convergence_mask,
        }


def jacobi_decode(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int = 20,
    max_iterations: int = 10,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    """Convenience wrapper for Jacobi decoding returning just output_ids.

    Args:
        model: Language model with forward(input_ids) -> logits or (loss, logits, pkv).
        input_ids: (B, T_prompt) prompt token ids.
        max_new_tokens: Number of new tokens to generate.
        max_iterations: Maximum Jacobi iterations.
        temperature: Sampling temperature.
        eos_token_id: Optional end-of-sequence token id.

    Returns:
        output_ids: (B, T_prompt + max_new_tokens) tensor.
    """
    decoder = JacobiDecoder(
        model=model,
        max_iterations=max_iterations,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )
    output_ids, _ = decoder.decode(input_ids, max_new_tokens=max_new_tokens)
    return output_ids
