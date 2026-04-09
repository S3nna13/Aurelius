"""Constitutional AI training loop: critique -> revision -> SFT on revised outputs."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CAIConfig:
    """Configuration for the CAI training loop."""
    n_critique_rounds: int = 3
    max_new_tokens_critique: int = 64
    max_new_tokens_revision: int = 128
    temperature: float = 1.0
    top_k: int = 50
    sft_lr: float = 1e-5
    sft_batch_size: int = 4


def top_k_sample(logits: Tensor, top_k: int, temperature: float) -> int:
    """Sample a token from logits using top-k filtering and temperature scaling.

    Args:
        logits: (vocab_size,) unnormalized log-probabilities.
        top_k: Number of top logits to keep (rest are zeroed out).
        temperature: Temperature for scaling (higher = more random).

    Returns:
        Sampled token id as a Python int.
    """
    # Clamp top_k to vocab size
    top_k = min(top_k, logits.size(-1))

    # Zero out all but top-k logits
    values, _ = torch.topk(logits, top_k)
    threshold = values[..., -1, None]
    logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Apply temperature
    logits = logits / temperature

    # Softmax and multinomial sample
    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return int(token_id)


def greedy_generate(model: nn.Module, input_ids: Tensor, max_new_tokens: int) -> Tensor:
    """Autoregressive greedy decoding.

    Args:
        model: AureliusTransformer. Forward returns (loss, logits, past_key_values).
        input_ids: (1, T) prompt token ids.
        max_new_tokens: Number of tokens to generate.

    Returns:
        (1, max_new_tokens) newly generated token ids.
    """
    generated = []
    past_key_values = None
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            _, logits, past_key_values = model(
                current_ids, past_key_values=past_key_values
            )
            # Greedy: take argmax of last token's logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            generated.append(next_token)
            current_ids = next_token  # feed only new token with KV cache

    return torch.cat(generated, dim=1)  # (1, max_new_tokens)


def stochastic_generate(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
) -> Tensor:
    """Autoregressive stochastic decoding with top-k sampling.

    Args:
        model: AureliusTransformer. Forward returns (loss, logits, past_key_values).
        input_ids: (1, T) prompt token ids.
        max_new_tokens: Number of tokens to generate.
        top_k: Number of top logits to keep.
        temperature: Sampling temperature.

    Returns:
        (1, max_new_tokens) newly generated token ids.
    """
    generated = []
    past_key_values = None
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            _, logits, past_key_values = model(
                current_ids, past_key_values=past_key_values
            )
            last_logits = logits[0, -1, :]  # (vocab_size,)
            token_id = top_k_sample(last_logits, top_k=top_k, temperature=temperature)
            next_token = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
            generated.append(next_token)
            current_ids = next_token  # feed only new token with KV cache

    return torch.cat(generated, dim=1)  # (1, max_new_tokens)


@dataclass
class CritiqueRevisionPair:
    """Stores a single critique-revision example for SFT training."""
    prompt_ids: Tensor
    initial_ids: Tensor
    critique_ids: Tensor
    revised_ids: Tensor


class CAITrainer:
    """Constitutional AI training loop: critique -> revision -> SFT on revised outputs.

    This is distinct from ConstitutionalReviser (which operates on text strings
    via a tokenizer and applies principle-by-principle revision at inference time).
    CAITrainer works entirely in token-id space, performs multiple critique-revision
    rounds, and computes SFT loss on the revised outputs for training.

    Args:
        model: AureliusTransformer (forward returns (loss, logits, past_key_values)).
        config: CAIConfig.
        tokenizer_encode: Callable[str] -> list[int].
        tokenizer_decode: Callable[list[int]] -> str.
    """

    def __init__(
        self,
        model: nn.Module,
        config: CAIConfig,
        tokenizer_encode,
        tokenizer_decode,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

        # Pre-encode the fixed prompt suffixes
        critique_suffix_ids = tokenizer_encode(
            "Critique the above response for harmlessness and helpfulness:"
        )
        revision_suffix_ids = tokenizer_encode("Revised response:")

        device = next(model.parameters()).device
        self._critique_suffix = torch.tensor(
            [critique_suffix_ids], dtype=torch.long, device=device
        )
        self._revision_suffix = torch.tensor(
            [revision_suffix_ids], dtype=torch.long, device=device
        )

    def generate_critique_prompt(self, prompt_ids: Tensor, response_ids: Tensor) -> Tensor:
        """Build a critique prompt by concatenating prompt + response + critique suffix.

        Args:
            prompt_ids: (1, T_p) original prompt token ids.
            response_ids: (1, T_r) initial response token ids.

        Returns:
            (1, T_p + T_r + T_suffix) concatenated ids.
        """
        device = prompt_ids.device
        suffix = self._critique_suffix.to(device)
        return torch.cat([prompt_ids, response_ids, suffix], dim=1)

    def generate_revision_prompt(
        self,
        prompt_ids: Tensor,
        response_ids: Tensor,
        critique_ids: Tensor,
    ) -> Tensor:
        """Build a revision prompt extending the critique prompt with critique ids + revision suffix.

        Concatenates: prompt + response + critique_suffix + critique_ids + revision_suffix.
        This is structurally a superset of generate_critique_prompt output, so the result
        is always longer than the critique prompt.

        Args:
            prompt_ids: (1, T_p) original prompt token ids.
            response_ids: (1, T_r) initial response token ids.
            critique_ids: (1, T_c) generated critique token ids.

        Returns:
            (1, T_p + T_r + T_critique_suffix + T_c + T_revision_suffix) concatenated ids.
        """
        device = prompt_ids.device
        critique_suffix = self._critique_suffix.to(device)
        revision_suffix = self._revision_suffix.to(device)
        return torch.cat(
            [prompt_ids, response_ids, critique_suffix, critique_ids, revision_suffix], dim=1
        )

    def run_critique_revision(self, prompt_ids: Tensor) -> CritiqueRevisionPair:
        """Run one full critique-revision cycle.

        Steps:
          1. Generate initial response (greedy).
          2. Build critique prompt and generate critique (stochastic).
          3. Build revision prompt and generate revised response (greedy).

        Args:
            prompt_ids: (1, T) prompt token ids.

        Returns:
            CritiqueRevisionPair with all four tensors.
        """
        # 1. Generate initial response greedily
        initial_ids = greedy_generate(
            self.model, prompt_ids, self.config.max_new_tokens_revision
        )

        # 2. Generate critique stochastically
        critique_prompt_ids = self.generate_critique_prompt(prompt_ids, initial_ids)
        critique_ids = stochastic_generate(
            self.model,
            critique_prompt_ids,
            self.config.max_new_tokens_critique,
            top_k=self.config.top_k,
            temperature=self.config.temperature,
        )

        # 3. Generate revised response greedily
        revision_prompt_ids = self.generate_revision_prompt(
            prompt_ids, initial_ids, critique_ids
        )
        revised_ids = greedy_generate(
            self.model, revision_prompt_ids, self.config.max_new_tokens_revision
        )

        return CritiqueRevisionPair(
            prompt_ids=prompt_ids,
            initial_ids=initial_ids,
            critique_ids=critique_ids,
            revised_ids=revised_ids,
        )

    def sft_loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Compute SFT cross-entropy loss on the target portion only.

        Concatenates input_ids and target_ids, runs a forward pass, then
        computes cross-entropy loss only on the positions corresponding to
        target_ids (teacher-forcing on target tokens).

        Args:
            input_ids: (1, T_in) context (prompt) token ids.
            target_ids: (1, T_tgt) target (revised response) token ids.

        Returns:
            Scalar loss tensor.
        """
        T_in = input_ids.shape[1]
        T_tgt = target_ids.shape[1]

        # Concatenate for a single forward pass
        full_ids = torch.cat([input_ids, target_ids], dim=1)  # (1, T_in + T_tgt)

        _, logits, _ = self.model(full_ids)
        # logits: (1, T_in + T_tgt, vocab_size)

        # We want to predict target_ids from positions [T_in-1 : T_in-1+T_tgt]
        # i.e. logits at positions T_in-1 .. T_in+T_tgt-2 predict target tokens 0..T_tgt-1
        target_logits = logits[0, T_in - 1 : T_in - 1 + T_tgt, :]  # (T_tgt, vocab_size)
        target_labels = target_ids[0]  # (T_tgt,)

        loss = F.cross_entropy(target_logits, target_labels)
        return loss

    def train_step(self, prompt_ids_batch: list[Tensor]) -> dict[str, float]:
        """Run critique-revision for each prompt and compute mean SFT loss.

        Does NOT perform an optimizer step — the caller is responsible for
        optimization. Returns loss and pair count for logging.

        Args:
            prompt_ids_batch: List of (1, T_i) prompt tensors.

        Returns:
            {"loss": float, "n_pairs": int}
        """
        total_loss = torch.tensor(0.0)
        device = next(self.model.parameters()).device

        for prompt_ids in prompt_ids_batch:
            prompt_ids = prompt_ids.to(device)
            pair = self.run_critique_revision(prompt_ids)
            loss = self.sft_loss(pair.prompt_ids, pair.revised_ids)
            total_loss = total_loss + loss.cpu()

        mean_loss = total_loss / max(len(prompt_ids_batch), 1)
        return {"loss": float(mean_loss.item()), "n_pairs": len(prompt_ids_batch)}
