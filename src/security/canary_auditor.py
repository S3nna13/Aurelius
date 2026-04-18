"""Canary-based memorization auditor for the Aurelius LLM research platform."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.model.transformer import AureliusTransformer


class CanaryAuditor:
    """Measures training-data memorization by injecting unique canary token sequences.

    Canary sequences are deterministic pseudo-random token strings injected into
    training data. After training, the model's ability to reproduce those sequences
    (measured by cross-entropy loss and perplexity) lower-bounds how much verbatim
    content was memorized. The exposure metric ranks the canary against random
    sequences of equal length to provide a scale-invariant signal.

    Args:
        model: The language model under evaluation.
        vocab_size: Number of tokens in the vocabulary.
        canary_len: Length of each generated canary sequence in tokens.
    """

    def __init__(
        self,
        model: AureliusTransformer,
        vocab_size: int,
        canary_len: int = 16,
    ) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.canary_len = canary_len

    def generate_canary(self, seed: int) -> torch.LongTensor:
        """Generate a deterministic canary token sequence from a seed.

        Args:
            seed: Integer seed controlling the random draw.

        Returns:
            LongTensor of shape (1, canary_len) with values in [0, vocab_size).
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(1, self.canary_len),
            generator=gen,
        )
        return ids

    def canary_loss(self, input_ids: torch.LongTensor) -> float:
        """Cross-entropy loss of the model predicting the canary as a next-token task.

        The sequence is shifted by one position so that each token predicts the
        next. A lower loss indicates that the model has memorized this sequence
        more strongly.

        Args:
            input_ids: LongTensor of shape (1, seq_len) containing canary tokens.

        Returns:
            Scalar float cross-entropy loss. Lower means more memorized.
        """
        self.model.train(False)
        with torch.no_grad():
            loss, _logits, _pkv = self.model(input_ids, labels=input_ids)
        return loss.item()

    def perplexity(self, input_ids: torch.LongTensor) -> float:
        """Per-token perplexity of the model on the canary sequence.

        Perplexity is exp(canary_loss). A value close to 1.0 means the model
        assigns near-certainty to each token, indicating strong memorization.

        Args:
            input_ids: LongTensor of shape (1, seq_len) containing canary tokens.

        Returns:
            Scalar float >= 1.0.
        """
        return math.exp(self.canary_loss(input_ids))

    def exposure(
        self,
        canary_ids: torch.LongTensor,
        n_random_trials: int,
        seed: int,
    ) -> float:
        """Estimate the exposure of a canary sequence relative to random baselines.

        Generates n_random_trials random sequences of the same length, computes
        the cross-entropy loss for each (including the canary), sorts all losses
        in ascending order (lower loss = more memorized), and returns the log2
        of the canary's rank in that sorted list.

        A rank of 1 means the canary is the easiest sequence for the model to
        predict, giving exposure = log2(1) = 0. Higher rank means less
        memorization relative to the random pool.

        Args:
            canary_ids: LongTensor of shape (1, canary_len).
            n_random_trials: Number of random comparison sequences to generate.
            seed: Seed used to generate the random comparison pool.

        Returns:
            Exposure value as a non-negative float.
        """
        seq_len = canary_ids.shape[1]
        canary_loss_val = self.canary_loss(canary_ids)

        gen = torch.Generator()
        gen.manual_seed(seed)

        all_losses: list[float] = [canary_loss_val]
        for _ in range(n_random_trials):
            rand_ids = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(1, seq_len),
                generator=gen,
            )
            all_losses.append(self.canary_loss(rand_ids))

        sorted_losses = sorted(all_losses)
        rank = sorted_losses.index(canary_loss_val) + 1
        return math.log2(rank)

    def is_memorized(
        self,
        canary_ids: torch.LongTensor,
        threshold_perplexity: float,
    ) -> bool:
        """Return True if the canary's perplexity falls below the given threshold.

        A low perplexity means the model assigns high probability to each canary
        token, consistent with verbatim memorization.

        Args:
            canary_ids: LongTensor of shape (1, canary_len).
            threshold_perplexity: Perplexity boundary; sequences below this are
                considered memorized.

        Returns:
            True if perplexity(canary_ids) < threshold_perplexity, else False.
        """
        return self.perplexity(canary_ids) < threshold_perplexity
