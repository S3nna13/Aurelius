"""Adversarial text augmentation for training robustness.

Applies token-level perturbations to input sequences to generate diverse
adversarial training examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AugConfig:
    """Configuration for adversarial text augmentation."""

    p_swap: float = 0.1
    p_delete: float = 0.05
    p_insert: float = 0.05
    p_replace: float = 0.1
    vocab_size: int = 256
    seed: int = 42


class AdversarialAugmenter:
    """Applies stochastic token-level perturbations to produce adversarial variants."""

    def __init__(self, config: AugConfig) -> None:
        self.config = config
        self.generator = torch.Generator()
        self.generator.manual_seed(config.seed)

    # ------------------------------------------------------------------
    # Individual perturbation operations
    # ------------------------------------------------------------------

    def random_swap(self, ids: torch.Tensor) -> torch.Tensor:
        """For each adjacent pair with probability p_swap, swap them.

        Applies left-to-right; a position involved in a swap is not
        re-evaluated for cascading swaps.

        Args:
            ids: 1-D LongTensor of token ids.

        Returns:
            1-D LongTensor of the same length with some adjacent pairs swapped.
        """
        ids = ids.clone()
        n = ids.shape[0]
        if n < 2:
            return ids
        i = 0
        while i < n - 1:
            p = torch.rand(1, generator=self.generator).item()
            if p < self.config.p_swap:
                ids[i], ids[i + 1] = ids[i + 1].clone(), ids[i].clone()
                i += 2  # skip the swapped pair to avoid cascading
            else:
                i += 1
        return ids

    def random_delete(self, ids: torch.Tensor) -> torch.Tensor:
        """Drop each token independently with probability p_delete.

        If all tokens would be dropped, the original sequence is returned.

        Args:
            ids: 1-D LongTensor of token ids.

        Returns:
            1-D LongTensor with length <= len(ids).
        """
        n = ids.shape[0]
        keep_mask = torch.rand(n, generator=self.generator) >= self.config.p_delete
        if keep_mask.sum() == 0:
            return ids.clone()
        return ids[keep_mask]

    def random_insert(self, ids: torch.Tensor) -> torch.Tensor:
        """For each position, with probability p_insert insert a random token before it.

        Args:
            ids: 1-D LongTensor of token ids.

        Returns:
            1-D LongTensor with length >= len(ids).
        """
        n = ids.shape[0]
        insert_mask = torch.rand(n, generator=self.generator) < self.config.p_insert
        n_inserts = int(insert_mask.sum().item())
        if n_inserts == 0:
            return ids.clone()

        random_tokens = torch.randint(
            0,
            self.config.vocab_size,
            (n_inserts,),
            generator=self.generator,
            dtype=torch.long,
        )

        out_tokens = []
        insert_idx = 0
        for pos in range(n):
            if insert_mask[pos]:
                out_tokens.append(random_tokens[insert_idx].unsqueeze(0))
                insert_idx += 1
            out_tokens.append(ids[pos].unsqueeze(0))
        return torch.cat(out_tokens)

    def random_replace(self, ids: torch.Tensor) -> torch.Tensor:
        """Replace each token with a random token with probability p_replace.

        Args:
            ids: 1-D LongTensor of token ids.

        Returns:
            1-D LongTensor of the same length.
        """
        n = ids.shape[0]
        replace_mask = torch.rand(n, generator=self.generator) < self.config.p_replace
        n_replace = int(replace_mask.sum().item())
        result = ids.clone()
        if n_replace == 0:
            return result
        random_tokens = torch.randint(
            0,
            self.config.vocab_size,
            (n_replace,),
            generator=self.generator,
            dtype=torch.long,
        )
        result[replace_mask] = random_tokens
        return result

    # ------------------------------------------------------------------
    # Composed augmentation
    # ------------------------------------------------------------------

    def augment(
        self,
        ids: torch.Tensor,
        operations: tuple[str, ...] = ("swap", "delete", "insert", "replace"),
    ) -> torch.Tensor:
        """Apply specified operations in order.

        Args:
            ids: 1-D LongTensor of token ids.
            operations: Ordered tuple of operation names to apply.  Valid
                values are ``"swap"``, ``"delete"``, ``"insert"``,
                ``"replace"``.

        Returns:
            1-D LongTensor after all requested perturbations have been applied.
        """
        result = ids.clone()
        op_map = {
            "swap": self.random_swap,
            "delete": self.random_delete,
            "insert": self.random_insert,
            "replace": self.random_replace,
        }
        for op in operations:
            result = op_map[op](result)
        return result

    def augment_batch(
        self,
        batch_ids: torch.Tensor,
        n_augments: int = 1,
    ) -> torch.Tensor:
        """Augment each sample in a batch n_augments times.

        Each sample is augmented independently n_augments times.  The results
        are padded with 0s to the maximum sequence length in the output batch.

        Args:
            batch_ids: 2-D LongTensor of shape (B, T).
            n_augments: Number of augmented versions to produce per sample.

        Returns:
            2-D LongTensor of shape (B * n_augments, T_max) where T_max is the
            maximum sequence length across all augmented samples.
        """
        b = batch_ids.shape[0]
        augmented_seqs = []
        for i in range(b):
            for _ in range(n_augments):
                aug = self.augment(batch_ids[i])
                augmented_seqs.append(aug)

        t_max = max(seq.shape[0] for seq in augmented_seqs)
        out = torch.zeros(b * n_augments, t_max, dtype=torch.long)
        for idx, seq in enumerate(augmented_seqs):
            out[idx, : seq.shape[0]] = seq
        return out
