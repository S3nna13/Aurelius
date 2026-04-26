"""Model fingerprinting for the Aurelius LLM research platform.

Embeds a verifiable ownership signature into model weights via a deterministic
backdoor-style approach: specific key inputs always produce a target output
pattern, distinguishing the owner's model from copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FingerprintConfig:
    """Configuration for model fingerprinting."""

    n_keys: int = 8
    key_len: int = 4
    seed: int = 42
    embed_strength: float = 0.1


class ModelFingerprint:
    """Embeds and verifies an ownership signature in model weights."""

    def __init__(self, model: nn.Module, config: FingerprintConfig) -> None:
        self.model = model
        self.config = config
        self.key_ids, self.target_ids = self.generate_keys()

    def generate_keys(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate deterministic key sequences and target class ids from seed.

        Returns:
            key_ids: (n_keys, key_len) LongTensor of random token ids.
            target_ids: (n_keys,) LongTensor of random class ids.
        """
        rng = torch.Generator()
        rng.manual_seed(self.config.seed)

        vocab_size = self._vocab_size()

        key_ids = torch.randint(
            0,
            vocab_size,
            (self.config.n_keys, self.config.key_len),
            generator=rng,
        )
        target_ids = torch.randint(
            0,
            vocab_size,
            (self.config.n_keys,),
            generator=rng,
        )
        return key_ids, target_ids

    def embed_fingerprint(self, optimizer: torch.optim.Optimizer, n_steps: int = 50) -> list[float]:
        """Fine-tune the model to memorize key-to-target mappings.

        For each step, runs a forward pass on key_ids, computes cross-entropy
        loss between last-token logits and target_ids, and updates weights.

        Args:
            optimizer: A PyTorch optimizer already initialized on the model.
            n_steps: Number of gradient steps to take.

        Returns:
            List of per-step scalar loss values.
        """
        self.model.train()
        losses: list[float] = []

        device = self._device()
        key_ids = self.key_ids.to(device)
        target_ids = self.target_ids.to(device)

        for _ in range(n_steps):
            optimizer.zero_grad()
            _, logits, _ = self.model(key_ids)
            # logits: (n_keys, key_len, vocab_size) -- use last token position
            last_logits = logits[:, -1, :]  # (n_keys, vocab_size)
            loss = F.cross_entropy(last_logits, target_ids)
            loss = loss * self.config.embed_strength
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    def verify(self, threshold: float = 0.5) -> tuple[bool, float]:
        """Verify the fingerprint by checking key-to-target prediction accuracy.

        Args:
            threshold: Minimum accuracy to consider the fingerprint verified.

        Returns:
            (verified, accuracy): verified is True if accuracy > threshold.
        """
        self.model.train(False)
        device = self._device()
        key_ids = self.key_ids.to(device)
        target_ids = self.target_ids.to(device)

        with torch.no_grad():
            _, logits, _ = self.model(key_ids)
            last_logits = logits[:, -1, :]  # (n_keys, vocab_size)
            predictions = last_logits.argmax(dim=-1)  # (n_keys,)
            correct = (predictions == target_ids).sum().item()

        accuracy = float(correct) / self.config.n_keys
        verified = accuracy > threshold
        return verified, accuracy

    def extract_signature(self, layer_idx: int = 0) -> torch.Tensor:
        """Extract a compact signature from the first linear weight matrix.

        Args:
            layer_idx: Index used to select among linear layers.

        Returns:
            A 2-element tensor [mean, std] of the selected weight matrix.
        """
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        layer = linear_layers[layer_idx]
        w = layer.weight.detach().float()
        mean_val = torch.mean(w)
        std_val = torch.std(w)
        return torch.stack([mean_val, std_val])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vocab_size(self) -> int:
        """Infer vocab size from the model config or embedding layer."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
            return self.model.config.vocab_size
        for m in self.model.modules():
            if isinstance(m, nn.Embedding):
                return m.num_embeddings
        raise RuntimeError("Cannot determine vocab_size from model.")

    def _device(self) -> torch.device:
        """Return the device of the first model parameter."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
