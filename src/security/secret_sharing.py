"""Additive secret sharing for gradient privacy in federated learning."""

from __future__ import annotations

import torch
from torch import Tensor


class SecretSharing:
    """Splits tensors into additive shares for federated gradient privacy.

    Uses a pure additive scheme: n_parties-1 shares are sampled uniformly at
    random and the final share is set to secret - sum(random_shares), so all
    shares sum exactly to the original secret. Any strict subset of shares
    reveals nothing about the secret because the missing share(s) form a
    one-time pad over the remaining ones.
    """

    def __init__(self, n_parties: int, device: str = "cpu") -> None:
        if n_parties < 2:
            raise ValueError("n_parties must be at least 2")
        self.n_parties = n_parties
        self.device = device

    def share(self, secret: Tensor) -> list[Tensor]:
        """Split *secret* into n_parties additive shares.

        Args:
            secret: Tensor to split.

        Returns:
            List of n_parties tensors that sum exactly to *secret*.
        """
        shares: list[Tensor] = []
        running_sum = torch.zeros_like(secret)
        for _ in range(self.n_parties - 1):
            r = torch.rand_like(secret) * 2 - 1  # uniform in (-1, 1)
            shares.append(r)
            running_sum = running_sum + r
        last_share = secret - running_sum
        shares.append(last_share)
        return shares

    def reconstruct(self, shares: list[Tensor]) -> Tensor:
        """Reconstruct a secret by summing all shares.

        Args:
            shares: List of share tensors produced by :meth:`share`.

        Returns:
            Tensor equal to the sum of all shares.
        """
        result = shares[0].clone()
        for s in shares[1:]:
            result = result + s
        return result

    def share_gradients(self, state_dict: dict[str, Tensor]) -> list[dict[str, Tensor]]:
        """Apply additive sharing to every tensor in *state_dict*.

        Args:
            state_dict: Mapping of parameter names to gradient tensors.

        Returns:
            List of n_parties dicts, one per party. Each dict has the same
            keys as *state_dict* and contains that party's share of each
            tensor.
        """
        per_party: list[dict[str, Tensor]] = [{} for _ in range(self.n_parties)]
        for key, tensor in state_dict.items():
            shares = self.share(tensor)
            for i, s in enumerate(shares):
                per_party[i][key] = s
        return per_party

    def reconstruct_gradients(self, party_dicts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Reconstruct a state dict from per-party share dicts.

        Args:
            party_dicts: List returned by :meth:`share_gradients`.

        Returns:
            Dict mapping each key to the reconstructed tensor.
        """
        keys = list(party_dicts[0].keys())
        result: dict[str, Tensor] = {}
        for key in keys:
            shares = [pd[key] for pd in party_dicts]
            result[key] = self.reconstruct(shares)
        return result

    def verify_reconstruction(self, secret: Tensor, shares: list[Tensor]) -> bool:
        """Return True if *shares* reconstruct to a value close to *secret*.

        Args:
            secret: The original tensor.
            shares: List of share tensors.

        Returns:
            True when the reconstructed tensor matches *secret* within atol=1e-5.
        """
        reconstructed = self.reconstruct(shares)
        return bool(torch.allclose(reconstructed, secret, atol=1e-5))
