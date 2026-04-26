"""Neural Process Language Model for few-shot in-context meta-learning.

Encodes K context examples into a distribution over latent vectors z,
then conditions generation on samples from that distribution.

References:
    Garnelo et al. 2018 (CNP/NP) — https://arxiv.org/abs/1807.01622
    Gordon et al. 2019 (ConvCNP)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# ContextEncoder
# ---------------------------------------------------------------------------


class ContextEncoder(nn.Module):
    """Encodes individual (input_emb, output_emb) pairs to fixed-size representations.

    Each pair is encoded through a small MLP, then the K representations are
    mean-aggregated to produce a single context representation.
    """

    def __init__(self, d_input: int, d_output: int, d_repr: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_input + d_output, d_repr),
            nn.GELU(),
            nn.Linear(d_repr, d_repr),
        )

    def forward(self, input_embs: Tensor, output_embs: Tensor) -> Tensor:
        """Encode K context pairs into a single aggregate representation.

        Args:
            input_embs: Context inputs of shape (K, d_input).
            output_embs: Context outputs of shape (K, d_output).

        Returns:
            Aggregate context representation of shape (d_repr,).
        """
        # Concatenate along last dim: (K, d_input + d_output)
        pairs = torch.cat([input_embs, output_embs], dim=-1)
        # Encode each pair: (K, d_repr)
        representations = self.encoder(pairs)
        # Mean-aggregate over K: (d_repr,)
        return representations.mean(dim=0)


# ---------------------------------------------------------------------------
# LatentEncoder
# ---------------------------------------------------------------------------


class LatentEncoder(nn.Module):
    """Maps aggregate representation to latent distribution parameters (mu, log_sigma)."""

    def __init__(self, d_repr: int, d_latent: int = 16) -> None:
        super().__init__()
        self.mu_net = nn.Linear(d_repr, d_latent)
        self.log_sigma_net = nn.Linear(d_repr, d_latent)

    def forward(self, r: Tensor) -> tuple[Tensor, Tensor]:
        """Map aggregate representation to latent distribution parameters.

        Args:
            r: Aggregate representation of shape (d_repr,) or (B, d_repr).

        Returns:
            Tuple of (mu, log_sigma), each of shape (d_latent,) or (B, d_latent).
        """
        mu = self.mu_net(r)
        log_sigma = self.log_sigma_net(r)
        return mu, log_sigma

    def sample(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
        """Sample from the latent distribution via reparameterization.

        Args:
            mu: Mean of shape (d_latent,) or (B, d_latent).
            log_sigma: Log standard deviation, same shape as mu.

        Returns:
            Latent sample z = mu + exp(log_sigma) * eps, same shape as mu.
        """
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma) * eps


# ---------------------------------------------------------------------------
# NPDecoder
# ---------------------------------------------------------------------------


class NPDecoder(nn.Module):
    """Decodes given a query embedding and latent z to produce output embeddings."""

    def __init__(self, d_query: int, d_latent: int, d_output: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_query + d_latent, 64),
            nn.GELU(),
            nn.Linear(64, d_output),
        )

    def forward(self, query_emb: Tensor, z: Tensor) -> Tensor:
        """Decode query embeddings conditioned on latent z.

        Args:
            query_emb: Query embeddings of shape (B, d_query).
            z: Latent vector of shape (B, d_latent) or (d_latent,) for broadcast.

        Returns:
            Decoded output of shape (B, d_output).
        """
        # Broadcast z to (B, d_latent) if needed
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(query_emb.size(0), -1)
        combined = torch.cat([query_emb, z], dim=-1)
        return self.decoder(combined)


# ---------------------------------------------------------------------------
# NeuralProcessLM
# ---------------------------------------------------------------------------


class NeuralProcessLM(nn.Module):
    """Full Neural Process Language Model.

    Combines context encoding, latent sampling, and decoding for few-shot
    in-context meta-learning.
    """

    def __init__(self, d_model: int, d_latent: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.context_encoder = ContextEncoder(d_model, d_model, d_repr=32)
        self.latent_encoder = LatentEncoder(32, d_latent)
        self.decoder = NPDecoder(d_model, d_latent, d_model)

    def encode_context(
        self, context_inputs: Tensor, context_outputs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode context examples into latent distribution parameters.

        Args:
            context_inputs: Context input embeddings of shape (K, d_model).
            context_outputs: Context output embeddings of shape (K, d_model).

        Returns:
            Tuple of (mu, log_sigma), each of shape (d_latent,).
        """
        r = self.context_encoder(context_inputs, context_outputs)
        mu, log_sigma = self.latent_encoder(r)
        return mu, log_sigma

    def forward(
        self,
        query_embs: Tensor,
        context_inputs: Tensor,
        context_outputs: Tensor,
        n_samples: int = 1,
    ) -> Tensor:
        """Encode context, sample latent z, decode query embeddings.

        Args:
            query_embs: Query embeddings of shape (B, d_model).
            context_inputs: Context input embeddings of shape (K, d_model).
            context_outputs: Context output embeddings of shape (K, d_model).
            n_samples: Number of latent samples; if > 1, returns mean of decoded outputs.

        Returns:
            Decoded output of shape (B, d_model).
        """
        mu, log_sigma = self.encode_context(context_inputs, context_outputs)

        if n_samples == 1:
            z = self.latent_encoder.sample(mu, log_sigma)
            return self.decoder(query_embs, z)

        # Average over multiple latent samples
        outputs = []
        for _ in range(n_samples):
            z = self.latent_encoder.sample(mu, log_sigma)
            outputs.append(self.decoder(query_embs, z))
        return torch.stack(outputs, dim=0).mean(dim=0)

    def elbo_loss(
        self,
        query_embs: Tensor,
        query_targets: Tensor,
        context_inputs: Tensor,
        context_outputs: Tensor,
    ) -> dict[str, Tensor]:
        """Compute Evidence Lower BOund (ELBO) loss for training.

        Encodes the full context+query set for the posterior and the context
        only for the prior, then computes reconstruction loss + KL divergence.

        Args:
            query_embs: Query input embeddings of shape (B, d_model).
            query_targets: Query target embeddings of shape (B, d_model).
            context_inputs: Context input embeddings of shape (K, d_model).
            context_outputs: Context output embeddings of shape (K, d_model).

        Returns:
            Dict with keys 'recon_loss', 'kl_loss', 'total_loss'.
        """
        # Prior: encode context only
        mu_prior, log_sigma_prior = self.encode_context(context_inputs, context_outputs)

        # Posterior: encode context + query jointly
        # Concatenate context and query inputs/outputs to form the full set
        all_inputs = torch.cat([context_inputs, query_embs], dim=0)
        all_outputs = torch.cat([context_outputs, query_targets], dim=0)
        mu_post, log_sigma_post = self.encode_context(all_inputs, all_outputs)

        # Sample from posterior for reconstruction
        z_post = self.latent_encoder.sample(mu_post, log_sigma_post)

        # Reconstruction loss: MSE between decoded output and targets
        decoded = self.decoder(query_embs, z_post)
        recon_loss = nn.functional.mse_loss(decoded, query_targets)

        # KL divergence: KL(q(z|C,T) || p(z|C))
        # = sum(log_sigma_post - log_sigma_prior
        #       + (exp(2*log_sigma_prior) + (mu_prior - mu_post)^2) / (2*exp(2*log_sigma_post))
        #       - 0.5)
        kl_loss = (
            log_sigma_post
            - log_sigma_prior
            + (torch.exp(2 * log_sigma_prior) + (mu_prior - mu_post) ** 2)
            / (2 * torch.exp(2 * log_sigma_post))
            - 0.5
        ).sum()

        total_loss = recon_loss + kl_loss

        return {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }
