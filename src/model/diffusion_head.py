"""Masked Diffusion Language Model head and utilities.

Implements the masked diffusion (MDLM) approach from Austin et al. 2021:
  - Forward process: gradually replace tokens with [MASK] tokens
  - Backward process: learn to predict original tokens from partially masked sequence
  - Inference: start with all-MASK, iteratively unmask confident tokens
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_TOKEN_ID = 0


class DiffusionSchedule:
    """Noise schedule for masked diffusion.

    At timestep t (0=clean, T=fully masked):
      linear:  mask_rate(t) = t / T
      cosine:  mask_rate(t) = 1 - cos(t/T * pi/2)^2

    Args:
        T: total diffusion steps (default 100)
        schedule: 'linear' | 'cosine'
        mask_token_id: ID of the mask token
    """

    def __init__(
        self,
        T: int = 100,
        schedule: str = "linear",
        mask_token_id: int = MASK_TOKEN_ID,
    ) -> None:
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"schedule must be 'linear' or 'cosine', got {schedule!r}")
        self.T = T
        self.schedule = schedule
        self.mask_token_id = mask_token_id

    def mask_rate(self, t: int | torch.Tensor) -> torch.Tensor:
        """Return mask probability at timestep t. Returns tensor in [0, 1]."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.float()
        ratio = t / self.T
        if self.schedule == "linear":
            rate = ratio
        else:  # cosine
            rate = 1.0 - torch.cos(ratio * (math.pi / 2)) ** 2
        return rate.clamp(0.0, 1.0)

    def add_noise(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Add noise (masking) to token sequence.

        Args:
            x: (B, L) clean token IDs
            t: timestep

        Each token is independently masked with probability mask_rate(t).
        Returns: (B, L) noisy token IDs with some positions == mask_token_id
        """
        rate = self.mask_rate(t).item()
        mask = torch.bernoulli(torch.full_like(x, rate, dtype=torch.float32)).bool()
        noisy = x.clone()
        noisy[mask] = self.mask_token_id
        return noisy

    def get_timesteps(self, batch_size: int, device=None) -> torch.Tensor:
        """Sample random timesteps uniformly from [1, T]. Returns (B,)."""
        return torch.randint(1, self.T + 1, (batch_size,), device=device)


class DiffusionLMHead(nn.Module):
    """Diffusion LM head: predicts original tokens from noisy input.

    Can be added on top of any transformer backbone.
    The backbone processes the noisy token sequence; this head
    predicts the denoised tokens.

    Args:
        d_model: input dimension (from transformer)
        vocab_size: output vocabulary size
        mask_token_id: ID of mask token
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        mask_token_id: int = MASK_TOKEN_ID,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.mask_token_id = mask_token_id

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict token logits from hidden states.

        Args:
            hidden_states: (B, L, d_model)

        Returns:
            (B, L, vocab_size) logits
        """
        return self.proj(self.norm(hidden_states))

    def diffusion_loss(
        self,
        logits: torch.Tensor,
        original_ids: torch.Tensor,
        noisy_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion loss: CE only on MASKED positions.

        We only compute loss where noisy_ids == mask_token_id,
        since we want to predict what was masked.

        Args:
            logits: (B, L, vocab_size) predicted logits
            original_ids: (B, L) clean original tokens
            noisy_ids: (B, L) noisy input (with masks)

        Returns:
            Scalar loss (0.0 if no masked positions)
        """
        mask = noisy_ids == self.mask_token_id  # (B, L)
        n_masked = mask.sum().item()
        if n_masked == 0:
            return logits.sum() * 0.0  # differentiable zero

        B, L, V = logits.shape
        logits_flat = logits.view(B * L, V)
        targets_flat = original_ids.view(B * L)
        mask_flat = mask.view(B * L)

        masked_logits = logits_flat[mask_flat]
        masked_targets = targets_flat[mask_flat]
        return F.cross_entropy(masked_logits, masked_targets)


class MaskedDiffusionTrainer:
    """Train a model with masked diffusion objective.

    Args:
        model: backbone transformer (AureliusTransformer or compatible)
        diffusion_head: DiffusionLMHead
        schedule: DiffusionSchedule
        optimizer: optimizer covering both model and head parameters
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion_head: DiffusionLMHead,
        schedule: DiffusionSchedule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.diffusion_head = diffusion_head
        self.schedule = schedule
        self.optimizer = optimizer

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Run one training step with masked diffusion objective.

        Steps:
            1. Sample random timestep t
            2. Add noise: noisy_ids = schedule.add_noise(input_ids, t)
            3. Run model on noisy_ids to get hidden states
            4. Predict denoised with diffusion_head
            5. Compute diffusion_loss
            6. Backward + step

        Args:
            input_ids: (B, L) clean token IDs

        Returns:
            dict with keys: 'loss' (float), 'mask_rate' (float), 'n_masked' (int)
        """
        self.model.train()
        self.diffusion_head.train()
        self.optimizer.zero_grad()

        device = input_ids.device
        B = input_ids.size(0)

        # 1. Sample one timestep per batch (use the first for noise rate; could extend to per-sample)  # noqa: E501
        t_batch = self.schedule.get_timesteps(B, device=device)
        t = t_batch[0].item()

        # 2. Add noise
        noisy_ids = self.schedule.add_noise(input_ids, t)

        # 3. Run backbone — AureliusTransformer returns (loss, logits, kv)
        result = self.model(noisy_ids)
        if isinstance(result, tuple):
            # AureliusTransformer: (loss_or_none, logits, kv)
            hidden_states = result[1]  # logits slot is actually logits; we need hidden states
            # For a generic backbone that returns logits we need to go via the embedding layer.
            # Since AureliusTransformer exposes internal hidden states only through lm_head,
            # we obtain them by running up to the final norm ourselves.
            hidden_states = self._get_hidden_states(noisy_ids)
        else:
            hidden_states = result

        # 4. Predict
        logits = self.diffusion_head(hidden_states)

        # 5. Loss
        mask_rate_val = self.schedule.mask_rate(t).item()
        n_masked = int((noisy_ids == self.schedule.mask_token_id).sum().item())
        loss = self.diffusion_head.diffusion_loss(logits, input_ids, noisy_ids)

        # 6. Backward
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mask_rate": mask_rate_val,
            "n_masked": n_masked,
        }

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states before the lm_head from the backbone.

        Works with AureliusTransformer by temporarily bypassing lm_head,
        and falls back to running the full forward pass otherwise.
        """
        model = self.model
        # Check if it looks like AureliusTransformer (has embed, layers, norm)
        if hasattr(model, "embed") and hasattr(model, "layers") and hasattr(model, "norm"):
            x = model.embed(input_ids)
            S = input_ids.size(1)
            freqs_cis = model.freqs_cis[:S]
            for layer in model.layers:
                x, _, _ = layer(x, freqs_cis, None, None)
            x = model.norm(x)
            return x
        # Generic fallback: run full forward and return logits as proxy hidden states
        result = model(input_ids)
        if isinstance(result, tuple):
            return result[1]
        return result


class DiffusionDecoder:
    """Decode using masked diffusion: iteratively unmask tokens.

    Args:
        model: backbone transformer
        diffusion_head: DiffusionLMHead
        schedule: DiffusionSchedule
        n_steps: number of decoding steps (less than T for speed)
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion_head: DiffusionLMHead,
        schedule: DiffusionSchedule,
        n_steps: int = 10,
    ) -> None:
        self.model = model
        self.diffusion_head = diffusion_head
        self.schedule = schedule
        self.n_steps = n_steps

    @torch.no_grad()
    def decode(
        self,
        prompt_ids: torch.Tensor,
        gen_len: int,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Iterative decoding via masked diffusion.

        Algorithm:
            1. Start with all-MASK tokens appended to prompt
            2. For each step (from T to 0):
               a. Run model on current sequence
               b. Get logits for masked positions
               c. Sample tokens from distribution
               d. Accept top-confidence tokens (keep others masked)
            3. Return final generated token IDs

        Args:
            prompt_ids: (1, prompt_len) conditioning tokens
            gen_len: number of tokens to generate
            n_steps: number of denoising steps

        Returns:
            (1, gen_len) generated token IDs
        """
        device = prompt_ids.device
        prompt_len = prompt_ids.size(1)

        # 1. Start with all MASK for the generation portion
        gen_ids = torch.full(
            (1, gen_len),
            self.schedule.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # Build timestep sequence from T down to 1
        T = self.schedule.T
        steps = torch.linspace(T, 1, n_steps).long().tolist()

        for step_idx, t in enumerate(steps):
            # Current full sequence: prompt + partially-denoised generation
            full_seq = torch.cat([prompt_ids, gen_ids], dim=1)

            # Get hidden states
            hidden = self._get_hidden_states(full_seq)

            # Logits for generation portion only
            gen_hidden = hidden[:, prompt_len:, :]  # (1, gen_len, d_model)
            logits = self.diffusion_head(gen_hidden)  # (1, gen_len, vocab_size)

            # Mask positions in current generation
            is_masked = gen_ids == self.schedule.mask_token_id  # (1, gen_len)

            if not is_masked.any():
                break

            # Sample from distribution for all masked positions
            probs = logits.softmax(dim=-1)  # (1, gen_len, vocab_size)

            # Get confidence (max prob) for each position
            max_probs, sampled_tokens = probs.max(dim=-1)  # (1, gen_len)

            # Determine how many to unmask at this step
            # Unmask a fraction proportional to progress through the schedule
            remaining_steps = n_steps - step_idx
            n_currently_masked = is_masked.sum().item()
            if remaining_steps <= 1:
                # Last step: unmask everything
                n_to_unmask = n_currently_masked
            else:
                n_to_unmask = max(1, int(n_currently_masked / remaining_steps))

            # Accept the top-confidence masked positions
            masked_confidences = max_probs.clone()
            masked_confidences[~is_masked] = -1.0  # exclude already-unmasked

            # Get top-n indices to unmask
            flat_conf = masked_confidences.view(-1)
            topk_vals, topk_idx = flat_conf.topk(min(n_to_unmask, int(n_currently_masked)))

            # Unmask selected positions
            for idx in topk_idx:
                b = idx.item() // gen_len
                lvl = idx.item() % gen_len
                gen_ids[b, lvl] = sampled_tokens[b, lvl]

        return gen_ids

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states before lm_head from the backbone."""
        model = self.model
        if hasattr(model, "embed") and hasattr(model, "layers") and hasattr(model, "norm"):
            x = model.embed(input_ids)
            S = input_ids.size(1)
            freqs_cis = model.freqs_cis[:S]
            for layer in model.layers:
                x, _, _ = layer(x, freqs_cis, None, None)
            x = model.norm(x)
            return x
        result = model(input_ids)
        if isinstance(result, tuple):
            return result[1]
        return result
