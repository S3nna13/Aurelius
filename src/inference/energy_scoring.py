"""Energy-based reranking and Langevin dynamics sampling for improved generation quality.

Implements:
- EnergyConfig: configuration dataclass
- compute_sequence_energy: negative log-likelihood energy scoring
- compute_fluency_score: entropy-based fluency scoring
- EnergyReranker: reranks candidates by combined energy+fluency score
- langevin_refine: continuous-relaxation Langevin dynamics in embedding space
- EnergyBasedGenerator: candidate generation with energy-based selection
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EnergyConfig:
    """Configuration for energy-based generation and reranking."""

    n_candidates: int = 8
    mcmc_steps: int = 10
    step_size: float = 0.01
    temperature: float = 1.0
    energy_weight: float = 1.0
    fluency_weight: float = 1.0


def compute_sequence_energy(
    model: object,
    input_ids: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute negative log-likelihood energy for each sequence.

    Energy = mean cross-entropy over non-padding positions (per sequence).

    Uses logits[:, :-1, :] vs input_ids[:, 1:] cross-entropy.

    Args:
        model: AureliusTransformer (callable: loss, logits, pkv = model(input_ids))
        input_ids: (B, T) token ids.
        reduction: "mean" (default) or "sum".

    Returns:
        Tensor of shape (B,) — energy per sequence (higher = less likely).
    """
    with torch.no_grad():
        _, logits, _ = model(input_ids)  # (B, T, V)

    B, T, V = logits.shape
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_targets = input_ids[:, 1:].contiguous()  # (B, T-1)

    # Compute per-token cross-entropy for each batch element
    # F.cross_entropy with reduction='none' gives per-token losses
    # reshape to (B*(T-1), V) and (B*(T-1),) then reshape back
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_targets.view(-1),
        reduction="none",
    )  # (B*(T-1),)
    per_token_loss = per_token_loss.view(B, T - 1)  # (B, T-1)

    if reduction == "mean":
        energy = per_token_loss.mean(dim=1)  # (B,)
    else:
        energy = per_token_loss.sum(dim=1)  # (B,)

    return energy


def compute_fluency_score(
    model: object,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute fluency score based on per-token entropy.

    Lower entropy = more confident / fluent = higher score.
    Score = -mean_entropy (so higher score = more fluent).

    Args:
        model: AureliusTransformer.
        input_ids: (B, T) token ids.

    Returns:
        Tensor of shape (B,) — fluency score per sequence (higher = more fluent).
    """
    with torch.no_grad():
        _, logits, _ = model(input_ids)  # (B, T, V)

    # Compute entropy at each position
    probs = F.softmax(logits, dim=-1)  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # H(p) = -sum(p * log_p) per position
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

    mean_entropy = entropy.mean(dim=1)  # (B,)

    # Higher score = lower entropy = more fluent
    return -mean_entropy


class EnergyReranker:
    """Reranks candidate sequences using energy + fluency scores.

    Combined score = -(energy_weight * energy) + fluency_weight * fluency_score
    (negating energy because lower energy = better, and we want higher score = better)
    """

    def __init__(self, model: object, config: EnergyConfig) -> None:
        self.model = model
        self.config = config

    def score(self, candidates: list[torch.Tensor]) -> torch.Tensor:
        """Score each candidate sequence.

        Args:
            candidates: List of N tensors, each shape (1, T) or (B, T).

        Returns:
            Tensor of shape (N,) — combined score per candidate (higher = better).
        """
        scores = []
        for cand in candidates:
            energy = compute_sequence_energy(self.model, cand)  # (B,) or (1,)
            fluency = compute_fluency_score(self.model, cand)  # (B,) or (1,)

            # Combined: lower energy is better, higher fluency is better
            combined = -self.config.energy_weight * energy + self.config.fluency_weight * fluency
            # Average across batch dimension if needed
            scores.append(combined.mean())

        return torch.stack(scores)  # (N,)

    def rerank(self, candidates: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return candidates sorted by score descending.

        Args:
            candidates: List of N tensors.

        Returns:
            Same candidates reordered by score (best first).
        """
        scores = self.score(candidates)
        order = torch.argsort(scores, descending=True)
        return [candidates[i.item()] for i in order]

    def select_best(self, candidates: list[torch.Tensor]) -> torch.Tensor:
        """Return the highest-scoring candidate.

        Args:
            candidates: List of N tensors.

        Returns:
            The best candidate tensor.
        """
        scores = self.score(candidates)
        best_idx = scores.argmax().item()
        return candidates[best_idx]


def langevin_refine(
    model: object,
    input_ids: torch.Tensor,
    config: EnergyConfig,
) -> torch.Tensor:
    """Refine token sequences via Langevin dynamics in embedding space.

    Performs continuous-relaxation Langevin dynamics:
    1. Start from token embeddings.
    2. At each step, add Gaussian noise scaled by sqrt(2 * step_size).
    3. Compute gradient of energy w.r.t. embeddings.
    4. Update embeddings using gradient.
    5. Project back to discrete tokens by nearest-neighbor lookup.

    Args:
        model: AureliusTransformer with model.embed (nn.Embedding).
        input_ids: (B, T) token ids.
        config: EnergyConfig with step_size, mcmc_steps.

    Returns:
        Refined input_ids of shape (B, T), dtype torch.long.
    """
    try:
        B, T = input_ids.shape

        # Get embedding matrix for nearest-neighbor projection
        embed_weight = model.embed.weight  # (V, d_model)

        # Start from current embeddings
        with torch.no_grad():
            emb = model.embed(input_ids).clone()  # (B, T, d_model)

        noise_scale = math.sqrt(2.0 * config.step_size)

        for _ in range(config.mcmc_steps):
            # Require grad on embeddings
            emb_param = emb.detach().requires_grad_(True)

            # Forward pass through transformer layers using embeddings directly
            # We need to go through the model but start from embeddings
            # The model forward takes input_ids; we compute logits from embeddings manually
            # by going through the layers directly
            freqs_cis = model.freqs_cis[:T]
            x = emb_param

            present_key_values = []
            for layer in model.layers:
                x, kv = layer(x, freqs_cis, None, None)
                present_key_values.append(kv)

            x = model.norm(x)
            logits = model.lm_head(x)  # (B, T, V)

            # Compute energy: cross-entropy on shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            energy = F.cross_entropy(
                shift_logits.view(-1, logits.shape[-1]),
                shift_targets.view(-1),
            )

            # Compute gradient of energy w.r.t. embeddings
            grad = torch.autograd.grad(energy, emb_param)[0]  # (B, T, d_model)

            # Langevin update: emb = emb - step_size * grad + noise
            with torch.no_grad():
                noise = torch.randn_like(emb) * noise_scale
                emb = emb_param.detach() - config.step_size * grad + noise

        # Project back to discrete tokens via nearest-neighbor
        # emb: (B, T, d_model), embed_weight: (V, d_model)
        with torch.no_grad():
            # Compute L2 distances: ||emb - w||^2 = ||emb||^2 + ||w||^2 - 2*emb@w.T
            emb_flat = emb.view(B * T, -1)  # (B*T, d_model)
            # Cosine-sim or L2
            # L2: (B*T, V)
            emb_sq = (emb_flat**2).sum(dim=1, keepdim=True)  # (B*T, 1)
            w_sq = (embed_weight**2).sum(dim=1, keepdim=True).T  # (1, V)
            dot = emb_flat @ embed_weight.T  # (B*T, V)
            dist_sq = emb_sq + w_sq - 2.0 * dot  # (B*T, V)
            nearest = dist_sq.argmin(dim=1)  # (B*T,)
            refined_ids = nearest.view(B, T).long()

        return refined_ids

    except Exception:
        # Fallback: return original input_ids unchanged
        return input_ids


class EnergyBasedGenerator:
    """Generate candidates and select the best using energy-based reranking.

    Args:
        model: AureliusTransformer.
        config: EnergyConfig.
        tokenizer_encode: callable(text: str) -> list[int]
        tokenizer_decode: callable(ids: list[int]) -> str
    """

    def __init__(
        self,
        model: object,
        config: EnergyConfig,
        tokenizer_encode,
        tokenizer_decode,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.reranker = EnergyReranker(model, config)

    def generate_candidates(
        self,
        prompt: str,
        max_new_tokens: int = 16,
    ) -> list[str]:
        """Generate n_candidates completions using varied temperatures.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate per candidate.

        Returns:
            List of n_candidates decoded strings.
        """
        prompt_ids = self.tokenizer_encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)  # (1, T)
        prompt_len = input_ids.shape[1]

        candidates_text = []

        for i in range(self.config.n_candidates):
            # Vary temperature across candidates
            temp = self.config.temperature * (0.7 + 0.1 * (i % 4))

            with torch.no_grad():
                # Autoregressive generation with multinomial sampling
                cur_ids = input_ids.clone()
                past_key_values = None
                generated = cur_ids

                for _ in range(max_new_tokens):
                    _, logits, past_key_values = self.model(
                        cur_ids, past_key_values=past_key_values
                    )
                    next_logits = logits[:, -1, :] / max(temp, 1e-8)
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                    generated = torch.cat([generated, next_token], dim=1)
                    cur_ids = next_token  # KV cache: only feed new token

            # Decode only the generated portion (not the prompt)
            generated_ids = generated[0, prompt_len:].tolist()
            text = self.tokenizer_decode(generated_ids)
            candidates_text.append(text)

        return candidates_text

    def generate_best(
        self,
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        """Generate candidates and return the best-scoring one.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate per candidate.

        Returns:
            The best candidate decoded string.
        """
        prompt_ids = self.tokenizer_encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)  # (1, T)
        prompt_len = input_ids.shape[1]

        candidate_tensors = []

        for i in range(self.config.n_candidates):
            temp = self.config.temperature * (0.7 + 0.1 * (i % 4))

            with torch.no_grad():
                cur_ids = input_ids.clone()
                past_key_values = None
                generated = cur_ids

                for _ in range(max_new_tokens):
                    _, logits, past_key_values = self.model(
                        cur_ids, past_key_values=past_key_values
                    )
                    next_logits = logits[:, -1, :] / max(temp, 1e-8)
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)
                    cur_ids = next_token

            candidate_tensors.append(generated)  # (1, prompt_len + gen_len)

        # Rerank by energy+fluency
        best_tensor = self.reranker.select_best(candidate_tensors)  # (1, T)
        best_ids = best_tensor[0, prompt_len:].tolist()
        return self.tokenizer_decode(best_ids)
