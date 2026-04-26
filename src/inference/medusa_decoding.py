"""
Medusa (Cai et al. 2024) -- multi-head speculative decoding.

K additional prediction heads are attached to the base model, each predicting
tokens at offset +1, +2, ..., +K without any separate draft model.

Pure PyTorch only -- no transformers, einops, trl, etc.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# MedusaHead
# ---------------------------------------------------------------------------


class MedusaHead(nn.Module):
    """Single prediction head for one future token offset.

    Architecture: (Linear(d_model, d_model) -> SiLU) * n_layers -> Linear(d_model, vocab_size)

    Args:
        d_model:    Hidden dimension of the base model.
        vocab_size: Vocabulary size.
        head_idx:   Which future token this head predicts (1-indexed).
        n_layers:   Number of hidden layers before the final projection.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        head_idx: int,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.head_idx = head_idx
        self.d_model = d_model
        self.vocab_size = vocab_size

        hidden_layers: list[nn.Module] = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(d_model, d_model))
            hidden_layers.append(nn.SiLU())
        self.hidden = nn.Sequential(*hidden_layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Predict logits from hidden states.

        Args:
            hidden_states: (B, T, D)

        Returns:
            logits: (B, T, V)
        """
        x = self.hidden(hidden_states)
        return self.proj(x)


# ---------------------------------------------------------------------------
# MedusaModel
# ---------------------------------------------------------------------------


class MedusaModel(nn.Module):
    """Base model wrapped with K Medusa speculation heads.

    Args:
        base_model: Callable (input_ids) -> (logits, hidden_states)
                    where logits is (B, T, V) and hidden_states is (B, T, D).
        d_model:    Hidden dimension.
        vocab_size: Vocabulary size.
        n_heads:    Number of Medusa heads (one per future offset).
    """

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        vocab_size: int,
        n_heads: int = 3,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.medusa_heads = nn.ModuleList(
            [MedusaHead(d_model, vocab_size, head_idx=i + 1) for i in range(n_heads)]
        )

    def forward(self, input_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Run base model and all Medusa heads.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            base_logits:      (B, T, V)
            medusa_logits:    list of K tensors each (B, T, V)
        """
        base_logits, hidden_states = self.base_model(input_ids)
        medusa_logits = [head(hidden_states) for head in self.medusa_heads]
        return base_logits, medusa_logits


# ---------------------------------------------------------------------------
# MedusaLoss
# ---------------------------------------------------------------------------


class MedusaLoss(nn.Module):
    """Training loss for Medusa heads.

    Args:
        n_heads:      Number of Medusa heads.
        head_weights: Per-head loss weights. Default: geometric decay
                      [1, 0.8, 0.64, ...].
    """

    def __init__(
        self,
        n_heads: int,
        head_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        if head_weights is None:
            head_weights = [0.8**i for i in range(n_heads)]
        if len(head_weights) != n_heads:
            raise ValueError(f"head_weights length {len(head_weights)} != n_heads {n_heads}")
        self.head_weights = head_weights

    def forward(
        self,
        base_logits: Tensor,
        medusa_logits_list: list[Tensor],
        input_ids: Tensor,
    ) -> tuple[Tensor, list[float]]:
        """Compute total Medusa training loss.

        Args:
            base_logits:        (B, T, V) -- base model logits.
            medusa_logits_list: list of K tensors each (B, T, V).
            input_ids:          (B, T) token ids (targets derived from these).

        Returns:
            total_loss:   scalar tensor
            head_losses:  list of K floats (individual per-head losses)
        """
        # Base LM loss: predict token t+1 from position t
        base_loss = F.cross_entropy(
            base_logits[:, :-1, :].reshape(-1, base_logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        total_loss = base_loss
        head_losses: list[float] = []

        for k, (head_logits, weight) in enumerate(zip(medusa_logits_list, self.head_weights)):
            # Per spec: targets = input_ids[:, k+1:], logits = medusa_logits[k][:, :-k-1, :]
            targets = input_ids[:, k + 1 :]  # (B, T - k - 1)
            logits = head_logits[:, : -(k + 1), :]  # (B, T - k - 1, V)

            if targets.numel() == 0 or logits.size(1) == 0:
                head_losses.append(0.0)
                continue

            h_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_loss = total_loss + weight * h_loss
            head_losses.append(h_loss.item())

        return total_loss, head_losses


# ---------------------------------------------------------------------------
# MedusaTreeDecoder
# ---------------------------------------------------------------------------


class MedusaTreeDecoder:
    """Tree-based decoding using Medusa heads.

    Args:
        medusa_model: A MedusaModel instance.
        top_k:        How many candidates per head to consider.
    """

    def __init__(self, medusa_model: MedusaModel, top_k: int = 3) -> None:
        self.medusa_model = medusa_model
        self.top_k = top_k

    @torch.no_grad()
    def draft_candidates(self, input_ids: Tensor) -> Tensor:
        """Generate candidate token sequences using Medusa heads.

        Takes the top_k predictions from each head at the last position and
        forms all combinations (cartesian product).

        Args:
            input_ids: (B, T) -- only batch dimension 0 is used for drafting.

        Returns:
            candidates: (top_k^K, K) integer tensor of candidate token ids.
        """
        _, medusa_logits = self.medusa_model(input_ids)
        len(medusa_logits)

        per_head_top_k: list[Tensor] = []
        for head_logits in medusa_logits:
            # head_logits: (B, T, V) -- take last position of first batch item
            last_logits = head_logits[0, -1, :]  # (V,)
            top_ids = torch.topk(last_logits, self.top_k, dim=-1).indices  # (top_k,)
            per_head_top_k.append(top_ids)

        # Cartesian product -> (top_k^K, K)
        combos = list(itertools.product(*[t.tolist() for t in per_head_top_k]))
        candidates = torch.tensor(combos, dtype=torch.long, device=input_ids.device)
        return candidates

    @torch.no_grad()
    def verify(self, input_ids: Tensor, candidates: Tensor) -> tuple[Tensor, int]:
        """Verify candidate tokens using the base model.

        Uses the greedy (top-1) candidate and runs the base model on the
        extended sequence to check acceptance token by token.

        Args:
            input_ids:  (B, T)
            candidates: (top_k^K, K) -- from draft_candidates

        Returns:
            accepted_tokens: (n_accepted,) tokens that were accepted
            n_accepted:      number of accepted tokens in [0, K]
        """
        K = candidates.size(1)
        if K == 0 or candidates.size(0) == 0:
            return torch.empty(0, dtype=torch.long, device=input_ids.device), 0

        # Greedy candidate: first candidate (top-1 from each head)
        best_candidate = candidates[0]  # (K,)

        # Extend input with candidate tokens (use batch item 0)
        extended = torch.cat([input_ids[0:1, :], best_candidate.unsqueeze(0)], dim=1)  # (1, T+K)

        base_logits, _ = self.medusa_model.base_model(extended)
        # base_logits: (1, T+K, V)

        T = input_ids.size(1)
        accepted_tokens: list[int] = []
        for i in range(K):
            pred_token = base_logits[0, T - 1 + i, :].argmax(dim=-1).item()
            candidate_token = best_candidate[i].item()
            if pred_token == candidate_token:
                accepted_tokens.append(candidate_token)
            else:
                break

        n_accepted = len(accepted_tokens)
        if n_accepted == 0:
            return torch.empty(0, dtype=torch.long, device=input_ids.device), 0

        accepted_tensor = torch.tensor(accepted_tokens, dtype=torch.long, device=input_ids.device)
        return accepted_tensor, n_accepted

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 10) -> Tensor:
        """Autoregressive generation with Medusa speculation.

        Args:
            input_ids:      (B, T)
            max_new_tokens: Number of new tokens to generate.

        Returns:
            output_ids: (B, T + max_new_tokens)
        """
        # Work on a single sequence (batch dim 0) for speculation
        current = input_ids[0:1, :].clone()  # (1, T_cur)
        generated = 0

        while generated < max_new_tokens:
            candidates = self.draft_candidates(current)
            accepted_tokens, n_accepted = self.verify(current, candidates)

            if n_accepted > 0:
                tokens_to_add = min(n_accepted, max_new_tokens - generated)
                current = torch.cat([current, accepted_tokens[:tokens_to_add].unsqueeze(0)], dim=1)
                generated += tokens_to_add
            else:
                # Fall back: greedy decode one token from base model
                base_logits, _ = self.medusa_model.base_model(current)
                next_token = base_logits[0, -1, :].argmax(dim=-1, keepdim=True)
                current = torch.cat([current, next_token.unsqueeze(0)], dim=1)
                generated += 1

        # Expand back to original batch size
        B = input_ids.size(0)
        output = current.expand(B, -1)
        return output


# ---------------------------------------------------------------------------
# MedusaTrainer
# ---------------------------------------------------------------------------


class MedusaTrainer:
    """Fine-tune Medusa heads with base model optionally frozen.

    Args:
        medusa_model: A MedusaModel instance.
        optimizer:    A torch optimizer (should cover medusa_heads params).
        freeze_base:  If True, freeze base_model params on init.
    """

    def __init__(
        self,
        medusa_model: MedusaModel,
        optimizer: torch.optim.Optimizer,
        freeze_base: bool = True,
    ) -> None:
        self.medusa_model = medusa_model
        self.optimizer = optimizer
        self.loss_fn = MedusaLoss(n_heads=medusa_model.n_heads)

        if freeze_base:
            self.freeze_base()

    def freeze_base(self) -> None:
        """Freeze all parameters of the base model."""
        for param in self.medusa_model.base_model.parameters():
            param.requires_grad = False

    def train_step(self, input_ids: Tensor) -> dict:
        """Perform one training step on Medusa heads.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            dict with keys: total_loss (float), head_losses (list[float]),
            grad_norm (float).
        """
        self.medusa_model.train()
        self.optimizer.zero_grad()

        base_logits, medusa_logits = self.medusa_model(input_ids)
        total_loss, head_losses = self.loss_fn(base_logits, medusa_logits, input_ids)

        total_loss.backward()

        # Compute gradient norm over Medusa head params only
        medusa_params = list(self.medusa_model.medusa_heads.parameters())
        grad_norm = 0.0
        for p in medusa_params:
            if p.grad is not None:
                grad_norm += p.grad.detach().norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "head_losses": head_losses,
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def acceptance_rate_estimate(self, input_ids: Tensor) -> float:
        """Offline estimate of head-1 acceptance rate.

        Computes the fraction of positions where head-1's top-1 prediction
        matches the actual next token in input_ids.

        Args:
            input_ids: (B, T)

        Returns:
            acceptance_rate: float in [0, 1]
        """
        self.medusa_model.train(False)
        _, medusa_logits = self.medusa_model(input_ids)

        head1_logits = medusa_logits[0]  # (B, T, V)
        # Head 1 predicts the token at offset +1 from each position
        predicted = head1_logits[:, :-1, :].argmax(dim=-1)  # (B, T-1)
        targets = input_ids[:, 1:]  # (B, T-1)

        if targets.numel() == 0:
            return 0.0

        correct = (predicted == targets).float().mean().item()
        return float(correct)
