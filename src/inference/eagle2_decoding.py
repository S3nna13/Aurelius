"""EAGLE-2 dynamic draft-tree decoding utilities.

This module implements a compact, pure-PyTorch approximation of the EAGLE-2
drafting stack with variable names that follow the paper notation:

- ``H``: target-model hidden states
- ``E``: token embeddings for the current prefix
- ``q``: multi-step draft logits
- ``beta``: dynamic tree-width logits
- ``k``: per-depth branching widths derived from ``beta``

The implementation focuses on the trainable drafting side of EAGLE-2 and keeps
the tree builder explicit so it can be tested independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EAGLE2Config:
    """Configuration for the EAGLE-2 drafting module."""

    d_model: int = 64
    vocab_size: int = 256
    D: int = 4
    K: int = 4
    d_hidden: int = 128
    beta_threshold: float = 0.25
    max_nodes: int = 64
    layer_norm_eps: float = 1e-5


@dataclass
class DynamicDraftTree:
    """Dense representation of a batch of dynamic draft trees."""

    token_ids: torch.Tensor
    parent_index: torch.Tensor
    depth: torch.Tensor
    log_prob: torch.Tensor
    node_mask: torch.Tensor


def dynamic_branch_width(
    beta: torch.Tensor,
    K: int,
    threshold: float = 0.25,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map ``beta`` to per-depth branching widths ``k``.

    The paper uses a dynamic tree width. Here we parameterize it with a smooth
    sigmoid gate and a hard threshold for pruning small branches.
    """
    if K <= 0:
        raise ValueError("K must be positive")

    sigma = torch.sigmoid(beta)
    k = torch.ceil(sigma * float(K)).to(dtype=torch.long)
    k = torch.clamp(k, min=1, max=K)
    k = torch.where(sigma >= threshold, k, torch.zeros_like(k))

    if attention_mask is not None:
        attn = attention_mask.to(dtype=torch.bool)
        while attn.ndim < k.ndim:
            attn = attn.unsqueeze(-1)
        k = torch.where(attn, k, torch.zeros_like(k))

    return k


def _future_labels(
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    D: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct targets for the ``D`` future draft depths."""
    B, T = labels.shape
    targets = torch.full(
        (B, T, D),
        fill_value=ignore_index,
        dtype=labels.dtype,
        device=labels.device,
    )
    valid = torch.zeros(B, T, D, dtype=torch.bool, device=labels.device)

    if attention_mask is None:
        mask = torch.ones(B, T, dtype=torch.bool, device=labels.device)
    else:
        mask = attention_mask.to(dtype=torch.bool)

    for d in range(1, D + 1):
        source_slice = slice(0, T - d)
        target_slice = slice(d, T)
        valid[:, source_slice, d - 1] = mask[:, source_slice] & mask[:, target_slice]
        targets[:, source_slice, d - 1] = labels[:, target_slice]

    targets = torch.where(valid, targets, torch.full_like(targets, ignore_index))
    return targets, valid


def eagle2_loss_reference(
    q: torch.Tensor,
    beta: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Reference loss for validating the vectorized formulation.

    ``q`` has shape ``(B, T, D, V)`` and predicts the next ``D`` tokens from each
    position. ``beta`` has shape ``(B, T, D)``.
    """
    if q.ndim != 4:
        raise ValueError("q must have shape (B, T, D, V)")
    if beta.shape != q.shape[:3]:
        raise ValueError("beta must have shape (B, T, D)")

    _, _, D, V = q.shape
    targets, valid = _future_labels(labels, attention_mask, D=D, ignore_index=ignore_index)

    flat_q = q.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    valid_targets = flat_targets != ignore_index
    if valid_targets.any():
        L_q = F.cross_entropy(
            flat_q[valid_targets],
            flat_targets[valid_targets],
        )
    else:
        L_q = q.new_zeros(())

    beta_target = valid.to(dtype=beta.dtype)
    beta_loss = F.binary_cross_entropy_with_logits(beta, beta_target, reduction="none")

    source_mask = valid.any(dim=-1, keepdim=True)
    denom = source_mask.sum().clamp_min(1).to(dtype=beta.dtype)
    L_beta = (beta_loss * source_mask.to(dtype=beta.dtype)).sum() / denom
    return L_q + L_beta


def build_dynamic_draft_tree(
    q: torch.Tensor,
    beta: torch.Tensor,
    config: EAGLE2Config,
) -> DynamicDraftTree:
    """Build a dynamic draft tree from endpoint logits.

    Args:
        q: ``(B, D, V)`` multi-step logits for the current prefix endpoint.
        beta: ``(B, D)`` dynamic width logits.
        config: Tree hyperparameters ``D``, ``K``, and ``max_nodes``.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape (B, D, V)")
    if beta.shape != q.shape[:2]:
        raise ValueError("beta must have shape (B, D)")

    B, D, _ = q.shape
    if D != config.D:
        raise ValueError("q depth must match config.D")

    k = dynamic_branch_width(beta, K=config.K, threshold=config.beta_threshold)

    token_ids = torch.zeros((B, config.max_nodes), dtype=torch.long, device=q.device)
    parent_index = torch.full((B, config.max_nodes), -1, dtype=torch.long, device=q.device)
    depth = torch.zeros((B, config.max_nodes), dtype=torch.long, device=q.device)
    log_prob = torch.zeros((B, config.max_nodes), dtype=q.dtype, device=q.device)
    node_mask = torch.zeros((B, config.max_nodes), dtype=torch.bool, device=q.device)

    for b in range(B):
        cursor = 0
        frontier = [-1]
        for d in range(config.D):
            width = int(k[b, d].item())
            if width <= 0 or not frontier or cursor >= config.max_nodes:
                break

            log_q = F.log_softmax(q[b, d], dim=-1)
            top_log_q, top_token = torch.topk(log_q, k=min(width, log_q.shape[0]))

            next_frontier: list[int] = []
            for parent in frontier:
                for branch in range(top_token.numel()):
                    if cursor >= config.max_nodes:
                        break
                    token_ids[b, cursor] = top_token[branch]
                    parent_index[b, cursor] = parent
                    depth[b, cursor] = d + 1
                    log_prob[b, cursor] = top_log_q[branch]
                    node_mask[b, cursor] = True
                    next_frontier.append(cursor)
                    cursor += 1
                if cursor >= config.max_nodes:
                    break
            frontier = next_frontier

    return DynamicDraftTree(
        token_ids=token_ids,
        parent_index=parent_index,
        depth=depth,
        log_prob=log_prob,
        node_mask=node_mask,
    )


class EAGLE2Decoder(nn.Module):
    """Trainable EAGLE-2 drafting head with dynamic tree proposal."""

    def __init__(self, config: EAGLE2Config) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.depth_embedding = nn.Parameter(torch.empty(config.D, config.d_model))
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.W_1 = nn.Linear(config.d_model, config.d_hidden)
        self.W_2 = nn.Linear(config.d_hidden, config.d_model)
        self.W_q = nn.Linear(config.d_model, config.vocab_size)
        self.W_beta = nn.Linear(config.d_model, 1)
        self.act = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.depth_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.W_1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_1.bias)
        nn.init.normal_(self.W_2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_2.bias)
        nn.init.normal_(self.W_q.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_q.bias)
        nn.init.normal_(self.W_beta.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_beta.bias)

    def _draft_state(self, H: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Construct the EAGLE-2 draft state from ``H`` and ``E``."""
        E = self.token_embedding(input_ids)
        S = H.unsqueeze(2) + E.unsqueeze(2) + self.depth_embedding.view(1, 1, self.config.D, -1)
        S = self.norm(S)
        S = self.W_2(self.act(self.W_1(S)))
        return S

    def forward(
        self,
        H: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> dict[str, torch.Tensor | None]:
        if H.ndim != 3:
            raise ValueError("H must have shape (B, T, d_model)")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, T)")
        if H.shape[:2] != input_ids.shape:
            raise ValueError("H and input_ids must agree on batch and sequence dimensions")

        S = self._draft_state(H, input_ids)
        q = self.W_q(S)
        beta = self.W_beta(S).squeeze(-1)

        if attention_mask is not None:
            source_mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)
            q = q.masked_fill(~source_mask, 0.0)
            beta = beta.masked_fill(~attention_mask.to(dtype=torch.bool).unsqueeze(-1), 0.0)

        k = dynamic_branch_width(
            beta,
            K=self.config.K,
            threshold=self.config.beta_threshold,
            attention_mask=attention_mask,
        )

        loss = None
        if labels is not None:
            loss = eagle2_loss_reference(
                q=q,
                beta=beta,
                labels=labels,
                attention_mask=attention_mask,
                ignore_index=ignore_index,
            )

        return {
            "q": q,
            "beta": beta,
            "k": k,
            "loss": loss,
        }

    @torch.no_grad()
    def propose_tree(
        self,
        H: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DynamicDraftTree:
        """Draft a dynamic tree from the last valid prefix position."""
        outputs = self.forward(H=H, input_ids=input_ids, attention_mask=attention_mask)
        q = outputs["q"]
        beta = outputs["beta"]
        assert isinstance(q, torch.Tensor)
        assert isinstance(beta, torch.Tensor)

        B, T, _, _ = q.shape
        if attention_mask is None:
            last = torch.full((B,), T - 1, dtype=torch.long, device=q.device)
        else:
            mask = attention_mask.to(dtype=torch.bool)
            has_token = mask.any(dim=-1)
            last = mask.long().sum(dim=-1) - 1
            last = torch.where(has_token, last, torch.zeros_like(last))

        batch = torch.arange(B, device=q.device)
        q_last = q[batch, last]
        beta_last = beta[batch, last]

        if attention_mask is not None:
            active = attention_mask.to(dtype=torch.bool).any(dim=-1)
            beta_last = beta_last.masked_fill(~active.unsqueeze(-1), -math.inf)

        return build_dynamic_draft_tree(q=q_last, beta=beta_last, config=self.config)
