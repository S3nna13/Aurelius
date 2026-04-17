"""TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding.

Based on Sun et al., arXiv:2404.11912.

TriForce uses two-level speculative decoding:
  - Level 2 (fast draft): retrieval-based KV cache approximation via top-K block lookup
  - Level 1 (slow draft): small draft model with full KV cache produces γ_1 draft tokens
  - Verification: target model verifies all speculative tokens

Public API
----------
KVBlockStore        — stores KV cache blocks for retrieval
TriForceDraftCache  — fast Level-2 draft context via retrieved KV blocks
TriForceVerifier    — standard speculative verification (Algorithm 1, Leviathan et al. 2023)
TriForceDecoder     — orchestrates the two-level speculation loop
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# KVBlockStore
# ---------------------------------------------------------------------------

class KVBlockStore:
    """Stores KV cache blocks for retrieval by query similarity.

    Parameters
    ----------
    block_size:
        Number of tokens per KV block.
    max_blocks:
        Maximum number of blocks to retain in the store.
    """

    def __init__(self, block_size: int = 16, max_blocks: int = 256) -> None:
        self.block_size = block_size
        self.max_blocks = max_blocks

        # Each element: (block_size, n_heads, d_head)
        self._key_blocks: List[Tensor] = []
        self._val_blocks: List[Tensor] = []
        # Mean key per block for fast similarity lookup: (n_heads, d_head)
        self._key_means: List[Tensor] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_blocks(self) -> int:
        """Current number of stored blocks."""
        return len(self._key_blocks)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_block(self, keys: Tensor, values: Tensor) -> None:
        """Add a KV block to the store.

        Parameters
        ----------
        keys, values:
            Tensors of shape ``(block_size, n_heads, d_head)``.
        """
        if keys.shape[0] != self.block_size or values.shape[0] != self.block_size:
            raise ValueError(
                f"Expected block_size={self.block_size} on dim-0, "
                f"got keys={keys.shape[0]}, values={values.shape[0]}"
            )

        # Evict oldest block when at capacity
        if len(self._key_blocks) >= self.max_blocks:
            self._key_blocks.pop(0)
            self._val_blocks.pop(0)
            self._key_means.pop(0)

        self._key_blocks.append(keys.detach().clone())
        self._val_blocks.append(values.detach().clone())
        # Mean over the token dimension: (n_heads, d_head)
        self._key_means.append(keys.detach().clone().mean(dim=0))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_top_k(
        self, query: Tensor, k: int = 8
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve the top-k KV blocks by mean cosine similarity.

        Parameters
        ----------
        query:
            Shape ``(n_heads, d_head)``.
        k:
            Number of blocks to retrieve.

        Returns
        -------
        top_keys : ``(k, block_size, n_heads, d_head)``
        top_values : ``(k, block_size, n_heads, d_head)``

        If fewer than *k* blocks are stored, all stored blocks are returned
        (the leading dimension equals the number of stored blocks).
        """
        if self.n_blocks == 0:
            n_heads, d_head = query.shape
            empty_k = torch.zeros(0, self.block_size, n_heads, d_head, device=query.device)
            return empty_k, empty_k.clone()

        actual_k = min(k, self.n_blocks)

        # Stack means: (n_blocks, n_heads, d_head)
        means = torch.stack(self._key_means, dim=0).to(query.device)  # (B_n, n_h, d_h)

        # Flatten heads for cosine similarity: (n_blocks, n_heads * d_head)
        means_flat = means.reshape(self.n_blocks, -1)
        query_flat = query.reshape(1, -1)  # (1, n_heads * d_head)

        similarities = F.cosine_similarity(query_flat, means_flat, dim=-1)  # (n_blocks,)

        top_indices = torch.topk(similarities, k=actual_k).indices  # (actual_k,)

        top_keys = torch.stack(
            [self._key_blocks[i.item()].to(query.device) for i in top_indices], dim=0
        )  # (actual_k, block_size, n_heads, d_head)
        top_values = torch.stack(
            [self._val_blocks[i.item()].to(query.device) for i in top_indices], dim=0
        )  # (actual_k, block_size, n_heads, d_head)

        return top_keys, top_values


# ---------------------------------------------------------------------------
# TriForceDraftCache
# ---------------------------------------------------------------------------

class TriForceDraftCache:
    """Fast Level-2 draft context built by retrieving top-K KV blocks.

    Parameters
    ----------
    n_heads:
        Number of attention heads.
    d_head:
        Head dimension.
    block_size:
        Token block size (must match the ``KVBlockStore``).
    top_k:
        Number of blocks to retrieve per query.
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        block_size: int = 16,
        top_k: int = 8,
    ) -> None:
        self.n_heads = n_heads
        self.d_head = d_head
        self.block_size = block_size
        self.top_k = top_k

    def build_attn_context(
        self, query_states: Tensor, kv_store: KVBlockStore
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve the top-k blocks and concatenate into a draft context.

        Parameters
        ----------
        query_states:
            Shape ``(n_heads, d_head)`` — the current query vector.
        kv_store:
            A populated :class:`KVBlockStore`.

        Returns
        -------
        ctx_keys : ``(k * block_size, n_heads, d_head)``
        ctx_values : ``(k * block_size, n_heads, d_head)``
        """
        top_keys, top_values = kv_store.retrieve_top_k(query_states, k=self.top_k)
        # top_keys: (actual_k, block_size, n_heads, d_head)
        actual_k = top_keys.shape[0]

        if actual_k == 0:
            ctx_len = 0
        else:
            ctx_len = actual_k * self.block_size

        if actual_k == 0:
            ctx_keys = torch.zeros(
                ctx_len, self.n_heads, self.d_head,
                device=query_states.device,
                dtype=query_states.dtype,
            )
            ctx_values = ctx_keys.clone()
        else:
            # Reshape: (actual_k, block_size, n_heads, d_head) -> (actual_k * block_size, n_heads, d_head)
            ctx_keys = top_keys.reshape(ctx_len, self.n_heads, self.d_head)
            ctx_values = top_values.reshape(ctx_len, self.n_heads, self.d_head)

        return ctx_keys, ctx_values


# ---------------------------------------------------------------------------
# TriForceVerifier
# ---------------------------------------------------------------------------

class TriForceVerifier:
    """Standard speculative verification (Algorithm 1, Leviathan et al. 2023).

    Parameters
    ----------
    beta:
        Temperature for the resampling / corrected distribution.
        ``beta=0.0`` means greedy resampling from the corrected distribution.
    """

    def __init__(self, beta: float = 0.0) -> None:
        self.beta = beta

    def verify(
        self,
        draft_tokens: Tensor,
        draft_logits: Tensor,
        target_logits: Tensor,
    ) -> Tuple[Tensor, int]:
        """Verify γ draft tokens against the target model's logits.

        Implements standard rejection sampling:
          1. For position i with draft token t:
             accept_prob = min(1, p_target[t] / p_draft[t])
          2. On rejection: sample from max(0, p_target - p_draft) (normalised).
          3. If all γ draft tokens are accepted, sample a bonus token from
             p_target at position γ.

        Parameters
        ----------
        draft_tokens:
            Shape ``(γ,)`` int64 — the proposed token ids.
        draft_logits:
            Shape ``(γ, V)`` — logits from the draft model at each position.
        target_logits:
            Shape ``(γ, V)`` — logits from the target model at each position
            (including position γ for the bonus token, so shape is ``(γ+1, V)``
            or exactly ``(γ, V)`` with the understanding that position γ is
            included only when a bonus token is needed).

            .. note::
                We accept ``(γ, V)`` or ``(γ+1, V)``. If ``(γ, V)`` is
                provided the bonus token is sampled from ``target_logits[-1]``.

        Returns
        -------
        accepted_tokens : 1-D int64 ``Tensor`` of length between 1 and γ+1.
        n_accepted : int — number of draft tokens that were accepted (0..γ).
        """
        gamma = draft_tokens.shape[0]
        device = draft_tokens.device

        # Compute probabilities
        draft_probs = F.softmax(draft_logits.float(), dim=-1)   # (γ, V)
        target_probs = F.softmax(target_logits.float(), dim=-1) # (γ or γ+1, V)

        accepted: List[Tensor] = []
        n_accepted = 0

        for i in range(gamma):
            t = draft_tokens[i].item()
            p_d = draft_probs[i, t]
            p_t = target_probs[i, t]

            accept_prob = torch.clamp(p_t / (p_d + 1e-10), max=1.0)
            u = torch.rand((), device=device)

            if u.item() <= accept_prob.item():
                accepted.append(draft_tokens[i])
                n_accepted += 1
            else:
                # Corrected distribution: max(0, p_target - p_draft)
                adj = (target_probs[i] - draft_probs[i]).clamp(min=0.0)  # (V,)
                adj_sum = adj.sum()
                if adj_sum < 1e-10:
                    adj = target_probs[i]
                    adj_sum = adj.sum()
                adj = adj / (adj_sum + 1e-10)

                if self.beta <= 1e-8:
                    # Greedy from corrected distribution
                    resample = adj.argmax()
                else:
                    adj_tempered = (adj + 1e-10).log() / self.beta
                    adj_tempered = F.softmax(adj_tempered, dim=-1)
                    resample = torch.multinomial(adj_tempered, num_samples=1).squeeze(0)

                accepted.append(resample)
                return torch.stack(accepted), n_accepted

        # All γ draft tokens accepted — emit bonus token from target
        # Use last row of target_logits (index γ if available, else γ-1)
        if target_logits.shape[0] > gamma:
            bonus_probs = target_probs[gamma]
        else:
            bonus_probs = target_probs[-1]

        if self.beta <= 1e-8:
            bonus = bonus_probs.argmax()
        else:
            bonus_tempered = (bonus_probs + 1e-10).log() / self.beta
            bonus_tempered = F.softmax(bonus_tempered, dim=-1)
            bonus = torch.multinomial(bonus_tempered, num_samples=1).squeeze(0)

        accepted.append(bonus)
        return torch.stack(accepted), n_accepted


# ---------------------------------------------------------------------------
# TriForceDecoder
# ---------------------------------------------------------------------------

class TriForceDecoder:
    """Orchestrates the two-level TriForce speculation loop.

    Parameters
    ----------
    target_fn:
        Callable ``(input_ids: Tensor) -> logits: Tensor`` where
        ``input_ids`` is a 1-D ``(T,)`` int64 tensor and ``logits`` is a
        1-D ``(V,)`` float tensor (single-token greedy interface).
    draft_fn:
        Same signature as *target_fn* — the small draft model.
    kv_store:
        A :class:`KVBlockStore` instance (may be pre-populated or empty).
    gamma:
        Number of draft tokens per speculation step (γ).
    """

    def __init__(
        self,
        target_fn: Callable[[Tensor], Tensor],
        draft_fn: Callable[[Tensor], Tensor],
        kv_store: KVBlockStore,
        gamma: int = 4,
    ) -> None:
        self.target_fn = target_fn
        self.draft_fn = draft_fn
        self.kv_store = kv_store
        self.gamma = gamma
        self._verifier = TriForceVerifier(beta=0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_fn(self, fn: Callable[[Tensor], Tensor], ids: Tensor) -> Tensor:
        """Call a model fn with a 1-D ids tensor; return (V,) logits."""
        with torch.no_grad():
            return fn(ids)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Generate *max_new_tokens* tokens using two-level speculative decoding.

        Parameters
        ----------
        prompt_ids:
            1-D int64 tensor ``(T,)`` — the prompt token ids.
        max_new_tokens:
            Exact number of new tokens to generate.

        Returns
        -------
        generated : 1-D int64 ``Tensor`` of length ``max_new_tokens``.
        """
        prompt_ids = prompt_ids.long()
        generated: List[Tensor] = []
        context = prompt_ids.clone()

        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            remaining = max_new_tokens - tokens_generated
            gamma = min(self.gamma, remaining)

            # ---------------------------------------------------------------
            # Level 1: draft model generates γ draft tokens
            # ---------------------------------------------------------------
            draft_token_list: List[Tensor] = []
            draft_logit_list: List[Tensor] = []

            draft_context = context.clone()
            for _ in range(gamma):
                d_logits = self._call_fn(self.draft_fn, draft_context)  # (V,)
                d_token = d_logits.argmax()
                draft_token_list.append(d_token)
                draft_logit_list.append(d_logits)
                draft_context = torch.cat([draft_context, d_token.unsqueeze(0)], dim=0)

            draft_toks = torch.stack(draft_token_list, dim=0)   # (γ,)
            draft_lgts = torch.stack(draft_logit_list, dim=0)   # (γ, V)

            # ---------------------------------------------------------------
            # Level 0: target model verifies draft tokens in γ+1 forward passes
            # (one pass per position, single-token interface as specified)
            # ---------------------------------------------------------------
            target_logit_list: List[Tensor] = []

            # For position i, target sees context + draft_toks[:i]
            verify_context = context.clone()
            for i in range(gamma + 1):
                t_logits = self._call_fn(self.target_fn, verify_context)  # (V,)
                target_logit_list.append(t_logits)
                if i < gamma:
                    verify_context = torch.cat(
                        [verify_context, draft_toks[i].unsqueeze(0)], dim=0
                    )

            target_lgts = torch.stack(target_logit_list, dim=0)  # (γ+1, V)

            # ---------------------------------------------------------------
            # Verify
            # ---------------------------------------------------------------
            accepted_toks, n_accepted = self._verifier.verify(
                draft_toks, draft_lgts, target_lgts
            )
            # accepted_toks: 1-D, length between 1 and γ+1

            # Clamp to remaining budget
            n_to_add = min(accepted_toks.shape[0], remaining)
            chosen = accepted_toks[:n_to_add]

            generated.append(chosen)
            context = torch.cat([context, chosen], dim=0)
            tokens_generated += n_to_add

        if generated:
            return torch.cat(generated, dim=0).long()
        return torch.zeros(0, dtype=torch.long, device=prompt_ids.device)
