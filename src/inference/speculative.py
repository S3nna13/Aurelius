"""Speculative decoding with tree-based draft + acceptance (Yggdrasil-inspired).

Core reference: Leviathan et al. 2023 (arXiv:2211.17192) for the basic
speculative decoding algorithm. Yggdrasil (NeurIPS 2025, arXiv:2512.23858)
for the equal-growth tree structure and tree-based verification.

Key differences from standard speculative decoding:
1. Draft proposals use an **equal-growth tree** instead of a linear sequence.
   The tree has roughly equal nodes per level, capped at a budget, so the
   target model verifies multiple candidate continuations in one forward pass.
2. The tree's **best leaf path** is selected after verification, giving higher
   expected acceptance than any single linear draft of equivalent length.
3. Stage-based scheduling: draft model runs are interleaved with verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .draft_tree import (
    DraftNode,
    equal_growth_parent_indices,
    root_to_leaf_paths,
)


@dataclass
class TreeSpeculativeConfig:
    """Configuration for tree-based speculative decoding.

    Attributes:
        tree_budget: Total nodes in the draft tree (including root).
            Yggdrasil uses 8-16 for typical setups.
        nodes_per_level: How many new nodes per tree level.
            Lower = deeper but narrower trees. Higher = shallower but wider.
        temperature: Sampling temperature for both draft and target models.
        top_p: Nucleus sampling threshold.
        max_new_tokens: Maximum tokens to generate.
        eos_token_id: Stop token (None = no early stopping).
    """

    tree_budget: int = 8
    nodes_per_level: int = 3
    temperature: float = 1.0
    top_p: float = 0.9
    max_new_tokens: int = 256
    eos_token_id: int | None = None


def _sample_from_probs(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sample one token from a probability vector with optional top-p."""
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        s = sorted_probs.sum()
        if s < 1e-10:
            sorted_probs = probs[sorted_indices]
        else:
            sorted_probs /= s
        probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)
    return torch.multinomial(probs, 1).squeeze(-1)


def _get_probs(model: nn.Module, input_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """Run model forward and return softmax probabilities at each position."""
    _, logits, _ = model(input_ids)
    if temperature != 1.0:
        logits = logits / temperature
    return F.softmax(logits[0], dim=-1)


def _build_draft_tree(
    draft_model: nn.Module,
    ctx: torch.Tensor,
    parents: list[int | None],
    temperature: float,
    top_p: float,
) -> list[DraftNode]:
    """Build a draft tree by scoring each node with the draft model.

    Walks the tree breadth-first, running the draft model at each node's
    position with the appropriate ancestor context.

    Args:
        draft_model: Small model for proposing tokens.
        ctx: (1, seq_len) — base context before the tree.
        parents: Equal-growth parent indices for the tree.
        temperature: Sampling temperature.
        top_p: Nucleus threshold.

    Returns:
        List of DraftNodes with sampled token_ids and draft model scores.
    """
    nodes: list[DraftNode] = []

    for i in range(len(parents)):
        # Build ancestor chain for this node (parent -> grandparent -> ...)
        ancestor_path: list[int] = []
        p = parents[i]
        while p is not None:
            ancestor_path.append(nodes[p].token_id)
            p = nodes[p].parent

        # Create input: context + ancestor tokens
        if ancestor_path:
            tree_ctx = torch.cat(
                [ctx, torch.tensor(ancestor_path[::-1], device=ctx.device).view(1, -1)],
                dim=1,
            )
        else:
            tree_ctx = ctx

        # Draft model scores the last position
        dist = _get_probs(draft_model, tree_ctx, temperature)[-1]
        token = _sample_from_probs(dist, top_p=top_p)

        nodes.append(
            DraftNode(
                token_id=token.item(),
                score=dist[token].item(),
                parent=parents[i],
            )
        )

    return nodes


class TreeSpeculativeDecoder:
    """Generates tokens using tree-based speculative decoding.

    Inspired by Yggdrasil (NeurIPS 2025): drafts an equal-growth tree of
    candidate tokens, then selects the best path and runs standard rejection
    sampling against the target model's distribution.

    Args:
        draft_model: Small fast model for proposing tokens.
        target_model: Large accurate model for verification.
        cfg: Configuration for tree-based speculative decoding.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        cfg: TreeSpeculativeConfig | None = None,
    ) -> None:
        self.draft = draft_model
        self.target = target_model
        self.cfg = cfg or TreeSpeculativeConfig()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate tokens with tree-based speculative decoding.

        Args:
            input_ids: (1, prompt_len) — input token ids.

        Returns:
            (1, prompt_len + generated_len) — input + generated tokens.
        """
        cfg = self.cfg
        generated = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < cfg.max_new_tokens:
            # Step 1: Build equal-growth tree structure
            parents = equal_growth_parent_indices(
                total_budget=cfg.tree_budget,
                nodes_per_level=cfg.nodes_per_level,
            )

            # Step 2: Score tree nodes with draft model
            nodes = _build_draft_tree(
                self.draft,
                generated,
                parents,
                cfg.temperature,
                cfg.top_p,
            )

            # Step 3: Find best root-to-leaf path
            all_paths = root_to_leaf_paths(nodes)
            if not all_paths:
                # Fallback: single token generation
                _, logits, _ = self.target(generated)
                next_tok = _sample_from_probs(
                    F.softmax(logits[0, -1] / cfg.temperature, dim=-1),
                    top_p=cfg.top_p,
                )
                generated = torch.cat([generated, next_tok.view(1, 1)], dim=1)
                tokens_generated += 1
                continue

            # Score each path by sum of log-probs
            best_path = max(
                all_paths,
                key=lambda path: sum(nodes[idx].score for idx in range(len(path))),
            )

            # Step 4: Verify best path with target model
            path_tensor = torch.tensor(best_path, device=generated.device).view(1, -1)
            verify_input = torch.cat([generated, path_tensor], dim=1)
            target_probs = _get_probs(self.target, verify_input, cfg.temperature)

            # Step 5: Rejection sampling on best path
            current_len = generated.shape[1]
            for i, token_id in enumerate(best_path):
                target_pos = current_len - 1 + i
                t_prob = target_probs[target_pos, token_id].clamp(min=1e-10)
                # Draft probability at this position
                d_prob_val = nodes[i].score if i < len(nodes) else 1.0
                d_prob = max(d_prob_val, 1e-10)

                if t_prob >= d_prob:
                    # Always accept
                    tok_tensor = torch.tensor([[token_id]], device=generated.device)
                    generated = torch.cat([generated, tok_tensor], dim=1)
                    tokens_generated += 1
                else:
                    # Accept with probability p_target / p_draft
                    ratio = (t_prob / d_prob).item()
                    if torch.rand(1, device=generated.device).item() <= ratio:
                        tok_tensor = torch.tensor([[token_id]], device=generated.device)
                        generated = torch.cat([generated, tok_tensor], dim=1)
                        tokens_generated += 1
                    else:
                        # Reject: sample from residual (p_target - p_draft)+
                        t_full = target_probs[target_pos]
                        adjusted = (t_full - t_prob).clamp(min=0.0)
                        if adjusted.sum() < 1e-10:
                            adjusted = t_full.clone()
                        adjusted /= adjusted.sum()
                        corrected = torch.multinomial(adjusted, 1).squeeze(-1)
                        generated = torch.cat([generated, corrected.view(1, 1)], dim=1)
                        tokens_generated += 1
                        break

                if cfg.eos_token_id is not None and token_id == cfg.eos_token_id:
                    return generated
                if tokens_generated >= cfg.max_new_tokens:
                    return generated
            else:
                # All tokens accepted: sample bonus token
                bonus_pos = current_len + len(best_path) - 1
                bonus_tok = _sample_from_probs(target_probs[bonus_pos], top_p=cfg.top_p)
                generated = torch.cat([generated, bonus_tok.view(1, 1)], dim=1)
                tokens_generated += 1

        return generated
