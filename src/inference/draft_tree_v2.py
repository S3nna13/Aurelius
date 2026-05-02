"""Draft Tree Speculative Decoding — tree-structured draft generation and verification."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configs / data structures
# ---------------------------------------------------------------------------


@dataclass
class DraftTreeConfig:
    """Configuration for draft-tree speculative decoding."""

    branch_factor: int = 3
    depth: int = 4
    acceptance_threshold: float = 0.8
    vocab_size: int = 256


@dataclass
class TreeNode:
    """A single node in the draft tree."""

    token_id: int
    log_prob: float
    children: list[TreeNode] = field(default_factory=list)
    depth: int = 0
    is_accepted: bool = False


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_draft_tree(
    model: torch.nn.Module,
    input_ids: Tensor,
    config: DraftTreeConfig,
) -> TreeNode:
    """Expand a draft tree from *input_ids* using greedy top-k branching.

    At each node the draft model is called once; the top *branch_factor*
    tokens become children.  Recursion stops at *config.depth*.

    Returns the root TreeNode (whose token_id is the last token of
    *input_ids*).
    """
    # Root node is the last token already in the context.
    root_token = int(input_ids[0, -1].item())

    # Get log-probs for the root position.
    loss, logits, _ = model(input_ids)  # (B, T, V)
    last_logits = logits[0, -1]  # (V,)
    log_probs = F.log_softmax(last_logits, dim=-1)

    root = TreeNode(token_id=root_token, log_prob=0.0, depth=0)

    def _expand(parent: TreeNode, context: Tensor, parent_log_probs: Tensor) -> None:
        if parent.depth >= config.depth:
            return

        k = min(config.branch_factor, parent_log_probs.size(-1))
        top_lps, top_ids = parent_log_probs.topk(k)

        for i in range(k):
            tid = int(top_ids[i].item())
            lp = float(top_lps[i].item())
            child = TreeNode(
                token_id=tid,
                log_prob=lp,
                depth=parent.depth + 1,
            )
            parent.children.append(child)

            if child.depth < config.depth:
                # Extend context with child token and run model.
                new_ctx = torch.cat(
                    [context, torch.tensor([[tid]], device=context.device)],
                    dim=1,
                )
                _, child_logits, _ = model(new_ctx)
                child_lp = F.log_softmax(child_logits[0, -1], dim=-1)
                _expand(child, new_ctx, child_lp)

    _expand(root, input_ids, log_probs)
    return root


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def score_path(path: list[TreeNode]) -> float:
    """Sum of log_prob values along *path*."""
    return sum(node.log_prob for node in path)


def extract_all_paths(root: TreeNode) -> list[list[TreeNode]]:
    """DFS to extract every root-to-leaf path."""
    if not root.children:
        return [[root]]
    paths: list[list[TreeNode]] = []
    for child in root.children:
        for sub in extract_all_paths(child):
            paths.append([root] + sub)
    return paths


def best_path(root: TreeNode) -> list[int]:
    """Return token ids of the highest-scoring root-to-leaf path."""
    all_paths = extract_all_paths(root)
    best = max(all_paths, key=score_path)
    return [n.token_id for n in best]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_path(
    model: torch.nn.Module,
    input_ids: Tensor,
    path: list[int],
    threshold: float,
) -> tuple[list[int], int]:
    """Verify each token in *path* against the target model.

    Feeds *input_ids* (optionally extended one token at a time) through
    *model* and checks whether the draft token's probability exceeds
    *threshold*.

    Returns (accepted_tokens, n_accepted).
    """
    accepted: list[int] = []
    ctx = input_ids.clone()

    for token_id in path:
        loss, logits, _ = model(ctx)
        probs = F.softmax(logits[0, -1], dim=-1)
        p = float(probs[token_id].item())
        if p >= threshold:
            accepted.append(token_id)
            ctx = torch.cat(
                [ctx, torch.tensor([[token_id]], device=ctx.device)],
                dim=1,
            )
        else:
            break

    return accepted, len(accepted)


# ---------------------------------------------------------------------------
# Decoder class
# ---------------------------------------------------------------------------


class DraftTreeDecoder:
    """Iteratively build draft trees and verify them to generate tokens."""

    def __init__(self, config: DraftTreeConfig | None = None) -> None:
        self.config = config or DraftTreeConfig()

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        input_ids: Tensor,
        max_new_tokens: int = 32,
    ) -> Tensor:
        """Generate up to *max_new_tokens* new tokens.

        Each iteration:
        1. Build a draft tree.
        2. Find the best path.
        3. Verify it against the model.
        4. Append accepted tokens (at least one fallback greedy token).

        Returns an int64 tensor of shape (1, original_len + generated).
        """
        ctx = input_ids.clone()
        generated = 0

        while generated < max_new_tokens:
            root = build_draft_tree(model, ctx, self.config)
            path_ids = best_path(root)

            # Skip root token (already in context) for verification.
            candidate = path_ids[1:] if len(path_ids) > 1 else path_ids

            accepted, n_acc = verify_path(model, ctx, candidate, self.config.acceptance_threshold)

            if n_acc > 0:
                new_tokens = torch.tensor([accepted], dtype=torch.long, device=ctx.device)
                ctx = torch.cat([ctx, new_tokens], dim=1)
                generated += n_acc
            else:
                # Fallback: greedily pick the single most-likely token.
                loss, logits, _ = model(ctx)
                next_tok = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                ctx = torch.cat([ctx, next_tok], dim=1)
                generated += 1

        return ctx
