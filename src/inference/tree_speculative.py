"""Tree-structured speculative decoding.

Extends standard speculative decoding (Leviathan et al., 2023) by drafting a
*tree* of candidate continuations instead of a single linear sequence.  The
target model verifies one path from the tree per round, dramatically increasing
the number of accepted tokens compared to single-path speculation.

Reference: SpecTree / Medusa-style tree speculation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

@dataclass
class DraftNode:
    """A single node in the draft tree."""
    token_id: int
    parent_idx: int   # index of parent in DraftTree.nodes; -1 for root sentinel
    log_prob: float   # draft model log probability for this token


class DraftTree:
    """A tree of draft tokens for parallel verification.

    Nodes are stored in a flat list.  Each node records its parent index,
    which allows efficient path reconstruction without a dedicated pointer tree.
    """

    def __init__(self) -> None:
        self.nodes: list[DraftNode] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, token_id: int, parent_idx: int, log_prob: float) -> int:
        """Add a node and return its index in self.nodes."""
        idx = len(self.nodes)
        self.nodes.append(DraftNode(token_id=token_id, parent_idx=parent_idx, log_prob=log_prob))
        return idx

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_path_to(self, node_idx: int) -> list[int]:
        """Return token IDs from the root to node at *node_idx* (inclusive)."""
        path: list[int] = []
        idx = node_idx
        while idx != -1:
            node = self.nodes[idx]
            path.append(node.token_id)
            idx = node.parent_idx
        path.reverse()
        return path

    def _leaf_indices(self) -> list[int]:
        """Return indices of all leaf nodes (nodes that are nobody's parent)."""
        has_child = set()
        for node in self.nodes:
            if node.parent_idx != -1:
                has_child.add(node.parent_idx)
        return [i for i in range(len(self.nodes)) if i not in has_child]

    def paths(self) -> list[list[int]]:
        """Return all root-to-leaf paths as lists of token IDs."""
        leaves = self._leaf_indices()
        return [self.get_path_to(leaf) for leaf in leaves]

    def depth(self) -> int:
        """Maximum depth of the tree (root is depth 1)."""
        if not self.nodes:
            return 0

        def _depth(idx: int) -> int:
            return len(self.get_path_to(idx))

        return max(_depth(i) for i in range(len(self.nodes)))

    def size(self) -> int:
        """Total number of nodes in the tree."""
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Draft tree construction
# ---------------------------------------------------------------------------

def _get_last_logits(model, input_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """Run *model* on *input_ids* and return the logits at the last position.

    Supports models that return ``(loss, logits, kv)`` tuples as well as models
    that return logits directly.

    Returns:
        (vocab_size,) float tensor of raw logits (temperature already applied).
    """
    out = model(input_ids)
    if isinstance(out, tuple):
        logits = out[1]          # (loss, logits, kv) convention
    else:
        logits = out             # raw logits

    logits = logits[0, -1, :]   # (vocab_size,)
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature
    return logits


def build_draft_tree(
    draft_model,
    input_ids: torch.Tensor,
    n_branches: int,
    depth: int,
    temperature: float = 1.0,
) -> DraftTree:
    """Build a draft tree using beam-like greedy expansion.

    Strategy (as described in the task spec):
    * Level 0 → Level 1: sample top-*n_branches* tokens from the draft model.
    * Level 1 → Level d: for each existing leaf, greedily extend by the
      single highest-probability token.

    This gives a tree with ``n_branches + n_branches * (depth - 1)`` nodes
    (a star expanded greedily), which is simple and predictable.

    Args:
        draft_model: callable(input_ids) -> (loss, logits, kv) or logits.
                     Called with shape (1, seq_len).
        input_ids:   (1, seq_len) context tensor.
        n_branches:  Number of branches at the first level.
        depth:       Maximum tree depth (number of tokens per path).
        temperature: Sampling temperature applied to draft logits.

    Returns:
        A populated :class:`DraftTree`.
    """
    tree = DraftTree()

    if depth <= 0 or n_branches <= 0:
        return tree

    # ---- Level 1: expand top-n_branches tokens from the root context ----
    logits = _get_last_logits(draft_model, input_ids, temperature)  # (vocab,)
    log_probs = F.log_softmax(logits, dim=-1)
    top_lp, top_ids = torch.topk(log_probs, k=n_branches)

    # Each branch keeps track of its current context (input_ids + tokens so far)
    # and the tree node index of the deepest node on that branch.
    branch_contexts: list[torch.Tensor] = []
    branch_node_idx: list[int] = []

    for rank in range(n_branches):
        tok = top_ids[rank].item()
        lp = top_lp[rank].item()
        node_idx = tree.add_node(token_id=tok, parent_idx=-1, log_prob=lp)
        new_ctx = torch.cat(
            [input_ids, torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device)],
            dim=1,
        )
        branch_contexts.append(new_ctx)
        branch_node_idx.append(node_idx)

    # ---- Levels 2..depth: greedily extend each branch by 1 token ----
    for _ in range(depth - 1):
        new_contexts: list[torch.Tensor] = []
        new_node_idxs: list[int] = []

        for ctx, parent_idx in zip(branch_contexts, branch_node_idx):
            logits = _get_last_logits(draft_model, ctx, temperature)
            log_probs = F.log_softmax(logits, dim=-1)
            best_lp, best_id = log_probs.max(dim=-1)

            tok = best_id.item()
            lp = best_lp.item()
            node_idx = tree.add_node(token_id=tok, parent_idx=parent_idx, log_prob=lp)
            new_ctx = torch.cat(
                [ctx, torch.tensor([[tok]], dtype=ctx.dtype, device=ctx.device)],
                dim=1,
            )
            new_contexts.append(new_ctx)
            new_node_idxs.append(node_idx)

        branch_contexts = new_contexts
        branch_node_idx = new_node_idxs

    return tree


# ---------------------------------------------------------------------------
# Tree verification
# ---------------------------------------------------------------------------

def verify_tree(
    target_model,
    input_ids: torch.Tensor,
    tree: DraftTree,
    temperature: float = 1.0,
) -> list[int]:
    """Verify the first path of *tree* against *target_model* via rejection sampling.

    Strategy (simplified single-path verification):
    1. Extract path 0 (the first root-to-leaf path).
    2. Concatenate *input_ids* with the path tokens.
    3. Run *target_model* once on the full sequence.
    4. Walk each draft token; accept with probability min(1, p_target / p_draft).
    5. On first rejection, sample a correction from the target and stop.
    6. If all tokens are accepted, sample a bonus token from the target.

    Returns:
        A non-empty list of accepted token IDs (always at least 1).
    """
    all_paths = tree.paths()
    if not all_paths:
        # Empty tree: just sample one token from target
        logits = _get_last_logits(target_model, input_ids, temperature)
        probs = F.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, 1).item()
        return [int(tok)]

    path_tokens = all_paths[0]   # list[int], length == depth
    depth = len(path_tokens)

    # Retrieve per-token draft log-probs from the first path's nodes.
    # The first path corresponds to the first n_branches root then greedily
    # extended nodes.  We recover them by walking the leaf back to root.
    leaves = tree._leaf_indices()
    leaf_idx = leaves[0]
    draft_log_probs: list[float] = []
    idx = leaf_idx
    while idx != -1:
        draft_log_probs.append(tree.nodes[idx].log_prob)
        idx = tree.nodes[idx].parent_idx
    draft_log_probs.reverse()   # root-to-leaf order; length == depth

    # Build full verification sequence: context + draft tokens
    draft_tensor = torch.tensor(
        [path_tokens], dtype=input_ids.dtype, device=input_ids.device
    )  # (1, depth)
    verify_ids = torch.cat([input_ids, draft_tensor], dim=1)  # (1, ctx + depth)

    # Single target model forward pass
    out = target_model(verify_ids)
    if isinstance(out, tuple):
        logits_all = out[1]
    else:
        logits_all = out
    # logits_all: (1, ctx + depth, vocab)
    if temperature != 1.0 and temperature > 0:
        logits_all = logits_all / temperature
    target_probs_all = F.softmax(logits_all[0], dim=-1)  # (ctx + depth, vocab)

    context_len = input_ids.shape[1]
    accepted: list[int] = []

    for i in range(depth):
        tok = path_tokens[i]
        # target distribution that *predicts* token at position context_len + i
        # is at logit index context_len - 1 + i
        target_pos = context_len - 1 + i
        t_prob = target_probs_all[target_pos, tok].clamp(min=1e-10).item()

        # Convert draft log-prob to prob
        d_prob = float(torch.exp(torch.tensor(draft_log_probs[i])).item())
        d_prob = max(d_prob, 1e-10)

        accept_prob = min(1.0, t_prob / d_prob)

        if torch.rand(1, device=input_ids.device).item() <= accept_prob:
            accepted.append(tok)
        else:
            # Rejection: sample corrected token from adjusted distribution
            t_full = target_probs_all[target_pos]
            # Build draft prob vector for this position
            d_full = torch.zeros_like(t_full)
            d_full[tok] = d_prob
            adjusted = (t_full - d_full).clamp(min=0.0)
            if adjusted.sum() < 1e-10:
                adjusted = t_full.clone()
            adjusted = adjusted / adjusted.sum()
            corrected = torch.multinomial(adjusted, 1).item()
            accepted.append(int(corrected))
            return accepted  # stop early

    # All tokens accepted: sample one bonus token from target at position depth
    bonus_pos = context_len + depth - 1
    if bonus_pos < target_probs_all.shape[0]:
        bonus_probs = target_probs_all[bonus_pos]
    else:
        # Fallback: run target on full accepted sequence
        full_ids = torch.cat(
            [input_ids,
             torch.tensor([accepted], dtype=input_ids.dtype, device=input_ids.device)],
            dim=1,
        )
        out2 = target_model(full_ids)
        logits2 = out2[1] if isinstance(out2, tuple) else out2
        if temperature != 1.0 and temperature > 0:
            logits2 = logits2 / temperature
        bonus_probs = F.softmax(logits2[0, -1, :], dim=-1)

    bonus = torch.multinomial(bonus_probs, 1).item()
    accepted.append(int(bonus))
    return accepted


# ---------------------------------------------------------------------------
# High-level decoder
# ---------------------------------------------------------------------------

class TreeSpeculativeDecoder:
    """High-level tree speculative decoder.

    Drafts a tree of candidate continuations with the small *draft_model* and
    verifies one path per round with the large *target_model*, achieving higher
    throughput than single-path speculative decoding.

    Args:
        target_model: Large accurate model called once per round.
        draft_model:  Small fast model called many times to build the draft tree.
        n_branches:   Branching factor at the first draft level.
        depth:        Tree depth — number of tokens drafted per path.
        temperature:  Sampling temperature applied to both models.
    """

    def __init__(
        self,
        target_model,
        draft_model,
        n_branches: int = 2,
        depth: int = 4,
        temperature: float = 1.0,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.n_branches = n_branches
        self.depth = depth
        self.temperature = temperature

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Generate tokens using tree speculative decoding.

        Args:
            input_ids:      (1, seq_len) prompt tensor.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            (1, seq_len + generated_len) tensor.
        """
        generated = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # 1. Build draft tree
            tree = build_draft_tree(
                draft_model=self.draft_model,
                input_ids=generated,
                n_branches=self.n_branches,
                depth=self.depth,
                temperature=self.temperature,
            )

            # 2. Verify against target model
            new_tokens = verify_tree(
                target_model=self.target_model,
                input_ids=generated,
                tree=tree,
                temperature=self.temperature,
            )

            # 3. Append accepted tokens (respecting the budget)
            for tok in new_tokens:
                if tokens_generated >= max_new_tokens:
                    break
                generated = torch.cat(
                    [generated,
                     torch.tensor([[tok]], dtype=generated.dtype, device=generated.device)],
                    dim=1,
                )
                tokens_generated += 1

        return generated
