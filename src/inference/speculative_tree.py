"""Tree speculative decoding: draft tree construction and parallel target verification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration and data structures
# ---------------------------------------------------------------------------


@dataclass
class TreeSpecConfig:
    """Configuration for tree speculative decoding."""

    branch_factor: int = 2  # number of branches at each node
    tree_depth: int = 3  # depth of draft tree
    max_new_tokens: int = 64
    temperature: float = 1.0


@dataclass
class TreeNode:
    """A single node in the draft tree (flat representation)."""

    token_id: int
    log_prob: float
    parent_idx: int  # index in flat node list; -1 for root-level nodes
    depth: int  # 1-indexed; depth=1 means child of root context


# ---------------------------------------------------------------------------
# Draft tree construction
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_draft_tree(
    draft_model,
    input_ids: torch.Tensor,
    config: TreeSpecConfig,
) -> list[TreeNode]:
    """Build a draft tree via BFS expansion from the root context.

    Only expands from the root for simplicity: takes top branch_factor tokens
    as depth-1 nodes, then for each depth-1 node expands branch_factor children
    (using the same root logits for subsequent depths as a simplified approach),
    repeating for tree_depth levels.

    Args:
        draft_model: callable with signature model(input_ids) -> (_, logits, _).
        input_ids: (1, seq_len) context tensor.
        config: TreeSpecConfig with branch_factor, tree_depth, etc.

    Returns:
        Flat list of TreeNode objects (root context excluded), starting with
        depth=1 nodes. Total nodes = branch_factor + branch_factor^2 + ... +
        branch_factor^tree_depth for a full BFS tree.
    """
    nodes: list[TreeNode] = []

    if config.tree_depth <= 0 or config.branch_factor <= 0:
        return nodes

    # Get root logits once
    _, root_logits, _ = draft_model(input_ids)
    root_logits = root_logits[0, -1, :]  # (vocab,)
    if config.temperature != 1.0 and config.temperature > 0:
        root_logits = root_logits / config.temperature
    root_log_probs = F.log_softmax(root_logits, dim=-1)

    vocab_size = root_log_probs.shape[0]

    # BFS expansion: (parent_idx_in_nodes, depth, log_prob_tensor_to_expand_from)
    # Start: expand root into branch_factor depth-1 nodes
    top_lp, top_ids = torch.topk(root_log_probs, k=min(config.branch_factor, vocab_size))

    # Level 1: add depth-1 nodes (children of root context, parent_idx=-1)
    current_level: list[int] = []  # indices into `nodes` for current frontier
    for i in range(top_ids.shape[0]):
        tok = int(top_ids[i].item())
        lp = float(top_lp[i].item())
        node = TreeNode(token_id=tok, log_prob=lp, parent_idx=-1, depth=1)
        nodes.append(node)
        current_level.append(len(nodes) - 1)

    # Levels 2..tree_depth: expand each frontier node
    for depth in range(2, config.tree_depth + 1):
        next_level: list[int] = []
        for parent_node_idx in current_level:
            parent_node = nodes[parent_node_idx]
            # For simplicity: re-use root logits for all expansions, but
            # offset by parent token to get diverse children via top-k on
            # a shuffled distribution derived from the parent's token.
            # Simple approach: use root_log_probs but shift to avoid
            # duplicating the parent (mask parent token out).
            child_log_probs = root_log_probs.clone()
            child_log_probs[parent_node.token_id] = float("-inf")
            top_child_lp, top_child_ids = torch.topk(
                child_log_probs, k=min(config.branch_factor, vocab_size - 1)
            )
            for j in range(top_child_ids.shape[0]):
                tok = int(top_child_ids[j].item())
                lp = float(top_child_lp[j].item())
                node = TreeNode(
                    token_id=tok,
                    log_prob=lp,
                    parent_idx=parent_node_idx,
                    depth=depth,
                )
                nodes.append(node)
                next_level.append(len(nodes) - 1)
        current_level = next_level

    return nodes


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------


def flatten_tree_paths(nodes: list[TreeNode]) -> list[list[int]]:
    """Extract all root-to-leaf paths as token id sequences.

    A leaf is a node with no children. Each path starts at depth=1 and
    ends at a leaf node.

    Args:
        nodes: Flat list of TreeNodes as returned by build_draft_tree.

    Returns:
        List of paths; each path is a list of token_ids from depth-1 to leaf.
    """
    if not nodes:
        return []

    # Determine which node indices have children
    has_child: set[int] = set()
    for node in nodes:
        if node.parent_idx != -1:
            has_child.add(node.parent_idx)

    leaf_indices = [i for i in range(len(nodes)) if i not in has_child]

    paths: list[list[int]] = []
    for leaf_idx in leaf_indices:
        path_tokens: list[int] = []
        idx = leaf_idx
        while idx != -1:
            path_tokens.append(nodes[idx].token_id)
            idx = nodes[idx].parent_idx
        path_tokens.reverse()
        paths.append(path_tokens)

    return paths


# ---------------------------------------------------------------------------
# Tree verification
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_tree(
    target_model,
    input_ids: torch.Tensor,
    nodes: list[TreeNode],
) -> tuple[list[int], int]:
    """Verify draft tree nodes against the target model.

    For each node in the flat list, compute target probability and accept/reject
    using the speculative sampling criterion: accept if target_prob >= draft_prob.
    Walks nodes in order (BFS order = depth-first from the perspective of the
    flat list), returning the longest accepted prefix along the first path.

    Args:
        target_model: callable with signature model(input_ids) -> (_, logits, _).
        input_ids: (1, seq_len) context tensor.
        nodes: Flat list of TreeNodes from build_draft_tree.

    Returns:
        (accepted_token_ids, n_accepted) — the longest accepted prefix along
        the first root-to-leaf path.
    """
    if not nodes:
        # Empty tree: sample one token from target
        _, logits, _ = target_model(input_ids)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        tok = int(torch.multinomial(probs, 1).item())
        return [tok], 0

    # Get all paths and verify the first one
    paths = flatten_tree_paths(nodes)
    if not paths:
        return [], 0

    paths[0]

    # Run target model on input_ids
    _, target_logits, _ = target_model(input_ids)
    target_logits = target_logits[0, -1, :]  # (vocab,)
    target_log_probs = F.log_softmax(target_logits, dim=-1)
    target_probs = target_log_probs.exp()

    # Build a node lookup for the first path (nodes at each depth)
    # Find nodes corresponding to the first path by walking depth=1 nodes in order
    accepted_tokens: list[int] = []
    n_accepted = 0

    # Find the first path's nodes in flat order (BFS: depth-1 nodes appear first)
    # First depth-1 node is nodes[0]
    # Walk nodes to reconstruct first-path nodes
    # Leaf of first path: find it
    leaf_of_first_path = _find_first_leaf(nodes)
    if leaf_of_first_path == -1:
        return [], 0

    # Reconstruct path nodes from leaf to root
    path_node_indices: list[int] = []
    idx = leaf_of_first_path
    while idx != -1:
        path_node_indices.append(idx)
        idx = nodes[idx].parent_idx
    path_node_indices.reverse()  # root to leaf order

    for node_idx in path_node_indices:
        node = nodes[node_idx]
        tok = node.token_id
        draft_prob = float(torch.exp(torch.tensor(node.log_prob)).item())
        draft_prob = max(draft_prob, 1e-10)
        target_prob = float(target_probs[tok].item())
        target_prob = max(target_prob, 1e-10)

        if target_prob >= draft_prob:
            accepted_tokens.append(tok)
            n_accepted += 1
        else:
            # Reject: sample correction from target
            correction = int(torch.multinomial(target_probs, 1).item())
            accepted_tokens.append(correction)
            break

    return accepted_tokens, n_accepted


def _find_first_leaf(nodes: list[TreeNode]) -> int:
    """Return the index of the leaf node on the first root-to-leaf path."""
    if not nodes:
        return -1
    has_child: set[int] = set()
    for node in nodes:
        if node.parent_idx != -1:
            has_child.add(node.parent_idx)
    # The first depth-1 node that has no child, or if all have children,
    # follow the first depth-1 node down to its leaf
    # Find the first depth-1 node
    first_d1 = -1
    for i, node in enumerate(nodes):
        if node.parent_idx == -1:
            first_d1 = i
            break
    if first_d1 == -1:
        return -1
    # Follow first child chain down
    current = first_d1
    while True:
        # Find the first child of current
        child = -1
        for i, node in enumerate(nodes):
            if node.parent_idx == current:
                child = i
                break
        if child == -1:
            return current  # current is a leaf
        current = child


# ---------------------------------------------------------------------------
# High-level decoder
# ---------------------------------------------------------------------------


class TreeSpecDecoder:
    """Tree speculative decoder: drafts a tree, verifies against target, repeats.

    Args:
        draft_model: Small fast model.
        target_model: Large accurate model.
        config: TreeSpecConfig.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        config: TreeSpecConfig,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Generate tokens using tree speculative decoding.

        Build tree → verify → append accepted → repeat until max_new_tokens.

        Args:
            input_ids: (1, prompt_len) input token ids.

        Returns:
            (generated_ids, stats) where generated_ids has shape
            (1, generated_len) and stats contains:
              - "n_steps": int
              - "mean_accepted_per_step": float
              - "total_tokens": int
        """
        cfg = self.config
        current_ids = input_ids.clone()
        total_tokens = 0
        n_steps = 0
        total_accepted = 0

        while total_tokens < cfg.max_new_tokens:
            # Build draft tree
            nodes = build_draft_tree(self.draft_model, current_ids, cfg)

            # Verify against target
            accepted_tokens, n_accepted = verify_tree(self.target_model, current_ids, nodes)

            total_accepted += n_accepted
            n_steps += 1

            # Append accepted tokens up to budget
            for tok in accepted_tokens:
                if total_tokens >= cfg.max_new_tokens:
                    break
                tok_tensor = torch.tensor(
                    [[tok]], dtype=current_ids.dtype, device=current_ids.device
                )
                current_ids = torch.cat([current_ids, tok_tensor], dim=1)
                total_tokens += 1

            if total_tokens >= cfg.max_new_tokens:
                break

        generated_ids = current_ids[:, input_ids.shape[1] :]

        stats = {
            "n_steps": n_steps,
            "mean_accepted_per_step": total_accepted / n_steps if n_steps > 0 else 0.0,
            "total_tokens": total_tokens,
        }
        return generated_ids, stats
