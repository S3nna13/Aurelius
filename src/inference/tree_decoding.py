"""Speculative tree decoding — draft a tree of candidates, verify with one target forward."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TreeConfig:
    """Configuration for speculative tree decoding."""
    branching_factor: int = 2   # number of draft tokens per node
    tree_depth: int = 3         # depth of draft tree
    temperature: float = 1.0
    top_p: float = 0.9


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A single node in the draft tree."""
    token_id: int
    log_prob: float
    parent: "TreeNode | None" = None
    children: list["TreeNode"] = field(default_factory=list)
    depth: int = 0

    def path(self) -> list[int]:
        """Return token_ids from root to this node (walk parent chain).

        The sentinel root (token_id == -1) is excluded from the result so that
        paths contain only real draft tokens.
        """
        tokens: list[int] = []
        node: TreeNode | None = self
        while node is not None and node.token_id != -1:
            tokens.append(node.token_id)
            node = node.parent
        tokens.reverse()
        return tokens


class DraftTree:
    """A tree of draft token candidates for parallel speculative verification.

    The tree has a sentinel root node (token_id=-1, no parent).  Children of
    the sentinel are the level-0 draft tokens.
    """

    def __init__(self, root_ids: list[int]) -> None:
        # Sentinel root: token_id=-1, no parent, depth=0
        self.root = TreeNode(token_id=-1, log_prob=0.0, parent=None, depth=0)
        self._root_ids = root_ids  # prompt token ids (kept for reference)
        self.leaves: list[TreeNode] = []

    def add_children(
        self,
        node: TreeNode,
        token_ids: list[int],
        log_probs: list[float],
    ) -> None:
        """Add children to *node* and update leaves list.

        Removes *node* from leaves (it is now an internal node) and appends
        the new children.
        """
        # Remove node from leaves if it's there
        if node in self.leaves:
            self.leaves.remove(node)

        for tok, lp in zip(token_ids, log_probs):
            child = TreeNode(
                token_id=tok,
                log_prob=lp,
                parent=node,
                depth=node.depth + 1,
            )
            node.children.append(child)
            self.leaves.append(child)

    def all_paths(self) -> list[list[int]]:
        """Return all root-to-leaf paths as lists of token_ids (excluding sentinel root)."""
        paths: list[list[int]] = []
        for leaf in self.leaves:
            # Walk parent chain, stop before the sentinel root (token_id == -1)
            tokens: list[int] = []
            node: TreeNode | None = leaf
            while node is not None and node.token_id != -1:
                tokens.append(node.token_id)
                node = node.parent
            tokens.reverse()
            paths.append(tokens)
        return paths

    def num_nodes(self) -> int:
        """Total number of non-sentinel nodes in the tree."""
        count = 0
        stack = list(self.root.children)
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count


# ---------------------------------------------------------------------------
# Draft tree construction
# ---------------------------------------------------------------------------

def _get_last_logits(model, input_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """Run *model* on *input_ids* and return logits at the last position.

    Handles models returning ``(loss, logits, kv)`` tuples and raw logits.

    Returns:
        (vocab_size,) float tensor of raw logits (temperature already applied).
    """
    out = model(input_ids)
    if isinstance(out, tuple):
        logits = out[1]
    else:
        logits = out
    logits = logits[0, -1, :]   # (vocab_size,)
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature
    return logits


def build_draft_tree(
    draft_model,
    prompt_ids: list[int],
    config: TreeConfig,
) -> DraftTree:
    """Build a draft tree by expanding each leaf with top-branching_factor tokens.

    At each depth level, for each leaf node, run draft_model on
    ``prompt_ids + leaf.path()`` and take the top-``branching_factor`` tokens.

    Returns a completed DraftTree with ``branching_factor^tree_depth`` leaves.

    Args:
        draft_model: callable(input_ids: Tensor) -> (loss, logits, kv) or logits.
        prompt_ids:  Prompt token IDs as a plain Python list.
        config:      TreeConfig controlling branching factor, depth, etc.

    Returns:
        A populated :class:`DraftTree`.
    """
    tree = DraftTree(root_ids=prompt_ids)

    if config.tree_depth <= 0 or config.branching_factor <= 0:
        return tree

    # Determine device from a quick model probe
    device = torch.device("cpu")

    def _make_ids(prefix: list[int]) -> torch.Tensor:
        return torch.tensor([prefix], dtype=torch.long, device=device)

    # Initialise: expand the sentinel root using the raw prompt
    root_input = _make_ids(prompt_ids)
    logits = _get_last_logits(draft_model, root_input, config.temperature)
    log_probs = F.log_softmax(logits, dim=-1)
    top_lp, top_ids = torch.topk(log_probs, k=config.branching_factor)

    tree.add_children(
        tree.root,
        token_ids=[int(t) for t in top_ids],
        log_probs=[float(lp) for lp in top_lp],
    )

    # Expand depth-1 through depth-(tree_depth-1)
    for _depth in range(1, config.tree_depth):
        # snapshot current leaves so we iterate over them, not newly added ones
        current_leaves = list(tree.leaves)
        for leaf in current_leaves:
            path = leaf.path()  # token_ids from level-0 to leaf (excludes sentinel)
            leaf_input = _make_ids(prompt_ids + path)
            logits = _get_last_logits(draft_model, leaf_input, config.temperature)
            log_probs = F.log_softmax(logits, dim=-1)
            top_lp, top_ids = torch.topk(log_probs, k=config.branching_factor)

            tree.add_children(
                leaf,
                token_ids=[int(t) for t in top_ids],
                log_probs=[float(lp) for lp in top_lp],
            )

    return tree


# ---------------------------------------------------------------------------
# Tree verification
# ---------------------------------------------------------------------------

def verify_tree(
    target_model,
    prompt_ids: list[int],
    tree: DraftTree,
) -> list[int]:
    """Verify draft tree paths against target_model; return accepted tokens.

    Simplified verification strategy:
    * For each path in tree.all_paths():
        - Run target_model on ``prompt_ids + path``.
        - Accept tokens that agree with target_model's greedy prediction at each
          position.
        - Track the longest accepted prefix.
    * Return the longest accepted path plus 1 bonus token from target.
    * If no path has any agreement, return [target_model greedy next token].

    Args:
        target_model: callable returning ``(loss, logits, kv)`` or raw logits.
        prompt_ids:   Prompt token IDs as a plain Python list.
        tree:         Populated DraftTree.

    Returns:
        Non-empty list of accepted token IDs.
    """
    device = torch.device("cpu")
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    context_len = len(prompt_ids)

    all_paths = tree.all_paths()

    if not all_paths:
        # Empty tree: just return greedy next token from target
        logits = _get_last_logits(target_model, prompt_tensor, temperature=1.0)
        next_tok = int(logits.argmax(dim=-1).item())
        return [next_tok]

    best_accepted: list[int] = []
    best_bonus_tok: int | None = None

    for path in all_paths:
        if not path:
            continue
        path_tensor = torch.tensor([path], dtype=torch.long, device=device)
        verify_ids = torch.cat([prompt_tensor, path_tensor], dim=1)

        out = target_model(verify_ids)
        logits_all = out[1] if isinstance(out, tuple) else out
        # logits_all: (1, context_len + len(path), vocab)
        # position i in logits_all predicts token at position i+1

        accepted: list[int] = []
        agreed = True
        for i, tok in enumerate(path):
            target_pos = context_len - 1 + i   # logits position predicting path[i]
            greedy = int(logits_all[0, target_pos, :].argmax(dim=-1).item())
            if greedy == tok:
                accepted.append(tok)
            else:
                agreed = False
                break

        if len(accepted) > len(best_accepted):
            best_accepted = accepted
            # Bonus token: target's greedy prediction after the last accepted token
            if agreed:
                bonus_pos = context_len + len(accepted) - 1
                if bonus_pos < logits_all.shape[1]:
                    best_bonus_tok = int(logits_all[0, bonus_pos, :].argmax(dim=-1).item())
                else:
                    best_bonus_tok = None
            else:
                # Greedy token at the rejection position
                reject_pos = context_len - 1 + len(accepted)
                best_bonus_tok = int(logits_all[0, reject_pos, :].argmax(dim=-1).item())

    if not best_accepted and best_bonus_tok is None:
        # No path matched at all — return target greedy next token
        logits = _get_last_logits(target_model, prompt_tensor, temperature=1.0)
        return [int(logits.argmax(dim=-1).item())]

    result = best_accepted[:]
    if best_bonus_tok is not None:
        result.append(best_bonus_tok)

    if not result:
        logits = _get_last_logits(target_model, prompt_tensor, temperature=1.0)
        return [int(logits.argmax(dim=-1).item())]

    return result


# ---------------------------------------------------------------------------
# High-level decoder
# ---------------------------------------------------------------------------

class TreeSpeculativeDecoder:
    """High-level speculative decoder using tree-structured drafting.

    Drafts a tree of candidate continuations with *draft_model* and verifies
    the best path with *target_model*, accepting the longest matching prefix
    plus a bonus token each round.

    Args:
        target_model: Large accurate model called once per generation round.
        draft_model:  Small fast model used to build the draft tree.
        config:       TreeConfig controlling branching factor, depth, etc.
    """

    def __init__(
        self,
        target_model,
        draft_model,
        config: TreeConfig | None = None,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config or TreeConfig()
        self._total_steps = 0
        self._total_accepted = 0

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """Generate tokens using speculative tree decoding.

        Args:
            input_ids:      (1, seq_len) prompt tensor.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            (1, seq_len + n_accepted) tensor.
        """
        generated = input_ids.clone()
        tokens_generated = 0
        self._total_steps = 0
        self._total_accepted = 0

        while tokens_generated < max_new_tokens:
            prompt_ids = generated[0].tolist()

            tree = build_draft_tree(
                draft_model=self.draft_model,
                prompt_ids=prompt_ids,
                config=self.config,
            )

            new_tokens = verify_tree(
                target_model=self.target_model,
                prompt_ids=prompt_ids,
                tree=tree,
            )

            self._total_steps += 1
            n_to_add = min(len(new_tokens), max_new_tokens - tokens_generated)
            new_tokens = new_tokens[:n_to_add]
            self._total_accepted += len(new_tokens)

            for tok in new_tokens:
                generated = torch.cat(
                    [
                        generated,
                        torch.tensor([[tok]], dtype=generated.dtype, device=generated.device),
                    ],
                    dim=1,
                )
            tokens_generated += len(new_tokens)

        return generated

    def stats(self) -> dict[str, float]:
        """Return generation statistics.

        Returns:
            dict with keys:
                ``mean_accepted_per_step``: average tokens accepted per round.
                ``total_steps``: number of draft-verify rounds performed.
        """
        mean = (
            self._total_accepted / self._total_steps
            if self._total_steps > 0
            else 0.0
        )
        return {
            "mean_accepted_per_step": float(mean),
            "total_steps": int(self._total_steps),
        }
