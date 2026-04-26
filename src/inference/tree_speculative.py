"""Tree-structured speculative decoding.

Extends standard speculative decoding by drafting a *tree* of candidate
continuations instead of a single linear sequence.  The target model
verifies multiple paths from the tree per round, picking the best accepted
sequence.

Reference: SpecTree / Medusa-style tree speculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TreeConfig:
    """Configuration for tree-structured speculative decoding."""

    branching_factor: int = 2
    tree_depth: int = 4
    max_new_tokens: int = 128
    temperature: float = 1.0
    typical_acceptance_rate: float = 0.8


# ---------------------------------------------------------------------------
# Draft tree data structure
# ---------------------------------------------------------------------------


@dataclass
class DraftNode:
    """A single node in the draft tree."""

    token_id: int
    log_prob: float
    depth: int
    children: list[DraftNode] = field(default_factory=list)
    parent: DraftNode | None = None

    def path_to_root(self) -> list[int]:
        """Return token IDs from root to this node (inclusive)."""
        path: list[int] = []
        node: DraftNode | None = self
        while node is not None:
            path.append(node.token_id)
            node = node.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Helper: run model and get last-position logits
# ---------------------------------------------------------------------------


def _last_logits(model, input_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """Run model on input_ids and return (possibly temperature-scaled) logits
    at the final sequence position. Shape: (vocab_size,)."""
    out = model(input_ids)
    logits = out[1] if isinstance(out, tuple) else out  # (1, S, V)
    logits = logits[0, -1, :]  # (V,)
    if temperature != 1.0 and temperature > 0.0:
        logits = logits / temperature
    return logits


# ---------------------------------------------------------------------------
# Build draft tree
# ---------------------------------------------------------------------------


def build_draft_tree(
    model,
    prefix_ids: list[int],
    config: TreeConfig,
) -> DraftNode:
    """Build a tree of draft tokens.

    Strategy:
    - The root node represents prefix[-1] at depth 0.
    - At each depth level, sample the top-k (k=branching_factor) tokens from
      the model logits and attach them as children.
    - Recurse up to tree_depth levels deep.

    Args:
        model:      Callable returning (loss, logits, kv) or logits directly.
        prefix_ids: Context token IDs (list[int]).
        config:     TreeConfig controlling branching_factor, tree_depth, etc.

    Returns:
        Root DraftNode (token = prefix[-1], depth = 0).
    """
    device = torch.device("cpu")

    # Create a sentinel root that represents the last token in the prefix.
    root_token = prefix_ids[-1] if prefix_ids else 0
    root = DraftNode(token_id=root_token, log_prob=0.0, depth=0, parent=None)

    if config.tree_depth <= 0 or config.branching_factor <= 0:
        return root

    # BFS / recursive expansion — we use a simple recursive helper.
    def _expand(node: DraftNode, context: list[int]) -> None:
        if node.depth >= config.tree_depth:
            return
        # Build input tensor
        ids_tensor = torch.tensor([context], dtype=torch.long, device=device)
        logits = _last_logits(model, ids_tensor, config.temperature)
        log_probs = F.log_softmax(logits, dim=-1)
        k = min(config.branching_factor, log_probs.shape[0])
        top_lp, top_ids = torch.topk(log_probs, k=k)
        for rank in range(k):
            tok = int(top_ids[rank].item())
            lp = float(top_lp[rank].item())
            child = DraftNode(
                token_id=tok,
                log_prob=lp,
                depth=node.depth + 1,
                parent=node,
            )
            node.children.append(child)
            _expand(child, context + [tok])

    _expand(root, prefix_ids)
    return root


# ---------------------------------------------------------------------------
# Extract sequences from tree
# ---------------------------------------------------------------------------


def tree_to_sequences(root: DraftNode) -> list[list[int]]:
    """Enumerate all root-to-leaf paths in the tree.

    Returns a list of token sequences, each from the root node to a leaf node
    (inclusive).  If the root has no children it is its own leaf.
    """
    sequences: list[list[int]] = []

    def _collect(node: DraftNode) -> None:
        if not node.children:
            # Leaf — record the path from root to here
            sequences.append(node.path_to_root())
        else:
            for child in node.children:
                _collect(child)

    _collect(root)
    return sequences


# ---------------------------------------------------------------------------
# Verify tree sequences against the target model
# ---------------------------------------------------------------------------


def verify_tree(
    model,
    prefix_ids: list[int],
    tree_sequences: list[list[int]],
) -> tuple[list[int], int]:
    """Verify multiple draft sequences in parallel against the target model.

    For each sequence in tree_sequences:
    - Prepend prefix_ids to form the full input.
    - Run a single model forward pass.
    - Greedily accept tokens where the model's argmax matches the draft token.

    Returns:
        (best_accepted_sequence, max_tokens_accepted)

    where best_accepted_sequence is the sequence (without the prefix) that
    achieved the highest number of accepted tokens, and max_tokens_accepted
    is that count.
    """
    device = torch.device("cpu")

    if not tree_sequences:
        # No drafts — return empty
        return [], 0

    best_seq: list[int] = []
    best_count: int = 0

    torch.tensor(prefix_ids, dtype=torch.long, device=device)
    prefix_len = len(prefix_ids)

    for seq in tree_sequences:
        # seq includes the root token (= prefix[-1]); the draft tokens start
        # after the root, i.e. seq[1:] are the new tokens drafted.
        # We verify the draft tokens against the model run on prefix + draft.
        draft_tokens = seq[1:]  # exclude root (already in prefix)

        if not draft_tokens:
            continue

        full_ids = torch.tensor(
            [prefix_ids + draft_tokens], dtype=torch.long, device=device
        )  # (1, prefix_len + n_draft)

        out = model(full_ids)
        logits_all = out[1] if isinstance(out, tuple) else out  # (1, S, V)
        # logits_all[0, i, :] predicts token at position i+1
        # so to predict draft_tokens[j] (at position prefix_len + j),
        # we look at logits_all[0, prefix_len - 1 + j, :]
        logits_seq = logits_all[0]  # (S, V)

        accepted: list[int] = []
        for j, tok in enumerate(draft_tokens):
            pred_pos = prefix_len - 1 + j
            if pred_pos >= logits_seq.shape[0]:
                break
            argmax_tok = int(logits_seq[pred_pos].argmax().item())
            if argmax_tok == tok:
                accepted.append(tok)
            else:
                # Mismatch — stop
                break

        n_accepted = len(accepted)
        if n_accepted > best_count:
            best_count = n_accepted
            best_seq = accepted

    return best_seq, best_count


# ---------------------------------------------------------------------------
# High-level TreeSpeculativeDecoder
# ---------------------------------------------------------------------------


class TreeSpeculativeDecoder:
    """High-level tree speculative decoder with tokenizer interface.

    Args:
        model:            Model callable returning (loss, logits, kv).
        config:           TreeConfig.
        tokenizer_encode: Callable(str) -> list[int].
        tokenizer_decode: Callable(list[int]) -> str.
    """

    def __init__(
        self,
        model,
        config: TreeConfig,
        tokenizer_encode,
        tokenizer_decode,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    @torch.no_grad()
    def decode(self, prompt: str) -> tuple[str, dict]:
        """Full tree speculative decoding loop.

        Args:
            prompt: Input text.

        Returns:
            (text, stats) where stats contains:
                - tokens_generated: int
                - mean_accepted_per_step: float
                - n_steps: int
        """
        prefix_ids: list[int] = self.tokenizer_encode(prompt)
        generated_tokens: list[int] = []
        accepted_counts: list[int] = []
        n_steps = 0

        while len(generated_tokens) < self.config.max_new_tokens:
            current_prefix = prefix_ids + generated_tokens
            accepted, n_accepted = self._decode_step(current_prefix)

            n_steps += 1
            if n_accepted == 0:
                # Fall back: greedily sample one token from the model
                ids_tensor = torch.tensor([current_prefix], dtype=torch.long)
                logits = _last_logits(self.model, ids_tensor, self.config.temperature)
                tok = int(logits.argmax().item())
                accepted = [tok]
                n_accepted = 1

            # Respect budget
            remaining = self.config.max_new_tokens - len(generated_tokens)
            accepted = accepted[:remaining]
            generated_tokens.extend(accepted)
            accepted_counts.append(len(accepted))

            if len(generated_tokens) >= self.config.max_new_tokens:
                break

        mean_accepted = float(sum(accepted_counts)) / max(len(accepted_counts), 1)
        stats = {
            "tokens_generated": len(generated_tokens),
            "mean_accepted_per_step": mean_accepted,
            "n_steps": n_steps,
        }
        text = self.tokenizer_decode(prefix_ids + generated_tokens)
        return text, stats

    @torch.no_grad()
    def _decode_step(self, prefix_ids: list[int]) -> tuple[list[int], int]:
        """One step: build tree, extract sequences, verify, return accepted tokens.

        Args:
            prefix_ids: Current token context.

        Returns:
            (accepted_tokens, n_accepted)
        """
        root = build_draft_tree(self.model, prefix_ids, self.config)
        sequences = tree_to_sequences(root)
        accepted, n_accepted = verify_tree(self.model, prefix_ids, sequences)
        return accepted, n_accepted
