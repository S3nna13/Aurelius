"""Trie-based constrained decoding: enforce valid outputs via prefix tree."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# TrieDecodingConfig
# ---------------------------------------------------------------------------


@dataclass
class TrieDecodingConfig:
    eos_token_id: int = 0
    pad_token_id: int = 0
    max_new_tokens: int = 32
    temperature: float = 1.0
    allow_eos_anywhere: bool = False  # allow EOS even if trie not exhausted


# ---------------------------------------------------------------------------
# TrieNode
# ---------------------------------------------------------------------------


class TrieNode:
    """Node in a prefix tree."""

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.is_terminal: bool = False

    def insert(self, tokens: list[int]) -> None:
        """Insert a token sequence into the trie (starting from this node)."""
        node = self
        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        node.is_terminal = True

    def has_child(self, token_id: int) -> bool:
        return token_id in self.children

    def get_child(self, token_id: int) -> TrieNode | None:
        return self.children.get(token_id)

    def valid_next_tokens(self) -> list[int]:
        """Return list of valid next token ids from this node."""
        return list(self.children.keys())

    def depth(self) -> int:
        """Max depth of subtree rooted at this node."""
        if not self.children:
            return 0
        return 1 + max(child.depth() for child in self.children.values())


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------


class Trie:
    """Prefix tree for constrained generation."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self._count: int = 0

    def insert(self, tokens: list[int]) -> None:
        """Insert a valid token sequence."""
        self.root.insert(tokens)
        self._count += 1

    def insert_batch(self, token_sequences: list[list[int]]) -> None:
        """Insert multiple sequences."""
        for seq in token_sequences:
            self.insert(seq)

    def prefix_allowed_tokens(self, prefix: list[int]) -> list[int]:
        """Given a prefix of generated tokens, return valid next token ids.

        Returns [] if prefix is not in trie (invalid prefix).
        Returns [] with the node being terminal if prefix matches a terminal node
        that has no children (fully matched).
        """
        node = self.root
        for tok in prefix:
            child = node.get_child(tok)
            if child is None:
                return []  # invalid prefix
            node = child
        return node.valid_next_tokens()

    def __len__(self) -> int:
        """Number of sequences in the trie."""
        return self._count

    def __contains__(self, tokens: object) -> bool:
        """Check if exact sequence is in the trie."""
        if not isinstance(tokens, list):
            return False
        node = self.root
        for tok in tokens:
            child = node.get_child(tok)
            if child is None:
                return False
            node = child
        return node.is_terminal


# ---------------------------------------------------------------------------
# constrained_logits
# ---------------------------------------------------------------------------


def constrained_logits(
    logits: Tensor,  # (vocab_size,) for one token position
    allowed_tokens: list[int],
    temperature: float = 1.0,
) -> Tensor:
    """Mask logits to only allow specified tokens (set others to -inf).

    Apply temperature scaling. Returns (vocab_size,) tensor.
    If allowed_tokens is empty, returns temperature-scaled logits unchanged
    (fall back to unconstrained).
    """
    if temperature != 1.0:
        logits = logits / temperature

    if not allowed_tokens:
        # Fall back: unconstrained (no masking)
        return logits

    mask = torch.full_like(logits, float("-inf"))
    for tok in allowed_tokens:
        if 0 <= tok < logits.shape[-1]:
            mask[tok] = logits[tok]
    return mask


# ---------------------------------------------------------------------------
# TrieDecoder
# ---------------------------------------------------------------------------


class TrieDecoder:
    """Trie-constrained generation."""

    def __init__(self, model: nn.Module, trie: Trie, config: TrieDecodingConfig) -> None:
        self.model = model
        self.trie = trie
        self.config = config

    def _forward(self, input_ids: Tensor) -> Tensor:
        """Run model and return last-step logits, shape (vocab_size,)."""
        loss, logits, pkv = self.model(input_ids)
        # logits: (B, T, V) or (B, V)
        if logits.dim() == 3:
            logits = logits[0, -1, :]
        elif logits.dim() == 2:
            logits = logits[0, -1] if logits.shape[0] == 1 else logits[-1]
        return logits  # (V,)

    def generate(self, input_ids: Tensor) -> tuple[Tensor, dict]:
        """Constrained greedy decode.

        At each step, get valid next tokens from trie, mask logits, sample.
        Stop when: trie exhausted (terminal) or max_new_tokens reached.

        Returns (generated_ids (max_new_tokens,), stats dict):
            stats keys: 'n_tokens', 'constrained', 'trie_terminal_reached'
        """
        cfg = self.config
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        current_ids = input_ids.clone()
        generated: list[int] = []
        trie_terminal_reached = False

        with torch.no_grad():
            for step in range(cfg.max_new_tokens):
                # Determine valid next tokens from current trie prefix
                allowed = self.trie.prefix_allowed_tokens(generated)

                # Check if we're at a terminal with no further children
                if not allowed:
                    # Either invalid path or terminal with no children — stop
                    # Pad remaining with pad_token_id
                    # Check if it's actually a terminal
                    node = self.trie.root
                    valid_path = True
                    for tok in generated:
                        child = node.get_child(tok)
                        if child is None:
                            valid_path = False
                            break
                        node = child
                    if valid_path and node.is_terminal:
                        trie_terminal_reached = True
                    break

                # Allow EOS if configured
                if cfg.allow_eos_anywhere and cfg.eos_token_id not in allowed:
                    allowed = allowed + [cfg.eos_token_id]

                raw_logits = self._forward(current_ids)  # (V,)
                masked = constrained_logits(raw_logits, allowed, temperature=cfg.temperature)
                next_token = int(torch.argmax(masked).item())
                generated.append(next_token)

                # Stop on EOS
                if next_token == cfg.eos_token_id and cfg.allow_eos_anywhere:
                    break

                next_tensor = torch.tensor(
                    [[next_token]], dtype=torch.long, device=current_ids.device
                )
                current_ids = torch.cat([current_ids, next_tensor], dim=1)

        # Pad to max_new_tokens
        pad_len = cfg.max_new_tokens - len(generated)
        generated_ids = torch.tensor(
            generated + [cfg.pad_token_id] * pad_len,
            dtype=torch.long,
        )

        stats = {
            "n_tokens": len(generated),
            "constrained": True,
            "trie_terminal_reached": trie_terminal_reached,
        }
        return generated_ids, stats

    def batch_generate(self, input_ids: Tensor, n_sequences: int) -> list[Tensor]:
        """Generate n_sequences constrained outputs independently.

        Returns list of generated id tensors.
        """
        results: list[Tensor] = []
        for _ in range(n_sequences):
            gen_ids, _ = self.generate(input_ids)
            results.append(gen_ids)
        return results
