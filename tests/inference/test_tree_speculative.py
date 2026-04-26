"""Tests for tree-structured speculative decoding (tree_speculative.py).

Covers TreeConfig, DraftNode, build_draft_tree, tree_to_sequences,
verify_tree, and TreeSpeculativeDecoder — 14+ tests in total.
"""

from __future__ import annotations

import pytest
import torch

from src.inference.tree_speculative import (
    DraftNode,
    TreeConfig,
    TreeSpeculativeDecoder,
    build_draft_tree,
    tree_to_sequences,
    verify_tree,
)

# ---------------------------------------------------------------------------
# Shared mock model — matches (loss, logits, kv) tuple API
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256


class MockModel:
    """Tiny mock that returns deterministic-ish random logits."""

    def __call__(self, input_ids: torch.Tensor):
        B, S = input_ids.shape
        torch.manual_seed(42)
        logits = torch.randn(B, S, VOCAB_SIZE)
        return (torch.tensor(0.0), logits, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> TreeConfig:
    defaults = dict(
        branching_factor=2,
        tree_depth=2,
        max_new_tokens=4,
        temperature=1.0,
        typical_acceptance_rate=0.8,
    )
    defaults.update(kwargs)
    return TreeConfig(**defaults)


def _dummy_encode(text: str) -> list[int]:
    return [ord(c) % VOCAB_SIZE for c in text] or [0]


def _dummy_decode(ids: list[int]) -> str:
    return "".join(chr(max(32, i % 128)) for i in ids)


# ---------------------------------------------------------------------------
# 1. TreeConfig defaults
# ---------------------------------------------------------------------------


def test_tree_config_defaults():
    cfg = TreeConfig()
    assert cfg.branching_factor == 2
    assert cfg.tree_depth == 4
    assert cfg.max_new_tokens == 128
    assert cfg.temperature == 1.0
    assert cfg.typical_acceptance_rate == 0.8


# ---------------------------------------------------------------------------
# 2. DraftNode fields
# ---------------------------------------------------------------------------


def test_draft_node_fields():
    node = DraftNode(token_id=7, log_prob=-0.3, depth=1)
    assert node.token_id == 7
    assert node.log_prob == pytest.approx(-0.3)
    assert node.depth == 1
    assert node.children == []
    assert node.parent is None


# ---------------------------------------------------------------------------
# 3. DraftNode.path_to_root for leaf node
# ---------------------------------------------------------------------------


def test_draft_node_path_to_root_leaf():
    root = DraftNode(token_id=1, log_prob=0.0, depth=0)
    child = DraftNode(token_id=2, log_prob=-0.5, depth=1, parent=root)
    leaf = DraftNode(token_id=3, log_prob=-0.8, depth=2, parent=child)
    root.children.append(child)
    child.children.append(leaf)

    assert leaf.path_to_root() == [1, 2, 3]


# ---------------------------------------------------------------------------
# 4. DraftNode.path_to_root for root node
# ---------------------------------------------------------------------------


def test_draft_node_path_to_root_root():
    root = DraftNode(token_id=42, log_prob=0.0, depth=0)
    assert root.path_to_root() == [42]


# ---------------------------------------------------------------------------
# 5. build_draft_tree returns DraftNode
# ---------------------------------------------------------------------------


def test_build_draft_tree_returns_draft_node():
    model = MockModel()
    cfg = _make_config()
    result = build_draft_tree(model, [1, 2, 3], cfg)
    assert isinstance(result, DraftNode)


# ---------------------------------------------------------------------------
# 6. build_draft_tree correct tree depth
# ---------------------------------------------------------------------------


def test_build_draft_tree_correct_depth():
    model = MockModel()
    cfg = _make_config(branching_factor=2, tree_depth=2)
    root = build_draft_tree(model, [0, 1, 2], cfg)

    # Gather max depth across all nodes reachable from root
    def max_depth(node: DraftNode) -> int:
        if not node.children:
            return node.depth
        return max(max_depth(c) for c in node.children)

    assert max_depth(root) == cfg.tree_depth


# ---------------------------------------------------------------------------
# 7. build_draft_tree branching factor correct
# ---------------------------------------------------------------------------


def test_build_draft_tree_branching_factor():
    model = MockModel()
    cfg = _make_config(branching_factor=2, tree_depth=2)
    root = build_draft_tree(model, [0, 1], cfg)
    # Root's immediate children == branching_factor
    assert len(root.children) == cfg.branching_factor


# ---------------------------------------------------------------------------
# 8. tree_to_sequences returns list of lists
# ---------------------------------------------------------------------------


def test_tree_to_sequences_returns_list_of_lists():
    model = MockModel()
    cfg = _make_config(branching_factor=2, tree_depth=2)
    root = build_draft_tree(model, [0, 1, 2], cfg)
    seqs = tree_to_sequences(root)
    assert isinstance(seqs, list)
    assert all(isinstance(s, list) for s in seqs)


# ---------------------------------------------------------------------------
# 9. tree_to_sequences count = branching_factor^tree_depth
# ---------------------------------------------------------------------------


def test_tree_to_sequences_count():
    model = MockModel()
    bf = 2
    td = 2
    cfg = _make_config(branching_factor=bf, tree_depth=td)
    root = build_draft_tree(model, [0, 1], cfg)
    seqs = tree_to_sequences(root)
    expected = bf**td
    assert len(seqs) == expected


# ---------------------------------------------------------------------------
# 10. verify_tree returns (list, int)
# ---------------------------------------------------------------------------


def test_verify_tree_return_type():
    model = MockModel()
    cfg = _make_config(branching_factor=2, tree_depth=2)
    root = build_draft_tree(model, [0, 1, 2], cfg)
    seqs = tree_to_sequences(root)
    result = verify_tree(model, [0, 1, 2], seqs)
    assert isinstance(result, tuple)
    assert len(result) == 2
    accepted_seq, n_accepted = result
    assert isinstance(accepted_seq, list)
    assert isinstance(n_accepted, int)


# ---------------------------------------------------------------------------
# 11. verify_tree n_accepted <= max draft length
# ---------------------------------------------------------------------------


def test_verify_tree_accepted_bound():
    model = MockModel()
    cfg = _make_config(branching_factor=2, tree_depth=2)
    root = build_draft_tree(model, [0, 1, 2], cfg)
    seqs = tree_to_sequences(root)
    # Max draft tokens per sequence = tree_depth (seq[1:] strips the root)
    max_draft_len = cfg.tree_depth
    _, n_accepted = verify_tree(model, [0, 1, 2], seqs)
    assert n_accepted <= max_draft_len


# ---------------------------------------------------------------------------
# 12. TreeSpeculativeDecoder.decode returns (str, dict)
# ---------------------------------------------------------------------------


def test_tree_speculative_decoder_decode_return_type():
    model = MockModel()
    cfg = _make_config(max_new_tokens=4, branching_factor=2, tree_depth=2)
    decoder = TreeSpeculativeDecoder(
        model=model,
        config=cfg,
        tokenizer_encode=_dummy_encode,
        tokenizer_decode=_dummy_decode,
    )
    result = decoder.decode("hello")
    assert isinstance(result, tuple)
    assert len(result) == 2
    text, stats = result
    assert isinstance(text, str)
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# 13. TreeSpeculativeDecoder.decode stats has required keys
# ---------------------------------------------------------------------------


def test_tree_speculative_decoder_stats_keys():
    model = MockModel()
    cfg = _make_config(max_new_tokens=4, branching_factor=2, tree_depth=2)
    decoder = TreeSpeculativeDecoder(
        model=model,
        config=cfg,
        tokenizer_encode=_dummy_encode,
        tokenizer_decode=_dummy_decode,
    )
    _, stats = decoder.decode("hi")
    assert "tokens_generated" in stats
    assert "mean_accepted_per_step" in stats
    assert "n_steps" in stats


# ---------------------------------------------------------------------------
# 14. TreeSpeculativeDecoder._decode_step returns (list, int)
# ---------------------------------------------------------------------------


def test_tree_speculative_decoder_decode_step_return_type():
    model = MockModel()
    cfg = _make_config(branching_factor=2, tree_depth=2)
    decoder = TreeSpeculativeDecoder(
        model=model,
        config=cfg,
        tokenizer_encode=_dummy_encode,
        tokenizer_decode=_dummy_decode,
    )
    prefix = [1, 2, 3, 4]
    result = decoder._decode_step(prefix)
    assert isinstance(result, tuple)
    assert len(result) == 2
    tokens, n = result
    assert isinstance(tokens, list)
    assert isinstance(n, int)


# ---------------------------------------------------------------------------
# 15. Bonus: stats tokens_generated matches max_new_tokens budget
# ---------------------------------------------------------------------------


def test_tree_speculative_decoder_tokens_generated_bounded():
    model = MockModel()
    max_new = 4
    cfg = _make_config(max_new_tokens=max_new, branching_factor=2, tree_depth=2)
    decoder = TreeSpeculativeDecoder(
        model=model,
        config=cfg,
        tokenizer_encode=_dummy_encode,
        tokenizer_decode=_dummy_decode,
    )
    _, stats = decoder.decode("test")
    assert stats["tokens_generated"] <= max_new


# ---------------------------------------------------------------------------
# 16. Bonus: n_steps is positive after decode
# ---------------------------------------------------------------------------


def test_tree_speculative_decoder_n_steps_positive():
    model = MockModel()
    cfg = _make_config(max_new_tokens=4, branching_factor=2, tree_depth=2)
    decoder = TreeSpeculativeDecoder(
        model=model,
        config=cfg,
        tokenizer_encode=_dummy_encode,
        tokenizer_decode=_dummy_decode,
    )
    _, stats = decoder.decode("abc")
    assert stats["n_steps"] >= 1
