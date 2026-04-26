"""Tests for speculative tree decoding (tree_decoding.py)."""

from __future__ import annotations

import torch

from src.inference.tree_decoding import (
    DraftTree,
    TreeConfig,
    TreeNode,
    TreeSpeculativeDecoder,
    build_draft_tree,
    verify_tree,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
PROMPT_IDS = [1, 2, 3, 4]


class MockModel:
    """Tiny mock that returns random logits; matches the (loss, logits, kv) API."""

    def __call__(self, input_ids: torch.Tensor):
        B, S = input_ids.shape
        logits = torch.randn(B, S, VOCAB_SIZE)
        return (torch.tensor(0.0), logits, None)


def _make_real_model() -> AureliusTransformer:
    """Create a tiny AureliusTransformer for integration-style tests."""
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


# ---------------------------------------------------------------------------
# 1. test_tree_config_defaults
# ---------------------------------------------------------------------------


def test_tree_config_defaults():
    """TreeConfig should have the specified default values."""
    cfg = TreeConfig()
    assert cfg.branching_factor == 2
    assert cfg.tree_depth == 3
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.9


# ---------------------------------------------------------------------------
# 2. test_tree_node_path_single
# ---------------------------------------------------------------------------


def test_tree_node_path_single():
    """A single TreeNode with no parent should have path == [token_id]."""
    node = TreeNode(token_id=42, log_prob=-0.5)
    assert node.path() == [42]


# ---------------------------------------------------------------------------
# 3. test_tree_node_path_chain
# ---------------------------------------------------------------------------


def test_tree_node_path_chain():
    """A 3-node chain should produce a path with 3 elements in root-to-leaf order."""
    root = TreeNode(token_id=10, log_prob=-0.1)
    mid = TreeNode(token_id=20, log_prob=-0.2, parent=root, depth=1)
    leaf = TreeNode(token_id=30, log_prob=-0.3, parent=mid, depth=2)
    assert leaf.path() == [10, 20, 30]


# ---------------------------------------------------------------------------
# 4. test_draft_tree_add_children
# ---------------------------------------------------------------------------


def test_draft_tree_add_children():
    """add_children should increase num_nodes by the number of children added."""
    tree = DraftTree(root_ids=PROMPT_IDS)
    assert tree.num_nodes() == 0

    tree.add_children(tree.root, token_ids=[1, 2], log_probs=[-0.3, -0.7])
    assert tree.num_nodes() == 2

    # Add children to first leaf
    first_leaf = tree.leaves[0]
    tree.add_children(first_leaf, token_ids=[3, 4], log_probs=[-0.1, -0.2])
    assert tree.num_nodes() == 4


# ---------------------------------------------------------------------------
# 5. test_draft_tree_all_paths_count
# ---------------------------------------------------------------------------


def test_draft_tree_all_paths_count():
    """branching_factor=2, depth=2 → 4 root-to-leaf paths."""
    tree = DraftTree(root_ids=PROMPT_IDS)
    # Level 1: 2 children of sentinel root
    tree.add_children(tree.root, token_ids=[10, 20], log_probs=[-0.1, -0.2])
    # Level 2: 2 children per level-1 node → 4 leaves
    for leaf in list(tree.leaves):
        tree.add_children(leaf, token_ids=[30, 40], log_probs=[-0.3, -0.4])

    paths = tree.all_paths()
    assert len(paths) == 4
    for path in paths:
        assert len(path) == 2  # each path has 2 non-sentinel tokens


# ---------------------------------------------------------------------------
# 6. test_build_draft_tree_leaf_count
# ---------------------------------------------------------------------------


def test_build_draft_tree_leaf_count():
    """build_draft_tree should produce branching_factor^tree_depth leaves."""
    torch.manual_seed(0)
    model = MockModel()
    config = TreeConfig(branching_factor=2, tree_depth=2)
    tree = build_draft_tree(model, PROMPT_IDS, config)
    # 2^2 = 4 leaves
    assert len(tree.leaves) == 4


# ---------------------------------------------------------------------------
# 7. test_build_draft_tree_path_length
# ---------------------------------------------------------------------------


def test_build_draft_tree_path_length():
    """Every root-to-leaf path should have length == tree_depth."""
    torch.manual_seed(0)
    model = MockModel()
    config = TreeConfig(branching_factor=2, tree_depth=2)
    tree = build_draft_tree(model, PROMPT_IDS, config)
    for path in tree.all_paths():
        assert len(path) == config.tree_depth


# ---------------------------------------------------------------------------
# 8. test_verify_tree_returns_list
# ---------------------------------------------------------------------------


def test_verify_tree_returns_list():
    """verify_tree should return a list of ints."""
    torch.manual_seed(0)
    model = MockModel()
    config = TreeConfig(branching_factor=2, tree_depth=2)
    tree = build_draft_tree(model, PROMPT_IDS, config)
    result = verify_tree(model, PROMPT_IDS, tree)
    assert isinstance(result, list)
    assert all(isinstance(t, int) for t in result)


# ---------------------------------------------------------------------------
# 9. test_verify_tree_nonempty
# ---------------------------------------------------------------------------


def test_verify_tree_nonempty():
    """verify_tree should always return at least 1 token."""
    torch.manual_seed(0)
    model = MockModel()
    config = TreeConfig(branching_factor=2, tree_depth=2)
    tree = build_draft_tree(model, PROMPT_IDS, config)
    result = verify_tree(model, PROMPT_IDS, tree)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# 10. test_tree_speculative_decoder_output_shape
# ---------------------------------------------------------------------------


def test_tree_speculative_decoder_output_shape():
    """TreeSpeculativeDecoder.generate should return (1, prompt_len + max_new_tokens)."""
    torch.manual_seed(0)
    target = _make_real_model()
    draft = _make_real_model()

    config = TreeConfig(branching_factor=2, tree_depth=2)
    decoder = TreeSpeculativeDecoder(target_model=target, draft_model=draft, config=config)

    input_ids = torch.tensor([PROMPT_IDS], dtype=torch.long)  # shape (1, 4)
    max_new_tokens = 6
    out = decoder.generate(input_ids, max_new_tokens=max_new_tokens)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 1
    # Output should have exactly prompt_len + max_new_tokens columns
    assert out.shape[1] == len(PROMPT_IDS) + max_new_tokens


# ---------------------------------------------------------------------------
# 11. test_tree_speculative_decoder_stats_keys
# ---------------------------------------------------------------------------


def test_tree_speculative_decoder_stats_keys():
    """TreeSpeculativeDecoder.stats() should contain the expected keys with correct types."""
    torch.manual_seed(0)
    target = _make_real_model()
    draft = _make_real_model()

    config = TreeConfig(branching_factor=2, tree_depth=2)
    decoder = TreeSpeculativeDecoder(target_model=target, draft_model=draft, config=config)

    input_ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
    decoder.generate(input_ids, max_new_tokens=6)

    s = decoder.stats()
    assert "mean_accepted_per_step" in s
    assert "total_steps" in s
    assert isinstance(s["mean_accepted_per_step"], float)
    assert isinstance(s["total_steps"], int)
    assert s["total_steps"] > 0
