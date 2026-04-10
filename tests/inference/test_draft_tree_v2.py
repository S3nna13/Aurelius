"""Tests for src/inference/draft_tree_v2.py — Draft Tree Speculative Decoding."""

from __future__ import annotations

import torch
import pytest

from src.inference.draft_tree_v2 import (
    DraftTreeConfig,
    TreeNode,
    build_draft_tree,
    score_path,
    extract_all_paths,
    best_path,
    verify_path,
    DraftTreeDecoder,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_model():
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )
    return AureliusTransformer(cfg).eval()


def make_input_ids(length: int = 8, vocab_size: int = 256) -> torch.Tensor:
    return torch.randint(0, vocab_size, (1, length))


# ---------------------------------------------------------------------------
# 1. DraftTreeConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = DraftTreeConfig()
    assert cfg.branch_factor == 3
    assert cfg.depth == 4
    assert cfg.acceptance_threshold == 0.8
    assert cfg.vocab_size == 256


# ---------------------------------------------------------------------------
# 2. DraftTreeConfig custom values
# ---------------------------------------------------------------------------

def test_config_custom():
    cfg = DraftTreeConfig(branch_factor=5, depth=2, acceptance_threshold=0.5, vocab_size=1024)
    assert cfg.branch_factor == 5
    assert cfg.depth == 2
    assert cfg.acceptance_threshold == 0.5
    assert cfg.vocab_size == 1024


# ---------------------------------------------------------------------------
# 3. TreeNode defaults
# ---------------------------------------------------------------------------

def test_tree_node_defaults():
    n = TreeNode(token_id=42, log_prob=-0.5)
    assert n.token_id == 42
    assert n.log_prob == -0.5
    assert n.children == []
    assert n.depth == 0
    assert n.is_accepted is False


# ---------------------------------------------------------------------------
# 4. TreeNode with children
# ---------------------------------------------------------------------------

def test_tree_node_with_children():
    child = TreeNode(token_id=10, log_prob=-1.0, depth=1)
    parent = TreeNode(token_id=5, log_prob=0.0, children=[child])
    assert len(parent.children) == 1
    assert parent.children[0].token_id == 10


# ---------------------------------------------------------------------------
# 5. score_path sums log probs
# ---------------------------------------------------------------------------

def test_score_path():
    path = [
        TreeNode(token_id=0, log_prob=-1.0),
        TreeNode(token_id=1, log_prob=-0.5),
        TreeNode(token_id=2, log_prob=-0.3),
    ]
    assert abs(score_path(path) - (-1.8)) < 1e-6


# ---------------------------------------------------------------------------
# 6. score_path empty
# ---------------------------------------------------------------------------

def test_score_path_empty():
    assert score_path([]) == 0.0


# ---------------------------------------------------------------------------
# 7. extract_all_paths single node (leaf root)
# ---------------------------------------------------------------------------

def test_extract_all_paths_single():
    root = TreeNode(token_id=0, log_prob=0.0)
    paths = extract_all_paths(root)
    assert len(paths) == 1
    assert paths[0][0].token_id == 0


# ---------------------------------------------------------------------------
# 8. extract_all_paths with branches
# ---------------------------------------------------------------------------

def test_extract_all_paths_branches():
    c1 = TreeNode(token_id=1, log_prob=-0.5, depth=1)
    c2 = TreeNode(token_id=2, log_prob=-1.0, depth=1)
    root = TreeNode(token_id=0, log_prob=0.0, children=[c1, c2])
    paths = extract_all_paths(root)
    assert len(paths) == 2
    # Each path starts at root
    for p in paths:
        assert p[0].token_id == 0


# ---------------------------------------------------------------------------
# 9. extract_all_paths deeper tree
# ---------------------------------------------------------------------------

def test_extract_all_paths_deeper():
    gc1 = TreeNode(token_id=10, log_prob=-0.2, depth=2)
    gc2 = TreeNode(token_id=11, log_prob=-0.3, depth=2)
    c1 = TreeNode(token_id=1, log_prob=-0.5, depth=1, children=[gc1, gc2])
    c2 = TreeNode(token_id=2, log_prob=-1.0, depth=1)
    root = TreeNode(token_id=0, log_prob=0.0, children=[c1, c2])
    paths = extract_all_paths(root)
    assert len(paths) == 3  # 2 through c1, 1 through c2


# ---------------------------------------------------------------------------
# 10. best_path picks highest score
# ---------------------------------------------------------------------------

def test_best_path_picks_highest():
    c1 = TreeNode(token_id=1, log_prob=-0.1, depth=1)
    c2 = TreeNode(token_id=2, log_prob=-5.0, depth=1)
    root = TreeNode(token_id=0, log_prob=0.0, children=[c1, c2])
    bp = best_path(root)
    assert bp == [0, 1]


# ---------------------------------------------------------------------------
# 11. best_path single node
# ---------------------------------------------------------------------------

def test_best_path_single_node():
    root = TreeNode(token_id=7, log_prob=0.0)
    assert best_path(root) == [7]


# ---------------------------------------------------------------------------
# 12. build_draft_tree returns TreeNode
# ---------------------------------------------------------------------------

def test_build_draft_tree_returns_tree_node():
    torch.manual_seed(0)
    model = make_tiny_model()
    ids = make_input_ids()
    cfg = DraftTreeConfig(branch_factor=2, depth=2, vocab_size=256)
    root = build_draft_tree(model, ids, cfg)
    assert isinstance(root, TreeNode)
    assert root.depth == 0


# ---------------------------------------------------------------------------
# 13. build_draft_tree respects branch_factor
# ---------------------------------------------------------------------------

def test_build_draft_tree_branch_factor():
    torch.manual_seed(1)
    model = make_tiny_model()
    ids = make_input_ids()
    cfg = DraftTreeConfig(branch_factor=2, depth=1, vocab_size=256)
    root = build_draft_tree(model, ids, cfg)
    assert len(root.children) == 2


# ---------------------------------------------------------------------------
# 14. build_draft_tree respects depth
# ---------------------------------------------------------------------------

def test_build_draft_tree_depth():
    torch.manual_seed(2)
    model = make_tiny_model()
    ids = make_input_ids()
    cfg = DraftTreeConfig(branch_factor=2, depth=2, vocab_size=256)
    root = build_draft_tree(model, ids, cfg)
    # Every leaf should be at depth 2
    paths = extract_all_paths(root)
    for path in paths:
        assert path[-1].depth == 2


# ---------------------------------------------------------------------------
# 15. verify_path accepts tokens above threshold
# ---------------------------------------------------------------------------

def test_verify_path_accepts():
    torch.manual_seed(3)
    model = make_tiny_model()
    ids = make_input_ids()
    # Get the model's own top prediction (it should agree with itself).
    with torch.no_grad():
        loss, logits, _ = model(ids)
    top_token = int(logits[0, -1].argmax().item())
    # A very low threshold should accept this token.
    accepted, n = verify_path(model, ids, [top_token], threshold=0.0)
    assert n >= 1
    assert accepted[0] == top_token


# ---------------------------------------------------------------------------
# 16. verify_path rejects with high threshold
# ---------------------------------------------------------------------------

def test_verify_path_rejects_high_threshold():
    torch.manual_seed(4)
    model = make_tiny_model()
    ids = make_input_ids()
    # Threshold of 1.0 means the model must give probability >= 1.0 — impossible.
    accepted, n = verify_path(model, ids, [0, 1, 2], threshold=1.0)
    assert n == 0
    assert accepted == []


# ---------------------------------------------------------------------------
# 17. DraftTreeDecoder produces correct shape
# ---------------------------------------------------------------------------

def test_decoder_generate_shape():
    torch.manual_seed(5)
    model = make_tiny_model()
    ids = make_input_ids(length=4)
    cfg = DraftTreeConfig(branch_factor=2, depth=1, acceptance_threshold=0.0, vocab_size=256)
    decoder = DraftTreeDecoder(config=cfg)
    out = decoder.generate(model, ids, max_new_tokens=3)
    assert out.shape[0] == 1
    assert out.shape[1] >= 4 + 3  # at least original + 3 new


# ---------------------------------------------------------------------------
# 18. DraftTreeDecoder preserves prompt prefix
# ---------------------------------------------------------------------------

def test_decoder_preserves_prefix():
    torch.manual_seed(6)
    model = make_tiny_model()
    ids = make_input_ids(length=4)
    cfg = DraftTreeConfig(branch_factor=2, depth=1, acceptance_threshold=0.0, vocab_size=256)
    decoder = DraftTreeDecoder(config=cfg)
    out = decoder.generate(model, ids, max_new_tokens=2)
    assert torch.equal(out[0, :4], ids[0, :4])


# ---------------------------------------------------------------------------
# 19. DraftTreeDecoder default config
# ---------------------------------------------------------------------------

def test_decoder_default_config():
    decoder = DraftTreeDecoder()
    assert decoder.config.branch_factor == 3
    assert decoder.config.depth == 4


# ---------------------------------------------------------------------------
# 20. verify_path empty path
# ---------------------------------------------------------------------------

def test_verify_path_empty():
    model = make_tiny_model()
    ids = make_input_ids()
    accepted, n = verify_path(model, ids, [], threshold=0.5)
    assert n == 0
    assert accepted == []
