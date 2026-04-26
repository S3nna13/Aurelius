"""Tests for speculative_tree: tree speculative decoding."""

import torch

from src.inference.speculative_tree import (
    TreeNode,
    TreeSpecConfig,
    TreeSpecDecoder,
    build_draft_tree,
    flatten_tree_paths,
    verify_tree,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model() -> AureliusTransformer:
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


def _make_prompt(length: int = 4) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 256, (1, length))


def _small_config() -> TreeSpecConfig:
    return TreeSpecConfig(branch_factor=2, tree_depth=2, max_new_tokens=4)


# ---------------------------------------------------------------------------
# 1. TreeSpecConfig defaults
# ---------------------------------------------------------------------------


def test_tree_spec_config_defaults():
    cfg = TreeSpecConfig()
    assert cfg.branch_factor == 2
    assert cfg.tree_depth == 3
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# 2. TreeNode fields
# ---------------------------------------------------------------------------


def test_tree_node_fields():
    node = TreeNode(token_id=42, log_prob=-1.5, parent_idx=-1, depth=1)
    assert node.token_id == 42
    assert node.log_prob == -1.5
    assert node.parent_idx == -1
    assert node.depth == 1


# ---------------------------------------------------------------------------
# 3. build_draft_tree returns list of TreeNodes
# ---------------------------------------------------------------------------


def test_build_draft_tree_returns_list_of_tree_nodes():
    model = _make_model()
    model.train(False)
    prompt = _make_prompt()
    cfg = _small_config()
    nodes = build_draft_tree(model, prompt, cfg)
    assert isinstance(nodes, list)
    assert len(nodes) > 0
    for node in nodes:
        assert isinstance(node, TreeNode)


# ---------------------------------------------------------------------------
# 4. build_draft_tree correct depth range
# ---------------------------------------------------------------------------


def test_build_draft_tree_correct_depth_range():
    model = _make_model()
    model.train(False)
    prompt = _make_prompt()
    cfg = _small_config()
    nodes = build_draft_tree(model, prompt, cfg)
    for node in nodes:
        assert 1 <= node.depth <= cfg.tree_depth


# ---------------------------------------------------------------------------
# 5. build_draft_tree token_ids in valid range
# ---------------------------------------------------------------------------


def test_build_draft_tree_token_ids_valid_range():
    model = _make_model()
    model.train(False)
    prompt = _make_prompt()
    cfg = _small_config()
    nodes = build_draft_tree(model, prompt, cfg)
    vocab_size = 256
    for node in nodes:
        assert 0 <= node.token_id < vocab_size


# ---------------------------------------------------------------------------
# 6. flatten_tree_paths returns list of lists
# ---------------------------------------------------------------------------


def test_flatten_tree_paths_returns_list_of_lists():
    model = _make_model()
    model.train(False)
    prompt = _make_prompt()
    cfg = _small_config()
    nodes = build_draft_tree(model, prompt, cfg)
    paths = flatten_tree_paths(nodes)
    assert isinstance(paths, list)
    assert len(paths) > 0
    for path in paths:
        assert isinstance(path, list)
        assert all(isinstance(tok, int) for tok in path)


# ---------------------------------------------------------------------------
# 7. flatten_tree_paths path lengths <= tree_depth
# ---------------------------------------------------------------------------


def test_flatten_tree_paths_lengths_le_tree_depth():
    model = _make_model()
    model.train(False)
    prompt = _make_prompt()
    cfg = _small_config()
    nodes = build_draft_tree(model, prompt, cfg)
    paths = flatten_tree_paths(nodes)
    for path in paths:
        assert len(path) <= cfg.tree_depth


# ---------------------------------------------------------------------------
# 8. verify_tree returns tuple
# ---------------------------------------------------------------------------


def test_verify_tree_returns_tuple():
    model = _make_model()
    model.train(False)
    prompt = _make_prompt()
    cfg = _small_config()
    nodes = build_draft_tree(model, prompt, cfg)
    result = verify_tree(model, prompt, nodes)
    assert isinstance(result, tuple)
    assert len(result) == 2
    accepted_ids, n_accepted = result
    assert isinstance(accepted_ids, list)
    assert isinstance(n_accepted, int)


# ---------------------------------------------------------------------------
# 9. verify_tree n_accepted <= len(nodes)
# ---------------------------------------------------------------------------


def test_verify_tree_n_accepted_le_len_nodes():
    model = _make_model()
    model.train(False)
    prompt = _make_prompt()
    cfg = _small_config()
    nodes = build_draft_tree(model, prompt, cfg)
    _, n_accepted = verify_tree(model, prompt, nodes)
    assert n_accepted <= len(nodes)


# ---------------------------------------------------------------------------
# 10. TreeSpecDecoder.generate output shape (1, generated_len)
# ---------------------------------------------------------------------------


def test_tree_spec_decoder_generate_output_shape():
    torch.manual_seed(42)
    draft = _make_model()
    target = _make_model()
    draft.train(False)
    target.train(False)
    cfg = _small_config()
    decoder = TreeSpecDecoder(draft, target, cfg)
    prompt = _make_prompt()
    generated_ids, _ = decoder.generate(prompt)
    assert generated_ids.shape[0] == 1
    assert generated_ids.shape[1] > 0


# ---------------------------------------------------------------------------
# 11. TreeSpecDecoder.generate stats keys present
# ---------------------------------------------------------------------------


def test_tree_spec_decoder_generate_stats_keys():
    torch.manual_seed(42)
    draft = _make_model()
    target = _make_model()
    draft.train(False)
    target.train(False)
    cfg = _small_config()
    decoder = TreeSpecDecoder(draft, target, cfg)
    prompt = _make_prompt()
    _, stats = decoder.generate(prompt)
    assert "n_steps" in stats
    assert "mean_accepted_per_step" in stats
    assert "total_tokens" in stats


# ---------------------------------------------------------------------------
# 12. TreeSpecDecoder.generate total_tokens > 0
# ---------------------------------------------------------------------------


def test_tree_spec_decoder_generate_total_tokens_positive():
    torch.manual_seed(42)
    draft = _make_model()
    target = _make_model()
    draft.train(False)
    target.train(False)
    cfg = _small_config()
    decoder = TreeSpecDecoder(draft, target, cfg)
    prompt = _make_prompt()
    _, stats = decoder.generate(prompt)
    assert stats["total_tokens"] > 0
