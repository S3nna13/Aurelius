"""Tests for hierarchical_generation.py."""

from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.hierarchical_generation import (
    HierarchicalConfig,
    OutlineNode,
    HierarchicalGenerator,
    generate_outline,
    compute_coherence_score,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    """Tiny model for fast tests."""
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
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def encode(s: str) -> list[int]:
    """Encode string to token IDs (byte values, clamped to 0-255)."""
    return [min(ord(c), 255) for c in s[:64]]


def decode(ids: list[int]) -> str:
    """Decode token IDs to string (printable ASCII, clamped)."""
    return "".join(chr(max(32, min(126, i))) for i in ids)


@pytest.fixture(scope="module")
def generator(small_model):
    """HierarchicalGenerator with n_sections=3 for reuse across tests."""
    config = HierarchicalConfig(
        n_sections=3,
        section_tokens=8,
        outline_tokens=16,
        temperature=0.8,
    )
    return HierarchicalGenerator(small_model, config, encode, decode)


# ---------------------------------------------------------------------------
# 1. HierarchicalConfig defaults
# ---------------------------------------------------------------------------

def test_hierarchical_config_defaults():
    cfg = HierarchicalConfig()
    assert cfg.n_sections == 3
    assert cfg.section_tokens == 64
    assert cfg.outline_tokens == 32
    assert cfg.coherence_weight == 0.5
    assert cfg.temperature == 0.8


# ---------------------------------------------------------------------------
# 2. OutlineNode fields
# ---------------------------------------------------------------------------

def test_outline_node_fields():
    node = OutlineNode(title="Introduction")
    assert node.title == "Introduction"
    assert node.level == 1
    assert node.children == []
    assert node.content == ""


# ---------------------------------------------------------------------------
# 3. OutlineNode.to_flat_list returns a flat list (leaf node)
# ---------------------------------------------------------------------------

def test_outline_node_to_flat_list_single():
    node = OutlineNode(title="Root")
    flat = node.to_flat_list()
    assert isinstance(flat, list)
    assert len(flat) == 1
    assert flat[0] is node


# ---------------------------------------------------------------------------
# 4. OutlineNode.to_flat_list with children includes all nodes
# ---------------------------------------------------------------------------

def test_outline_node_to_flat_list_with_children():
    child1 = OutlineNode(title="Child A", level=2)
    child2 = OutlineNode(title="Child B", level=2)
    grandchild = OutlineNode(title="Grandchild", level=2)
    child1.children.append(grandchild)
    root = OutlineNode(title="Root", level=1, children=[child1, child2])

    flat = root.to_flat_list()
    titles = [n.title for n in flat]
    assert "Root" in titles
    assert "Child A" in titles
    assert "Child B" in titles
    assert "Grandchild" in titles
    assert len(flat) == 4


# ---------------------------------------------------------------------------
# 5. generate_outline returns list of OutlineNode
# ---------------------------------------------------------------------------

def test_generate_outline_returns_list_of_nodes(small_model):
    result = generate_outline(
        small_model,
        prompt="climate change",
        n_sections=3,
        tokenizer_encode=encode,
        tokenizer_decode=decode,
        max_new_tokens=4,
    )
    assert isinstance(result, list)
    assert all(isinstance(n, OutlineNode) for n in result)


# ---------------------------------------------------------------------------
# 6. generate_outline length <= n_sections
# ---------------------------------------------------------------------------

def test_generate_outline_length_at_most_n_sections(small_model):
    n = 4
    result = generate_outline(
        small_model,
        prompt="artificial intelligence",
        n_sections=n,
        tokenizer_encode=encode,
        tokenizer_decode=decode,
        max_new_tokens=4,
    )
    assert len(result) <= n
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# 7. generate_outline fallback provides exactly n_sections nodes
# ---------------------------------------------------------------------------

def test_generate_outline_fallback_provides_n_sections(small_model):
    """When parsing yields < n_sections titles, fallback gives exactly n_sections."""
    n = 10
    result = generate_outline(
        small_model,
        prompt="x",
        n_sections=n,
        tokenizer_encode=encode,
        tokenizer_decode=decode,
        max_new_tokens=4,
    )
    assert len(result) == n


# ---------------------------------------------------------------------------
# 8. compute_coherence_score identical text -> 1.0
# ---------------------------------------------------------------------------

def test_coherence_score_identical_text():
    text = "The quick brown fox jumps over the lazy dog"
    score = compute_coherence_score(text, text)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. compute_coherence_score unrelated text -> low score
# ---------------------------------------------------------------------------

def test_coherence_score_unrelated_text():
    prev = "aaaaaaaaaaaaaaaaaaaaaaaaa"
    nxt = "zzzzzzzzzzzzzzzzzzzzzzzzz"
    score = compute_coherence_score(prev, nxt)
    assert score < 0.3


# ---------------------------------------------------------------------------
# 10. compute_coherence_score returns value in [0, 1]
# ---------------------------------------------------------------------------

def test_coherence_score_range():
    pairs = [
        ("hello world", "world hello"),
        ("abc", "xyz"),
        ("The cat sat on the mat", "A dog lay on the rug"),
        ("", "something"),
        ("something", ""),
    ]
    for prev, nxt in pairs:
        score = compute_coherence_score(prev, nxt)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for ({prev!r}, {nxt!r})"


# ---------------------------------------------------------------------------
# 11. HierarchicalGenerator.generate_section returns string
# ---------------------------------------------------------------------------

def test_generate_section_returns_string(generator):
    node = OutlineNode(title="Introduction")
    result = generator.generate_section(node, context="", max_new_tokens=4)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 12. HierarchicalGenerator.generate_document returns dict with required keys
# ---------------------------------------------------------------------------

def test_generate_document_returns_dict_with_keys(generator):
    result = generator.generate_document("machine learning")
    assert isinstance(result, dict)
    required_keys = {"outline", "sections", "document", "coherence_scores"}
    assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# 13. HierarchicalGenerator.generate_document document is string
# ---------------------------------------------------------------------------

def test_generate_document_document_is_string(generator):
    result = generator.generate_document("space exploration")
    assert isinstance(result["document"], str)


# ---------------------------------------------------------------------------
# 14. HierarchicalGenerator.generate_document coherence_scores length = n_sections - 1
# ---------------------------------------------------------------------------

def test_generate_document_coherence_scores_length(generator):
    result = generator.generate_document("renewable energy")
    n = generator.config.n_sections
    assert len(result["coherence_scores"]) == n - 1


# ---------------------------------------------------------------------------
# 15. HierarchicalGenerator.refine_section returns string
# ---------------------------------------------------------------------------

def test_refine_section_returns_string(generator):
    node = OutlineNode(title="Conclusion")
    original = "This section summarizes the findings."
    result = generator.refine_section(original, node)
    assert isinstance(result, str)
