"""Tests for src/training/model_editing.py."""
import pytest
import torch

from src.model.transformer import AureliusTransformer
from src.model.config import AureliusConfig
from src.training.model_editing import (
    EditConfig,
    FactEdit,
    ModelEditor,
    compute_key_vector,
    rank_one_update,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def model(cfg):
    m = AureliusTransformer(cfg)
    m.eval()
    return m


@pytest.fixture
def edit_cfg():
    return EditConfig()


@pytest.fixture
def subject_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 4))


@pytest.fixture
def target_ids():
    torch.manual_seed(1)
    return torch.randint(0, 256, (1, 4))


@pytest.fixture
def fact_edit(subject_ids, target_ids):
    return FactEdit(subject_ids=subject_ids, target_ids=target_ids, relation="located_in")


@pytest.fixture
def editor(model, edit_cfg):
    return ModelEditor(model, edit_cfg)


# ---------------------------------------------------------------------------
# 1. EditConfig defaults
# ---------------------------------------------------------------------------

def test_edit_config_defaults():
    cfg = EditConfig()
    assert cfg.n_edits == 1
    assert cfg.layer_idx == -1
    assert cfg.edit_method == "rank1"
    assert cfg.learning_rate == 1e-4
    assert cfg.n_steps == 10


# ---------------------------------------------------------------------------
# 2. FactEdit fields accessible
# ---------------------------------------------------------------------------

def test_fact_edit_fields(subject_ids, target_ids):
    fe = FactEdit(subject_ids=subject_ids, target_ids=target_ids, relation="rel")
    assert torch.equal(fe.subject_ids, subject_ids)
    assert torch.equal(fe.target_ids, target_ids)
    assert fe.relation == "rel"


# ---------------------------------------------------------------------------
# 3. compute_key_vector returns shape (d_model,)
# ---------------------------------------------------------------------------

def test_compute_key_vector_shape(model, subject_ids):
    key = compute_key_vector(model, subject_ids, layer_idx=-1)
    assert key.shape == (64,)


# ---------------------------------------------------------------------------
# 4. compute_key_vector different inputs -> different vectors
# ---------------------------------------------------------------------------

def test_compute_key_vector_different_inputs(model):
    torch.manual_seed(42)
    ids_a = torch.randint(0, 256, (1, 4))
    ids_b = torch.randint(0, 256, (1, 4))
    # Make sure they're actually different
    while torch.equal(ids_a, ids_b):
        ids_b = torch.randint(0, 256, (1, 4))
    key_a = compute_key_vector(model, ids_a, layer_idx=-1)
    key_b = compute_key_vector(model, ids_b, layer_idx=-1)
    assert not torch.allclose(key_a, key_b)


# ---------------------------------------------------------------------------
# 5. rank_one_update returns same shape
# ---------------------------------------------------------------------------

def test_rank_one_update_shape():
    W = torch.randn(64, 128)
    k = torch.randn(128)
    v = torch.randn(64)
    W_new = rank_one_update(W, k, v, lambda_reg=0.1)
    assert W_new.shape == W.shape


# ---------------------------------------------------------------------------
# 6. rank_one_update: W @ k after update approximates v (key recall)
# ---------------------------------------------------------------------------

def test_rank_one_update_key_recall():
    torch.manual_seed(7)
    W = torch.randn(64, 128)
    k = torch.randn(128)
    v = torch.randn(64)
    W_new = rank_one_update(W, k, v, lambda_reg=1e-6)
    # W_new @ k should be close to v (within numerical tolerance given lambda_reg)
    recalled = W_new @ k
    assert torch.allclose(recalled, v, atol=1e-3), f"max diff: {(recalled - v).abs().max()}"


# ---------------------------------------------------------------------------
# 7. ModelEditor.apply_edit returns required keys
# ---------------------------------------------------------------------------

def test_apply_edit_returns_required_keys(editor, fact_edit):
    result = editor.apply_edit(fact_edit)
    assert "layer" in result
    assert "edit_norm" in result


# ---------------------------------------------------------------------------
# 8. apply_edit changes FFN weights
# ---------------------------------------------------------------------------

def test_apply_edit_changes_weights(model, edit_cfg, fact_edit):
    editor = ModelEditor(model, edit_cfg)
    layer_idx = len(model.layers) - 1  # resolved -1
    original = model.layers[layer_idx].ffn.down_proj.weight.data.clone()
    editor.apply_edit(fact_edit)
    updated = model.layers[layer_idx].ffn.down_proj.weight.data
    assert not torch.equal(original, updated)


# ---------------------------------------------------------------------------
# 9. revert_edit restores original weights
# ---------------------------------------------------------------------------

def test_revert_edit_restores_weights(model, edit_cfg, fact_edit):
    editor = ModelEditor(model, edit_cfg)
    layer_idx = len(model.layers) - 1
    original = model.layers[layer_idx].ffn.down_proj.weight.data.clone()
    editor.apply_edit(fact_edit)
    editor.revert_edit(-1)
    restored = model.layers[layer_idx].ffn.down_proj.weight.data
    assert torch.allclose(original, restored)


# ---------------------------------------------------------------------------
# 10. evaluate_edit returns success and score keys
# ---------------------------------------------------------------------------

def test_evaluate_edit_returns_keys(editor, fact_edit):
    result = editor.evaluate_edit(fact_edit)
    assert "success" in result
    assert "score" in result


# ---------------------------------------------------------------------------
# 11. evaluate_edit score in [0, 1]
# ---------------------------------------------------------------------------

def test_evaluate_edit_score_range(editor, fact_edit):
    result = editor.evaluate_edit(fact_edit)
    assert 0.0 <= result["score"] <= 1.0


# ---------------------------------------------------------------------------
# 12. batch_edit returns list of same length as edits
# ---------------------------------------------------------------------------

def test_batch_edit_length(model, edit_cfg, subject_ids, target_ids):
    editor = ModelEditor(model, edit_cfg)
    edits = [
        FactEdit(subject_ids=subject_ids, target_ids=target_ids),
        FactEdit(subject_ids=target_ids, target_ids=subject_ids),
        FactEdit(subject_ids=subject_ids, target_ids=subject_ids),
    ]
    results = editor.batch_edit(edits)
    assert len(results) == len(edits)


# ---------------------------------------------------------------------------
# 13. apply_edit at different layers works
# ---------------------------------------------------------------------------

def test_apply_edit_different_layers(model, fact_edit):
    # Edit layer 0 explicitly
    cfg0 = EditConfig(layer_idx=0)
    editor0 = ModelEditor(model, cfg0)
    result0 = editor0.apply_edit(fact_edit)
    assert result0["layer"] == 0

    # Edit layer 1 explicitly
    cfg1 = EditConfig(layer_idx=1)
    editor1 = ModelEditor(model, cfg1)
    result1 = editor1.apply_edit(fact_edit)
    assert result1["layer"] == 1


# ---------------------------------------------------------------------------
# 14. edit_norm > 0 after non-trivial edit
# ---------------------------------------------------------------------------

def test_edit_norm_positive(model, edit_cfg):
    editor = ModelEditor(model, edit_cfg)
    # Use very different subject and target to ensure non-trivial edit
    torch.manual_seed(99)
    subject_ids = torch.randint(0, 128, (1, 4))
    target_ids = torch.randint(128, 256, (1, 4))
    fact = FactEdit(subject_ids=subject_ids, target_ids=target_ids)
    result = editor.apply_edit(fact)
    assert result["edit_norm"] > 0.0
