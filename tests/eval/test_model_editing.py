"""Tests for MEMIT-style mass editing, edit evaluation metrics, and editing-specific tests."""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.model_editing import (
    EditRequest,
    EditResult,
    ModelEditor,
    batch_rank_one_updates,
    compute_edit_efficacy,
    compute_sequence_probability,
    rank_one_update,
)


def make_model() -> AureliusTransformer:
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2, head_dim=32,
        d_ff=128, vocab_size=256, max_seq_len=512, dropout=0.0, tie_embeddings=False,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def encode(text: str) -> list[int]:
    return [min(ord(c), 255) for c in text]


def decode(ids: list[int]) -> str:
    return "".join(chr(min(i, 127)) for i in ids)


# 1. EditRequest required fields
def test_edit_request_required_fields():
    req = EditRequest(subject="Paris", prompt="Paris is in", target_new=" France")
    assert req.subject == "Paris"
    assert req.prompt == "Paris is in"
    assert req.target_new == " France"


# 2. EditRequest optional fields default to None
def test_edit_request_optional_fields_default_none():
    req = EditRequest(subject="X", prompt="X is", target_new="Y")
    assert req.target_old is None
    assert req.paraphrase_prompts is None


# 3. EditRequest optional fields can be set
def test_edit_request_optional_fields_set():
    req = EditRequest(
        subject="Rome", prompt="Rome is in", target_new=" Italy",
        target_old=" Germany", paraphrase_prompts=["The city of Rome is located in"],
    )
    assert req.target_old == " Germany"
    assert len(req.paraphrase_prompts) == 1


# 4. EditResult fields
def test_edit_result_fields():
    result = EditResult(success=True, efficacy=0.9, generalization=0.7,
                        specificity=0.8, edit_id="abc-123")
    assert result.success is True
    assert result.efficacy == pytest.approx(0.9)
    assert result.generalization == pytest.approx(0.7)
    assert result.specificity == pytest.approx(0.8)
    assert result.edit_id == "abc-123"


# 5. compute_sequence_probability returns float
def test_compute_sequence_probability_returns_float():
    model = make_model()
    input_ids = torch.randint(0, 256, (1, 5))
    target_ids = torch.randint(0, 256, (1, 3))
    result = compute_sequence_probability(model, input_ids, target_ids)
    assert isinstance(result, float)


# 6. compute_sequence_probability is <= 0 (log-prob)
def test_compute_sequence_probability_is_nonpositive():
    model = make_model()
    input_ids = torch.randint(0, 256, (1, 5))
    target_ids = torch.randint(0, 256, (1, 3))
    result = compute_sequence_probability(model, input_ids, target_ids)
    assert result <= 0.0


# 7. rank_one_update output shape matches W
def test_rank_one_update_output_shape():
    d_out, d_in = 64, 128
    W = torch.randn(d_out, d_in)
    k = torch.randn(d_in)
    v = torch.randn(d_out)
    W_new = rank_one_update(W, k, v)
    assert W_new.shape == W.shape


# 8. rank_one_update correctness: W_new @ k == v
def test_rank_one_update_correctness():
    d_out, d_in = 32, 64
    W = torch.randn(d_out, d_in)
    k = torch.randn(d_in)
    v = torch.randn(d_out)
    W_new = rank_one_update(W, k, v)
    result = W_new @ k
    assert torch.allclose(result, v, atol=1e-4), f"max err={(result - v).abs().max()}"


# 9. rank_one_update preserves other directions
def test_rank_one_update_preserves_other_directions():
    d_out, d_in = 32, 64
    torch.manual_seed(0)
    W = torch.randn(d_out, d_in)
    k = torch.randn(d_in)
    v = torch.randn(d_out)
    z = torch.randn(d_in)
    z = z - (z @ k / (k @ k)) * k  # Gram-Schmidt orthogonalization
    W_new = rank_one_update(W, k, v)
    assert torch.allclose(W_new @ z, W @ z, atol=1e-4)


# 10. batch_rank_one_updates output shape matches W
def test_batch_rank_one_updates_output_shape():
    d_out, d_in, n_edits = 64, 128, 3
    W = torch.randn(d_out, d_in)
    keys = torch.randn(n_edits, d_in)
    values = torch.randn(n_edits, d_out)
    W_new = batch_rank_one_updates(W, keys, values)
    assert W_new.shape == W.shape


# 11. batch_rank_one_updates with n_edits=1: W_new @ k ≈ v
def test_batch_rank_one_updates_single_edit_parity():
    d_out, d_in = 32, 64
    torch.manual_seed(42)
    W = torch.randn(d_out, d_in)
    k = torch.randn(d_in)
    v = torch.randn(d_out)
    keys = k.unsqueeze(0)
    values = v.unsqueeze(0)
    W_new = batch_rank_one_updates(W, keys, values)
    result = W_new @ k
    assert torch.allclose(result, v, atol=1e-3), f"max err={(result - v).abs().max()}"


# 12. compute_edit_efficacy returns float in [0, 1]
def test_compute_edit_efficacy_returns_float_in_range():
    model = make_model()
    edit = EditRequest(subject="Paris", prompt="Paris is in", target_new="France")
    result = compute_edit_efficacy(model, encode, edit)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# 13. ModelEditor.apply_edit returns EditResult
def test_model_editor_apply_edit_returns_edit_result():
    model = make_model()
    editor = ModelEditor(model, encode, decode)
    edit = EditRequest(subject="Berlin", prompt="Berlin is in", target_new="Germany")
    result = editor.apply_edit(edit, layer_idx=0)
    assert isinstance(result, EditResult)
    assert result.success is True
    assert isinstance(result.edit_id, str) and len(result.edit_id) > 0
    assert 0.0 <= result.efficacy <= 1.0


# 14. ModelEditor.apply_batch_edits returns list of EditResult
def test_model_editor_apply_batch_edits_returns_list():
    model = make_model()
    editor = ModelEditor(model, encode, decode)
    edits = [
        EditRequest(subject="Tokyo", prompt="Tokyo is in", target_new="Japan"),
        EditRequest(subject="Cairo", prompt="Cairo is in", target_new="Egypt"),
    ]
    results = editor.apply_batch_edits(edits, layer_idx=0)
    assert isinstance(results, list)
    assert len(results) == len(edits)
    for r in results:
        assert isinstance(r, EditResult)
        assert r.success is True


# 15. ModelEditor.restore restores weights
def test_model_editor_restore_restores_weights():
    model = make_model()
    editor = ModelEditor(model, encode, decode)

    original_weights = {n: p.data.clone() for n, p in model.named_parameters()}
    W_before = model.layers[0].ffn.down_proj.weight.data.clone()

    edit = EditRequest(subject="London", prompt="London is in", target_new="UK")
    editor.apply_edit(edit, layer_idx=0)

    W_after = model.layers[0].ffn.down_proj.weight.data.clone()
    assert not torch.allclose(W_before, W_after), "Weight should have changed after edit"

    editor.restore(original_weights)
    W_restored = model.layers[0].ffn.down_proj.weight.data.clone()
    assert torch.allclose(W_before, W_restored, atol=1e-6), "Weights not fully restored"
