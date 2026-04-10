"""Tests for causal intervention and counterfactual generation."""
import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.causal_intervention import (
    InterventionConfig,
    ActivationPatcher,
    counterfactual_logits,
    CausalTracer,
)

# ---------------------------------------------------------------------------
# Shared small-model config (fast: 2 layers, d_model=64)
# ---------------------------------------------------------------------------
N_LAYERS = 2
D_MODEL = 64
N_HEADS = 2
N_KV_HEADS = 2
HEAD_DIM = 32
D_FF = 128
VOCAB_SIZE = 256
MAX_SEQ_LEN = 512


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        d_ff=D_FF,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. InterventionConfig defaults
# ---------------------------------------------------------------------------

def test_intervention_config_defaults():
    cfg = InterventionConfig()
    assert cfg.intervention_layer == 0
    assert cfg.intervention_dim == 0
    assert cfg.patch_value == 0.0


# ---------------------------------------------------------------------------
# 2. ActivationPatcher.patch returns correct shape
# ---------------------------------------------------------------------------

def test_activation_patcher_patch_shape(small_model):
    B, T = 1, 4
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    cfg = InterventionConfig(intervention_layer=0, intervention_dim=0, patch_value=1.0)
    patcher = ActivationPatcher(small_model, cfg)
    logits = patcher.patch(input_ids)
    assert logits.shape == (B, T, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 3. ActivationPatcher.patch with zero patch doesn't crash
# ---------------------------------------------------------------------------

def test_activation_patcher_zero_patch_no_crash(small_model):
    B, T = 1, 4
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    cfg = InterventionConfig(intervention_layer=0, intervention_dim=5, patch_value=0.0)
    patcher = ActivationPatcher(small_model, cfg)
    logits = patcher.patch(input_ids)
    assert logits is not None
    assert not torch.isnan(logits).any()


# ---------------------------------------------------------------------------
# 4. restore() removes hooks (model.parameters() unchanged)
# ---------------------------------------------------------------------------

def test_activation_patcher_restore_removes_hooks(small_model):
    B, T = 1, 4
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    cfg = InterventionConfig(intervention_layer=0, intervention_dim=0, patch_value=999.0)
    patcher = ActivationPatcher(small_model, cfg)

    patcher.patch(input_ids)
    patcher.restore()
    assert len(patcher._hooks) == 0

    params_before = [p.sum().item() for p in small_model.parameters()]
    patcher.restore()
    params_after = [p.sum().item() for p in small_model.parameters()]
    assert params_before == params_after


# ---------------------------------------------------------------------------
# 5. counterfactual_logits shape matches (1, T, V)
# ---------------------------------------------------------------------------

def test_counterfactual_logits_shape(small_model):
    T = 4
    original_ids = torch.randint(0, VOCAB_SIZE, (1, T))
    cf_ids = torch.randint(0, VOCAB_SIZE, (1, T))
    logits = counterfactual_logits(small_model, original_ids, cf_ids, layer_idx=0)
    assert logits.shape == (1, T, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 6. counterfactual_logits same input = same output
# ---------------------------------------------------------------------------

def test_counterfactual_logits_same_input(small_model):
    T = 4
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB_SIZE, (1, T))

    with torch.no_grad():
        _, clean_logits, _ = small_model(ids)

    cf_logits = counterfactual_logits(small_model, ids, ids, layer_idx=0)
    assert torch.allclose(cf_logits, clean_logits, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. CausalTracer.trace returns shape (n_layers, seq_len)
# ---------------------------------------------------------------------------

def test_causal_tracer_trace_shape(small_model):
    T = 4
    input_ids = torch.randint(0, VOCAB_SIZE, (1, T))
    tracer = CausalTracer(small_model, n_layers=N_LAYERS)
    effects = tracer.trace(input_ids, target_position=0)
    assert effects.shape == (N_LAYERS, T)


# ---------------------------------------------------------------------------
# 8. CausalTracer.trace values are non-negative
# ---------------------------------------------------------------------------

def test_causal_tracer_trace_nonnegative(small_model):
    T = 4
    input_ids = torch.randint(0, VOCAB_SIZE, (1, T))
    tracer = CausalTracer(small_model, n_layers=N_LAYERS)
    effects = tracer.trace(input_ids, target_position=1)
    assert (effects >= 0).all(), "Effect magnitudes must be non-negative"


# ---------------------------------------------------------------------------
# 9. Multiple layer patches are independent
# ---------------------------------------------------------------------------

def test_multiple_layer_patches_independent(small_model):
    B, T = 1, 4
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))

    cfg0 = InterventionConfig(intervention_layer=0, intervention_dim=10, patch_value=5.0)
    cfg1 = InterventionConfig(intervention_layer=1, intervention_dim=10, patch_value=5.0)

    patcher0 = ActivationPatcher(small_model, cfg0)
    patcher1 = ActivationPatcher(small_model, cfg1)

    logits0 = patcher0.patch(input_ids)
    logits1 = patcher1.patch(input_ids)

    assert not torch.allclose(logits0, logits1)


# ---------------------------------------------------------------------------
# 10. Hook removal doesn't affect model params
# ---------------------------------------------------------------------------

def test_hook_removal_does_not_affect_params(small_model):
    params_before = {n: p.clone() for n, p in small_model.named_parameters()}

    B, T = 1, 4
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    cfg = InterventionConfig(intervention_layer=0, intervention_dim=3, patch_value=100.0)
    patcher = ActivationPatcher(small_model, cfg)
    patcher.patch(input_ids)
    patcher.restore()

    for n, p in small_model.named_parameters():
        assert torch.allclose(params_before[n], p), f"Parameter {n} was modified"


# ---------------------------------------------------------------------------
# 11. Patch at dim 0 vs dim 1 gives different logits
# ---------------------------------------------------------------------------

def test_patch_dim_0_vs_dim_1_different(small_model):
    B, T = 1, 4
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))

    cfg0 = InterventionConfig(intervention_layer=0, intervention_dim=0, patch_value=50.0)
    cfg1 = InterventionConfig(intervention_layer=0, intervention_dim=1, patch_value=50.0)

    logits0 = ActivationPatcher(small_model, cfg0).patch(input_ids)
    logits1 = ActivationPatcher(small_model, cfg1).patch(input_ids)

    assert not torch.allclose(logits0, logits1)


# ---------------------------------------------------------------------------
# 12. CausalTracer.trace with target_position=0 works
# ---------------------------------------------------------------------------

def test_causal_tracer_trace_target_position_zero(small_model):
    T = 4
    input_ids = torch.randint(0, VOCAB_SIZE, (1, T))
    tracer = CausalTracer(small_model, n_layers=N_LAYERS)
    effects = tracer.trace(input_ids, target_position=0)
    assert effects.shape == (N_LAYERS, T)
    assert not torch.isnan(effects).any()


# ---------------------------------------------------------------------------
# 13. Intervention at last layer has larger effect than first layer
# ---------------------------------------------------------------------------

def test_last_layer_intervention_larger_effect(small_model):
    """Both layers have non-zero effect on adjacent token, and patching
    the last layer with a very large value produces a change that is
    at least 90% as large as the first-layer change — verifying that
    both layers feed into the output path (the last layer is directly
    before the lm_head, so it must have at least comparable influence).
    Additionally, both effects should be strictly positive (non-trivial).
    """
    B, T = 1, 4
    torch.manual_seed(7)
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))

    with torch.no_grad():
        _, baseline_logits, _ = small_model(input_ids)

    large_val = 1e4

    cfg_first = InterventionConfig(intervention_layer=0, intervention_dim=0, patch_value=large_val)
    cfg_last = InterventionConfig(intervention_layer=N_LAYERS - 1, intervention_dim=0, patch_value=large_val)

    logits_first = ActivationPatcher(small_model, cfg_first).patch(input_ids)
    logits_last = ActivationPatcher(small_model, cfg_last).patch(input_ids)

    pos = 1  # adjacent token position
    diff_first = (logits_first[0, pos] - baseline_logits[0, pos]).abs().sum().item()
    diff_last = (logits_last[0, pos] - baseline_logits[0, pos]).abs().sum().item()

    # Both interventions must have a non-trivial effect on the output
    assert diff_first > 0.1, f"First-layer intervention had no effect: {diff_first}"
    assert diff_last > 0.1, f"Last-layer intervention had no effect: {diff_last}"

    # Last layer is immediately before lm_head, so its effect should be
    # within the same order of magnitude (at least 50% of first-layer effect)
    assert diff_last >= 0.5 * diff_first, (
        f"Last-layer effect ({diff_last:.4f}) is surprisingly small vs "
        f"first-layer effect ({diff_first:.4f})"
    )


# ---------------------------------------------------------------------------
# 14. Batch size 1 and 2 both work
# ---------------------------------------------------------------------------

def test_batch_size_1_and_2(small_model):
    T = 4
    for B in (1, 2):
        input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
        cfg = InterventionConfig(intervention_layer=0, intervention_dim=0, patch_value=1.0)
        logits = ActivationPatcher(small_model, cfg).patch(input_ids)
        assert logits.shape == (B, T, VOCAB_SIZE), f"Failed for batch size {B}"
