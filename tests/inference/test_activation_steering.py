"""Tests for activation steering / representation engineering."""
import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.activation_steering import (
    SteeringConfig,
    compute_steering_vector,
    SteeringHook,
    ActivationSteerer,
    contrastive_activation_addition,
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

SEQ_LEN = 4
MAX_NEW_TOKENS = 3


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


@pytest.fixture
def input_ids():
    torch.manual_seed(7)
    return torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))


@pytest.fixture
def pos_ids():
    torch.manual_seed(1)
    return torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))


@pytest.fixture
def neg_ids():
    torch.manual_seed(2)
    return torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))


# ---------------------------------------------------------------------------
# 1. SteeringConfig defaults
# ---------------------------------------------------------------------------

def test_steering_config_defaults():
    cfg = SteeringConfig()
    assert cfg.layer_idx == 0
    assert cfg.coeff == 1.0
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# 2. compute_steering_vector returns shape (d_model,)
# ---------------------------------------------------------------------------

def test_compute_steering_vector_shape(small_model, pos_ids, neg_ids):
    vec = compute_steering_vector(small_model, pos_ids, neg_ids, layer_idx=0)
    assert vec.shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# 3. compute_steering_vector same positive/negative near-zero vector
# ---------------------------------------------------------------------------

def test_compute_steering_vector_same_inputs_near_zero(small_model, pos_ids):
    vec = compute_steering_vector(small_model, pos_ids, pos_ids, layer_idx=0)
    assert vec.norm().item() < 1e-5


# ---------------------------------------------------------------------------
# 4. compute_steering_vector different inputs nonzero
# ---------------------------------------------------------------------------

def test_compute_steering_vector_different_inputs_nonzero(small_model, pos_ids, neg_ids):
    vec = compute_steering_vector(small_model, pos_ids, neg_ids, layer_idx=0)
    assert vec.norm().item() > 1e-6


# ---------------------------------------------------------------------------
# 5. SteeringHook modifies output (different from unsteered)
# ---------------------------------------------------------------------------

def test_steering_hook_modifies_output(small_model, input_ids):
    with torch.no_grad():
        _, baseline_logits, _ = small_model(input_ids)

    vec = torch.randn(D_MODEL)
    cfg = SteeringConfig(layer_idx=0, coeff=10.0, normalize=False)
    hook = SteeringHook(vec, cfg)
    handle = hook.register(small_model)
    try:
        with torch.no_grad():
            _, steered_logits, _ = small_model(input_ids)
    finally:
        handle.remove()

    assert not torch.allclose(baseline_logits, steered_logits)


# ---------------------------------------------------------------------------
# 6. SteeringHook with coeff=0 same as unsteered
# ---------------------------------------------------------------------------

def test_steering_hook_coeff_zero_no_change(small_model, input_ids):
    with torch.no_grad():
        _, baseline_logits, _ = small_model(input_ids)

    vec = torch.randn(D_MODEL)
    cfg = SteeringConfig(layer_idx=0, coeff=0.0, normalize=False)
    hook = SteeringHook(vec, cfg)
    handle = hook.register(small_model)
    try:
        with torch.no_grad():
            _, steered_logits, _ = small_model(input_ids)
    finally:
        handle.remove()

    assert torch.allclose(baseline_logits, steered_logits, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. SteeringHook handles tuple output correctly
# ---------------------------------------------------------------------------

def test_steering_hook_handles_tuple_output(small_model):
    """Ensure hook does not crash and layer output remains a tuple."""
    results = []

    def capture_hook(module, input, output):
        results.append(type(output))
        return output

    ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    handle = small_model.layers[0].register_forward_hook(capture_hook)
    with torch.no_grad():
        small_model(ids)
    handle.remove()

    # Apply steering hook and verify no crash
    vec = torch.zeros(D_MODEL)
    cfg = SteeringConfig(layer_idx=0, coeff=1.0, normalize=False)
    hook = SteeringHook(vec, cfg)
    handle = hook.register(small_model)
    try:
        with torch.no_grad():
            _, logits, _ = small_model(ids)
    finally:
        handle.remove()

    assert logits is not None
    assert results[0] is tuple


# ---------------------------------------------------------------------------
# 8. ActivationSteerer.generate returns shape (1, max_new_tokens)
# ---------------------------------------------------------------------------

def test_activation_steerer_generate_shape(small_model, input_ids):
    vec = torch.randn(D_MODEL)
    cfg = SteeringConfig(layer_idx=0, coeff=1.0, normalize=True)
    steerer = ActivationSteerer(small_model, cfg)
    steerer.add_steering_vector(vec)
    out = steerer.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    assert out.shape == (1, MAX_NEW_TOKENS)


# ---------------------------------------------------------------------------
# 9. ActivationSteerer.generate token ids in [0, vocab_size)
# ---------------------------------------------------------------------------

def test_activation_steerer_generate_valid_tokens(small_model, input_ids):
    vec = torch.randn(D_MODEL)
    cfg = SteeringConfig(layer_idx=0, coeff=1.0, normalize=True)
    steerer = ActivationSteerer(small_model, cfg)
    steerer.add_steering_vector(vec)
    out = steerer.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    assert (out >= 0).all()
    assert (out < VOCAB_SIZE).all()


# ---------------------------------------------------------------------------
# 10. ActivationSteerer.remove_steering removes hooks
# ---------------------------------------------------------------------------

def test_activation_steerer_remove_steering_clears_hooks(small_model, input_ids):
    vec = torch.randn(D_MODEL)
    cfg = SteeringConfig(layer_idx=0, coeff=1.0, normalize=True)
    steerer = ActivationSteerer(small_model, cfg)
    steerer.add_steering_vector(vec)
    steerer.remove_steering()
    assert len(steerer._handles) == 0
    assert len(small_model.layers[0]._forward_hooks) == 0


# ---------------------------------------------------------------------------
# 11. Steering at different layers gives different outputs
# ---------------------------------------------------------------------------

def test_steering_different_layers_different_outputs(small_model, input_ids):
    vec = torch.randn(D_MODEL)

    cfg0 = SteeringConfig(layer_idx=0, coeff=5.0, normalize=False)
    hook0 = SteeringHook(vec, cfg0)
    handle0 = hook0.register(small_model)
    with torch.no_grad():
        _, logits0, _ = small_model(input_ids)
    handle0.remove()

    cfg1 = SteeringConfig(layer_idx=1, coeff=5.0, normalize=False)
    hook1 = SteeringHook(vec, cfg1)
    handle1 = hook1.register(small_model)
    with torch.no_grad():
        _, logits1, _ = small_model(input_ids)
    handle1.remove()

    assert not torch.allclose(logits0, logits1)


# ---------------------------------------------------------------------------
# 12. contrastive_activation_addition returns shape (1, T, V)
# ---------------------------------------------------------------------------

def test_contrastive_activation_addition_shape(small_model, input_ids, pos_ids, neg_ids):
    logits = contrastive_activation_addition(
        small_model, input_ids, pos_ids, neg_ids, layer_idx=0, coeff=1.0
    )
    assert logits.shape == (1, SEQ_LEN, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 13. Normalized vector has unit norm
# ---------------------------------------------------------------------------

def test_normalized_vector_unit_norm(small_model, pos_ids, neg_ids):
    vec = compute_steering_vector(small_model, pos_ids, neg_ids, layer_idx=0)
    cfg = SteeringConfig(layer_idx=0, coeff=1.0, normalize=True)
    steerer = ActivationSteerer(small_model, cfg)
    steerer.add_steering_vector(vec)
    stored = steerer._steering_vector
    assert stored is not None
    assert abs(stored.norm().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 14. High coeff changes output more than low coeff
# ---------------------------------------------------------------------------

def test_high_coeff_changes_output_more(small_model, input_ids):
    with torch.no_grad():
        _, baseline_logits, _ = small_model(input_ids)

    vec = torch.randn(D_MODEL)

    cfg_low = SteeringConfig(layer_idx=0, coeff=0.01, normalize=False)
    hook_low = SteeringHook(vec, cfg_low)
    handle_low = hook_low.register(small_model)
    with torch.no_grad():
        _, logits_low, _ = small_model(input_ids)
    handle_low.remove()

    cfg_high = SteeringConfig(layer_idx=0, coeff=100.0, normalize=False)
    hook_high = SteeringHook(vec, cfg_high)
    handle_high = hook_high.register(small_model)
    with torch.no_grad():
        _, logits_high, _ = small_model(input_ids)
    handle_high.remove()

    diff_low = (logits_low - baseline_logits).abs().sum().item()
    diff_high = (logits_high - baseline_logits).abs().sum().item()

    assert diff_high > diff_low, (
        f"High coeff diff ({diff_high:.4f}) should exceed low coeff diff ({diff_low:.4f})"
    )
