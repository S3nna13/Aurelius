"""Tests for activation steering vectors (arXiv:2310.01405)."""

import pytest
import torch

from src.inference.steering import (
    SteeringHook,
    SteeringVector,
    compute_steering_vector,
    extract_hidden_states_at_layer,
    generate_with_steering,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# Small model config shared across all tests
N_LAYERS = 4
D_MODEL = 64
N_HEADS = 2
N_KV_HEADS = 2
HEAD_DIM = 32
D_FF = 128
VOCAB_SIZE = 256
MAX_SEQ_LEN = 32


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
# extract_hidden_states_at_layer
# ---------------------------------------------------------------------------


def test_extract_hidden_states_shape(small_model):
    """Returns (B, D) tensor."""
    B, S = 3, 8
    input_ids = torch.randint(0, VOCAB_SIZE, (B, S))
    hs = extract_hidden_states_at_layer(small_model, input_ids, layer_idx=1)
    assert hs.shape == (B, D_MODEL)


def test_extract_hidden_states_mean_pool(small_model):
    """pool='mean' aggregates over sequence dimension."""
    B, S = 2, 10
    input_ids = torch.randint(0, VOCAB_SIZE, (B, S))
    hs = extract_hidden_states_at_layer(small_model, input_ids, layer_idx=0, pool="mean")
    assert hs.shape == (B, D_MODEL)


def test_extract_hidden_states_last_pool(small_model):
    """pool='last' returns the final token's hidden state."""
    B, S = 2, 10
    input_ids = torch.randint(0, VOCAB_SIZE, (B, S))
    hs = extract_hidden_states_at_layer(small_model, input_ids, layer_idx=0, pool="last")
    assert hs.shape == (B, D_MODEL)


# ---------------------------------------------------------------------------
# compute_steering_vector
# ---------------------------------------------------------------------------


def test_compute_steering_vector_shape(small_model):
    """Steering vector direction has shape (D,)."""
    N, S = 4, 8
    pos_ids = torch.randint(0, VOCAB_SIZE, (N, S))
    neg_ids = torch.randint(0, VOCAB_SIZE, (N, S))
    sv = compute_steering_vector(small_model, pos_ids, neg_ids, layer_idx=1)
    assert sv.direction.shape == (D_MODEL,)


def test_compute_steering_vector_normalized(small_model):
    """When normalize=True, direction should have unit norm."""
    N, S = 4, 8
    pos_ids = torch.randint(0, VOCAB_SIZE, (N, S))
    neg_ids = torch.randint(0, VOCAB_SIZE, (N, S))
    sv = compute_steering_vector(small_model, pos_ids, neg_ids, layer_idx=1, normalize=True)
    assert abs(sv.norm - 1.0) < 1e-4


def test_compute_steering_vector_not_normalized(small_model):
    """When normalize=False, direction norm is not constrained to 1."""
    N, S = 4, 8
    torch.manual_seed(7)
    pos_ids = torch.randint(0, VOCAB_SIZE, (N, S))
    neg_ids = torch.randint(0, VOCAB_SIZE, (N, S))
    sv = compute_steering_vector(small_model, pos_ids, neg_ids, layer_idx=1, normalize=False)
    # The raw difference is very unlikely to be unit norm
    norm_val = sv.direction.norm().item()
    assert abs(norm_val - 1.0) > 1e-2


# ---------------------------------------------------------------------------
# SteeringHook
# ---------------------------------------------------------------------------


def test_steering_hook_context_manager(small_model):
    """SteeringHook can be used as a context manager without crashing."""
    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=1, name="test")
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    with SteeringHook(small_model, sv, alpha=5.0):
        with torch.no_grad():
            _, logits, _ = small_model(input_ids)
    assert logits.shape == (1, 8, VOCAB_SIZE)


def test_steering_hook_modifies_output(small_model):
    """Output with steering hook should differ from unsteered output."""
    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=1, name="test")
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))

    with torch.no_grad():
        _, logits_clean, _ = small_model(input_ids)

    with SteeringHook(small_model, sv, alpha=20.0):
        with torch.no_grad():
            _, logits_steered, _ = small_model(input_ids)

    assert not torch.allclose(logits_clean, logits_steered)


# ---------------------------------------------------------------------------
# generate_with_steering
# ---------------------------------------------------------------------------


def test_generate_with_steering_returns_tensor(small_model):
    """generate_with_steering returns a Tensor."""
    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=1)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
    output = generate_with_steering(small_model, input_ids, sv, max_new_tokens=5, alpha=1.0)
    assert isinstance(output, torch.Tensor)


def test_generate_with_steering_length(small_model):
    """Generated tokens count does not exceed max_new_tokens."""
    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=1)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
    max_new = 8
    output = generate_with_steering(small_model, input_ids, sv, max_new_tokens=max_new, alpha=1.0)
    assert output.shape[0] <= max_new


# ---------------------------------------------------------------------------
# SteeringVector dataclass
# ---------------------------------------------------------------------------


def test_steering_vector_name():
    """name field is accessible on SteeringVector."""
    direction = torch.randn(D_MODEL)
    sv = SteeringVector(direction=direction, layer_idx=2, name="honesty")
    assert sv.name == "honesty"
    assert sv.layer_idx == 2
