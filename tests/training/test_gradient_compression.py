"""Tests for src/training/gradient_compression.py"""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.gradient_compression import (
    CompressedGradOptimizer,
    ErrorFeedbackBuffer,
    GradCompressConfig,
    GradCompressTrainer,
    compress_gradient,
    quantize_gradient,
    random_compress,
    topk_compress,
)

# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_cfg():
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
def tiny_model(tiny_cfg):
    model = AureliusTransformer(tiny_cfg)
    return model


# ---------------------------------------------------------------------------
# 1. GradCompressConfig defaults
# ---------------------------------------------------------------------------


def test_grad_compress_config_defaults():
    cfg = GradCompressConfig()
    assert cfg.method == "topk"
    assert cfg.compression_ratio == 0.1
    assert cfg.bits == 8
    assert cfg.use_error_feedback is True


# ---------------------------------------------------------------------------
# 2. topk_compress — output shape matches input
# ---------------------------------------------------------------------------


def test_topk_compress_output_shape():
    grad = torch.randn(4, 8)
    k = 5
    compressed, mask = topk_compress(grad, k)
    assert compressed.shape == grad.shape
    assert mask.shape == grad.shape


# ---------------------------------------------------------------------------
# 3. topk_compress — exactly k non-zero elements
# ---------------------------------------------------------------------------


def test_topk_compress_exactly_k_nonzero():
    torch.manual_seed(0)
    grad = torch.randn(100)
    k = 17
    compressed, mask = topk_compress(grad, k)
    nonzero_count = (compressed != 0).sum().item()
    assert nonzero_count == k


# ---------------------------------------------------------------------------
# 4. topk_compress — keeps largest magnitude values
# ---------------------------------------------------------------------------


def test_topk_compress_keeps_largest():
    grad = torch.tensor([0.1, -5.0, 0.3, 2.0, -0.05])
    k = 2
    compressed, mask = topk_compress(grad, k)
    # The two largest magnitudes are -5.0 (idx 1) and 2.0 (idx 3)
    assert compressed[1].item() == pytest.approx(-5.0)
    assert compressed[3].item() == pytest.approx(2.0)
    # All others should be zero
    assert compressed[0].item() == pytest.approx(0.0)
    assert compressed[2].item() == pytest.approx(0.0)
    assert compressed[4].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. random_compress — exactly k non-zero elements
# ---------------------------------------------------------------------------


def test_random_compress_exactly_k_nonzero():
    torch.manual_seed(42)
    grad = torch.randn(200)
    k = 30
    compressed, mask = random_compress(grad, k)
    nonzero_count = (compressed != 0).sum().item()
    assert nonzero_count == k


# ---------------------------------------------------------------------------
# 6. random_compress — mask is boolean
# ---------------------------------------------------------------------------


def test_random_compress_mask_is_bool():
    grad = torch.randn(50)
    k = 10
    compressed, mask = random_compress(grad, k)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 7. quantize_gradient — output shape matches input
# ---------------------------------------------------------------------------


def test_quantize_gradient_output_shape():
    grad = torch.randn(3, 4, 5)
    quantized = quantize_gradient(grad, bits=8)
    assert quantized.shape == grad.shape


# ---------------------------------------------------------------------------
# 8. quantize_gradient — output values bounded within original range
# ---------------------------------------------------------------------------


def test_quantize_gradient_values_bounded():
    torch.manual_seed(7)
    grad = torch.randn(1000)
    quantized = quantize_gradient(grad, bits=4)
    assert quantized.min().item() >= grad.min().item() - 1e-5
    assert quantized.max().item() <= grad.max().item() + 1e-5


# ---------------------------------------------------------------------------
# 9. compress_gradient topk method returns correct shapes
# ---------------------------------------------------------------------------


def test_compress_gradient_topk_shapes():
    cfg = GradCompressConfig(method="topk", compression_ratio=0.2)
    grad = torch.randn(50)
    compressed, mask = compress_gradient(grad, cfg)
    assert compressed.shape == grad.shape
    assert mask is not None
    assert mask.shape == grad.shape


# ---------------------------------------------------------------------------
# 10. compress_gradient random method returns correct shapes
# ---------------------------------------------------------------------------


def test_compress_gradient_random_shapes():
    cfg = GradCompressConfig(method="random", compression_ratio=0.3)
    grad = torch.randn(40)
    compressed, mask = compress_gradient(grad, cfg)
    assert compressed.shape == grad.shape
    assert mask is not None
    assert mask.shape == grad.shape


# ---------------------------------------------------------------------------
# 11. compress_gradient quantize method returns (grad, None)
# ---------------------------------------------------------------------------


def test_compress_gradient_quantize_no_mask():
    cfg = GradCompressConfig(method="quantize", bits=8)
    grad = torch.randn(20)
    compressed, mask = compress_gradient(grad, cfg)
    assert compressed.shape == grad.shape
    assert mask is None


# ---------------------------------------------------------------------------
# 12. ErrorFeedbackBuffer starts empty
# ---------------------------------------------------------------------------


def test_error_feedback_buffer_starts_empty():
    buf = ErrorFeedbackBuffer()
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# 13. ErrorFeedbackBuffer.update accumulates errors across steps
# ---------------------------------------------------------------------------


def test_error_feedback_buffer_accumulates():
    buf = ErrorFeedbackBuffer()
    grad = torch.ones(10)

    # Step 1: compress to zeros (maximum error)
    compressed1 = torch.zeros(10)
    corrected1 = buf.update("layer", grad, compressed1)
    # First call: no prior error, corrected == grad
    assert torch.allclose(corrected1, grad)
    # Buffer should now contain an error entry
    assert len(buf) == 1

    # Step 2: the accumulated error from step 1 should be added
    grad2 = torch.ones(10) * 0.5
    compressed2 = torch.zeros(10)
    corrected2 = buf.update("layer", grad2, compressed2)
    # corrected2 should be > grad2 because residual from step 1 is added
    assert corrected2.sum().item() > grad2.sum().item()


# ---------------------------------------------------------------------------
# 14. CompressedGradOptimizer.step returns dict with correct keys
# ---------------------------------------------------------------------------


def test_compressed_grad_optimizer_step_keys(tiny_model):
    named_params = [(n, p) for n, p in tiny_model.named_parameters() if p.requires_grad]
    base_opt = torch.optim.SGD([p for _, p in named_params], lr=1e-3)
    cfg = GradCompressConfig(method="topk", compression_ratio=0.1)
    opt = CompressedGradOptimizer(base_opt, cfg, named_params)

    # Create synthetic gradients
    for _, p in named_params:
        p.grad = torch.randn_like(p)

    info = opt.step()
    assert "n_params_compressed" in info
    assert "mean_compression_ratio" in info
    assert "n_error_feedback_applied" in info


# ---------------------------------------------------------------------------
# 15. GradCompressTrainer.train_step returns dict with 'loss'
# ---------------------------------------------------------------------------


def test_grad_compress_trainer_train_step_has_loss(tiny_model):
    cfg = GradCompressConfig(method="topk", compression_ratio=0.1)
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
    trainer = GradCompressTrainer(tiny_model, cfg, optimizer)

    input_ids = torch.randint(0, 256, (2, 16))
    result = trainer.train_step(input_ids)

    assert "loss" in result
    assert isinstance(result["loss"], float)


# ---------------------------------------------------------------------------
# 16. GradCompressTrainer — loss is finite over multiple steps
# ---------------------------------------------------------------------------


def test_grad_compress_trainer_loss_finite_over_steps(tiny_model):
    cfg = GradCompressConfig(method="topk", compression_ratio=0.1, use_error_feedback=True)
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
    trainer = GradCompressTrainer(tiny_model, cfg, optimizer)

    input_ids = torch.randint(0, 256, (2, 16))
    for step in range(5):
        result = trainer.train_step(input_ids)
        assert torch.isfinite(torch.tensor(result["loss"])), (
            f"Loss is not finite at step {step}: {result['loss']}"
        )


# ---------------------------------------------------------------------------
# Bonus: random compression with different seeds gives different masks
# ---------------------------------------------------------------------------


def test_random_compress_stochastic():
    grad = torch.randn(100)
    k = 10
    torch.manual_seed(1)
    _, mask1 = random_compress(grad, k)
    torch.manual_seed(2)
    _, mask2 = random_compress(grad, k)
    # With overwhelming probability, two random masks of size 10/100 differ
    assert not torch.equal(mask1, mask2)


# ---------------------------------------------------------------------------
# Bonus: quantize_gradient constant tensor (no division by zero)
# ---------------------------------------------------------------------------


def test_quantize_gradient_constant_tensor():
    grad = torch.ones(20) * 3.14
    quantized = quantize_gradient(grad, bits=8)
    assert quantized.shape == grad.shape
    # Should return a clone of the original without NaN/inf
    assert torch.all(torch.isfinite(quantized))


# ---------------------------------------------------------------------------
# Bonus: CompressedGradOptimizer zero_grad delegates to inner optimizer
# ---------------------------------------------------------------------------


def test_compressed_grad_optimizer_zero_grad(tiny_model):
    named_params = [(n, p) for n, p in tiny_model.named_parameters() if p.requires_grad]
    base_opt = torch.optim.SGD([p for _, p in named_params], lr=1e-3)
    cfg = GradCompressConfig(method="random", compression_ratio=0.2)
    opt = CompressedGradOptimizer(base_opt, cfg, named_params)

    # Plant synthetic gradients
    for _, p in named_params:
        p.grad = torch.ones_like(p)

    opt.zero_grad()

    for _, p in named_params:
        assert p.grad is None or (p.grad == 0).all()
