"""Tests for the unified quantization pipeline in src/quantization/quantize.py."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.quantization.quantize import quantize

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

TINY_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)


@pytest.fixture
def tiny_model() -> AureliusTransformer:
    model = AureliusTransformer(TINY_CONFIG)
    model.eval()
    return model


@pytest.fixture(autouse=True)
def _ensure_qengine() -> None:
    """Ensure a quantized engine is available for 8-bit tests on CPU."""
    if torch.backends.quantized.engine == "none" and "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"


def _forward_pass(model: AureliusTransformer) -> torch.Tensor:
    """Run a tiny forward pass and return logits."""
    tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))
    with torch.no_grad():
        loss, logits, _ = model(tokens)
    return logits


# ---------------------------------------------------------------------------
# Mode coverage
# ---------------------------------------------------------------------------


class TestQuantizePipelineModes:
    def test_quantize_8bit(self, tiny_model: AureliusTransformer) -> None:
        """8-bit dynamic quantization can be applied and model stays functional."""
        quantize(tiny_model, mode="8bit", device="cpu")
        logits = _forward_pass(tiny_model)
        assert logits.shape == (1, 8, TINY_CONFIG.vocab_size)

    def test_quantize_4bit_fallback_on_cpu(self, tiny_model: AureliusTransformer) -> None:
        """4-bit on CPU falls back to 8-bit but remains functional."""
        quantize(tiny_model, mode="4bit", device="cpu")
        logits = _forward_pass(tiny_model)
        assert logits.shape == (1, 8, TINY_CONFIG.vocab_size)

    def test_quantize_nf4(self, tiny_model: AureliusTransformer) -> None:
        """NF4 quantization can be applied and model stays functional."""
        quantize(tiny_model, mode="nf4", device="cpu")
        logits = _forward_pass(tiny_model)
        assert logits.shape == (1, 8, TINY_CONFIG.vocab_size)

    def test_quantize_2bit_ternary(self, tiny_model: AureliusTransformer) -> None:
        """2-bit ternary quantization can be applied and model stays functional."""
        quantize(tiny_model, mode="2bit_ternary", device="cpu")
        logits = _forward_pass(tiny_model)
        assert logits.shape == (1, 8, TINY_CONFIG.vocab_size)


# ---------------------------------------------------------------------------
# Invalid input handling
# ---------------------------------------------------------------------------


class TestQuantizePipelineValidation:
    def test_invalid_mode_raises_value_error(self, tiny_model: AureliusTransformer) -> None:
        """An unknown quantization mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown quant mode"):
            quantize(tiny_model, mode="16bit", device="cpu")


# ---------------------------------------------------------------------------
# Weight dtype / bit-width expectations
# ---------------------------------------------------------------------------


class TestQuantizePipelineWeightProperties:
    def test_8bit_changes_linear_to_quantized(self, tiny_model: AureliusTransformer) -> None:
        """After 8-bit quantization, Linear layers contain quantized packed params."""
        quantize(tiny_model, mode="8bit", device="cpu")
        packed_params_layers = [
            name
            for name, mod in tiny_model.named_modules()
            if "PackedParams" in type(mod).__name__
        ]
        assert len(packed_params_layers) > 0

    def test_2bit_ternary_weights_are_ternary(self, tiny_model: AureliusTransformer) -> None:
        """After ternary quantization, each weight is either +scale, -scale, or 0."""
        # Capture pre-quantization weights for comparison
        pre_weights = {
            name: param.data.clone()
            for name, param in tiny_model.named_parameters()
            if "weight" in name and param.dim() >= 2
        }

        quantize(tiny_model, mode="2bit_ternary", device="cpu")

        for name, param in tiny_model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                scale = pre_weights[name].abs().mean()
                unique_vals = param.data.unique().tolist()
                # Allow for floating-point tolerance around 0, +scale, -scale
                for val in unique_vals:
                    assert any(abs(val - target) < 1e-4 for target in [-scale.item(), 0.0, scale.item()])


# ---------------------------------------------------------------------------
# Device coverage
# ---------------------------------------------------------------------------


class TestQuantizePipelineDevice:
    def test_quantize_runs_on_cpu(self, tiny_model: AureliusTransformer) -> None:
        """Quantization with device='cpu' succeeds."""
        quantize(tiny_model, mode="8bit", device="cpu")
        logits = _forward_pass(tiny_model)
        assert logits.device.type == "cpu"
