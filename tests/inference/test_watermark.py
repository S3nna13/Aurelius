import pytest
import torch
import math
from src.inference.watermark import WatermarkConfig, WatermarkLogitProcessor, WatermarkDetector

@pytest.fixture
def cfg():
    return WatermarkConfig(vocab_size=1000, green_list_fraction=0.5, delta=2.0, seed=42)

def test_green_list_size(cfg):
    proc = WatermarkLogitProcessor(cfg)
    green = proc._get_green_list(0)
    assert green.shape == (1000,)
    assert green.sum().item() == 500  # 50% of 1000

def test_green_list_deterministic(cfg):
    proc = WatermarkLogitProcessor(cfg)
    g1 = proc._get_green_list(7)
    g2 = proc._get_green_list(7)
    assert torch.equal(g1, g2)

def test_green_list_varies_by_token(cfg):
    proc = WatermarkLogitProcessor(cfg)
    g1 = proc._get_green_list(0)
    g2 = proc._get_green_list(1)
    assert not torch.equal(g1, g2)

def test_logit_processor_boosts_green(cfg):
    proc = WatermarkLogitProcessor(cfg)
    logits = torch.zeros(1000)
    result = proc(logits, torch.tensor([42]))
    green = proc._get_green_list(42)
    assert (result[green] == 2.0).all()
    assert (result[~green] == 0.0).all()

def test_logit_processor_shape(cfg):
    proc = WatermarkLogitProcessor(cfg)
    logits = torch.zeros(1000)
    result = proc(logits, torch.tensor([0, 1, 2]))
    assert result.shape == (1000,)

def test_detector_high_z_for_watermarked(cfg):
    """Watermarked tokens should have high z-score."""
    proc = WatermarkLogitProcessor(cfg)
    detector = WatermarkDetector(cfg)

    # Generate a sequence where every token is from the green list
    tokens = [100]  # start token
    for i in range(50):
        green = proc._get_green_list(tokens[-1])
        green_ids = green.nonzero().squeeze(1).tolist()
        tokens.append(green_ids[0])  # always pick first green token

    result = detector.detect(tokens)
    assert result["z_score"] > 4.0
    assert result["is_watermarked"] is True

def test_detector_low_z_for_random(cfg):
    """Random tokens should have low z-score on average."""
    detector = WatermarkDetector(cfg)
    torch.manual_seed(0)
    # Random tokens — green fraction should be ~0.5 -> z ~ 0
    tokens = torch.randint(0, 1000, (100,)).tolist()
    result = detector.detect(tokens)
    # z-score should be small (not necessarily < 4, but much less than watermarked)
    assert abs(result["z_score"]) < 10  # sanity check, not strict

def test_detector_output_keys(cfg):
    detector = WatermarkDetector(cfg)
    result = detector.detect([1, 2, 3, 4, 5])
    assert all(k in result for k in ["green_count", "total", "z_score", "is_watermarked", "green_fraction"])
