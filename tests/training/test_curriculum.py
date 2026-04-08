import pytest
import torch
from src.training.curriculum import CurriculumConfig, WeightMode, TokenWeighter

def test_uniform_mode_equals_mean():
    weighter = TokenWeighter(CurriculumConfig(mode=WeightMode.UNIFORM))
    loss = torch.ones(2, 8)
    result = weighter(loss)
    assert result == pytest.approx(1.0)

def test_position_weights_increase():
    """Later positions should get higher weight -> loss on later tokens weighted more."""
    weighter = TokenWeighter(CurriculumConfig(mode=WeightMode.POSITION, position_exponent=1.0))
    # Make only the last token have high loss
    loss = torch.zeros(1, 10)
    loss[0, -1] = 10.0
    result_pos = weighter(loss)
    # Compare to uniform (which would give 10/10 = 1.0)
    result_uniform = TokenWeighter(CurriculumConfig())(loss)
    # Position weighting should give higher weight to the last token
    assert result_pos.item() > result_uniform.item()

def test_frequency_weights_upweight_rare():
    weighter = TokenWeighter(CurriculumConfig(mode=WeightMode.FREQUENCY))
    # Token 0 is rare (count=1), token 1 is common (count=1000)
    token_counts = torch.tensor([1.0, 1000.0])
    input_ids = torch.tensor([[0, 1]])  # batch=1, seq=2
    loss = torch.tensor([[1.0, 1.0]])   # equal loss
    result = weighter(loss, input_ids=input_ids, token_counts=token_counts)
    # Rare token should dominate -> result > 1.0 (weighted toward rare)
    # Actually with normalize_weights=True and equal losses, result = 1.0
    # But the rare token gets more weight, so if loss[rare] > loss[common]:
    loss2 = torch.tensor([[2.0, 1.0]])  # rare token has higher loss
    result2 = weighter(loss2, input_ids=input_ids, token_counts=token_counts)
    result_uniform2 = TokenWeighter()(torch.tensor([[2.0, 1.0]]))
    assert result2.item() > result_uniform2.item()

def test_custom_weights():
    weighter = TokenWeighter(CurriculumConfig(mode=WeightMode.CUSTOM, normalize_weights=False))
    loss = torch.tensor([[1.0, 2.0, 3.0]])
    weights = torch.tensor([[0.0, 0.0, 1.0]])
    result = weighter(loss, custom_weights=weights)
    # Only last token selected: weighted mean = 3.0
    assert result.item() == pytest.approx(3.0)

def test_padding_mask_excludes_padded():
    weighter = TokenWeighter(CurriculumConfig(mode=WeightMode.UNIFORM))
    loss = torch.tensor([[1.0, 1.0, 100.0]])
    mask = torch.tensor([[True, True, False]])  # last is padding
    result = weighter(loss, padding_mask=mask)
    assert result.item() == pytest.approx(1.0)

def test_returns_scalar():
    weighter = TokenWeighter()
    loss = torch.rand(4, 16)
    result = weighter(loss)
    assert result.ndim == 0
