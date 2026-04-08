import pytest
import torch
from src.training.ohem import OHEMConfig, OHEMMode, ohem_loss, ohem_mask

def test_ohem_token_selects_fraction():
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                            [0.5, 1.5, 2.5, 3.5]])  # (2, 4) = 8 tokens
    cfg = OHEMConfig(keep_fraction=0.5, mode=OHEMMode.TOKEN)
    mask = ohem_mask(losses, cfg)
    assert mask.shape == losses.shape
    assert mask.sum() == 4  # 50% of 8

def test_ohem_token_selects_hardest():
    losses = torch.tensor([[1.0, 2.0, 3.0, 10.0]])  # (1, 4)
    cfg = OHEMConfig(keep_fraction=0.5, mode=OHEMMode.TOKEN, min_keep=1)
    mask = ohem_mask(losses, cfg)
    # Top 50% = top 2 = [3.0, 10.0] at indices [2, 3]
    assert mask[0, 3]  # highest loss must be selected

def test_ohem_sequence_mode():
    # Sequence 0 has higher mean loss
    losses = torch.tensor([[5.0, 5.0], [1.0, 1.0]])  # (2, 2)
    cfg = OHEMConfig(keep_fraction=0.5, mode=OHEMMode.SEQUENCE)
    mask = ohem_mask(losses, cfg)
    # Should select sequence 0 (higher mean), all its tokens
    assert mask[0].all()
    assert not mask[1].any()

def test_ohem_loss_scalar():
    losses = torch.rand(3, 10)
    cfg = OHEMConfig(keep_fraction=0.7)
    loss = ohem_loss(losses, cfg)
    assert loss.ndim == 0
    assert loss.item() > 0

def test_ohem_padding_mask_respected():
    losses = torch.tensor([[1.0, 2.0, 100.0, 0.0]])
    # Mark last token as padding
    padding = torch.tensor([[True, True, True, False]])
    cfg = OHEMConfig(keep_fraction=0.5, mode=OHEMMode.TOKEN, min_keep=1)
    mask = ohem_mask(losses, cfg, padding_mask=padding)
    # Padded position should never be selected
    assert not mask[0, 3]

def test_ohem_min_keep_respected():
    losses = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    cfg = OHEMConfig(keep_fraction=0.01, min_keep=2, mode=OHEMMode.TOKEN)
    mask = ohem_mask(losses, cfg)
    assert mask.sum() >= 2

def test_ohem_full_fraction():
    losses = torch.rand(2, 8)
    cfg = OHEMConfig(keep_fraction=1.0, mode=OHEMMode.TOKEN)
    mask = ohem_mask(losses, cfg)
    assert mask.sum() == 16  # all selected
