import torch

from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.mtah import MultiTokenAlignmentHorizon


def test_mtah_extends_advantages_forward():
    cfg = PRAXISConfig(gamma_mtah=0.9, k_mtah=2)
    mtah = MultiTokenAlignmentHorizon(cfg)
    B, T = 1, 5
    adv = torch.zeros(B, T)
    adv[0, 3] = 1.0   # spike at position 3
    extended = mtah.extend(adv)
    # Positions 1 and 2 should receive discounted credit from position 3
    assert extended[0, 1].item() > 0.0, "position 1 gets credit from future spike"
    assert extended[0, 2].item() > extended[0, 1].item(), "closer = more credit"
    assert torch.isclose(extended[0, 3], adv[0, 3], atol=1e-6), "original spike preserved"

def test_mtah_shape_preserved():
    cfg = PRAXISConfig(gamma_mtah=0.95, k_mtah=3)
    mtah = MultiTokenAlignmentHorizon(cfg)
    adv = torch.randn(4, 16)
    extended = mtah.extend(adv)
    assert extended.shape == adv.shape

def test_mtah_k_zero_is_identity():
    cfg = PRAXISConfig(gamma_mtah=0.95, k_mtah=0)
    mtah = MultiTokenAlignmentHorizon(cfg)
    adv = torch.randn(2, 8)
    extended = mtah.extend(adv)
    assert torch.allclose(extended, adv)
