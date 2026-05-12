import torch

from src.alignment.praxis.precision_fusion import PrecisionFusion


def test_equal_uncertainty_gives_uniform_weights():
    fuser = PrecisionFusion(n_signals=3)
    B = 4
    means = [torch.zeros(B) + float(i) for i in range(3)]   # 0, 1, 2
    stds  = [torch.ones(B) for _ in range(3)]                # equal std=1
    result = fuser.fuse(means, stds)
    expected = torch.ones(B) * 1.0   # (0+1+2)/3
    assert torch.allclose(result, expected, atol=1e-5), f"got {result}"

def test_low_uncertainty_dominates():
    fuser = PrecisionFusion(n_signals=2)
    B = 2
    means = [torch.ones(B) * 10.0, torch.ones(B) * 0.0]
    stds  = [torch.ones(B) * 0.01, torch.ones(B) * 100.0]   # first is precise
    result = fuser.fuse(means, stds)
    assert (result > 9.0).all(), f"precise signal should dominate: {result}"