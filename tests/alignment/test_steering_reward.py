import torch
import torch.nn as nn
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.steering_reward import SteeringRewardCorrespondence

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def test_src_returns_negative_scalar():
    cfg = PRAXISConfig(d_model=8, steer_layers=[0, 1], steer_alpha=0.1, lambda_src=0.1)
    model = TinyModel()
    src = SteeringRewardCorrespondence(model, cfg)
    x = torch.randn(2, 4, 8)
    reward = src.compute(x)
    assert reward.shape == (), f"expected scalar, got {reward.shape}"
    assert reward.item() <= 0.0, f"SRC reward should be non-positive: {reward.item()}"

def test_src_zero_alpha_gives_zero():
    cfg = PRAXISConfig(d_model=8, steer_layers=[0], steer_alpha=0.0, lambda_src=0.1)
    model = TinyModel()
    src = SteeringRewardCorrespondence(model, cfg)
    x = torch.randn(2, 4, 8)
    reward = src.compute(x)
    assert abs(reward.item()) < 1e-5, "zero alpha → no steering → zero distance"