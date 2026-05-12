from __future__ import annotations

import pytest

pytest.skip("module removed during reorganization", allow_module_level=True)


import torch
from aurelius_model_1b import AureliusModel1B
from aurelius_model_3b import AureliusModel3B
from aurelius_model_7b import AureliusModel7B
from aurelius_model_14b import AureliusModel14B
from aurelius_model_32b import AureliusModel32B

from aurelius import aurelius_model_1b as package_model_1b


def tiny_config(**overrides):
    config = {
        "vocab_size": 32,
        "d_model": 16,
        "n_heads": 4,
        "d_ff": 32,
        "d_mem": 8,
        "episodic_slots": 2,
        "lts_capacity": 4,
        "n_layers": 1,
        "max_seq_len": 8,
        "gradient_checkpointing": False,
        "skill_dim": 16,
        "n_known_tools": 4,
        "n_simulations": 1,
        "d_brain": 32,
        "n_actions": 4,
        "replay_capacity": 8,
        "agent_interval": 1,
    }
    config.update(overrides)
    return config


def test_aurelius_model_1b_package_alias_and_forward_with_states():
    assert package_model_1b.AureliusModel1B is AureliusModel1B

    torch.manual_seed(0)
    model = AureliusModel1B(tiny_config())
    input_ids = torch.randint(0, 32, (1, 4))

    out = model.forward_with_states(input_ids)

    assert out["logits"].shape == (1, 4, 32)
    assert out["hidden"].shape == (1, 4, 16)
    assert "layer_0" in out["mem_states"]


@pytest.mark.parametrize(
    "model_cls",
    [AureliusModel3B, AureliusModel7B, AureliusModel14B, AureliusModel32B],
)
def test_aurelius_model_tiers_forward_smoke(model_cls):
    torch.manual_seed(0)
    model = model_cls(tiny_config())
    input_ids = torch.randint(0, 32, (1, 4))

    out = model(input_ids)

    assert out["logits"].shape == (1, 4, 32)
    assert sum(p.numel() for p in model.parameters()) > 0
