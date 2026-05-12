from src.alignment.praxis.config import PRAXISConfig


def test_praxis_config_defaults():
    cfg = PRAXISConfig()
    assert cfg.n_group == 8
    assert cfg.eps_low == 0.20
    assert cfg.eps_high == 0.28
    assert cfg.tau_gate == 0.4
    assert cfg.steer_layers == [12, 16, 20]
    assert cfg.safety_experts == [0, 1]
    assert cfg.k_mtah == 2
    assert cfg.warp_interval == 50


def test_praxis_config_custom():
    cfg = PRAXISConfig(n_group=4, eps_low=0.1, steer_layers=[8, 12])
    assert cfg.n_group == 4
    assert cfg.eps_low == 0.1
    assert cfg.steer_layers == [8, 12]
