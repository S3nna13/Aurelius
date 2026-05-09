"""Smoke test: import PRAXIS from both public surfaces and verify registry."""

def test_import_from_alignment():
    from src.alignment import ALIGNMENT_REGISTRY
    assert "praxis" in ALIGNMENT_REGISTRY, \
        f"'praxis' not in ALIGNMENT_REGISTRY. Keys: {list(ALIGNMENT_REGISTRY.keys())}"

def test_import_praxis_trainer_directly():
    from src.alignment.praxis.trainer import PRAXISTrainer
    assert PRAXISTrainer is not None

def test_import_praxis_config():
    from src.alignment.praxis.config import PRAXISConfig
    cfg = PRAXISConfig()
    assert cfg.eps_low == 0.20
    assert cfg.eps_high == 0.28

def test_precision_fusion_import():
    from src.alignment.praxis.precision_fusion import PrecisionFusion
    assert PrecisionFusion is not None

def test_all_novel_components_import():
    from src.alignment.praxis.steering_reward import SteeringRewardCorrespondence
    from src.alignment.praxis.expert_safety_affinity import ExpertSafetyAffinity
    from src.alignment.praxis.mtah import MultiTokenAlignmentHorizon
    assert SteeringRewardCorrespondence is not None
    assert ExpertSafetyAffinity is not None
    assert MultiTokenAlignmentHorizon is not None