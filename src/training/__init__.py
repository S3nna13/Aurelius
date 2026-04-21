from src.training.fsdp_lite import FSDPLite, ShardSpec, gather_tensor, shard_tensor
from src.training.loss_variance_monitor import LossStats, LossVarianceMonitor
from src.training.lr_range_test import LRRangeTest, LRRangeTestResult
from src.training.token_dropout import TokenDropout

__all__ = [
    "FSDPLite",
    "ShardSpec",
    "shard_tensor",
    "gather_tensor",
    "TokenDropout",
    "LossStats",
    "LossVarianceMonitor",
    "LRRangeTest",
    "LRRangeTestResult",
]

from src.training.tool_call_supervision_loss import ToolCallSupervisionLoss  # noqa: E402

AUXILIARY_LOSS_REGISTRY: dict[str, type] = {}
AUXILIARY_LOSS_REGISTRY.setdefault("tool_call_supervision", ToolCallSupervisionLoss)

__all__ += ["ToolCallSupervisionLoss", "AUXILIARY_LOSS_REGISTRY"]

# ---------------------------------------------------------------------------
# TRAINING_REGISTRY — additive: maps string keys to training strategy classes.
# async_rl: GLM-5 §4.1 Multi-Task Rollout Orchestrator (arXiv:2602.15763)
# ---------------------------------------------------------------------------
from src.training.async_rl_infra import RolloutOrchestrator  # noqa: E402

TRAINING_REGISTRY: dict[str, type] = {}
TRAINING_REGISTRY["async_rl"] = RolloutOrchestrator

__all__ += ["TRAINING_REGISTRY", "RolloutOrchestrator"]

# TITO Gateway — GLM-5 §4.1 (arXiv:2602.15763)
from src.training.tito_gateway import TITOConfig, TITOGateway  # noqa: E402

TRAINING_REGISTRY["tito"] = TITOGateway

__all__ += ["TITOConfig", "TITOGateway"]

# Slime RL Framework — GLM-5 §4 (arXiv:2602.15763)
from src.training.slime_framework import SlimeTaskRouter  # noqa: E402

TRAINING_REGISTRY["slime"] = SlimeTaskRouter

__all__ += ["SlimeTaskRouter"]

# STILL-3 Trainer — min-std filtering + entropy bonus (arXiv:2501.12599)
from src.training.still3_trainer import STILL3Config, STILL3Trainer  # noqa: E402

TRAINING_REGISTRY["still3"] = STILL3Trainer

__all__ += ["STILL3Config", "STILL3Trainer"]

# Curriculum RL Sampler — difficulty-adaptive task sampling (Cycle 130-E)
from src.training.curriculum_rl import (  # noqa: E402
    CurriculumRLConfig,
    CurriculumRLSampler,
    TaskDifficulty,
)

TRAINING_REGISTRY["curriculum_rl"] = CurriculumRLSampler

__all__ += ["TaskDifficulty", "CurriculumRLConfig", "CurriculumRLSampler"]
