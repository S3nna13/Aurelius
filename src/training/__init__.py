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

# SWE-RL Trainer — software engineering RL with test-verifier rewards (Cycle 131-B)
from src.training.swe_rl import (  # noqa: E402
    SWETask,
    SWEPatch,
    SWEResult,
    SWERLConfig,
    SWERLTrainer,
)

TRAINING_REGISTRY["swe_rl"] = SWERLTrainer

__all__ += ["SWETask", "SWEPatch", "SWEResult", "SWERLConfig", "SWERLTrainer"]

# PCGrad v2 — cosine-adaptive conflicting gradient projection (Cycle 132-E)
from src.training.pcgrad_v2 import (  # noqa: E402
    GradientBank,
    PCGradV2,
    PCGradV2Config,
)

TRAINING_REGISTRY["pcgrad_v2"] = PCGradV2

__all__ += ["PCGradV2Config", "GradientBank", "PCGradV2"]

# IPO Trainer — squared-loss regularised preference optimisation (Azar et al. 2024, Cycle 134-F)
from src.training.ipo_trainer import IPOBatch, IPOConfig, IPOTrainer  # noqa: E402

# Registry entry is set inside ipo_trainer.py after import.

__all__ += ["IPOConfig", "IPOBatch", "IPOTrainer"]

# Token Credit Assignment — uniform/discounted/GAE/end-decay (Cycle 135-E)
from src.training.token_credit_assignment import (  # noqa: E402
    TokenCreditConfig,
    TokenCreditAssigner,
)

# Registry entry is set inside token_credit_assignment.py after import.

__all__ += ["TokenCreditConfig", "TokenCreditAssigner"]

# Offline Reward Modeling — Bradley-Terry pairwise preference model (Cycle 136-D)
from src.training.offline_reward_modeling import (  # noqa: E402
    RewardModelConfig,
    RewardBatch,
    RewardHead,
    RewardModelTrainer,
)

# Registry entry is set inside offline_reward_modeling.py after import.

__all__ += ["RewardModelConfig", "RewardBatch", "RewardHead", "RewardModelTrainer"]

# MCTS RL Trainer — AlphaZero-style MCTS policy/value training (Cycle 137-C)
from src.training.mcts_rl_trainer import (  # noqa: E402
    MCTSRLConfig,
    MCTSNode,
    MCTSStats,
    MCTSRLTrainer,
)

# Registry entry is set inside mcts_rl_trainer.py after import.

__all__ += ["MCTSRLConfig", "MCTSNode", "MCTSStats", "MCTSRLTrainer"]
