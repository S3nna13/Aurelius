"""Aurelius alignment and safety evaluation."""

from .preference_ranking_loss import (
    bradley_terry_loss,
    margin_ranking_loss,
    listnet_loss,
    ordinal_ranking_loss,
    dpo_pair_loss,
)
from .kto_v2 import KTOv2Loss, kto_v2_loss_functional
from .step_dpo import StepPreferenceExample, step_dpo_loss, StepDPOTrainer
from src.training.process_reward_model import ProcessRewardModel
from .adversarial_code_battle import (
    AdversarialCodeBattle,
    BattleTranscript,
    BluePatch,
    RedFinding,
    Round as AdversarialCodeBattleRound,
    heuristic_blue_fn,
    heuristic_red_fn,
)

# Alignment component registry: maps string keys -> class
ALIGNMENT_REGISTRY: dict = {}
ALIGNMENT_REGISTRY["prm"] = ProcessRewardModel
ALIGNMENT_REGISTRY["adversarial_code_battle"] = AdversarialCodeBattle

from .constitution_dimensions import (
    ConstitutionScorer,
    ConstitutionLevel,
    ConstitutionDimension,
    DimensionGrade,
    ConstitutionReport,
    CONSTITUTION_DIMENSIONS,
    DEFAULT_CONSTITUTION_TEXT,
    DEFAULT_GRADERS,
)

ALIGNMENT_REGISTRY["constitution_dimensions"] = ConstitutionScorer

from .parl import PARLReward, AnnealedLambda  # noqa: E402

ALIGNMENT_REGISTRY["parl"] = PARLReward

from .toggle import ToggleReward  # noqa: E402

ALIGNMENT_REGISTRY["toggle"] = ToggleReward

from .grm import GenerativeRewardModel, GRMConfig  # noqa: E402

ALIGNMENT_REGISTRY["grm"] = GenerativeRewardModel

from .cross_stage_distillation import CrossStageDistillation  # noqa: E402

ALIGNMENT_REGISTRY["cross_stage_distillation"] = CrossStageDistillation

from .zero_vision_sft import ZeroVisionSFTTrainer, ZeroVisionSFTConfig  # noqa: E402

ALIGNMENT_REGISTRY["zero_vision_sft"] = ZeroVisionSFTTrainer

