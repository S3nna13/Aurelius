"""Aurelius alignment and safety evaluation."""

from src.training.process_reward_model import ProcessRewardModel as ProcessRewardModel

from .adversarial_code_battle import (
    AdversarialCodeBattle,
)
from .adversarial_code_battle import (
    BattleTranscript as BattleTranscript,
)
from .adversarial_code_battle import (
    BluePatch as BluePatch,
)
from .adversarial_code_battle import (
    RedFinding as RedFinding,
)
from .adversarial_code_battle import (
    Round as Round,
)
from .adversarial_code_battle import (
    heuristic_blue_fn as heuristic_blue_fn,
)
from .adversarial_code_battle import (
    heuristic_red_fn as heuristic_red_fn,
)
from .kto_v2 import KTOv2Loss as KTOv2Loss
from .kto_v2 import kto_v2_loss_functional as kto_v2_loss_functional
from .preference_ranking_loss import (
    bradley_terry_loss as bradley_terry_loss,
)
from .preference_ranking_loss import (
    dpo_pair_loss as dpo_pair_loss,
)
from .preference_ranking_loss import (
    listnet_loss as listnet_loss,
)
from .preference_ranking_loss import (
    margin_ranking_loss as margin_ranking_loss,
)
from .preference_ranking_loss import (
    ordinal_ranking_loss as ordinal_ranking_loss,
)
from .step_dpo import StepDPOTrainer as StepDPOTrainer
from .step_dpo import StepPreferenceExample as StepPreferenceExample
from .step_dpo import step_dpo_loss as step_dpo_loss

# Alignment component registry: maps string keys -> class
ALIGNMENT_REGISTRY: dict = {}
ALIGNMENT_REGISTRY["prm"] = ProcessRewardModel
ALIGNMENT_REGISTRY["adversarial_code_battle"] = AdversarialCodeBattle

from .constitution_dimensions import (
    CONSTITUTION_DIMENSIONS as CONSTITUTION_DIMENSIONS,
)
from .constitution_dimensions import (
    DEFAULT_CONSTITUTION_TEXT as DEFAULT_CONSTITUTION_TEXT,
)
from .constitution_dimensions import (
    DEFAULT_GRADERS as DEFAULT_GRADERS,
)
from .constitution_dimensions import (
    ConstitutionDimension as ConstitutionDimension,
)
from .constitution_dimensions import (
    ConstitutionLevel as ConstitutionLevel,
)
from .constitution_dimensions import (
    ConstitutionReport as ConstitutionReport,
)
from .constitution_dimensions import (
    ConstitutionScorer,
)
from .constitution_dimensions import (
    DimensionGrade as DimensionGrade,
)

ALIGNMENT_REGISTRY["constitution_dimensions"] = ConstitutionScorer

from .parl import AnnealedLambda as AnnealedLambda  # noqa: E402
from .parl import PARLReward as PARLReward

ALIGNMENT_REGISTRY["parl"] = PARLReward

from .toggle import ToggleReward as ToggleReward  # noqa: E402

ALIGNMENT_REGISTRY["toggle"] = ToggleReward

from .grm import GenerativeRewardModel as GenerativeRewardModel  # noqa: E402
from .grm import GRMConfig as GRMConfig

ALIGNMENT_REGISTRY["grm"] = GenerativeRewardModel

from .cross_stage_distillation import CrossStageDistillation as CrossStageDistillation  # noqa: E402

ALIGNMENT_REGISTRY["cross_stage_distillation"] = CrossStageDistillation

from .zero_vision_sft import ZeroVisionSFTConfig as ZeroVisionSFTConfig  # noqa: E402
from .zero_vision_sft import ZeroVisionSFTTrainer as ZeroVisionSFTTrainer

ALIGNMENT_REGISTRY["zero_vision_sft"] = ZeroVisionSFTTrainer

from .absolute_zero import AbsoluteZeroConfig as AbsoluteZeroConfig  # noqa: E402
from .absolute_zero import AbsoluteZeroTrainer as AbsoluteZeroTrainer

ALIGNMENT_REGISTRY["absolute_zero"] = AbsoluteZeroTrainer

from .length_reward import LengthReward as LengthReward  # noqa: E402
from .length_reward import LengthRewardConfig as LengthRewardConfig

ALIGNMENT_REGISTRY["length_reward"] = LengthReward

from .online_rft import OnlineRFTConfig as OnlineRFTConfig  # noqa: E402
from .online_rft import OnlineRFTTrainer as OnlineRFTTrainer
from .online_rft import RFTSample as RFTSample

ALIGNMENT_REGISTRY["online_rft"] = OnlineRFTTrainer

from .orpo_trainer import ORPOBatch as ORPOBatch  # noqa: E402
from .orpo_trainer import ORPOConfig as ORPOConfig
from .orpo_trainer import ORPOTrainer as ORPOTrainer

ALIGNMENT_REGISTRY["orpo"] = ORPOTrainer

from .self_reward_trainer import (  # noqa: E402
    ScoredCandidate as ScoredCandidate,
)
from .self_reward_trainer import (
    SelfRewardBatch as SelfRewardBatch,
)
from .self_reward_trainer import (
    SelfRewardConfig as SelfRewardConfig,
)
from .self_reward_trainer import (
    SelfRewardTrainer,
)

ALIGNMENT_REGISTRY["self_reward"] = SelfRewardTrainer

from .cpo_trainer import CPOBatch as CPOBatch  # noqa: E402
from .cpo_trainer import CPOConfig as CPOConfig
from .cpo_trainer import CPOTrainer as CPOTrainer

ALIGNMENT_REGISTRY["cpo"] = CPOTrainer

from .spin_trainer import SPINBatch as SPINBatch  # noqa: E402
from .spin_trainer import SPINConfig as SPINConfig
from .spin_trainer import SPINLoss as SPINLoss
from .spin_trainer import SPINTrainer as SPINTrainer

ALIGNMENT_REGISTRY["spin"] = SPINTrainer

from .kto_trainer import KTOBatch as KTOBatch  # noqa: E402
from .kto_trainer import KTOConfig as KTOConfig
from .kto_trainer import KTOLoss as KTOLoss
from .kto_trainer import KTOTrainer as KTOTrainer

ALIGNMENT_REGISTRY["kto"] = KTOTrainer

from .dpo_trainer import DPOConfig as DPOConfig  # noqa: E402
from .dpo_trainer import DPOLoss as DPOLoss
from .dpo_trainer import DPOTrainer as DPOTrainer

ALIGNMENT_REGISTRY["dpo"] = DPOTrainer()

from .reward_calibration import (  # noqa: E402
    REWARD_CALIBRATOR_REGISTRY as REWARD_CALIBRATOR_REGISTRY,
)
from .reward_calibration import (
    CalibrationMethod as CalibrationMethod,
)
from .reward_calibration import (
    RewardCalibrator as RewardCalibrator,
)
from .rlhf_pipeline import (  # noqa: E402
    RLHF_PIPELINE as RLHF_PIPELINE,
)
from .rlhf_pipeline import (
    PhaseConfig as PhaseConfig,
)
from .rlhf_pipeline import (
    PhaseResult as PhaseResult,
)
from .rlhf_pipeline import (
    RLHFPhase as RLHFPhase,
)
from .rlhf_pipeline import (
    RLHFPipeline,
)

ALIGNMENT_REGISTRY["rlhf_pipeline"] = RLHFPipeline

from .preference_collector import (  # noqa: E402
    PREFERENCE_COLLECTOR as PREFERENCE_COLLECTOR,
)
from .preference_collector import (
    PreferenceCollector,
)
from .preference_collector import (
    PreferenceItem as PreferenceItem,
)

ALIGNMENT_REGISTRY["preference_collector"] = PreferenceCollector
