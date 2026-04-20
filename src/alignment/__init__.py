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

# Alignment component registry: maps string keys -> class
ALIGNMENT_REGISTRY: dict = {}
ALIGNMENT_REGISTRY["prm"] = ProcessRewardModel

