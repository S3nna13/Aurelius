"""PEFT configuration for parameter-efficient fine-tuning (Aurelius).

Adapted from Heavens_Gate TRAINING_PIPELINE/peft_trainer.py with pydantic
replaced by stdlib dataclasses so this module stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class PEFTMethod(Enum):
    """Supported parameter-efficient fine-tuning methods."""

    LORA = "lora"
    P_TUNING = "p_tuning"
    ADAPTER = "adapter"
    IA3 = "ia3"


class TrainingMode(Enum):
    """Training objective / alignment method.

    Modes
    -----
    SFT
        Supervised fine-tuning on labelled (prompt, completion) pairs.
    DPO
        Direct Preference Optimisation -- learns from (prompt, chosen, rejected)
        triples without a separate reward model (Rafailov et al., 2023).
    GRPO
        Group Relative Policy Optimisation -- optimises reasoning via per-group
        relative rewards; used in DeepSeek-R1 (Shao et al., 2024).
    SIMPO
        Simple Preference Optimisation -- reference-free variant of DPO using
        length-normalised rewards; 40% less VRAM than DPO (Meng et al., 2024).
    ORPO
        Odds Ratio Preference Optimisation -- single-pass SFT + preference
        alignment without a reference model (Hong et al., 2024).
    """

    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    SIMPO = "simpo"
    ORPO = "orpo"


@dataclass
class PEFTConfig:
    """Configuration for a PEFT training run."""

    method: PEFTMethod = PEFTMethod.LORA
    rank: int = 16
    alpha: float = 32.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    training_mode: TrainingMode = TrainingMode.SFT
    output_dir: str = "./outputs"


def validate_config(cfg: PEFTConfig) -> list[str]:
    """Validate a PEFTConfig; return list of error strings (empty = valid)."""
    errors: list[str] = []
    if cfg.rank <= 0:
        errors.append(f"rank must be > 0 (got {cfg.rank})")
    if cfg.alpha <= 0:
        errors.append(f"alpha must be > 0 (got {cfg.alpha})")
    if not (0.0 <= cfg.dropout < 1.0):
        errors.append(f"dropout must be in [0, 1) (got {cfg.dropout})")
    if cfg.learning_rate <= 0:
        errors.append(f"learning_rate must be > 0 (got {cfg.learning_rate})")
    if cfg.num_epochs <= 0:
        errors.append(f"num_epochs must be > 0 (got {cfg.num_epochs})")
    if cfg.batch_size <= 0:
        errors.append(f"batch_size must be > 0 (got {cfg.batch_size})")
    return errors


def config_summary(cfg: PEFTConfig) -> dict:
    """Return a flat dict summary of the config for logging."""
    d = asdict(cfg)
    # Enums -> their .value for log serialisation
    d["method"] = cfg.method.value
    d["training_mode"] = cfg.training_mode.value
    return d


PEFT_CONFIG_REGISTRY = {"default": PEFTConfig}
