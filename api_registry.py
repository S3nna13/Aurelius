"""api_registry.py — contract surface for serving/training API endpoints.

Contract: Training scripts, config loaders, generation pipelines, and model
checkpoint surfaces. Each entry documents the import path, expected I/O shapes,
and config key invariants. Live path: train.py / agent_train.py / config yamls
-> torch tensor ops / reportlab PDF generation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import yaml
import logging
logger = logging.getLogger("api_registry")



@dataclass
class RegistryEntry:
    name: str
    version: str = "0.1.0"
    contract: str = ""
    live: bool = True
    path: str = ""
    test_command: str = ""


API_REGISTRY: dict[str, RegistryEntry] = {
    # ── Training entry points ──────────────────────────────────────────
    "train": RegistryEntry(
        name="TrainScript",
        contract="Main training loop: AureliusModel(150M) + MemoryAuxLoss + AdamW + cosine LR. "
        "Takes config.yaml, runs train_step(model, batch, optimizer, loss_fn). "
        "Output: logits (B,T,V) + mem_states dict per layer.",
        path="train",
        test_command="python3 -c 'from train import train_step, MemoryAuxLoss; print(\"ok\")'",
    ),
    "train_3b": RegistryEntry(
        name="Train3BScript",
        contract="3B-parameter training: AureliusModel3B + AgentTrainer + ImitationDataset. "
        "Loads config_3b.yaml, supports mixed-precision via MixedPrecisionTrainer.",
        path="train_3b",
        test_command="python3 -c 'from train_3b import count_parameters; print(\"ok\")'",
    ),
    "train_optimized": RegistryEntry(
        name="TrainOptimizedScript",
        contract="Optimized training: AureliusModel1B + MixedPrecisionTrainer + "
        "CpuOffloadManager + ActivationMemoryBudget + MemoryBudgetTracker. "
        "Memory-efficient variant with KV-cache quantization awareness.",
        path="train_optimized",
        test_command="python3 -c 'from train_optimized import MemoryAuxLoss; print(\"ok\")'",
    ),
    "agent_train": RegistryEntry(
        name="AgentTrainScript",
        contract="Agent-loop training: AgentAureliusModel (base + ToolFormerAdapter + "
        "PlanningModule + CriticHead + SkillLibrary + replay buffer). "
        "Provides AgentTrainer, ImitationDataset for supervised agent trajectory learning.",
        path="agent_train",
        test_command="python3 -c 'from agent_train import AgentAureliusModel, AgentTrainer; print(\"ok\")'",
    ),
    # ── Generation ────────────────────────────────────────────────────
    "generate_report": RegistryEntry(
        name="GenerateReport",
        contract="PDF report generation using reportlab. Produces AURELIUS_REPORT.pdf "
        "with project metrics, architecture diagrams, and training curves.",
        path="generate_report",
        test_command="python3 -c 'from generate_report import build_report; print(\"ok\")'",
    ),
    # ── Configuration files ───────────────────────────────────────────
    "config_150m": RegistryEntry(
        name="Config150M",
        contract="150M-parameter model config. Keys: d_model=768, n_heads=12, n_layers=12, "
        "d_ff=3072, max_seq_len=2048, d_mem=256, episodic_slots=512. "
        "Training section: batch_size=16, lr=3e-4, total_steps=100k.",
        path="config.yaml",
        test_command="python3 -c \"import yaml; c=yaml.safe_load(open('config.yaml')); "
        "assert c['aurelius_150m']['d_model']==768; print('ok')\"",
    ),
    "config_1b": RegistryEntry(
        name="Config1B",
        contract="1B-parameter model config. Keys: d_model=1536, n_heads=16, n_layers=24, "
        "d_ff=6144, max_seq_len=4096, d_mem=512, episodic_slots=1024. "
        "Training: batch_size=8, lr=2e-4, total_steps=200k.",
        path="config_1b.yaml",
        test_command="python3 -c \"import yaml; c=yaml.safe_load(open('config_1b.yaml')); "
        "assert c['aurelius_1b']['d_model']==1536; print('ok')\"",
    ),
    "config_3b": RegistryEntry(
        name="Config3B",
        contract="3B-parameter model config. Keys: d_model=2560, n_heads=32, n_layers=32, "
        "d_ff=10240, max_seq_len=8192, d_mem=768. Adds agent config: n_known_tools=128, "
        "n_simulations=16, skill_dim=256, max_skills=8192.",
        path="config_3b.yaml",
        test_command="python3 -c \"import yaml; c=yaml.safe_load(open('config_3b.yaml')); "
        "assert c['aurelius_3b']['d_model']==2560; print('ok')\"",
    ),
    "config_7b": RegistryEntry(
        name="Config7B",
        contract="7B-parameter model config. Keys: d_model=3584, n_heads=40, n_layers=40, "
        "d_ff=14336, max_seq_len=16384, d_mem=1024. Agent: n_known_tools=256, "
        "n_simulations=24, skill_dim=512, max_skills=16384, max_agent_context=32768.",
        path="config_7b.yaml",
        test_command="python3 -c \"import yaml; c=yaml.safe_load(open('config_7b.yaml')); "
        "assert c['aurelius_7b']['d_model']==3584; print('ok')\"",
    ),
    # ── Model checkpoint pipeline ─────────────────────────────────────
    "model_150m": RegistryEntry(
        name="AureliusModel",
        contract="150M decoder-only transformer with AurelianMemoryCore. "
        "Input: (B, T) token ids -> Output: (B, T, V) logits + mem_states. "
        "Components: RMSNorm, RotaryEmbedding, FlashAttention-like causal attn.",
        path="aurelius_model.AureliusModel",
        test_command="python3 -c \"from aurelius_model import AureliusModel; "
        "m=AureliusModel({'d_model':768,'n_heads':12,'d_ff':3072,'n_layers':12,"
        "'vocab_size':50257,'max_seq_len':2048,'d_mem':256,'episodic_slots':512,"
        "'lts_capacity':1024,'consolidation_freq':64,'graph_threshold':0.65,"
        "'dropout':0.0,'weight_init':'small_init'}); print('ok')\"",
    ),
    "model_1b": RegistryEntry(
        name="AureliusModel1B",
        contract="1B decoder-only transformer with AurelianMemoryCore. "
        "Input: (B, T) -> (B, T, V) logits + mem_states. Extended config for longer "
        "sequences (max_seq_len=4096), larger memory (d_mem=512, episodic_slots=1024).",
        path="aurelius_model_1b.AureliusModel1B",
        test_command="python3 -c \"from aurelius_model_1b import AureliusModel1B; "
        "m=AureliusModel1B({'d_model':1536,'n_heads':16,'d_ff':6144,'n_layers':24,"
        "'vocab_size':50257,'max_seq_len':4096,'d_mem':512,'episodic_slots':1024,"
        "'lts_capacity':2048,'consolidation_freq':128,'graph_threshold':0.65,"
        "'dropout':0.0,'weight_init':'small_init'}); print('ok')\"",
    ),
    "model_3b": RegistryEntry(
        name="AureliusModel3B",
        contract="3B decoder-only transformer. Input: (B, T) -> (B, T, V) logits + mem_states. "
        "Extended for agent capabilities: n_known_tools, n_simulations attached via agent_train.",
        path="aurelius_model_3b.AureliusModel3B",
        test_command="python3 -c \"from aurelius_model_3b import AureliusModel3B; print('ok')\"",
    ),
    "model_7b": RegistryEntry(
        name="AureliusModel7B",
        contract="7B decoder-only transformer. Largest variant with max_seq_len=16384, "
        "d_model=3584, n_layers=40, d_mem=1024. Full agent+skills pipeline support.",
        path="aurelius_model_7b.AureliusModel7B",
        test_command="python3 -c \"from aurelius_model_7b import AureliusModel7B; print('ok')\"",
    ),
    # ── Checkpoint helpers ───────────────────────────────────────────
    "recursive_mas_train": RegistryEntry(
        name="RecursiveMASTrainer",
        contract="Inner-outer loop training for recursive multi-agent systems. "
        "Inner loop: agent-local supervised/RL objectives. "
        "Outer loop: shared gradient-based credit assignment via RecursiveLink.",
        path="recursive_mas.InnerOuterOptimizer",
        test_command="python3 -m pytest test_recursive_mas.py::test_inner_outer_step_shapes -v",
    ),
    "recursive_mas_config": RegistryEntry(
        name="RecursiveMASConfig",
        contract="Configuration: d_model, d_latent, n_agents, n_rounds, collaboration_pattern, lrs",
        path="recursive_mas.RecursiveMASConfig",
        test_command="python3 -c \"from recursive_mas import RecursiveMASConfig; c=RecursiveMASConfig(); print('ok')\"",
    ),
    "checkpoint_pipeline": RegistryEntry(
        name="CheckpointPipeline",
        contract="Model checkpoint save/load pipeline. Train scripts save state_dict "
        "at save_interval. Load path: torch.load(ckpt.pt, map_location=device, weights_only=True). "
        "Never use weights_only=False unless full model deserialization is required. "
        "Config keys from training section control interval and micro_batch_size.",
        path="train.py::save_checkpoint / train_optimized.py::save_checkpoint",
        test_command="python3 -c \"import torch; print('ok')\"",
    ),
}


def get_registry() -> dict[str, RegistryEntry]:
    return API_REGISTRY


def lookup(name: str) -> RegistryEntry | None:
    return API_REGISTRY.get(name)


class ConfigContract:
    """Verify a config YAML loads and its model section has expected keys."""

    REQUIRED_MODEL_KEYS = {
        "d_model", "n_heads", "d_ff", "n_layers",
        "vocab_size", "max_seq_len", "d_mem",
        "episodic_slots", "lts_capacity",
        "consolidation_freq", "graph_threshold",
    }
    REQUIRED_TRAINING_KEYS = {
        "batch_size", "learning_rate", "weight_decay",
        "warmup_steps", "total_steps", "grad_clip",
    }

    def __init__(self, path: str, model_key: str):
        self.path = path
        self.model_key = model_key

    def verify_contract(self) -> None:
        with open(self.path) as f:
            cfg = yaml.safe_load(f)
        assert self.model_key in cfg, (
            f"Missing model key '{self.model_key}' in {self.path}"
        )
        model_cfg = cfg[self.model_key]
        for k in self.REQUIRED_MODEL_KEYS:
            assert k in model_cfg, (
                f"Missing required model key '{k}' in {self.path}[{self.model_key}]"
            )
        if "training" in cfg:
            for k in self.REQUIRED_TRAINING_KEYS:
                assert k in cfg["training"], (
                    f"Missing required training key '{k}' in {self.path}"
                )


class TrainContract:
    """Verify train.py imports resolve and core signatures hold."""

    def __init__(self):
        from train import MemoryAuxLoss, train_step
        self._loss_fn = MemoryAuxLoss
        self._train_step = train_step

    def verify_contract(self, d_model: int = 768, vocab_size: int = 50257) -> None:
        loss = self._loss_fn()
        logits = torch.randn(2, 8, vocab_size)
        labels = torch.randint(0, vocab_size, (2, 8))
        mem_states = {f"layer_{i}": {"surprise": torch.randn(2, 8, d_model)}
                       for i in range(4)}
        total, metrics = loss(logits, labels, mem_states)
        assert total.ndim == 0, f"loss should be scalar, got shape {total.shape}"
        for k in ("ce", "surprise", "total"):
            assert k in metrics, f"missing metric '{k}'"


class AgentTrainContract:
    """Verify agent_train.py imports resolve and AgentAureliusModel instantiates."""

    def __init__(self):
        from agent_train import AgentAureliusModel
        self._cls = AgentAureliusModel

    def verify_contract(self) -> None:
        with open("config_1b.yaml") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg["aurelius_1b"]
        model = self._cls(model_cfg)
        assert isinstance(model, torch.nn.Module)
        x = torch.randint(0, model_cfg["vocab_size"], (1, 32))
        out = model(x)
        assert isinstance(out, dict)


class GenerateReportContract:
    """Verify generate_report.py imports resolve."""

    def __init__(self):
        from generate_report import build_report
        self._build_report = build_report

    def verify_contract(self) -> None:
        assert callable(self._build_report)


class AureliusModelContract:
    """Verify a model class instantiates and forward pass has correct shape."""

    def __init__(self, model_cls: type, config: dict):
        self._impl = model_cls(config)
        self._config = config

    def verify_contract(self, batch: int = 2, time: int = 16) -> None:
        vocab_size = self._config["vocab_size"]
        x = torch.randint(0, vocab_size, (batch, time))
        logits, mem_states = self._impl(x, return_mem_state=True)
        expected_shape = (batch, time, vocab_size)
        assert logits.shape == expected_shape, (
            f"Expected {expected_shape}, got {logits.shape}"
        )
        assert isinstance(mem_states, dict), (
            f"mem_states should be dict, got {type(mem_states)}"
        )
