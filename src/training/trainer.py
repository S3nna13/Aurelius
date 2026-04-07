"""Aurelius 1.3B training loop.

Uses HuggingFace Accelerate as a lighter alternative to Megatron-LM for
smaller-scale runs, with full support for DeepSpeed ZeRO on H100 clusters
and MLX-based local training on Apple Silicon.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.training.muon import Muon
from src.training.zclip import ZClip

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Parsed training configuration from YAML."""

    # Model
    model_name: str = "aurelius-1.3b"

    # Optimizer
    lr: float = 3e-4
    min_lr: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1

    # Schedule
    warmup_steps: int = 2000
    total_steps: int = 143_000

    # Batch
    global_batch_tokens: int = 2_097_152
    micro_batch_size: int = 4
    seq_len: int = 8192

    # Gradient
    max_grad_norm: float = 1.0

    # Precision
    dtype: str = "bf16"
    flash_attention: bool = True
    gradient_checkpointing: bool = True

    # Checkpoint
    save_interval_steps: int = 4800
    save_dir: str = "checkpoints/aurelius-1.3b"
    max_checkpoints: int = 5
    checkpoint_format: str = "safetensors"

    # Logging
    log_interval_steps: int = 10
    wandb_enabled: bool = True
    wandb_project: str = "aurelius-1.3b"

    # Data
    train_data_dir: str = "data/pretrain/train"
    val_data_dir: str = "data/pretrain/val"
    num_workers: int = 8
    pin_memory: bool = True

    # Optimizer variant
    use_muon: bool = True   # Use Muon for matrix params + AdamW for embeddings
    muon_lr: float = 0.02   # Muon learning rate (much higher than AdamW's 3e-4)
    muon_momentum: float = 0.95

    # ZClip (adaptive gradient clipping)
    use_zclip: bool = False
    zclip_z_threshold: float = 2.5
    zclip_ema_alpha: float = 0.01

    # DeepSpeed
    deepspeed_config: str | None = None

    # Training
    seed: int = 42
    total_tokens: int = 300_000_000_000

    # Resume
    resume_from: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            model_name=raw.get("model", {}).get("name", cls.model_name),
            lr=raw.get("optimizer", {}).get("lr", cls.lr),
            min_lr=raw.get("scheduler", {}).get("min_lr", cls.min_lr),
            beta1=raw.get("optimizer", {}).get("beta1", cls.beta1),
            beta2=raw.get("optimizer", {}).get("beta2", cls.beta2),
            eps=raw.get("optimizer", {}).get("eps", cls.eps),
            weight_decay=raw.get("optimizer", {}).get("weight_decay", cls.weight_decay),
            warmup_steps=raw.get("scheduler", {}).get("warmup_steps", cls.warmup_steps),
            total_steps=raw.get("scheduler", {}).get("total_steps", cls.total_steps),
            global_batch_tokens=raw.get("batch", {}).get("global_batch_tokens", cls.global_batch_tokens),
            micro_batch_size=raw.get("batch", {}).get("micro_batch_size", cls.micro_batch_size),
            seq_len=raw.get("batch", {}).get("seq_len", cls.seq_len),
            max_grad_norm=raw.get("gradient", {}).get("max_norm", cls.max_grad_norm),
            dtype=raw.get("precision", {}).get("dtype", cls.dtype),
            flash_attention=raw.get("precision", {}).get("flash_attention", cls.flash_attention),
            gradient_checkpointing=raw.get("precision", {}).get("gradient_checkpointing", cls.gradient_checkpointing),
            save_interval_steps=raw.get("checkpoint", {}).get("save_interval_steps", cls.save_interval_steps),
            save_dir=raw.get("checkpoint", {}).get("save_dir", cls.save_dir),
            max_checkpoints=raw.get("checkpoint", {}).get("max_checkpoints", cls.max_checkpoints),
            checkpoint_format=raw.get("checkpoint", {}).get("format", cls.checkpoint_format),
            log_interval_steps=raw.get("logging", {}).get("log_interval_steps", cls.log_interval_steps),
            wandb_enabled=raw.get("logging", {}).get("wandb", {}).get("enabled", cls.wandb_enabled),
            wandb_project=raw.get("logging", {}).get("wandb", {}).get("project", cls.wandb_project),
            train_data_dir=raw.get("data", {}).get("train_data_dir", cls.train_data_dir),
            val_data_dir=raw.get("data", {}).get("val_data_dir", cls.val_data_dir),
            num_workers=raw.get("data", {}).get("num_workers", cls.num_workers),
            pin_memory=raw.get("data", {}).get("pin_memory", cls.pin_memory),
            deepspeed_config=raw.get("deepspeed", {}).get("config_file"),
            seed=raw.get("training", {}).get("seed", cls.seed),
            total_tokens=raw.get("training", {}).get("total_tokens", cls.total_tokens),
            use_muon=raw.get("optimizer", {}).get("use_muon", cls.use_muon),
            muon_lr=raw.get("optimizer", {}).get("muon_lr", cls.muon_lr),
            muon_momentum=raw.get("optimizer", {}).get("muon_momentum", cls.muon_momentum),
        )


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _cosine_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Return LR multiplier for a given step."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def build_scheduler(optimizer: AdamW, cfg: TrainConfig) -> LambdaLR:
    """Build cosine-with-warmup LR scheduler."""
    min_lr_ratio = cfg.min_lr / cfg.lr

    def lr_lambda(step: int) -> float:
        return _cosine_with_warmup(step, cfg.warmup_steps, cfg.total_steps, min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# MFU (Model FLOPs Utilization) estimator
# ---------------------------------------------------------------------------

def estimate_mfu(
    model_params: int,
    tokens_per_sec: float,
    n_gpus: int = 1,
    gpu_flops_bf16: float = 989e12,  # H100 SXM peak BF16 TFLOPS
) -> float:
    """Estimate model FLOPs utilization.

    Uses the 6*N*D approximation for transformer forward+backward FLOPs,
    where N = model parameters, D = tokens processed.
    """
    model_flops_per_token = 6 * model_params
    achieved_flops = model_flops_per_token * tokens_per_sec
    peak_flops = gpu_flops_bf16 * n_gpus
    return achieved_flops / peak_flops if peak_flops > 0 else 0.0


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

@dataclass
class CheckpointManager:
    """Manages checkpoint saving with rolling window and safetensors format."""

    save_dir: Path
    max_checkpoints: int = 5
    _saved: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: AdamW,
        scheduler: LambdaLR,
        step: int,
        tokens_seen: int,
        loss: float,
        accelerator: Accelerator,
    ) -> Path:
        """Save checkpoint as safetensors + metadata JSON."""
        ckpt_dir = self.save_dir / f"step-{step:07d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap model from Accelerate / DeepSpeed wrappers
        unwrapped = accelerator.unwrap_model(model)

        # Save model weights
        state_dict = {k: v.contiguous().cpu() for k, v in unwrapped.state_dict().items()}
        save_file(state_dict, ckpt_dir / "model.safetensors")

        # Save optimizer + scheduler state (torch format for complex state)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt_dir / "optim_state.pt",
        )

        # Save metadata
        metadata = {
            "step": step,
            "tokens_seen": tokens_seen,
            "loss": loss,
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self._saved.append(ckpt_dir)

        # Rolling window: remove oldest checkpoints
        while len(self._saved) > self.max_checkpoints:
            oldest = self._saved.pop(0)
            if oldest.exists():
                import shutil
                shutil.rmtree(oldest)
                logger.info("Removed old checkpoint: %s", oldest)

        logger.info("Saved checkpoint at step %d to %s", step, ckpt_dir)
        return ckpt_dir

    @staticmethod
    def load(
        ckpt_dir: str | Path,
        model: torch.nn.Module,
        optimizer: AdamW | None = None,
        scheduler: LambdaLR | None = None,
    ) -> dict[str, Any]:
        """Load checkpoint from safetensors + metadata."""
        ckpt_dir = Path(ckpt_dir)

        # Load model weights
        state_dict = load_file(ckpt_dir / "model.safetensors")
        model.load_state_dict(state_dict, strict=True)

        # Load optimizer + scheduler
        if optimizer is not None or scheduler is not None:
            optim_state = torch.load(
                ckpt_dir / "optim_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            if optimizer is not None:
                optimizer.load_state_dict(optim_state["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(optim_state["scheduler"])

        # Load metadata
        with open(ckpt_dir / "metadata.json") as f:
            metadata: dict[str, Any] = json.load(f)

        logger.info(
            "Resumed from %s (step=%d, tokens=%d)",
            ckpt_dir, metadata["step"], metadata["tokens_seen"],
        )
        return metadata


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

class AureliusTrainer:
    """Pre-training loop for Aurelius 1.3B using HuggingFace Accelerate."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        cfg: TrainConfig,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Count parameters
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Trainable parameters: %s (%.2fB)",
            f"{self.n_params:,}", self.n_params / 1e9,
        )

        # Optimizer
        self.optimizer = self._build_optimizer()
        # For Muon+AdamW pair, schedule only the AdamW optimizer
        if isinstance(self.optimizer, list):
            self.scheduler = build_scheduler(self.optimizer[1], cfg)  # AdamW
            self.muon_optimizer = self.optimizer[0]
            self.adamw_optimizer = self.optimizer[1]
        else:
            self.scheduler = build_scheduler(self.optimizer, cfg)
            self.muon_optimizer = None
            self.adamw_optimizer = self.optimizer

        # Accelerator (handles device placement, mixed precision, DeepSpeed)
        ds_plugin = None
        if cfg.deepspeed_config and os.path.exists(cfg.deepspeed_config):
            from accelerate import DeepSpeedPlugin
            ds_plugin = DeepSpeedPlugin(hf_ds_config=cfg.deepspeed_config)

        mixed_precision = "bf16" if cfg.dtype == "bf16" else "fp16" if cfg.dtype == "fp16" else "no"
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self._compute_grad_accum_steps(),
            deepspeed_plugin=ds_plugin,
            log_with="wandb" if cfg.wandb_enabled else None,
            project_dir=cfg.save_dir,
        )

        # Prepare with Accelerate
        if isinstance(self.optimizer, list):
            (
                self.model,
                self.muon_optimizer,
                self.adamw_optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, self.muon_optimizer, self.adamw_optimizer,
                self.train_dataloader, self.scheduler,
            )
            self.optimizer = [self.muon_optimizer, self.adamw_optimizer]
        else:
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.scheduler,
            )
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # Checkpoint manager
        self.ckpt_mgr = CheckpointManager(
            save_dir=cfg.save_dir,
            max_checkpoints=cfg.max_checkpoints,
        )

        # ZClip (adaptive gradient clipping, opt-in)
        self.zclip: ZClip | None = None
        if cfg.use_zclip:
            self.zclip = ZClip(
                params=list(self.model.parameters()),
                z_threshold=cfg.zclip_z_threshold,
                ema_alpha=cfg.zclip_ema_alpha,
            )
            logger.info(
                "ZClip enabled: z_threshold=%.1f, ema_alpha=%.4f",
                cfg.zclip_z_threshold, cfg.zclip_ema_alpha,
            )

        # Training state
        self.global_step: int = 0
        self.tokens_seen: int = 0
        self.best_val_loss: float = float("inf")

    def _build_optimizer(self) -> AdamW | list:
        """Build optimizer(s).

        When use_muon=True: returns a list [Muon, AdamW] — Muon for 2D weight
        matrices in transformer layers, AdamW for embeddings and 1D params.
        When use_muon=False: returns a single AdamW.
        """
        if self.cfg.use_muon:
            return self._build_muon_adamw()
        return self._build_adamw()

    def _build_adamw(self) -> AdamW:
        """Build standard AdamW with weight-decay exclusion for bias/norm params."""
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return AdamW(
            [
                {"params": decay_params, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            eps=self.cfg.eps,
        )

    def _build_muon_adamw(self) -> list:
        """Build [Muon, AdamW] optimizer pair.

        Muon: all 2D weight matrices in transformer projection layers.
        AdamW: embeddings, normalization weights, lm_head, 1D params.
        """
        muon_params: list[torch.nn.Parameter] = []
        adamw_decay: list[torch.nn.Parameter] = []
        adamw_no_decay: list[torch.nn.Parameter] = []

        muon_target_suffixes = (
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "gate_proj.weight", "up_proj.weight", "down_proj.weight",
        )

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and any(name.endswith(s) for s in muon_target_suffixes):
                muon_params.append(param)
            elif "bias" in name or "norm" in name or "ln" in name or param.ndim < 2:
                adamw_no_decay.append(param)
            else:
                adamw_decay.append(param)

        muon_opt = Muon(
            muon_params,
            lr=self.cfg.muon_lr,
            momentum=self.cfg.muon_momentum,
            weight_decay=self.cfg.weight_decay,
        )
        adamw_opt = AdamW(
            [
                {"params": adamw_decay, "weight_decay": self.cfg.weight_decay},
                {"params": adamw_no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            eps=self.cfg.eps,
        )
        logger.info(
            "Muon: %d params | AdamW: %d params",
            sum(p.numel() for p in muon_params),
            sum(p.numel() for p in adamw_decay) + sum(p.numel() for p in adamw_no_decay),
        )
        return [muon_opt, adamw_opt]

    def _compute_grad_accum_steps(self) -> int:
        """Derive gradient accumulation steps from global batch config."""
        tokens_per_micro = self.cfg.micro_batch_size * self.cfg.seq_len
        n_gpus = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        total_per_step = tokens_per_micro * n_gpus
        accum = max(1, self.cfg.global_batch_tokens // total_per_step)
        logger.info(
            "Gradient accumulation: %d (global_batch=%d tokens, per_micro=%d, gpus=%d)",
            accum, self.cfg.global_batch_tokens, tokens_per_micro, n_gpus,
        )
        return accum

    def train(self) -> None:
        """Execute the full training loop."""
        set_seed(self.cfg.seed)

        # Initialize W&B tracking
        if self.cfg.wandb_enabled and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.cfg.wandb_project,
                config={
                    "model_params": self.n_params,
                    "total_tokens": self.cfg.total_tokens,
                    "total_steps": self.cfg.total_steps,
                    "lr": self.cfg.lr,
                    "batch_tokens": self.cfg.global_batch_tokens,
                },
            )

        # Resume from checkpoint if specified
        if self.cfg.resume_from is not None:
            metadata = CheckpointManager.load(
                self.cfg.resume_from,
                self.accelerator.unwrap_model(self.model),
                self.optimizer,
                self.scheduler,
            )
            self.global_step = metadata["step"]
            self.tokens_seen = metadata["tokens_seen"]
            logger.info("Resumed training from step %d", self.global_step)

        logger.info(
            "Starting training for %d steps (%d tokens)",
            self.cfg.total_steps, self.cfg.total_tokens,
        )

        self.model.train()
        step_loss_accum = 0.0
        micro_steps_in_accum = 0
        step_start_time = time.perf_counter()

        data_iter = iter(self.train_dataloader)

        while self.global_step < self.cfg.total_steps:
            # Fetch next batch, cycling the dataloader if exhausted
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)

            with self.accelerator.accumulate(self.model):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

                self.accelerator.backward(loss)

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    if self.zclip is not None:
                        self.zclip.clip_grad_norm_(self.model.parameters())
                    else:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.cfg.max_grad_norm,
                        )

                if self.muon_optimizer is not None:
                    self.muon_optimizer.step()
                    self.muon_optimizer.zero_grad()
                self.adamw_optimizer.step()
                self.scheduler.step()
                self.adamw_optimizer.zero_grad()

            step_loss_accum += loss.detach().float().item()
            micro_steps_in_accum += 1

            # Check if a full global step has completed
            if self.accelerator.sync_gradients:
                self.global_step += 1
                batch_tokens = self.cfg.global_batch_tokens
                self.tokens_seen += batch_tokens

                # Logging
                if self.global_step % self.cfg.log_interval_steps == 0:
                    elapsed = time.perf_counter() - step_start_time
                    avg_loss = step_loss_accum / max(1, micro_steps_in_accum)
                    tokens_per_sec = (
                        batch_tokens * self.cfg.log_interval_steps
                    ) / max(elapsed, 1e-6)
                    current_lr = self.scheduler.get_last_lr()[0]
                    n_gpus = max(1, int(os.environ.get("WORLD_SIZE", "1")))
                    mfu = estimate_mfu(
                        self.n_params, tokens_per_sec, n_gpus=n_gpus,
                    )

                    metrics = {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/mfu": mfu,
                        "train/tokens_seen": self.tokens_seen,
                        "train/step": self.global_step,
                    }

                    if self.accelerator.is_main_process:
                        logger.info(
                            "step=%d  loss=%.4f  lr=%.2e  tok/s=%.0f"
                            "  mfu=%.2f%%  tokens=%s",
                            self.global_step,
                            avg_loss,
                            current_lr,
                            tokens_per_sec,
                            mfu * 100,
                            f"{self.tokens_seen:,}",
                        )
                        if self.cfg.wandb_enabled:
                            self.accelerator.log(metrics, step=self.global_step)

                    step_loss_accum = 0.0
                    micro_steps_in_accum = 0
                    step_start_time = time.perf_counter()

                # Checkpointing
                if self.global_step % self.cfg.save_interval_steps == 0:
                    self._save_checkpoint(
                        step_loss_accum / max(1, micro_steps_in_accum),
                    )

        # Final checkpoint
        self._save_checkpoint(step_loss_accum / max(1, micro_steps_in_accum))

        if self.cfg.wandb_enabled:
            self.accelerator.end_training()

        logger.info("Training complete. Total tokens: %s", f"{self.tokens_seen:,}")

    def _save_checkpoint(self, current_loss: float) -> None:
        """Save checkpoint on the main process only."""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.ckpt_mgr.save(
                model=self.model,
                optimizer=self.adamw_optimizer,  # save AdamW state (Muon state is momentum buffers only)
                scheduler=self.scheduler,
                step=self.global_step,
                tokens_seen=self.tokens_seen,
                loss=current_loss,
                accelerator=self.accelerator,
            )

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        if self.val_dataloader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_dataloader:
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            total_loss += loss.float().item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        self.model.train()

        if self.accelerator.is_main_process:
            logger.info(
                "Validation loss: %.4f (perplexity: %.2f)",
                avg_loss, math.exp(min(avg_loss, 20)),
            )

        return avg_loss


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Aurelius 1.3B Pre-training")
    parser.add_argument(
        "--config", type=str, default="configs/train_1b.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = TrainConfig.from_yaml(args.config)
    if args.resume:
        cfg.resume_from = args.resume

    logger.info("Loading config from %s", args.config)
    logger.info(
        "Config: total_steps=%d, lr=%.2e, batch_tokens=%d",
        cfg.total_steps, cfg.lr, cfg.global_batch_tokens,
    )

    # Placeholder: real usage would build the model from AureliusConfig
    # and construct dataloaders from the tokenized dataset.
    logger.info(
        "To launch training, instantiate AureliusTrainer with your model and "
        "dataloaders, then call trainer.train(). See scripts/run_training.sh."
    )


if __name__ == "__main__":
    main()
