import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml
import time

from aurelius_model_1b import AureliusModel1B
from memory_optimizer import (
    MixedPrecisionTrainer,
    CpuOffloadManager,
    ActivationMemoryBudget,
)
from kv_cache_quant import MemoryBudgetTracker
import logging
logger = logging.getLogger("train_optimized")



class MemoryAuxLoss(nn.Module):
    def __init__(self, memory_weight: float = 0.1, consolidation_weight: float = 0.05):
        super().__init__()
        self.memory_weight = memory_weight
        self.consolidation_weight = consolidation_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                mem_states: dict) -> tuple[torch.Tensor, dict]:
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        surprise_penalty = 0.0
        for ms in mem_states.values():
            s = ms['surprise']
            surprise_penalty = surprise_penalty + s.pow(2).mean()
        surprise_penalty = surprise_penalty / max(len(mem_states), 1)
        total = ce + self.memory_weight * surprise_penalty
        return total, {
            'ce': ce.item(),
            'surprise': surprise_penalty.item(),
            'total': total.item(),
        }


class CheckpointSaver:
    def __init__(self, save_dir: str = './checkpoints'):
        self.save_dir = save_dir

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: object, step: int, metrics: dict):
        path = f"{self.save_dir}/aurelius_1b_step_{step}.pt"
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': step,
            'metrics': metrics,
        }
        torch.save(state, path)
        return path


def build_optimizer(model: nn.Module, cfg: dict) -> tuple[AdamW, object]:
    opt = AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        betas=(cfg['beta1'], cfg['beta2']),
        weight_decay=cfg['weight_decay'],
    )
    warmup = LinearLR(opt, start_factor=0.01, total_iters=cfg['warmup_steps'])
    cosine = CosineAnnealingLR(opt, T_max=max(1, cfg['total_steps'] - cfg['warmup_steps']))
    scheduler = SequentialLR(
        opt, schedulers=[warmup, cosine],
        milestones=[cfg['warmup_steps']]
    )
    return opt, scheduler


class MemoryEfficientTrainer:
    def __init__(self, model: AureliusModel1B, config: dict):
        self.model = model
        self.cfg = config['training']
        self.infra = config.get('infrastructure', {})

        self.model.gradient_checkpointing = self.cfg.get('gradient_checkpointing', False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.optimizer, self.scheduler = build_optimizer(self.model, self.cfg)
        self.loss_fn = MemoryAuxLoss(
            memory_weight=self.cfg['memory_loss_weight'],
            consolidation_weight=self.cfg['consolidation_loss_weight'],
        )

        amp_enabled = self.cfg.get('mixed_precision', 'bf16') in ('bf16', 'fp16')
        self.mp_trainer = MixedPrecisionTrainer(self.model, enabled=amp_enabled)

        self.cpu_offload = None
        if self.cfg.get('cpu_offload', False):
            self.cpu_offload = CpuOffloadManager(self.model)

        self.mem_budget = ActivationMemoryBudget(
            max_act_mb=self.infra.get('activation_memory_budget_mb', 65536)
        )

        self.budget_tracker = MemoryBudgetTracker(
            target_mb=self.infra.get('activation_memory_budget_mb', 65536)
        )

        self.saver = CheckpointSaver()

    def train_epoch(self, dataloader: object, step: int = 0) -> dict:
        self.model.train()
        total_metrics = {'ce': 0.0, 'surprise': 0.0, 'total': 0.0}
        n_batches = 0
        tokens_processed = 0
        start_time = time.time()

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            metrics = self.mp_trainer.train_step(
                batch, self.loss_fn, self.optimizer,
                grad_clip=self.cfg['grad_clip'],
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if self.cpu_offload is not None:
                self.cpu_offload.step_callback()

            for k in total_metrics:
                if k in metrics:
                    total_metrics[k] += metrics[k]

            tokens_processed += batch['input_ids'].numel()
            n_batches += 1
            step += 1

            if step % self.cfg['eval_interval'] == 0:
                mem = self.budget_tracker.snapshot(f"step_{step}")
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"step {step} | loss {metrics['total']:.4f} | "
                      f"ce {metrics['ce']:.4f} | surprise {metrics['surprise']:.4f} | "
                      f"lr {lr:.2e} | tokens/s {tokens_processed / (time.time() - start_time):.0f}")

            if step % self.cfg['save_interval'] == 0:
                path = self.saver.save(
                    self.model, self.optimizer, self.scheduler,
                    step, metrics
                )
                logger.info(f"checkpoint saved: {path}")

        elapsed = time.time() - start_time
        for k in total_metrics:
            total_metrics[k] /= max(n_batches, 1)
        total_metrics['tokens_per_sec'] = tokens_processed / max(elapsed, 0.001)
        total_metrics['steps'] = step

        return total_metrics

    def print_memory_report(self):
        report = self.budget_tracker.report()
        logger.info(f"Memory Budget Report:\n{report}")

        d = self.model.config['d_model']
        n = self.model.config['n_layers']
        recommended = self.mem_budget.recommend_batch_size(
            seq_len=self.model.config['max_seq_len'],
            d_model=d,
            n_layers=n,
        )
        est = self.mem_budget.estimate_activation_size(
            batch_size=self.cfg['batch_size'],
            seq_len=self.model.config['max_seq_len'],
            d_model=d,
            n_layers=n,
        )
        logger.info(f"Estimated activation memory: {est / 1024 / 1024:.0f}MB")
        logger.info(f"Recommended batch size: {recommended}")


def print_infrastructure_plan(config: dict):
    infra = config.get('infrastructure', {})
    model_cfg = config['aurelius_1b']
    gpus = infra.get('n_gpus', 1)
    gpu_type = infra.get('gpu_type', 'Unknown')
    strat = infra.get('distributed_strategy', 'none')

    vocab_size = model_cfg.get('vocab_size', 50257)
    total_params = sum([
        vocab_size * model_cfg['d_model'],
        model_cfg['n_layers'] * (
            4 * model_cfg['d_model'] ** 2
            + 3 * model_cfg['d_model'] * model_cfg['d_ff']
            + 4 * model_cfg['d_model'] * model_cfg['d_mem']
            + 4 * model_cfg['d_mem'] ** 2
            + 3 * model_cfg['d_mem'] * (model_cfg['d_mem'] // 2) * 2
        ),
    ])
    total_b = total_params / 1e9

    logger.info(f"\n=== AURELIUS 1B INFRASTRUCTURE ===")
    logger.info(f"Total parameters: {total_b:.2f}B")
    logger.info(f"Distributed strategy: {strat}")
    logger.info(f"GPUs: {gpus} × {gpu_type}")
    logger.info(f"Total GPU memory: {gpus * 80}GB (H100-80GB)")

    mem_per_gpu = total_b * 2 * 2 / gpus  # fp16 weights + adam states
    act_mem = model_cfg['d_model'] * model_cfg['max_seq_len'] * model_cfg['n_layers'] * 2 * 8 / gpus
    mem_per_gpu += act_mem / (1024**3)
    logger.info(f"Estimated memory per GPU: {mem_per_gpu:.1f}GB")
    logger.info(f"Memory savings active:")
    if train_cfg := config.get('training'):
        if train_cfg.get('gradient_checkpointing'):
            logger.info(f"  - Gradient checkpointing (saves ~60% activation memory)")
        if train_cfg.get('mixed_precision') == 'bf16':
            logger.info(f"  - BF16 mixed precision (halves weight memory)")
        if train_cfg.get('cpu_offload'):
            logger.info(f"  - CPU offloading (moves optimizer states to CPU)")
    logger.info(f"  - AMC LTS paged memory table (LRU eviction)")
    if infra.get('kv_cache_quant_bits', 0) > 0:
        logger.info(f"  - KV cache quantization ({infra['kv_cache_quant_bits']}-bit)")


if __name__ == '__main__':
    with open('config_1b.yaml') as f:
        config = yaml.safe_load(f)

    print_infrastructure_plan(config)

    model = AureliusModel1B(config['aurelius_1b'])
    total = sum(p.numel() for p in model.parameters())
    mem_params = sum(p.numel() for n, p in model.named_parameters() if 'memory' in n)
    logger.info(f"\nModel instantiated: {total:,} total parameters")
    logger.info(f"Memory system: {mem_params:,} parameters ({100*mem_params/total:.1f}%)")

    trainer = MemoryEfficientTrainer(model, config)
    trainer.print_memory_report()

    logger.info("\nReady for training. Run:")
    logger.info("  trainer.train_epoch(your_dataloader)")
