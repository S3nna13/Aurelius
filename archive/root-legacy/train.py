import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml

from aurelius_model import AureliusModel
import logging
logger = logging.getLogger("train")



class MemoryAuxLoss(nn.Module):
    def __init__(self, memory_loss_weight: float = 0.1, consolidation_weight: float = 0.05):
        super().__init__()
        self.memory_weight = memory_loss_weight
        self.consolidation_weight = consolidation_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                mem_states: dict) -> torch.Tensor:
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        surprise_penalty = 0.0
        for layer, ms in mem_states.items():
            s = ms['surprise']
            surprise_penalty = surprise_penalty + s.pow(2).mean()
        surprise_penalty = surprise_penalty / max(len(mem_states), 1)
        total = ce + self.memory_weight * surprise_penalty
        return total, {'ce': ce.item(), 'surprise': surprise_penalty.item(), 'total': total.item()}


def train_step(model: nn.Module, batch: dict, optimizer: AdamW,
               loss_fn: MemoryAuxLoss, grad_clip: float = 1.0):
    model.train()
    optimizer.zero_grad()
    input_ids = batch['input_ids']
    labels = batch['labels']
    logits, mem_states = model(input_ids, return_mem_state=True)
    loss, metrics = loss_fn(logits, labels, mem_states)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return metrics


def build_optimizer(model: nn.Module, config: dict) -> tuple[AdamW, object]:
    opt = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        weight_decay=config['weight_decay'],
    )
    warmup = LinearLR(opt, start_factor=0.01, total_iters=config['warmup_steps'])
    cosine = CosineAnnealingLR(opt, T_max=max(1, config['total_steps'] - config['warmup_steps']))
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine],
                             milestones=[config['warmup_steps']])
    return opt, scheduler


if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        model_cfg = config['aurelius_150m']
        train_cfg = config['training']

    model = AureliusModel(model_cfg)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Aurelius total parameters: {total_params:,}")
    mem_params = sum(p.numel() for n, p in model.named_parameters() if 'memory' in n)
    logger.info(f"Memory-specific parameters: {mem_params:,} ({(mem_params/total_params)*100:.1f}%)")

    optim, scheduler = build_optimizer(model, train_cfg)
    loss_fn = MemoryAuxLoss(
        memory_loss_weight=train_cfg['memory_loss_weight'],
        consolidation_weight=train_cfg['consolidation_loss_weight'],
    )

    logger.info(f"Model ready. Memory system initialized.")
    logger.info(f"Memory tier: Working (2048 ctx) + Episodic (512 slots) + LTS (1024 entries)")
    logger.info(f"Surprise-gated writes | Graph-based consolidation | Cross-attention reads")
