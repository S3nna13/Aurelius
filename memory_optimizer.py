import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Iterator


class GradientCheckpointing:
    def __init__(self, module: nn.Module, segments: int = 4):
        self.module = module
        self.segments = segments

    @contextmanager
    def training_context(self) -> Iterator[None]:
        orig = self.module.training
        self.module.train()
        try:
            yield
        finally:
            self.module.train(orig)

    def forward_with_checkpoint(self, *args, **kwargs) -> torch.Tensor:
        return torch.utils.checkpoint.checkpoint(
            self.module,
            *args,
            use_reentrant=False,
            **kwargs,
        )


class CpuOffloadManager:
    def __init__(self, module: nn.Module, offload_freq: int = 100):
        self.module = module
        self.offload_freq = offload_freq
        self.step = 0
        self.offloaded = {}

    def step_callback(self):
        self.step += 1
        if self.step % self.offload_freq == 0:
            self._offload_idle_params()

    def _offload_idle_params(self):
        for name, param in self.module.named_parameters():
            if param.grad is None and param.device.type == 'cuda':
                if name not in self.offloaded:
                    orig_shape = param.data.shape
                    self.offloaded[name] = {
                        'data': param.data.to('cpu', non_blocking=True),
                        'shape': orig_shape,
                    }
                    param.data = torch.empty(orig_shape, device='cpu', dtype=param.dtype)

    def restore(self):
        for name, entry in self.offloaded.items():
            data = entry['data']
            for n, p in self.module.named_parameters():
                if n == name:
                    p.data = data.to('cuda', non_blocking=True)
        self.offloaded.clear()


class MixedPrecisionTrainer:
    def __init__(self, model: nn.Module, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler('cuda') if (enabled and torch.cuda.is_available()) else None

    def train_step(self, batch: dict, loss_fn: callable, optimizer: torch.optim.Optimizer,
                   grad_clip: float = 1.0) -> dict:
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.enabled):
            logits, mem_states = self.model(
                batch['input_ids'], return_mem_state=True
            )
            loss, metrics = loss_fn(logits, batch['labels'], mem_states)

        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

        return metrics

    def state_dict(self) -> dict:
        return {'scaler': self.scaler.state_dict()} if self.scaler else {}

    def load_state_dict(self, state: dict):
        if self.scaler and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])


class ActivationMemoryBudget:
    def __init__(self, max_act_mb: int = 2048):
        self.max_bytes = max_act_mb * 1024 * 1024
        self.peak = 0

    def estimate_activation_size(self, batch_size: int, seq_len: int, d_model: int,
                                  n_layers: int, precision_bytes: int = 2) -> int:
        per_layer_act = d_model * seq_len * batch_size * precision_bytes * 8
        total = per_layer_act * n_layers
        self.peak = max(self.peak, total)
        return total

    def recommend_batch_size(self, seq_len: int, d_model: int, n_layers: int,
                              precision_bytes: int = 2) -> int:
        per_token_per_layer = d_model * precision_bytes * 8
        total_per_seq = per_token_per_layer * seq_len * n_layers
        if total_per_seq == 0:
            return 1
        max_batch = self.max_bytes // total_per_seq
        return max(1, max_batch)


class ZeROOptimizerGroup:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 n_gpus: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.n_gpus = n_gpus

    def partition_optimizer_state(self):
        optim_state = self.optimizer.state_dict()['state']
        partitioned = {}
        for i, (param_id, state) in enumerate(optim_state.items()):
            if i % self.n_gpus == 0:
                partitioned[param_id] = state
        return partitioned

    def reduce_gradients(self, loss: torch.Tensor):
        loss.backward()
        if torch.distributed.is_initialized():
            for param in self.model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                    param.grad /= self.n_gpus
