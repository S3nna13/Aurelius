import torch
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any


class PagedOptimizerState:
    def __init__(self, gpu_budget: int = 4, device: Optional[torch.device] = None):
        self._gpu_budget = gpu_budget
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lru: OrderedDict = OrderedDict()
        self._gpu_count: int = 0
        self._param_to_group: Dict[str, int] = {}

    def register_param(self, name: str, group_index: int) -> None:
        self._param_to_group[name] = group_index

    def _load_single_to_gpu(self, name: str) -> None:
        state = self._states.get(name)
        if state is None:
            return
        exp_avg = state.get('exp_avg')
        if exp_avg is None:
            return
        if exp_avg.device == self._device:
            self._lru.move_to_end(name)
            return
        while self._gpu_count >= self._gpu_budget:
            if not self._lru:
                break
            evict_key, _ = self._lru.popitem(last=False)
            self._evict_single_to_cpu(evict_key)
        state['exp_avg'] = exp_avg.to(self._device)
        state['exp_avg_sq'] = state['exp_avg_sq'].to(self._device)
        self._gpu_count += 1
        self._lru[name] = None

    def _evict_single_to_cpu(self, name: str) -> None:
        state = self._states.get(name)
        if state is None:
            return
        exp_avg = state.get('exp_avg')
        if exp_avg is None or exp_avg.device == torch.device('cpu'):
            return
        state['exp_avg'] = exp_avg.to('cpu')
        state['exp_avg_sq'] = state['exp_avg_sq'].to('cpu')
        self._gpu_count -= 1
        if name in self._lru:
            del self._lru[name]

    def load_to_gpu(self, param_group_index: int) -> None:
        names = [n for n, g in self._param_to_group.items() if g == param_group_index]
        for name in names:
            self._load_single_to_gpu(name)

    def evict_to_cpu(self, param_group_index: int) -> None:
        names = [n for n, g in self._param_to_group.items() if g == param_group_index]
        for name in names:
            self._evict_single_to_cpu(name)

    def get_state(self, param_name: str) -> Optional[Dict[str, Any]]:
        if param_name in self._lru:
            self._lru.move_to_end(param_name)
        return self._states.get(param_name)

    def set_state(self, param_name: str, exp_avg: Optional[torch.Tensor], exp_avg_sq: Optional[torch.Tensor]) -> None:
        self._states[param_name] = {'exp_avg': exp_avg, 'exp_avg_sq': exp_avg_sq, 'step': 0}
        if exp_avg is not None and exp_avg.device != torch.device('cpu'):
            self._gpu_count += 1
            self._lru[param_name] = None


class PagedAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, amsgrad: bool = False,
                 gpu_budget: int = 4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

        device = None
        for group in self.param_groups:
            for p in group['params']:
                device = p.device
                break
            if device is not None:
                break
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._state_manager = PagedOptimizerState(gpu_budget=gpu_budget, device=device)
        self._param_id_to_name: Dict[int, str] = {}
        self._optimizer_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        for group_idx, group in enumerate(self.param_groups):
            for p in group['params']:
                name = f"p_{id(p)}"
                self._param_id_to_name[id(p)] = name
                self._state_manager.register_param(name, group_idx)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PagedAdamW does not support sparse gradients")

                name = self._param_id_to_name[id(p)]
                self._get_param_state(p)

                if not (grad != 0).any().item():
                    continue

                self._state_manager._load_single_to_gpu(name)

                state = self._state_manager.get_state(name)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                if exp_avg.device != grad.device:
                    exp_avg = exp_avg.to(grad.device)
                    exp_avg_sq = exp_avg_sq.to(grad.device)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)

                step = state['step'] + 1
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                state['step'] = step

                self._state_manager._evict_single_to_cpu(name)

        return loss


class OptimizerStateCompressor:
    @staticmethod
    def compress(state_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        compressed = {}
        for param_name, state in state_dict.items():
            compressed[param_name] = {
                'exp_avg': state['exp_avg'],
                'exp_avg_sq': state['exp_avg_sq'].half(),
                'step': state.get('step', 0),
            }
        return compressed

    @staticmethod
    def decompress(compressed_state_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        decompressed = {}
        for param_name, state in compressed_state_dict.items():
            decompressed[param_name] = {
                'exp_avg': state['exp_avg'],
                'exp_avg_sq': state['exp_avg_sq'].float(),
                'step': state.get('step', 0),
            }
        return decompressed


class GradientBucket:
    def __init__(self, bucket_size_mb: int = 25):
        self._bucket_size = bucket_size_mb * 1024 * 1024
        self._gradients: Dict[str, torch.Tensor] = {}
        self._buckets: List[Dict[str, torch.Tensor]] = []

    def add_grad(self, name: str, grad: torch.Tensor) -> None:
        if name in self._gradients:
            self._gradients[name].add_(grad)
        else:
            self._gradients[name] = grad.clone()

    def _build_buckets(self) -> None:
        self._buckets = []
        current_bucket: Dict[str, torch.Tensor] = {}
        current_size = 0
        for name, grad in self._gradients.items():
            grad_size = grad.numel() * grad.element_size()
            if current_size + grad_size > self._bucket_size and current_bucket:
                self._buckets.append(current_bucket)
                current_bucket = {}
                current_size = 0
            current_bucket[name] = grad
            current_size += grad_size
        if current_bucket:
            self._buckets.append(current_bucket)

    def flush(self) -> List[Dict[str, torch.Tensor]]:
        self._build_buckets()
        for bucket in self._buckets:
            if not bucket:
                continue
            flat = torch.cat([g.flatten() for g in bucket.values()])
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(flat)
            offset = 0
            for name, grad in bucket.items():
                numel = grad.numel()
                grad.copy_(flat[offset:offset + numel].reshape(grad.shape))
                offset += numel
        return self._buckets

    def reset(self) -> None:
        self._gradients.clear()
        self._buckets.clear()
