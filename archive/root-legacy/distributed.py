import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint
import contextlib
import os
from typing import Optional, Dict, Any, Tuple
from nn_utils import sample_with_top_p_top_k
import logging
logger = logging.getLogger("distributed")



class ModelParallelGroup:
    def __init__(self, tp_size: int = 1, dp_size: int = 1):
        self.tp_size = tp_size
        self.dp_size = dp_size
        self._tp_group = None
        self._dp_group = None
        self._initialized = False

        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if tp_size * dp_size != world_size:
            if world_size % tp_size == 0:
                self.dp_size = world_size // tp_size
                logger.warning(f adjusted dp_size from {dp_size} to {self.dp_size}")
            elif world_size % dp_size == 0:
                self.tp_size = world_size // dp_size
                logger.warning(f adjusted tp_size from {tp_size} to {self.tp_size}")
            else:
                self.tp_size = 1
                self.dp_size = world_size
                logger.warning(f tp_size*dp_size != world_size, falling back to tp=1, dp={world_size}")

        self._tp_group = self._create_tp_group(rank)
        self._dp_group = self._create_dp_group(rank)
        self._initialized = True

    def _create_tp_group(self, rank: int):
        num_tp_groups = dist.get_world_size() // self.tp_size
        groups = []
        for i in range(num_tp_groups):
            start = i * self.tp_size
            ranks = list(range(start, start + self.tp_size))
            group = dist.new_group(ranks)
            groups.append(group)
        group_idx = rank // self.tp_size
        return groups[group_idx]

    def _create_dp_group(self, rank: int):
        dp_groups = []
        for tp_rank in range(self.tp_size):
            ranks = [tp_rank + i * self.tp_size for i in range(self.dp_size)]
            group = dist.new_group(ranks)
            dp_groups.append(group)
        dp_idx = rank % self.tp_size
        return dp_groups[dp_idx]

    def get_tp_group(self):
        if not self._initialized:
            return None
        return self._tp_group

    def get_dp_group(self):
        if not self._initialized:
            return None
        return self._dp_group

    def get_world_info(self) -> Dict[str, Any]:
        if not self._initialized:
            return {
                'tp_size': 1,
                'dp_size': 1,
                'tp_rank': 0,
                'dp_rank': 0,
                'global_rank': 0,
                'world_size': 1,
            }
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tp_rank = rank % self.tp_size
        dp_rank = rank // self.tp_size
        return {
            'tp_size': self.tp_size,
            'dp_size': self.dp_size,
            'tp_rank': tp_rank,
            'dp_rank': dp_rank,
            'global_rank': rank,
            'world_size': world_size,
        }


def shard_parameter(param: torch.Tensor, dim: int, parallel_group) -> torch.Tensor:
    if parallel_group is None:
        return param
    tp_size = dist.get_world_size(parallel_group)
    tp_rank = dist.get_rank(parallel_group)
    size = param.shape[dim]
    shard_size = size // tp_size
    remainder = size % tp_size
    start = tp_rank * shard_size + min(tp_rank, remainder)
    end = start + shard_size + (1 if tp_rank < remainder else 0)
    indices = torch.arange(start, end, device=param.device)
    local_shard = param.index_select(dim, indices)
    return local_shard.contiguous()


def gather_output(local_output: torch.Tensor, dim: int, parallel_group) -> torch.Tensor:
    if parallel_group is None:
        return local_output
    tp_size = dist.get_world_size(parallel_group)
    if tp_size == 1:
        return local_output
    gathered = [torch.empty_like(local_output) for _ in range(tp_size)]
    dist.all_gather(gathered, local_output.contiguous(), group=parallel_group)
    return torch.cat(gathered, dim=dim)


class AureliusFSDPWrapper:
    def __init__(self, model: nn.Module, shard_strategy: str = 'full', mixed_precision: str = 'bf16'):
        self.original_model = model
        self.shard_strategy = shard_strategy
        self.mixed_precision = mixed_precision
        self.optimizer = None

        strategy_map = {
            'full': ShardingStrategy.FULL_SHARD,
            'grad': ShardingStrategy.SHARD_GRAD_OP,
            'none': ShardingStrategy.NO_SHARD,
        }
        fsdp_strategy = strategy_map.get(shard_strategy, ShardingStrategy.FULL_SHARD)

        dtype_map = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32,
        }
        compute_dtype = dtype_map.get(mixed_precision, torch.bfloat16)

        mp_policy = MixedPrecision(
            param_dtype=compute_dtype,
            reduce_dtype=compute_dtype,
            buffer_dtype=compute_dtype,
        )

        block_class = _get_block_class(model)
        auto_wrap = transformer_auto_wrap_policy(transformer_layer_set={block_class}) if block_class else None

        wrap_kwargs = {
            'sharding_strategy': fsdp_strategy,
            'mixed_precision': mp_policy,
            'device_id': torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
        if auto_wrap is not None:
            wrap_kwargs['auto_wrap_policy'] = auto_wrap

        self.model = FSDP(model, **wrap_kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward_and_step(self, loss: torch.Tensor):
        loss.backward()
        if self.optimizer is not None:
            self.model.clip_grad_norm_(1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def save_full_checkpoint(self, path: str):
        if not dist.is_initialized() or dist.get_rank() == 0:
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                state = self.model.state_dict()
            torch.save(state, path)
        if dist.is_initialized():
            dist.barrier()

    def load_full_checkpoint(self, path: str):
        state = torch.load(path, map_location='cpu', weights_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            self.model.load_state_dict(state)
        if dist.is_initialized():
            dist.barrier()


class ActivationCheckpointing:
    def __init__(self, model: nn.Module, checkpoint_blocks: Optional[list] = None):
        self.model = model
        self._original_forwards = {}
        self._active = False
        blocks = _get_blocks(model)

        if checkpoint_blocks is None:
            self.checkpoint_indices = list(range(0, len(blocks), 2))
        else:
            self.checkpoint_indices = checkpoint_blocks

        for idx in self.checkpoint_indices:
            if idx < len(blocks):
                block = blocks[idx]
                self._original_forwards[idx] = block.forward

    def enable(self):
        if self._active:
            return
        blocks = _get_blocks(self.model)
        for idx in self.checkpoint_indices:
            if idx < len(blocks):
                block = blocks[idx]
                original_fwd = self._original_forwards[idx]
                block.forward = _make_checkpointed_forward(original_fwd, block)
        self._active = True

    def disable(self):
        if not self._active:
            return
        blocks = _get_blocks(self.model)
        for idx in self.checkpoint_indices:
            if idx < len(blocks):
                blocks[idx].forward = self._original_forwards[idx]
        self._active = False


class GradientAccumulator:
    def __init__(self, micro_batch_size: int, total_batch_size: int):
        self.micro_batch_size = micro_batch_size
        self.total_batch_size = total_batch_size
        self.accumulation_steps = max(1, total_batch_size // micro_batch_size)
        if self.accumulation_steps == 0:
            self.accumulation_steps = 1

    def should_step(self, step: int) -> bool:
        return (step + 1) % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss / self.accumulation_steps


class MemoryEfficientWrapper:
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self._past_key_values = None
        self._seq_len = 0

    def _init_kv_cache(self, input_ids: torch.Tensor):
        self.model.eval()
        batch_size = input_ids.shape[0]
        config = self.model.config if hasattr(self.model, 'config') else {}
        n_layers = getattr(self.model, 'blocks', None)
        if n_layers is not None:
            n_layers = len(n_layers)
        else:
            n_layers = config.get('n_layers', 32)

        n_heads = config.get('n_heads', 32)
        head_dim = config.get('d_model', 4096) // n_heads
        max_len = config.get('max_seq_len', 16384)

        self._past_key_values = []
        for _ in range(n_layers):
            self._past_key_values.append({
                'key': torch.zeros(batch_size, n_heads, 0, head_dim, device=self.device, dtype=torch.float16),
                'value': torch.zeros(batch_size, n_heads, 0, head_dim, device=self.device, dtype=torch.float16),
                'max_len': max_len,
            })
        self._seq_len = 0

    def _update_kv_cache(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor):
        entry = self._past_key_values[layer_idx]
        entry['key'] = torch.cat([entry['key'], new_key], dim=2)
        entry['value'] = torch.cat([entry['value'], new_value], dim=2)
        if entry['key'].shape[2] > entry['max_len']:
            entry['key'] = entry['key'][:, :, -entry['max_len']:]
            entry['value'] = entry['value'][:, :, -entry['max_len']:]

    def _clear_kv_cache(self):
        self._past_key_values = None
        self._seq_len = 0

    @torch.no_grad()
    def generate(self, model: nn.Module, input_ids: torch.Tensor,
                 max_new_tokens: int = 256, temperature: float = 1.0,
                 top_p: float = 0.9) -> Generator[torch.Tensor, None, None]:
        model.eval()
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        self._init_kv_cache(input_ids)
        current_ids = input_ids

        for _ in range(max_new_tokens):
            if self._seq_len == 0:
                out = model(current_ids)
            else:
                out = model(current_ids[:, -1:])

            if isinstance(out, dict):
                logits = out['logits']
            else:
                logits = out

            next_token = sample_with_top_p_top_k(logits[:, -1, :], temperature, top_k=0, top_p=top_p)

            self._seq_len += 1
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            yield next_token

        self._clear_kv_cache()


def _get_block_class(model: nn.Module):
    block_attr = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            if parent_name in ('blocks', '') or name == 'blocks':
                block_attr = module
                break

    if block_attr is None and hasattr(model, 'blocks'):
        block_attr = model.blocks

    if block_attr is not None and len(block_attr) > 0:
        return type(block_attr[0])

    return None


def _get_blocks(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, 'module') and hasattr(model.module, 'blocks'):
        return model.module.blocks
    if hasattr(model, 'blocks'):
        return model.blocks
    if hasattr(model, 'module'):
        inner = model.module
        if hasattr(inner, 'module') and hasattr(inner.module, 'blocks'):
            return inner.module.blocks
        if hasattr(inner, 'blocks'):
            return inner.blocks
    return nn.ModuleList()


def _make_checkpointed_forward(original_forward, module):
    def checkpointed_forward(*args, **kwargs):
        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
    return checkpointed_forward