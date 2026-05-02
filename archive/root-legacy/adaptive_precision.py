import torch


class AdaptivePrecisionManager:
    def __init__(self):
        self.tier_config = {
            'working_memory': {'bits': 8, 'dtype': torch.float16},
            'episodic_buffer': {'bits': 16, 'dtype': torch.float16},
            'long_term_store': {'bits': 32, 'dtype': torch.float32},
            'graph_edges': {'bits': 8, 'dtype': torch.int8},
        }
        self.usage_stats = {tier: {'reads': 0, 'writes': 0} for tier in self.tier_config}

    def get_dtype(self, tier: str) -> torch.dtype:
        return self.tier_config.get(tier, {}).get('dtype', torch.float32)

    def quantize_to_tier(self, tensor: torch.Tensor, tier: str) -> torch.Tensor:
        cfg = self.tier_config.get(tier)
        if cfg is None:
            return tensor
        target = cfg['dtype']
        if tensor.dtype == target:
            return tensor
        return tensor.to(target)

    def auto_tune(self, tier: str, error_rate: float):
        cfg = self.tier_config.get(tier)
        if cfg is None:
            return
        if error_rate > 0.05 and cfg['bits'] < 32:
            self._promote_precision(tier)
        elif error_rate < 0.005 and cfg['bits'] > 8:
            self._demote_precision(tier)

    def _promote_precision(self, tier: str):
        cfg = self.tier_config[tier]
        if cfg['bits'] == 8:
            cfg['bits'], cfg['dtype'] = 16, torch.float16
        elif cfg['bits'] == 16:
            cfg['bits'], cfg['dtype'] = 32, torch.float32

    def _demote_precision(self, tier: str):
        cfg = self.tier_config[tier]
        if cfg['bits'] == 32:
            cfg['bits'], cfg['dtype'] = 16, torch.bfloat16
        elif cfg['bits'] == 16:
            cfg['bits'], cfg['dtype'] = 8, torch.float8_e4m3fn

    def report(self) -> str:
        lines = ["Adaptive Precision:"]
        for tier, cfg in self.tier_config.items():
            lines.append(f"  {tier}: {cfg['bits']}-bit ({cfg['dtype']})")
        return "\n".join(lines)


class FP8LTSMemory(torch.nn.Module):
    FP8_MAX = 448.0

    def __init__(self, d_mem: int, capacity: int):
        super().__init__()
        self.d_mem = d_mem
        self.capacity = capacity
        self.register_buffer('scale', torch.ones(capacity, 1))
        self.fp8_storage = torch.nn.Parameter(
            torch.zeros(1, capacity, d_mem, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )

    def store(self, data: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            amax = data.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
            per_element_scale = amax / self.FP8_MAX
            data_scaled = data / per_element_scale
            data_fp8 = data_scaled.to(torch.float8_e4m3fn)
            self.fp8_storage.data[0, indices] = data_fp8
            self.scale.data[indices] = per_element_scale

    def retrieve(self, query: torch.Tensor, top_k: int = 64) -> torch.Tensor:
        stored_fp32 = self.fp8_storage.to(torch.float32) * self.scale
        scores = query @ stored_fp32.transpose(-2, -1)
        top_scores, top_idx = scores.topk(top_k, dim=-1)
        B, T, K = top_idx.shape
        values = stored_fp32.unsqueeze(1).expand(B, T, -1, -1)
        index = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_mem)
        selected = torch.gather(values, 2, index)
        attn = torch.softmax(top_scores / (self.d_mem ** 0.5), dim=-1)
        return (attn.unsqueeze(-1) * selected).sum(dim=-2)

    def memory_saved(self) -> float:
        fp32_bytes = self.capacity * self.d_mem * 4
        fp8_bytes = self.capacity * self.d_mem * 1
        return (1 - fp8_bytes / fp32_bytes) * 100


class _ParameterWrapper(torch.nn.Module):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.param = torch.nn.Parameter(data)

class TieredMemoryBank(torch.nn.Module):
    def __init__(self, d_mem: int, capacities: dict):
        super().__init__()
        self.tiers = torch.nn.ModuleDict()
        for name, cap in capacities.items():
            if name == 'core':
                self.tiers[name] = _ParameterWrapper(torch.zeros(1, cap, d_mem))
            else:
                self.tiers[name] = FP8LTSMemory(d_mem, cap)
        self.routes = {}

    def route_to_tier(self, surprise: torch.Tensor) -> str:
        avg = surprise.mean().item()
        if avg > 0.7:
            return 'core'
        elif avg > 0.3:
            return 'working'
        return 'archive'

    def write(self, key: str, data: torch.Tensor, surprise: torch.Tensor):
        tier = self.route_to_tier(surprise)
        if tier in self.tiers:
            tier_obj = self.tiers[tier]
            if isinstance(tier_obj, _ParameterWrapper):
                cap = tier_obj.param.shape[1]
                idx = torch.randint(0, cap, (1,))
                tier_obj.param.data[0, idx] = data
            else:
                cap = tier_obj.capacity
                idx = torch.randint(0, cap, (1,))
                tier_obj.store(data, idx)

    def total_memory_mb(self) -> dict:
        result = {}
        for name, tier in self.tiers.items():
            if isinstance(tier, FP8LTSMemory):
                mb = tier.fp8_storage.numel() * 1 / (1024*1024)
            elif hasattr(tier, 'param'):
                mb = tier.param.numel() * 4 / (1024*1024)
            else:
                mb = tier.numel() * 4 / (1024*1024)
            result[name] = mb
        return result
