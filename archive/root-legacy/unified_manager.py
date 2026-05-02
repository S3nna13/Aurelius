import torch
import threading
import time
from threading import Lock

from async_memory import AsyncConsolidationPipeline, PagedLTSMemory, ConsolidationTask
from adaptive_precision import AdaptivePrecisionManager, TieredMemoryBank
from prefetch_router import PredictiveMemoryPrefetcher, SparseLTSRouter
from deduplication import (CosineDeduplicator, PriorityProportionalAllocator,
                           MemoryAwareGradientAccumulator, LZ4MemoryCompressor)
import logging
logger = logging.getLogger(__name__)


class UnifiedMemoryManager:
    def __init__(self, d_model: int, d_mem: int, n_layers: int,
                 lts_capacity: int, episodic_slots: int,
                 device: str = 'cuda', gpu_budget_mb: int = 65536):

        self.d_model = d_model
        self.d_mem = d_mem
        self.n_layers = n_layers
        self.device = device

        self.paged_lts = [
            PagedLTSMemory(d_mem, lts_capacity, device=device)
            for _ in range(n_layers)
        ]

        self.consolidation = AsyncConsolidationPipeline(
            d_mem, lts_capacity, n_layers, device
        )

        self.precision = AdaptivePrecisionManager()

        self.prefetcher = PredictiveMemoryPrefetcher(d_model, d_mem, n_layers)

        self.allocator = PriorityProportionalAllocator(
            lts_capacity, n_layers
        )

        self.deduplicator = CosineDeduplicator(threshold=0.92)

        self.grad_acc = MemoryAwareGradientAccumulator(
            target_memory_mb=gpu_budget_mb
        )
        self.compression = LZ4MemoryCompressor()

        self._lock = Lock()
        self.running = False
        self.metrics = {
            'total_reads': 0, 'total_writes': 0,
            'gpu_hits': 0, 'cpu_fallbacks': 0,
            'dedup_removed': 0,
        }

    def start_background(self):
        self.running = True
        self.consolidation.start()
        self._bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._bg_thread.start()

    def stop(self):
        self.running = False
        self.consolidation.stop()

    def _background_loop(self):
        while self.running:
            results = self.consolidation.drain_results()
            for r in results:
                with self._lock:
                    layer = r['layer']
                    if r['new_entries'] > 0:
                        consolidated = r['consolidated'].to(self.device)
                        dups = self.deduplicator.find_duplicates(consolidated)
                        if dups:
                            consolidated = self.deduplicator.merge(consolidated, dups)
                            self.metrics['dedup_removed'] += len(dups)
            time.sleep(0.01)

    def on_forward(self, layer: int, h: torch.Tensor, lts: torch.Tensor,
                   episodic: torch.Tensor, surprise: torch.Tensor) -> dict:
        with self._lock:
            self.metrics['total_reads'] += 1

            if self.prefetcher.should_prefetch(layer):
                prefetch_idx = self.prefetcher.get_prefetch_indices(lts, h)
                self.metrics['gpu_hits'] += 1
            else:
                self.metrics['cpu_fallbacks'] += 1

            self.prefetcher.observe(layer, surprise, h)
            self.allocator.update_priority(layer, surprise.mean().item())
            self.precision.auto_tune('long_term_store', torch.abs(surprise).mean().item())

            self._consolidation_counter = getattr(self, '_consolidation_counter', 0) + 1
            if self._consolidation_counter % 10 == 0:
                task = ConsolidationTask(
                    layer=layer,
                    episodic_slots=episodic.detach().cpu().clone(),
                    surprise_scores=surprise.detach().cpu().clone(),
                    timestamp=time.time(),
                    priority=surprise.mean().item(),
                )
                self.consolidation.submit(task)

            budget = self.grad_acc.step(
                torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
            )

        return {
            'should_accumulate': budget,
            'layer_priority': self.allocator.layer_priority[layer].item(),
            'precision_tier': self.precision.tier_config['long_term_store']['bits'],
        }

    def report(self) -> str:
        lines = ["=== Unified Memory Manager Report ==="]
        lines.append(f"Reads: {self.metrics['total_reads']}")
        lines.append(f"GPU hits: {self.metrics['gpu_hits']}")
        lines.append(f"CPU fallbacks: {self.metrics['cpu_fallbacks']}")
        lines.append(f"Dedup removed: {self.metrics['dedup_removed']}")
        lines.append("")
        lines.append(self.precision.report())
        lines.append("")
        lines.append(self.allocator.report())
        return "\n".join(lines)


class NUMAAwareMemoryPlacer:
    def __init__(self, n_nodes: int = 2):
        self.n_nodes = n_nodes
        self.node_assignments = {}

    def assign_to_node(self, tensor_name: str, tensor_size: int) -> int:
        import hashlib
        digest = hashlib.md5(tensor_name.encode()).hexdigest()
        node = int(digest[:8], 16) % self.n_nodes
        self.node_assignments[tensor_name] = node
        return node

    def place_on_node(self, tensor: torch.Tensor, node: int) -> torch.Tensor:
        if not torch.cuda.is_available():
            return tensor
        n_gpus = torch.cuda.device_count()
        if n_gpus < self.n_nodes:
            return tensor
        gpu_id = node % n_gpus
        return tensor.to(f'cuda:{gpu_id}', non_blocking=True)

    def distribute_lts_across_nodes(self, lts_list: list, node_map: list[int]):
        distributed = {}
        for lts, node in zip(lts_list, node_map):
            gpu = f'cuda:{node % torch.cuda.device_count()}'
            distributed[node] = lts.to(gpu, non_blocking=True)
        return distributed


class DistributedMemoryBalancer:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.balances = [0.0] * world_size

    def record(self, rank: int, memory_mb: float):
        self.balances[rank] = memory_mb

    def imbalance_ratio(self) -> float:
        if not self.balances:
            return 0.0
        avg = sum(self.balances) / len(self.balances)
        if avg == 0:
            return 0.0
        return max(abs(m - avg) / avg for m in self.balances)

    def should_rebalance(self, threshold: float = 0.15) -> bool:
        return self.imbalance_ratio() > threshold

    def suggest_rebalance(self) -> list[dict]:
        avg = sum(self.balances) / len(self.balances)
        suggestions = []
        for rank, mem in enumerate(self.balances):
            if mem > avg * 1.15:
                suggestions.append({
                    'source_rank': rank,
                    'excess_mb': mem - avg,
                    'action': 'offload_to_cpu',
                })
            elif mem < avg * 0.85:
                suggestions.append({
                    'target_rank': rank,
                    'deficit_mb': avg - mem,
                    'action': 'load_from_cpu',
                })
        return suggestions
