import torch
import torch.nn.functional as F
import math
import threading
import queue
import time
import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConsolidationTask:
    layer: int
    episodic_slots: torch.Tensor
    surprise_scores: torch.Tensor
    timestamp: float
    priority: float


class AsyncConsolidationPipeline:
    def __init__(self, d_mem: int, lts_capacity: int, n_layers: int,
                 device: str = 'cuda', num_workers: int = 2):
        self.d_mem = d_mem
        self.lts_capacity = lts_capacity
        self.n_layers = n_layers
        self.device = device
        self.task_queue = queue.PriorityQueue(maxsize=1000)
        self.result_queue = queue.Queue(maxsize=100)
        self.workers = []
        self.running = False
        self.num_workers = num_workers

    def start(self):
        self.running = True
        for i in range(self.num_workers):
            w = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            w.start()
            self.workers.append(w)

    def stop(self):
        self.running = False
        for w in self.workers:
            w.join(timeout=5.0)
        self.workers.clear()

    def submit(self, task: ConsolidationTask):
        try:
            negative_priority = -task.priority
            self.task_queue.put((negative_priority, time.time(), task), block=False)
        except queue.Full:
            pass

    def _worker_loop(self, worker_id: int):
        while self.running:
            try:
                _, _, task = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                result = self._consolidate(task)
                self.result_queue.put(result, block=False)
            except Exception as e:
                logging.warning(f"Worker {worker_id} consolidation failed: {e}")

    def _consolidate(self, task: ConsolidationTask) -> dict:
        slots = task.episodic_slots
        b, n, d = slots.shape
        if n < 2:
            return {'layer': task.layer, 'new_entries': 0}

        with torch.no_grad():
            slots_norm = slots / (slots.norm(dim=-1, keepdim=True) + 1e-8)
            sim = slots_norm @ slots_norm.transpose(-2, -1)
            adj = (sim > 0.65).float()
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            norm_adj = adj / deg
            cluster_feats = norm_adj @ slots

            surprise = task.surprise_scores.mean(dim=1, keepdim=True).unsqueeze(-1)
            weighted = cluster_feats * surprise

            if weighted.shape[1] > self.lts_capacity:
                scores = task.surprise_scores
                if scores.dim() > 2:
                    scores = scores.mean(dim=-1)
                keep_idx = torch.topk(scores, self.lts_capacity, dim=1).indices
                keep_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)
                weighted = torch.gather(weighted, 1, keep_idx)

        return {
            'layer': task.layer,
            'new_entries': weighted.shape[1],
            'consolidated': weighted.cpu(),
            'timestamp': task.timestamp,
        }

    def drain_results(self) -> list[dict]:
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results


class PagedLTSMemory(torch.nn.Module):
    def __init__(self, d_mem: int, capacity: int, n_pages: int = 64,
                 page_size: int = 32, device: str = 'cuda',
                 offload_to_cpu: bool = True):
        super().__init__()
        self.d_mem = d_mem
        self.capacity = capacity
        self.n_pages = n_pages
        self.page_size = page_size
        self.device = device
        self.offload_to_cpu = offload_to_cpu

        assert n_pages * page_size >= capacity

        self.register_buffer('page_table', torch.zeros(n_pages, dtype=torch.long))
        self.register_buffer('page_access_count', torch.zeros(n_pages, dtype=torch.long))
        self.register_buffer('page_dirty', torch.zeros(n_pages, dtype=torch.bool))

        self.gpu_pages = torch.nn.Parameter(
            torch.zeros(1, n_pages, page_size, d_mem), requires_grad=False
        )

        self.cpu_pages = torch.zeros(1, n_pages, page_size, d_mem, dtype=torch.float16)

        self.page_lru = list(range(n_pages))

    def page_for(self, entry_idx: int) -> tuple[int, int]:
        page = entry_idx // self.page_size
        offset = entry_idx % self.page_size
        return page, offset

    def read(self, query: torch.Tensor, top_k: int = 64) -> torch.Tensor:
        b, t, d = query.shape
        total_reads = min(top_k, self.capacity)
        mem_flat = self.gpu_pages.view(1, -1, self.d_mem)
        scores = query @ mem_flat.transpose(-2, -1)
        top_scores, top_indices = scores.topk(total_reads, dim=-1)
        flat_idx = top_indices.unsqueeze(-1).expand(b, t, total_reads, d)
        mem_expanded = mem_flat.unsqueeze(1).expand(b, t, -1, -1)
        selected = torch.gather(mem_expanded, 2, flat_idx)
        attn = F.softmax(top_scores / math.sqrt(self.d_mem), dim=-1)
        return (attn.unsqueeze(-1) * selected).sum(dim=2)

    def write(self, keys: torch.Tensor, values: torch.Tensor, importance: torch.Tensor):
        b, k, d = keys.shape
        importance_flat = importance.mean(dim=-1).view(-1)

        _, gpu_topk = importance_flat.topk(min(k, self.capacity))
        for i, idx in enumerate(gpu_topk):
            p, offset = self.page_for(idx.item())
            if p < self.n_pages:
                self.gpu_pages.data[0, p, offset] = values[0, i]
                self.page_dirty[p] = True

    def evict_lru_page(self) -> int:
        if not self.page_lru:
            return -1
        victim = self.page_lru.pop(0)
        if self.offload_to_cpu and self.page_dirty[0, victim]:
            self.cpu_pages[0, victim] = self.gpu_pages.data[0, victim].to(
                'cpu', dtype=torch.float16, non_blocking=True
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self.gpu_pages.data[0, victim].zero_()
        self.page_dirty[0, victim] = False
        self.page_lru.append(victim)
        return victim

    def restore_from_cpu(self, page_idx: int):
        if page_idx < self.n_pages:
            self.gpu_pages.data[0, page_idx] = self.cpu_pages[0, page_idx].to(
                self.device, non_blocking=True
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.cpu_pages[0, page_idx].zero_()

    def get_memory_usage(self) -> dict:
        gpu_mb = self.gpu_pages.numel() * 4 / (1024 * 1024)
        cpu_mb = self.cpu_pages.numel() * 2 / (1024 * 1024)
        dirty_pages = self.page_dirty.sum().item()
        return {
            'gpu_mb': gpu_mb,
            'cpu_mb': cpu_mb,
            'dirty_pages': int(dirty_pages),
            'total_pages': self.n_pages,
        }
