import torch
import torch.nn as nn
import torch.distributed as dist
import math


class CommunicationBudget:
    def __init__(self):
        self._bytes = 0
        self._step = 0

    def record(self, bytes):
        self._bytes += bytes
        self._step += 1

    def report(self):
        if self._step == 0:
            return "CommunicationBudget: 0 bytes communicated"
        avg = self._bytes / self._step
        return (
            f"CommunicationBudget: {self._bytes} bytes communicated over "
            f"{self._step} steps (avg {avg:.1f} bytes/step)"
        )

    def reset(self):
        self._bytes = 0
        self._step = 0


class FP8Compressor:
    def __init__(self):
        self._fp8_max = {
            torch.float8_e4m3fn: 448.0,
            torch.float8_e5m2: 57344.0,
        }

    def compress(self, tensor, dtype=torch.float8_e4m3fn):
        absmax = tensor.abs().max()
        if absmax == 0:
            scale = torch.ones(1, device=tensor.device, dtype=torch.float32)
            quantized = torch.zeros(tensor.shape, dtype=dtype, device=tensor.device)
            return quantized, scale
        fp8_max = self._fp8_max[dtype]
        scale = absmax / fp8_max
        scaled = tensor / scale
        quantized = scaled.to(dtype)
        return quantized, scale

    def decompress(self, quantized, scale):
        return quantized.to(torch.float32) * scale

    def compress_inplace(self, tensor):
        q, scale = self.compress(tensor)
        tensor.copy_(self.decompress(q, scale))

    def get_compression_ratio(self):
        return 4.0


class ErrorFeedbackBuffer:
    def __init__(self):
        self._errors = {}

    def push(self, name, original_grad, quantized_grad):
        error = original_grad - quantized_grad
        if name in self._errors:
            self._errors[name] = self._errors[name] + error
        else:
            self._errors[name] = error.detach().clone()

    def pop(self, name):
        return self._errors.pop(name, None)

    def clear(self):
        self._errors.clear()


class FP8AllReduce:
    def __init__(self, compressor=None, budget=None):
        self.compressor = compressor or FP8Compressor()
        self.budget = budget
        self._hierarchical_groups_initialized = False
        self._intra_groups = []
        self._inter_group = None
        self._intra_ranks_list = []
        self._inter_ranks = []
        self._node_size = 0

    def _init_hierarchical_groups(self, group, node_size):
        if self._hierarchical_groups_initialized and self._node_size == node_size:
            return
        world_size = dist.get_world_size(group)
        num_nodes = (world_size + node_size - 1) // node_size

        self._intra_ranks_list = []
        self._intra_groups = []
        for n in range(num_nodes):
            start = n * node_size
            end = min(start + node_size, world_size)
            ranks = list(range(start, end))
            self._intra_ranks_list.append(ranks)
            if len(ranks) > 1:
                g = dist.new_group(ranks)
            else:
                g = None
            self._intra_groups.append(g)

        self._inter_ranks = [r[0] for r in self._intra_ranks_list]
        if len(self._inter_ranks) > 1:
            self._inter_group = dist.new_group(self._inter_ranks)
        else:
            self._inter_group = None

        self._node_size = node_size
        self._hierarchical_groups_initialized = True

    def _count_bytes(self, tensor, group, compress):
        world_size = dist.get_world_size(group)
        numel = tensor.numel()
        if compress:
            bytes_per_element = 2
        else:
            bytes_per_element = 4
        total = numel * bytes_per_element * (world_size - 1)
        return total

    def all_reduce(self, gradients, group=None, compress=True, hierarchical=False, node_size=8):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return list(gradients)

        if group is None:
            group = dist.group.WORLD

        if group is None:
            return list(gradients)

        reduced = []

        if hierarchical:
            self._init_hierarchical_groups(group, node_size)
            rank = dist.get_rank(group)
            node_id = rank // node_size
            intra_group = self._intra_groups[node_id]
            inter_group = self._inter_group
            is_rep = rank % node_size == 0

            for g in gradients:
                if compress:
                    q, scale = self.compressor.compress(g)
                    q_fp16 = q.to(torch.float16)

                    if intra_group is not None:
                        dist.all_reduce(q_fp16, group=intra_group)
                        q_fp16.div_(intra_group.size())

                    node_result = self.compressor.decompress(q_fp16.to(q.dtype), scale)

                    if inter_group is not None:
                        if is_rep:
                            inter_fp16 = node_result.to(torch.float16)
                            dist.all_reduce(inter_fp16, group=inter_group)
                            inter_fp16.div_(inter_group.size())
                            global_result = inter_fp16.to(torch.float32)
                        else:
                            global_result = torch.zeros_like(g)

                        if intra_group is not None:
                            dist.broadcast(global_result, src=0, group=intra_group)
                        else:
                            global_result = node_result
                    else:
                        global_result = node_result
                else:
                    g_fp16 = g.to(torch.float16)
                    if intra_group is not None:
                        dist.all_reduce(g_fp16, group=intra_group)
                        g_fp16.div_(intra_group.size())
                    if inter_group is not None:
                        if is_rep:
                            dist.all_reduce(g_fp16, group=inter_group)
                            g_fp16.div_(inter_group.size())
                        if intra_group is not None:
                            dist.broadcast(g_fp16, src=0, group=intra_group)
                    global_result = g_fp16.to(torch.float32)

                reduced.append(global_result)
                if self.budget is not None:
                    self.budget.record(self._count_bytes(g, group, compress))
        else:
            for g in gradients:
                if compress:
                    q, scale = self.compressor.compress(g)
                    q_fp16 = q.to(torch.float16)
                    dist.all_reduce(q_fp16, group=group)
                    q_fp16.div_(dist.get_world_size(group))
                    result = self.compressor.decompress(q_fp16.to(q.dtype), scale)
                else:
                    g_fp16 = g.to(torch.float16)
                    dist.all_reduce(g_fp16, group=group)
                    result = g_fp16.to(torch.float32) / dist.get_world_size(group)

                reduced.append(result)
                if self.budget is not None:
                    self.budget.record(self._count_bytes(g, group, compress))

        return reduced


class GradientClippingWithCompression:
    def __init__(self, compressor=None):
        self.compressor = compressor or FP8Compressor()

    def clip_grad_norm_(self, model, max_norm, norm_type=2.0):
        if max_norm <= 0:
            return 0.0

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            q, scale = self.compressor.compress(p.grad.data)
            q_fp16 = q.to(torch.float16)
            dq = self.compressor.decompress(q_fp16.to(q.dtype), scale)
            total_norm += dq.float().pow(norm_type).sum().item()

        if dist.is_initialized() and dist.get_world_size() > 1:
            norm_tensor = torch.tensor(total_norm, device=next(model.parameters()).device)
            dist.all_reduce(norm_tensor)
            total_norm = norm_tensor.item()

        total_norm = math.pow(total_norm, 1.0 / norm_type)

        clip_coef = max_norm / (total_norm + 1e-8)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        return total_norm


class FP8DistributedTrainer:
    def __init__(
        self,
        compressor=None,
        error_buffer=None,
        all_reduce=None,
        clipping=None,
        budget=None,
        use_error_feedback=True,
        max_norm=1.0,
    ):
        self.compressor = compressor or FP8Compressor()
        self.error_buffer = error_buffer or ErrorFeedbackBuffer()
        self.all_reduce = all_reduce or FP8AllReduce(compressor=self.compressor, budget=budget)
        self.clipping = clipping or GradientClippingWithCompression(compressor=self.compressor)
        self.budget = budget or CommunicationBudget()
        self.use_error_feedback = use_error_feedback
        self.max_norm = max_norm

    def train_step(self, model, loss_fn, optimizer, batch, compress=True, hierarchical=False, node_size=8):
        inputs, targets = batch

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        grad_list = []
        param_names = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            param_names.append(name)
            if self.use_error_feedback:
                error = self.error_buffer.pop(name)
                if error is not None:
                    p.grad.data.add_(error)
            original = p.grad.data.clone()
            grad_list.append(original)

        if grad_list:
            reduced = self.all_reduce.all_reduce(
                grad_list,
                compress=compress,
                hierarchical=hierarchical,
                node_size=node_size,
            )
            for name, p, r in zip(param_names, model.parameters(), reduced):
                if p.grad is None:
                    continue
                if self.use_error_feedback:
                    self.error_buffer.push(name, p.grad.data, r)
                p.grad.data.copy_(r)

        if self.max_norm > 0:
            self.clipping.clip_grad_norm_(model, self.max_norm)

        optimizer.step()

        num_params = sum(p.numel() for p in model.parameters() if p.grad is not None)
        metrics = {
            "loss": loss.item(),
            "num_params_communicated": num_params,
            "budget": self.budget.report(),
        }
        return metrics
