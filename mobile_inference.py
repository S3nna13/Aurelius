import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import math
import os
from nn_utils import RMSNorm


class MobileQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def quantize_model(self, model):
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    def quantize_tensor(self, tensor, bits=8):
        qmax = 2 ** (bits - 1) - 1
        scale = tensor.abs().max().clamp(min=1e-6) / qmax
        quantized = (tensor / scale).round().clamp(
            -(2 ** (bits - 1)), qmax
        ).to(torch.int8)
        return quantized, scale

    def dequantize_tensor(self, quantized, scale):
        return quantized.float() * scale


class InferenceOptimizer(nn.Module):
    def __init__(self):
        super().__init__()

    def optimize(self, model, mode='reduce-overhead'):
        model.eval()
        return torch.compile(model, mode=mode)

    def _fuse_norm_linear(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Sequential):
                new_modules = []
                i = 0
                children_list = list(child.children())
                while i < len(children_list):
                    curr = children_list[i]
                    if isinstance(curr, (nn.LayerNorm, RMSNorm)) and i + 1 < len(children_list):
                        nxt = children_list[i + 1]
                        if isinstance(nxt, nn.Linear):
                            gamma = curr.weight.data
                            beta = curr.bias.data if hasattr(curr, 'bias') and curr.bias is not None else None
                            W = nxt.weight.data
                            b = nxt.bias.data if nxt.bias is not None else torch.zeros(nxt.out_features, device=W.device)
                            W_new = W * gamma.view(1, -1)
                            b_new = b.clone()
                            if beta is not None:
                                b_new = b_new + F.linear(beta, W)
                            nxt.weight.data = W_new
                            nxt.bias = nn.Parameter(b_new)
                            new_modules.append(nxt)
                            i += 2
                            continue
                    new_modules.append(curr)
                    i += 1
                setattr(module, name, nn.Sequential(*new_modules))
            else:
                self._fuse_norm_linear(child)
        return module

    def optimize_for_mobile(self, model):
        model.eval()
        model = self._fuse_norm_linear(model)
        qmodel = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        scripted = torch.jit.script(qmodel)
        return torch.utils.mobile_optimizer.optimize_for_mobile(scripted)


class MobileMemoryManager(nn.Module):
    def __init__(self, d_model, device_buffer_size=256):
        super().__init__()
        self.d_model = d_model
        self.device_buffer_size = device_buffer_size
        self.device_buffer = nn.Parameter(torch.zeros(device_buffer_size, d_model))
        self.mmap = None
        self.mmap_size = 0

    def set_memory_map(self, path, size):
        num_bytes = size * self.d_model * 4
        if os.path.exists(path):
            storage = torch.UntypedStorage.from_file(path, shared=True, size=num_bytes)
        else:
            storage = torch.UntypedStorage.from_file(path, shared=False, size=num_bytes)
            storage.zero_()
            storage = torch.UntypedStorage.from_file(path, shared=True, size=num_bytes)
        self.mmap = torch.tensor([], dtype=torch.float32).set_(storage).view(size, self.d_model)
        self.mmap_size = size

    def load_to_device(self, indices):
        count = min(len(indices), self.device_buffer_size)
        for i in range(count):
            self.device_buffer.data[i].copy_(self.mmap[indices[i]])

    def evict_from_device(self, indices):
        count = min(len(indices), self.device_buffer_size)
        for i in range(count):
            self.mmap[indices[i]].copy_(self.device_buffer.data[i])

    def extra_repr(self):
        return f'd_model={self.d_model}, buffer_size={self.device_buffer_size}, mmap_size={self.mmap_size}'


class PrunedHead(nn.Module):
    def __init__(self, d_model, vocab_size, rank=256):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.empty(vocab_size, rank))
        self.S = nn.Parameter(torch.empty(rank))
        self.Vh = nn.Parameter(torch.empty(rank, d_model))
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.U)
        nn.init.eye_(self.Vh[:min(self.rank, self.Vh.shape[1]), :])
        nn.init.ones_(self.S)

    def prune(self, weight, rank=None):
        if rank is not None:
            self.rank = min(rank, min(weight.shape[-2], weight.shape[-1]))
            U = nn.Parameter(torch.empty(weight.shape[-2], self.rank))
            S = nn.Parameter(torch.empty(self.rank))
            Vh = nn.Parameter(torch.empty(self.rank, weight.shape[-1]))
            self.U = U
            self.S = S
            self.Vh = Vh
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        with torch.no_grad():
            self.U.data.copy_(U[:, :self.rank])
            self.S.data.copy_(S[:self.rank])
            self.Vh.data.copy_(Vh[:self.rank])

    def forward(self, x):
        out = x @ self.Vh.T
        out = out * self.S
        out = out @ self.U.T
        return out + self.bias
