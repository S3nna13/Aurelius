import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NTMMemory(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M

    def read(self, memory, read_weight):
        return torch.bmm(read_weight.unsqueeze(1), memory).squeeze(1)

    def write(self, memory, write_weight, erase_vector, add_vector):
        erase = torch.bmm(write_weight.unsqueeze(2), erase_vector.unsqueeze(1))
        add = torch.bmm(write_weight.unsqueeze(2), add_vector.unsqueeze(1))
        return memory * (1 - erase) + add

    def reset(self, batch_size, device=None, dtype=None):
        return torch.zeros(batch_size, self.N, self.M, device=device, dtype=dtype)


class NTMReadHead(nn.Module):
    def __init__(self, d_mem, N, shift_radius=1):
        super().__init__()
        self.d_mem = d_mem
        self.N = N
        self.shift_radius = shift_radius
        shift_range = 2 * shift_radius + 1
        self.beta = nn.Linear(d_mem, 1)
        self.gate = nn.Linear(d_mem, 1)
        self.shift = nn.Linear(d_mem, shift_range)
        self.gamma = nn.Linear(d_mem, 1)

    def forward(self, query, key, memory, prev_weight):
        beta = F.softplus(self.beta(query))
        sim = F.cosine_similarity(key.unsqueeze(1), memory, dim=-1)
        w_c = F.softmax(beta * sim, dim=-1)

        g = torch.sigmoid(self.gate(query))
        w_g = g * w_c + (1 - g) * prev_weight

        s = F.softmax(self.shift(query), dim=-1)
        w_shifted = self._shift(w_g, s)

        gamma = 1 + F.softplus(self.gamma(query))
        w_sharp = w_shifted ** gamma
        return w_sharp / (w_sharp.sum(dim=-1, keepdim=True) + 1e-16)

    def _shift(self, w, s):
        pad = s.size(-1) // 2
        if pad == 0:
            return w
        w_padded = torch.cat([w[:, -pad:], w, w[:, :pad]], dim=-1)
        w_unfolded = w_padded.unfold(-1, s.size(-1), 1)
        return (w_unfolded * s.unsqueeze(1)).sum(-1)


class NTMWriteHead(nn.Module):
    def __init__(self, d_mem, N, shift_radius=1):
        super().__init__()
        self.d_mem = d_mem
        self.N = N
        self.shift_radius = shift_radius
        shift_range = 2 * shift_radius + 1
        self.beta = nn.Linear(d_mem, 1)
        self.gate = nn.Linear(d_mem, 1)
        self.shift = nn.Linear(d_mem, shift_range)
        self.gamma = nn.Linear(d_mem, 1)
        self.erase = nn.Linear(d_mem, d_mem)
        self.add = nn.Linear(d_mem, d_mem)

    def forward(self, query, key, memory, prev_weight):
        beta = F.softplus(self.beta(query))
        sim = F.cosine_similarity(key.unsqueeze(1), memory, dim=-1)
        w_c = F.softmax(beta * sim, dim=-1)

        g = torch.sigmoid(self.gate(query))
        w_g = g * w_c + (1 - g) * prev_weight

        s = F.softmax(self.shift(query), dim=-1)
        w_shifted = self._shift(w_g, s)

        gamma = 1 + F.softplus(self.gamma(query))
        write_weight = w_shifted ** gamma
        write_weight = write_weight / (write_weight.sum(dim=-1, keepdim=True) + 1e-16)

        erase_vector = torch.sigmoid(self.erase(query))
        add_vector = self.add(query)
        return write_weight, erase_vector, add_vector

    def _shift(self, w, s):
        pad = s.size(-1) // 2
        if pad == 0:
            return w
        w_padded = torch.cat([w[:, -pad:], w, w[:, :pad]], dim=-1)
        w_unfolded = w_padded.unfold(-1, s.size(-1), 1)
        return (w_unfolded * s.unsqueeze(1)).sum(-1)


class NTMController(nn.Module):
    def __init__(self, d_controller, d_mem, N, shift_radius=1):
        super().__init__()
        self.d_controller = d_controller
        self.d_mem = d_mem
        self.N = N
        self.memory = NTMMemory(N, d_mem)
        self.read_head = NTMReadHead(d_mem, N, shift_radius)
        self.write_head = NTMWriteHead(d_mem, N, shift_radius)
        self.read_query = nn.Linear(d_controller, d_mem)
        self.read_key = nn.Linear(d_controller, d_mem)
        self.write_query = nn.Linear(d_controller, d_mem)
        self.write_key = nn.Linear(d_controller, d_mem)

    def forward(self, controller_output, memory_state):
        if isinstance(memory_state, tuple):
            memory, prev_read_weight, prev_write_weight = memory_state
        else:
            memory = memory_state
            batch_size = memory.shape[0]
            prev_read_weight = torch.zeros(batch_size, self.N, device=memory.device, dtype=memory.dtype)
            prev_write_weight = torch.zeros(batch_size, self.N, device=memory.device, dtype=memory.dtype)

        rq = self.read_query(controller_output)
        rk = self.read_key(controller_output)
        read_weight = self.read_head(rq, rk, memory, prev_read_weight)

        wq = self.write_query(controller_output)
        wk = self.write_key(controller_output)
        write_weight, erase_vector, add_vector = self.write_head(wq, wk, memory, prev_write_weight)

        read_values = self.memory.read(memory, read_weight)
        updated_memory = self.memory.write(memory, write_weight, erase_vector, add_vector)

        return read_values, (updated_memory, read_weight, write_weight)


class DifferentiableMemoryAugmentedBlock(nn.Module):
    def __init__(self, d_model, d_mem, N, shift_radius=1):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.N = N
        self.controller = NTMController(d_model, d_mem, N, shift_radius)
        self.output = nn.Linear(d_model + d_mem, d_model)

    def forward(self, x, memory_state):
        read_values, updated_memory_state = self.controller(x, memory_state)
        return self.output(torch.cat([x, read_values], dim=-1)), updated_memory_state

    def init_state(self, batch_size, device=None, dtype=None):
        memory = torch.zeros(batch_size, self.N, self.d_mem, device=device, dtype=dtype)
        prev_read_weight = torch.zeros(batch_size, self.N, device=device, dtype=dtype)
        prev_read_weight[:, 0] = 1.0
        prev_write_weight = torch.zeros(batch_size, self.N, device=device, dtype=dtype)
        prev_write_weight[:, 0] = 1.0
        return memory, prev_read_weight, prev_write_weight
