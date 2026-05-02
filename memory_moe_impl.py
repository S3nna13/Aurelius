import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyTopKRouter(nn.Module):
    def __init__(self, d_model, n_experts=8, top_k=2, noise_std=0.01):
        super().__init__()
        self.top_k = top_k
        self.noise_std = noise_std
        self.n_experts = n_experts
        self.W_gate = nn.Parameter(torch.randn(d_model, n_experts) * 0.02)
        self.W_noise = nn.Parameter(torch.randn(d_model, n_experts) * 0.02)

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, x.size(-1))
        h = x_flat @ self.W_gate
        noise = x_flat @ self.W_noise
        noise = F.softplus(noise) * self.noise_std
        if self.training:
            noisy_logits = h + torch.randn_like(h) * noise
        else:
            noisy_logits = h
        top_k_logits, top_k_indices = noisy_logits.topk(self.top_k, dim=-1)
        dispatch_weights = F.softmax(top_k_logits, dim=-1)
        dispatch_weights = dispatch_weights.view(*orig_shape[:-1], self.top_k)
        top_k_indices = top_k_indices.view(*orig_shape[:-1], self.top_k)

        probs = F.softmax(h, dim=-1)
        probs = probs.view(*orig_shape[:-1], self.n_experts)

        mask = F.one_hot(top_k_indices, num_classes=self.n_experts).float()
        expert_util = mask.view(-1, self.n_experts).sum(0)
        importance = probs.view(-1, self.n_experts).sum(0)

        load = expert_util
        cv = (load.std() / (load.mean() + 1e-6) +
              importance.std() / (importance.mean() + 1e-6))
        auxiliary_loss = cv * 0.01

        return dispatch_weights, top_k_indices, auxiliary_loss


class ExpertChoiceRouter(nn.Module):
    def __init__(self, d_model, n_experts=8, capacity_factor=1.25):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, n_experts),
        )

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, x.size(-1))
        N = x_flat.size(0)
        scores = self.router(x_flat)
        capacity = math.ceil(self.capacity_factor * N / self.n_experts)

        top_scores, top_indices = torch.topk(scores.transpose(0, 1), capacity, dim=1)
        dispatched_tokens = torch.zeros(self.n_experts, capacity, x_flat.size(-1), device=x.device, dtype=x.dtype)
        for e in range(self.n_experts):
            idx = top_indices[e]
            dispatched_tokens[e] = x_flat[idx]

        expert_to_token_assignment = torch.zeros(self.n_experts, capacity, N, device=x.device, dtype=x.dtype)
        for e in range(self.n_experts):
            idx = top_indices[e]
            expert_to_token_assignment[e, torch.arange(capacity, device=x.device), idx] = 1.0

        probs = F.softmax(scores, dim=-1)
        importance = probs.sum(0)
        load = (expert_to_token_assignment.sum(dim=(1, 2)) + 1e-6)
        load_loss = load.var() / (load.mean() ** 2 + 1e-6)
        importance_loss = importance.var() / (importance.mean() ** 2 + 1e-6)

        return dispatched_tokens, expert_to_token_assignment, top_scores, top_indices, load_loss, importance_loss


class AttentionPoolCompressor(nn.Module):
    def __init__(self, d_model, compress_ratio=2):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, chunk):
        L, d = chunk.shape
        ratio = self.compress_ratio
        L_out = max(L // ratio, 1)
        if L_out == 0:
            return chunk[:0]

        chunks_list = chunk.chunk(L_out, dim=0)
        outputs = []
        for c in chunks_list:
            c_len = c.size(0)
            q = self.query.expand(1, -1, -1)
            c_t = c.unsqueeze(0)
            attn = F.softmax(q @ c_t.transpose(1, 2) / math.sqrt(d), dim=-1)
            out = attn @ c_t
            out = self.proj(out.squeeze(0))
            outputs.append(out)

        return torch.cat(outputs, dim=0)


class CompressiveMemory(nn.Module):
    def __init__(self, d_model, mem_len=256, cmem_len=128, compress_ratio=2):
        super().__init__()
        self.d_model = d_model
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.compress_ratio = compress_ratio

        self.compressor = AttentionPoolCompressor(d_model, compress_ratio)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.cross_k_proj = nn.Linear(d_model, d_model)
        self.cross_v_proj = nn.Linear(d_model, d_model)
        self.cmem_k_proj = nn.Linear(d_model, d_model)
        self.cmem_v_proj = nn.Linear(d_model, d_model)

    def compress(self, fine_mem):
        compressed_chunks = []
        for chunk in fine_mem:
            compressed = self.compressor(chunk)
            compressed_chunks.append(compressed)
        return compressed_chunks

    def forward(self, x, memory_state):
        fine_mem, cmem_seq = memory_state
        B, S, D = x.shape

        fine_keys = []
        fine_vals = []
        for i in range(len(fine_mem)):
            m = fine_mem[i]
            fine_keys.append(self.k_proj(m))
            fine_vals.append(self.v_proj(m))

        cmem_keys = []
        cmem_vals = []
        for i in range(len(cmem_seq)):
            cmem_keys.append(self.cmem_k_proj(cmem_seq[i]))
            cmem_vals.append(self.cmem_v_proj(cmem_seq[i]))

        reconstruction_loss = 0.0
        new_fine_mem = list(fine_mem)
        new_cmem_seq = list(cmem_seq)

        if len(new_fine_mem) > 0:
            cumsum = 0
            chunks_to_compress = []
            for i, m in enumerate(new_fine_mem):
                cumsum += m.size(0)
                chunks_to_compress.append(m)
                if cumsum >= self.mem_len * 0.5:
                    break
            if len(chunks_to_compress) > 0:
                combined = torch.cat(chunks_to_compress, dim=0)
                compressed = self.compressor(combined)
                decoded = F.interpolate(
                    compressed.unsqueeze(0).transpose(1, 2),
                    size=combined.size(0),
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2).squeeze(0)
                min_len = min(decoded.size(0), combined.size(0))
                reconstruction_loss = F.mse_loss(decoded[:min_len], combined[:min_len])
                new_cmem_seq.append(compressed.detach())
                for _ in range(len(chunks_to_compress)):
                    if len(new_fine_mem) > 0:
                        new_fine_mem.pop(0)

        while len(new_cmem_seq) > self.cmem_len:
            new_cmem_seq.pop(0)

        fine_k = torch.stack(fine_keys, dim=0) if fine_keys else None
        fine_v = torch.stack(fine_vals, dim=0) if fine_vals else None
        cmem_k = torch.stack(cmem_keys, dim=0) if cmem_keys else None
        cmem_v = torch.stack(cmem_vals, dim=0) if cmem_vals else None

        query = self.cross_k_proj(x)
        output = x

        if fine_k is not None or cmem_k is not None:
            all_keys = []
            all_vals = []
            if fine_k is not None:
                fk = fine_k.view(-1, D)
                fv = fine_v.view(-1, D)
                all_keys.append(fk)
                all_vals.append(fv)
            if cmem_k is not None:
                ck = cmem_k.view(-1, D)
                cv = cmem_v.view(-1, D)
                all_keys.append(ck)
                all_vals.append(cv)
            if all_keys:
                mem_k = torch.cat(all_keys, dim=0).unsqueeze(0).expand(B, -1, -1)
                mem_v = torch.cat(all_vals, dim=0).unsqueeze(0).expand(B, -1, -1)
                attn_scores = torch.bmm(query, mem_k.transpose(1, 2)) / math.sqrt(D)
                attn_weights = F.softmax(attn_scores, dim=-1)
                mem_out = torch.bmm(attn_weights, mem_v)
                output = x + mem_out

        return output, reconstruction_loss, (new_fine_mem, new_cmem_seq)


class KVStore:
    def __init__(self, max_size=65536):
        self.keys = []
        self.values = []
        self.max_size = max_size

    def insert(self, key, value):
        self.keys.append(key.detach().cpu())
        self.values.append(value.detach().cpu())
        total = sum(k.size(0) for k in self.keys)
        while total > self.max_size and len(self.keys) > 0:
            removed = self.keys.pop(0)
            total -= removed.size(0)
            self.values.pop(0)

    def search(self, query, k=32):
        if len(self.keys) == 0:
            return torch.zeros_like(query)

        all_keys = torch.cat(self.keys, dim=0).to(query.device)
        all_vals = torch.cat(self.values, dim=0).to(query.device)

        B, S, D = query.shape
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(all_keys, dim=-1)

        sims = q_norm.view(-1, D) @ k_norm.t()
        actual_k = min(k, all_keys.size(0))
        top_sims, top_idx = sims.topk(actual_k, dim=-1)

        top_vals = all_vals[top_idx]
        weights = F.softmax(top_sims / 0.1, dim=-1)
        result = (weights.unsqueeze(-1) * top_vals).sum(dim=1)
        result = result.view(B, S, D)
        return result


class KNNAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, k=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.k = k

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.knn_key = nn.Linear(d_model, d_model, bias=False)
        self.knn_val = nn.Linear(d_model, d_model, bias=False)
        self.knn_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, 1)
        self.dropout = nn.Dropout(dropout)

        self.store = KVStore()

    def forward(self, x, kv_store=None):
        B, S, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        q = self.W_q(x).view(B, S, H, hd).transpose(1, 2)
        k = self.W_k(x).view(B, S, H, hd).transpose(1, 2)
        v = self.W_v(x).view(B, S, H, hd).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.W_o(attn_out)

        knn_q = self.knn_key(x)
        store = kv_store if kv_store is not None else self.store
        knn_result = store.search(knn_q, k=self.k)
        knn_out = self.knn_proj(knn_result)

        gate_input = torch.cat([attn_out, knn_out], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        output = gate * attn_out + (1 - gate) * knn_out

        with torch.no_grad():
            store_keys = self.knn_key(x).view(-1, D)
            store_vals = self.knn_val(x).view(-1, D)
            store.insert(store_keys, store_vals)

        return output


class MoEMemoryLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, n_experts=8, capacity_factor=1.25,
                 mem_len=256, cmem_len=128, compress_ratio=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.dropout_attn = nn.Dropout(dropout)

        self.router = ExpertChoiceRouter(d_model, n_experts, capacity_factor)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

        self.memory = CompressiveMemory(d_model, mem_len, cmem_len, compress_ratio)
        self.mem_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _self_attention(self, x):
        B, S, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        q = self.W_q(x).view(B, S, H, hd).transpose(1, 2)
        k = self.W_k(x).view(B, S, H, hd).transpose(1, 2)
        v = self.W_v(x).view(B, S, H, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)

    def forward(self, x, memory_state=None):
        B, S, D = x.shape

        attn_out = self._self_attention(self.norm1(x))
        h = x + attn_out

        dispatched_tokens, assignment, top_scores, top_indices, load_loss, importance_loss = self.router(self.norm2(h))
        N_experts, capacity, _ = dispatched_tokens.shape

        expert_outputs = []
        for e in range(N_experts):
            tokens = dispatched_tokens[e, :capacity]
            valid_mask = (top_scores[e] > -1e9)
            valid_tokens = tokens[valid_mask]
            if valid_tokens.size(0) > 0:
                expert_out = self.experts[e](valid_tokens)
                padded = torch.zeros(capacity, D, device=x.device, dtype=x.dtype)
                padded[valid_mask] = expert_out
            else:
                padded = torch.zeros(capacity, D, device=x.device, dtype=x.dtype)
            expert_outputs.append(padded)

        expert_outputs = torch.stack(expert_outputs)

        assign_float = assignment.float()
        assign_sum = assign_float.sum(dim=(0, 1), keepdim=True).clamp(min=1)
        combined = torch.einsum('nct,ncd->td', assign_float, expert_outputs)
        combined = combined / assign_sum.squeeze(0).squeeze(0).unsqueeze(-1).clamp(min=1)
        combined = combined.view(B, S, D)
        h = h + combined

        if memory_state is not None:
            mem_out, recon_loss, new_mem_state = self.memory(self.norm3(h), memory_state)
        else:
            fine_mem = []
            cmem_seq = []
            with torch.no_grad():
                for i in range(min(S, 64)):
                    fine_mem.append(h[0:1, i:i+1, :].squeeze(0))
            mem_out, recon_loss, new_mem_state = self.memory(self.norm3(h), (fine_mem, cmem_seq))

        h = h + self.mem_proj(mem_out)

        aux_loss = load_loss + importance_loss + 0.1 * recon_loss

        return h, aux_loss, new_mem_state
