import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Generator, Tuple
from contextlib import contextmanager
from nn_utils import sample_with_top_p_top_k, validate_input_ids
import logging
logger = logging.getLogger(__name__)


class KVCacheManager:
    def __init__(self, n_layers, n_heads, head_dim, max_batch_size=1, max_seq_len=4096, device='cpu', dtype=torch.float32):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.key_cache = torch.zeros(
            n_layers, max_batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float16,
        )
        self.value_cache = torch.zeros(
            n_layers, max_batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float16,
        )
        self._occupied = 0

    def update(self, layer_idx, key, value, seq_pos):
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")
        if seq_pos < 0 or seq_pos >= self.max_seq_len:
            raise IndexError(f"seq_pos {seq_pos} out of range [0, {self.max_seq_len})")

        if key.dim() == 3:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        b = key.shape[0]
        seq_len = key.shape[2]

        start = seq_pos
        end = seq_pos + seq_len
        if end > self.max_seq_len:
            end = self.max_seq_len
            seq_len = end - start

        self.key_cache[layer_idx, :b, :, start:end, :] = key[:, :, :seq_len, :]
        self.value_cache[layer_idx, :b, :, start:end, :] = value[:, :, :seq_len, :]

        new_occupied = end
        if new_occupied > self._occupied:
            self._occupied = new_occupied

    def get(self, layer_idx, seq_len=None):
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")

        if seq_len is None:
            seq_len = self._occupied
        seq_len = min(seq_len, self.max_seq_len)
        if seq_len <= 0:
            return (
                torch.zeros(self.max_batch_size, self.n_heads, 0, self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(self.max_batch_size, self.n_heads, 0, self.head_dim, device=self.device, dtype=self.dtype),
            )

        k = self.key_cache[layer_idx, :, :, :seq_len, :].clone()
        v = self.value_cache[layer_idx, :, :, :seq_len, :].clone()
        return k, v

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self._occupied = 0

    def roll(self, n):
        if n <= 0 or self._occupied == 0:
            return
        n = min(n, self._occupied)

        shifted_k = torch.roll(self.key_cache, shifts=-n, dims=3)
        shifted_v = torch.roll(self.value_cache, shifts=-n, dims=3)
        self.key_cache.copy_(shifted_k)
        self.value_cache.copy_(shifted_v)

        self.key_cache[:, :, :, self.max_seq_len - n:, :] = 0
        self.value_cache[:, :, :, self.max_seq_len - n:, :] = 0

        self._occupied = max(0, self._occupied - n)

    @property
    def occupied(self):
        return self._occupied

    @property
    def capacity(self):
        return self.max_seq_len

    @property
    def memory_mb(self):
        element_bytes = self.key_cache.element_size()
        total_elements = (self.key_cache.numel() + self.value_cache.numel())
        return total_elements * element_bytes / (1024 * 1024)


class SpeculativeDecoder:
    def __init__(self, target_model, draft_model=None, max_spec_tokens=5, accept_threshold=0.1):
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_spec_tokens = max_spec_tokens
        self.accept_threshold = accept_threshold

    def _get_model_config(self, model):
        if hasattr(model, 'config'):
            return model.config
        return {}

    def _model_forward(self, model, input_ids):
        try:
            out = model(input_ids)
            if isinstance(out, dict):
                return out['logits']
            return out
        except (RuntimeError, TypeError) as e:
            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                raise
            pass
        if hasattr(model, 'blocks'):
            h = model.token_embedding(input_ids)
            cos, sin = model.rotary(h)
            b, t = input_ids.shape
            causal_mask = torch.tril(torch.ones(t, t, device=input_ids.device))
            causal_mask = causal_mask.view(1, 1, t, t)
            for block in model.blocks:
                if hasattr(block, 'has_agent') and block.has_agent:
                    h = block(h, cos, sin, causal_mask, None)
                else:
                    h = block(h, cos, sin, causal_mask)
            h = model.norm_out(h)
            logits = model.lm_head(h)
            return logits
        return model(input_ids)

    def decode_step(self, input_ids, temperature=0.8, top_p=0.9):
        model = self.target_model
        with torch.no_grad():
            logits = self._model_forward(model, input_ids)
            next_token = sample_with_top_p_top_k(logits[:, -1, :], temperature, top_k=0, top_p=top_p)
        return next_token

    def speculative_generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_p=0.9):
        if self.draft_model is None:
            return self._autoregressive_generate(input_ids, max_new_tokens, temperature, top_p)

        output = input_ids.clone()
        generated = 0
        vocab_size = self._get_vocab_size(self.target_model)

        while generated < max_new_tokens:
            gamma = min(self.max_spec_tokens, max_new_tokens - generated)

            with torch.no_grad():
                draft_tokens = self._generate_draft(output, gamma, temperature, vocab_size)

            draft_input = torch.cat([output, draft_tokens], dim=1)

            with torch.no_grad():
                target_logits = self._model_forward(self.target_model, draft_input)
                draft_logits = self._model_forward(self.draft_model, draft_input)

            target_probs = F.softmax(target_logits / max(temperature, 1e-8), dim=-1)
            draft_probs = F.softmax(draft_logits / max(temperature, 1e-8), dim=-1)

            accepted_count = 0
            cur_len = output.shape[1]

            for i in range(gamma):
                pos = cur_len + i - 1
                if pos < 0:
                    continue
                token = draft_tokens[:, i]

                p_target = target_probs[:, pos, :].gather(1, token.unsqueeze(-1)).squeeze(-1)
                p_draft = draft_probs[:, pos, :].gather(1, token.unsqueeze(-1)).squeeze(-1)

                ratio = torch.where(
                    p_draft > 1e-10,
                    p_target / p_draft,
                    torch.zeros_like(p_target),
                )
                rand = torch.rand_like(ratio)

                if (rand < ratio).all():
                    accepted_count += 1
                else:
                    residual = torch.clamp(target_probs[:, pos, :] - draft_probs[:, pos, :], min=0)
                    residual_sum = residual.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                    residual = residual / residual_sum
                    resampled = torch.multinomial(residual, num_samples=1)
                    output = torch.cat([output, draft_tokens[:, :accepted_count], resampled], dim=1)
                    generated += accepted_count + 1
                    break
            else:
                last_pos = cur_len + gamma - 1
                if last_pos >= 0:
                    residual = torch.clamp(target_probs[:, last_pos, :] - draft_probs[:, last_pos, :], min=0)
                    residual_sum = residual.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                    residual = residual / residual_sum
                    bonus = torch.multinomial(residual, num_samples=1)
                    output = torch.cat([output, draft_tokens, bonus], dim=1)
                    generated += gamma + 1
                else:
                    output = torch.cat([output, draft_tokens], dim=1)
                    generated += gamma

        if output.shape[1] > input_ids.shape[1] + max_new_tokens:
            output = output[:, :input_ids.shape[1] + max_new_tokens]

        return output

    def _generate_draft(self, input_ids, gamma, temperature, vocab_size):
        draft_tokens = []
        cur = input_ids
        for _ in range(gamma):
            with torch.no_grad():
                logits = self._model_forward(self.draft_model, cur)[:, -1, :]
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)
            draft_tokens.append(token)
            cur = torch.cat([cur, token], dim=1)
        return torch.cat(draft_tokens, dim=1)

    def _autoregressive_generate(self, input_ids, max_new_tokens, temperature, top_p):
        output = input_ids.clone()
        for _ in range(max_new_tokens):
            next_token = self.decode_step(output, temperature, top_p)
            output = torch.cat([output, next_token], dim=1)
        return output

    def _get_vocab_size(self, model):
        if hasattr(model, 'config') and 'vocab_size' in model.config:
            return model.config['vocab_size']
        if hasattr(model, 'lm_head'):
            return model.lm_head.weight.shape[0]
        if hasattr(model, 'token_embedding'):
            return model.token_embedding.weight.shape[0]
        return 32000


class MemoryEfficientInference:
    def __init__(self, model, dtype=torch.bfloat16, device='cpu'):
        self.model = model
        self.dtype = dtype
        self.device = device
        self._original_weights = {}
        self._quantized = False

    def quantize_weights(self, bits=8):
        if self._quantized:
            return
        self._original_weights = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                qmax = 127.0 if bits >= 8 else 7.0
                qmin = -128 if bits >= 8 else -8
                scale = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
                q_weight = (weight / scale).round().clamp(qmin, qmax)
                scaled = q_weight.float() * scale
                param = nn.Parameter(scaled)
                param._quant_scale = scale.squeeze(-1)
                param._quantized = True
                param._q_weight = q_weight.to(torch.int8)
                module.weight = param
        self._quantized = True

    def dequantize_weights(self):
        if not self._quantized:
            return
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module.weight, '_q_weight'):
                q = module.weight._q_weight
                scale = module.weight._quant_scale
                recovered = q.float() * scale.unsqueeze(-1)
                module.weight = nn.Parameter(recovered)
                for attr in ('_quant_scale', '_quantized', '_q_weight'):
                    if hasattr(module.weight, attr):
                        delattr(module.weight, attr)
        self._quantized = False

    @contextmanager
    def _temp_device(self, tensor_device):
        yield

    def batch_generate(self, prompts, max_new_tokens=100, **kwargs):
        model = self.model
        model.eval()

        pad_id = kwargs.get('pad_token_id', 0)
        temperature = kwargs.get('temperature', 0.8)
        top_p = kwargs.get('top_p', 0.9)

        max_len = max(p.shape[1] for p in prompts)
        padded = []
        attention_masks = []
        for p in prompts:
            pad_length = max_len - p.shape[1]
            if pad_length > 0:
                padding = torch.full((p.shape[0], pad_length), pad_id, dtype=p.dtype, device=p.device)
                padded.append(torch.cat([p, padding], dim=1))
                mask = torch.cat([
                    torch.ones(p.shape, dtype=torch.long, device=p.device),
                    torch.zeros(p.shape[0], pad_length, dtype=torch.long, device=p.device),
                ], dim=1)
            else:
                padded.append(p)
                mask = torch.ones(p.shape, dtype=torch.long, device=p.device)
            attention_masks.append(mask)

        batch_ids = torch.cat(padded, dim=0)
        batch_mask = torch.cat(attention_masks, dim=0)

        cur_len = max_len
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self._forward_model(model, batch_ids)

            next_token = sample_with_top_p_top_k(logits[:, -1, :], temperature, top_k=0, top_p=top_p)
            batch_ids = torch.cat([batch_ids, next_token], dim=1)
            batch_mask = torch.cat([
                batch_mask,
                torch.ones(batch_mask.shape[0], 1, dtype=torch.long, device=batch_ids.device),
            ], dim=1)
            cur_len += 1

        results = []
        batch_idx = 0
        for i, original_prompt in enumerate(prompts):
            orig_len = original_prompt.shape[1]
            generated = batch_ids[batch_idx, orig_len:].unsqueeze(0)
            results.append(torch.cat([original_prompt, generated[:, :max_new_tokens]], dim=1))
            batch_idx += 1

        return results

    def stream_generate(self, input_ids, max_new_tokens=100, yield_every=1, **kwargs):
        model = self.model
        model.eval()

        temperature = kwargs.get('temperature', 0.8)
        top_p = kwargs.get('top_p', 0.9)

        output = input_ids.clone()
        generated_tokens = []

        for i in range(max_new_tokens):
            with torch.no_grad():
                logits = self._forward_model(model, output)

            next_token = sample_with_top_p_top_k(logits[:, -1, :], temperature, top_k=0, top_p=top_p)

            output = torch.cat([output, next_token], dim=1)
            generated_tokens.append(next_token)

            if len(generated_tokens) % yield_every == 0 or i == max_new_tokens - 1:
                yield torch.cat(generated_tokens, dim=1)

        yield torch.cat(generated_tokens, dim=1)

    def profile_memory(self):
        result = {}
        total_params = 0
        total_bytes = 0

        component_bytes = {}
        component_params = {}

        for name, param in self.model.named_parameters():
            top_key = name.split('.')[0]
            nbytes = param.numel() * param.element_size()
            component_bytes[top_key] = component_bytes.get(top_key, 0) + nbytes
            component_params[top_key] = component_params.get(top_key, 0) + param.numel()
            total_params += param.numel()
            total_bytes += nbytes

        for name, buf in self.model.named_buffers():
            top_key = name.split('.')[0] if '.' in name else name
            nbytes = buf.numel() * buf.element_size()
            component_bytes[top_key] = component_bytes.get(top_key, 0) + nbytes

        result['total_params'] = total_params
        result['total_bytes'] = total_bytes
        result['total_mb'] = total_bytes / (1024 * 1024)
        result['components'] = {}

        for key in component_bytes:
            result['components'][key] = {
                'params': component_params.get(key, 0),
                'bytes': component_bytes[key],
                'mb': component_bytes[key] / (1024 * 1024),
            }

        if torch.cuda.is_available():
            result['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            result['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)

        import psutil
        import os
        process = psutil.Process(os.getpid())
        result['cpu_rss_mb'] = process.memory_info().rss / (1024 * 1024)

        return result

    def _forward_model(self, model, input_ids):
        try:
            out = model(input_ids)
            if isinstance(out, dict):
                return out['logits']
            return out
        except (RuntimeError, TypeError) as e:
            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                raise
            pass
        if hasattr(model, 'blocks') and hasattr(model, 'rotary'):
            h = model.token_embedding(input_ids)
            cos, sin = model.rotary(h)
            b, t = input_ids.shape
            causal_mask = torch.tril(torch.ones(t, t, device=input_ids.device))
            causal_mask = causal_mask.view(1, 1, t, t)
            for block in model.blocks:
                if hasattr(block, 'has_agent') and block.has_agent:
                    h = block(h, cos, sin, causal_mask, None)
                else:
                    h = block(h, cos, sin, causal_mask)
            h = model.norm_out(h)
            return model.lm_head(h)
        out = model(input_ids)
        if isinstance(out, dict):
            return out['logits']
        return out

    def _get_model_config(self, model):
        if hasattr(model, 'config'):
            return model.config
        return {}


class PagedAttention:
    def __init__(self, n_heads, head_dim, block_size=16, max_blocks=4096, device='cpu'):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device

        self.k_cache = torch.zeros(
            max_blocks, n_heads, block_size, head_dim,
            device=device, dtype=torch.float32,
        )
        self.v_cache = torch.zeros(
            max_blocks, n_heads, block_size, head_dim,
            device=device, dtype=torch.float32,
        )

        self.free_blocks = list(range(max_blocks))
        self.block_tables = {}
        self.seq_block_count = {}
        self._block_ref_count = [0] * max_blocks

    def allocate(self, seq_len):
        if len(self.free_blocks) == 0 and seq_len > 0:
            raise RuntimeError("PagedAttention: no free blocks available")

        n_blocks_needed = math.ceil(seq_len / self.block_size)
        n_blocks_needed = min(n_blocks_needed, len(self.free_blocks))

        allocated = []
        for _ in range(n_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)
            self._block_ref_count[block_id] = 1

        seq_id = id(allocated)
        self.block_tables[seq_id] = allocated
        self.seq_block_count[seq_id] = len(allocated)

        return allocated

    def free(self, block_ids):
        for block_id in block_ids:
            if block_id < 0 or block_id >= self.max_blocks:
                continue
            self._block_ref_count[block_id] = max(0, self._block_ref_count[block_id] - 1)
            if self._block_ref_count[block_id] == 0:
                self.k_cache[block_id].zero_()
                self.v_cache[block_id].zero_()
                self.free_blocks.append(block_id)

        to_remove = []
        for seq_id, blocks in self.block_tables.items():
            remaining = [b for b in blocks if self._block_ref_count[b] > 0]
            if remaining:
                self.block_tables[seq_id] = remaining
            else:
                to_remove.append(seq_id)

        for seq_id in to_remove:
            del self.block_tables[seq_id]
            if seq_id in self.seq_block_count:
                del self.seq_block_count[seq_id]

    def forward(self, query, key, value, block_table):
        b = query.shape[0]
        if query.dim() == 3:
            q_heads = query.shape[1] if query.shape[1] == self.n_heads else query.shape[2] // self.head_dim
            if query.shape[1] == self.n_heads:
                pass
            else:
                query = query.view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)

        seq_len = query.shape[1] if query.dim() == 3 else query.shape[2]
        n_q_heads = query.shape[1] if query.dim() == 3 else query.shape[1]

        if isinstance(block_table, list):
            active_blocks = block_table
        elif isinstance(block_table, torch.Tensor):
            active_blocks = block_table.tolist()
        else:
            active_blocks = []

        if len(active_blocks) == 0:
            return torch.zeros_like(query)

        for i, block_id in enumerate(active_blocks):
            start = i * self.block_size
            end = start + min(self.block_size, seq_len - start)
            k_block = key[:, start:end] if key.dim() == 3 else key[:, :, start:end, :]
            v_block = value[:, start:end] if value.dim() == 3 else value[:, :, start:end, :]
            if k_block.dim() == 3:
                k_block = k_block.unsqueeze(1)
                v_block = v_block.unsqueeze(1)
            actual_len = k_block.shape[-2] if k_block.dim() == 4 else k_block.shape[-2]
            self.k_cache[block_id, :, :actual_len, :] = k_block.squeeze(1) if k_block.dim() == 4 else k_block
            self.v_cache[block_id, :, :actual_len, :] = v_block.squeeze(1) if v_block.dim() == 4 else v_block

        k_pages = self.k_cache[active_blocks]
        v_pages = self.v_cache[active_blocks]
        k_full = k_pages.reshape(self.n_heads, -1, self.head_dim).unsqueeze(0)
        v_full = v_pages.reshape(self.n_heads, -1, self.head_dim).unsqueeze(0)

        kv_len = k_full.shape[2]

        if query.dim() == 3:
            q = query.unsqueeze(1)
        else:
            q = query

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k_full.transpose(-2, -1)) * scale

        mask = torch.tril(
            torch.ones(seq_len, kv_len, device=self.device),
            diagonal=kv_len - seq_len,
        )
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v_full)
        if query.dim() == 3:
            return out.squeeze(1)
        return out

    def defrag(self):
        active_seqs = list(self.block_tables.items())
        new_k_cache = torch.zeros_like(self.k_cache)
        new_v_cache = torch.zeros_like(self.v_cache)
        new_free = list(range(self.max_blocks))
        new_block_count = 0

        new_block_tables = {}
        used_blocks = []

        for seq_id, blocks in active_seqs:
            new_blocks = list(range(new_block_count, new_block_count + len(blocks)))
            for new_id, old_id in zip(new_blocks, blocks):
                new_k_cache[new_id] = self.k_cache[old_id]
                new_v_cache[new_id] = self.v_cache[old_id]
                used_blocks.append(new_id)
            new_block_tables[seq_id] = new_blocks
            new_block_count += len(blocks)

        self.k_cache.copy_(new_k_cache)
        self.v_cache.copy_(new_v_cache)
        self.block_tables = new_block_tables

        self.free_blocks = []
        for i in range(self.max_blocks):
            if i >= new_block_count:
                self.free_blocks.append(i)
            self._block_ref_count[i] = 1 if i < new_block_count else 0

        self.k_cache[new_block_count:].zero_()
        self.v_cache[new_block_count:].zero_()