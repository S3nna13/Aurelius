# Security Audit: Aurelius AI Model

**Date:** 2026-04-28  
**Scope:** Python (`memory_core.py`, `async_memory.py`, `hierarchical_kv_cache.py`, `agent_loop.py`, `fp8_allreduce.py`, `adaptive_precision.py`, `ntm_memory.py`, `paged_optimizer.py`, `kv_cache_quant.py`, `deduplication.py`, `train_optimized.py`) and Rust (`rust_memory/src/lib.rs`, `rust_memory/src/checkpoint.rs`)  
**Methodology:** Trail of Bits — treat all inputs as hostile, assume all unsafe blocks are guilty until proven safe.

---

## 1. Memory Safety in the Rust Crate

### 1.1 — Missing `#[repr(C)]` on `CheckpointHeader` in `checkpoint.rs`

**File:** `rust_memory/src/checkpoint.rs:9-17`

```rust
#[derive(Clone, Debug)]
struct CheckpointHeader {
    magic: [u8; 8],
    version: u32,
    num_tensors: u32,
    total_bytes: u64,
    step: u64,
    timestamp: u64,
}
```

This struct is used with `std::mem::transmute` (line 63) to reinterpret it as `&[u8; size_of::<CheckpointHeader>()]` for direct disk I/O. **It lacks `#[repr(C)]`.**

Without `#[repr(C)]`, the Rust compiler is free to reorder fields and insert padding. The resulting byte stream is layout-dependent, meaning:
- A checkpoint written by debug builds may be unreadable by release builds (or vice versa).
- Alignment padding bytes may leak uninitialized stack data into the checkpoint file.
- The `size_of` value could differ between compilation targets, causing truncated or over-long writes.

Compare with `rust_memory/src/lib.rs:196-204`, which correctly adds `#[repr(C)]` to its copy of `CheckpointHeader`.

**Severity:** Medium  
**Mitigation:** Add `#[repr(C)]` to `CheckpointHeader` in `checkpoint.rs`. Delete the duplicate definition in `lib.rs` and share a single definition across both modules.

---

### 1.2 — `transmute` of a local stack variable

**File:** `rust_memory/src/checkpoint.rs:63-64` (and `lib.rs:235-236`, `lib.rs:276-277`)

```rust
let header_bytes: &[u8; size_of::<CheckpointHeader>()] =
    unsafe { transmute(&header) };
```

`header` is a stack-local `CheckpointHeader`. `transmute` creates a reference `&[u8; 40]` whose lifetime is tied to `header`. The reference is passed to `write_all` in the same scope and is not stored, so it does not dangle. This use is sound.

However, `transmute` from an arbitrary reference to `&[u8; N]` is a code smell. The layout guarantees come from `#[repr(C)]`, which is missing in `checkpoint.rs` (see §1.1).

**Severity:** Low (when `#[repr(C)]` is added)  
**Mitigation:** Replace with `bytemuck::bytes_of(&header)` or `core::slice::from_raw_parts(&header as *const _ as *const u8, size_of::<CheckpointHeader>())`, which make the provenance and size explicit.

---

### 1.3 — `from_raw_parts` for tensor data

**File:** `rust_memory/src/lib.rs:251-253`, `checkpoint.rs:85-86`, `checkpoint.rs:214-216`

```rust
let bytes: &[u8] = unsafe {
    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
};
```

This reinterprets a `Vec<f32>` as `&[u8]` for raw byte I/O. The size calculation `data.len() * 4` assumes `f32` is exactly 4 bytes, which is guaranteed by the Rust specification (`size_of::<f32>() == 4`). The provenance is valid: the original `Vec<f32>` owns the allocation; the raw-bytes slice lives only within the `write_all` scope.

**No vulnerability here.** The slice bounds are correct and the aliasing constraints are satisfied because the write is performed synchronously and the borrow ends immediately.

**Severity:** None  
**Mitigation:** None required. Consider adding `debug_assert_eq!(size_of::<f32>(), 4);` for defense-in-depth.

---

### 1.4 — No validation of integer casts in checkpoint headers

**File:** `rust_memory/src/checkpoint.rs:96`, `lib.rs:259`

`data.len() * 4` is computed as `usize`, then cast via `as u64`. On 32-bit platforms, a tensor with more than 2^32 / 4 elements would silently truncate. The `data` comes from Python via PyO3; a carefully crafted input could cause size truncation.

**Severity:** Low (32-bit platforms are not the primary target; PyO3 Vec<f32> size is bounded by Python's object model)  
**Mitigation:** Use `u64::try_from(data.len()).unwrap()` or `data.len().checked_mul(4).unwrap()` with an explicit error path.

---

## 2. Data Handling — Tensor Corruption Paths

### 2.1 — `DifferentialCheckpointer` writes without header validation

**File:** `rust_memory/src/lib.rs:331-356`, `checkpoint.rs:189-223`

The differential checkpoint format concatenates `(name_len: u32, name: [u8], tensor_size: u64, tensor_data: [u8])` entries with no magic bytes, version, or checksum. A corrupted or truncated file is indistinguishable from valid data. If a read path is later added for this format, it would have no integrity guarantees.

**Severity:** Medium (latent — no read path exists yet, but the format is defined for it)  
**Mitigation:** Add a 4-byte magic prefix and a trailing CRC32 checksum. Validate on read.

---

### 2.2 — `serialize_index` writes untrusted-length name prefix

**File:** `rust_memory/src/lib.rs:299-310`, `checkpoint.rs:154-165`

```rust
bytes.extend_from_slice(&(nb.len() as u32).to_le_bytes());
bytes.extend_from_slice(nb);
```

The name length (`nb.len()`) is written as `u32`. A name longer than 2^32 bytes is impossible in practice (PyO3 strings are bounded), but a malicious checkpoint writer or memory corruption could produce a length field that causes the reader to allocate a huge buffer. This is a standard binary serialization risk.

**Severity:** Low (no reader implemented yet)  
**Mitigation:** When implementing a reader, cap the name length at a reasonable maximum (e.g. 1024 bytes) and reject anything larger.

---

## 3. Buffer Overflows and Bounds Errors

### 3.1 — Negative slot count in `HierarchicalKVCache` eviction logic

**File:** `hierarchical_kv_cache.py:103-104, 141, 182`

```python
need_evict = max(0, t1_n + n_new - self.cap1)
...
n_keep_old = self.cap1 - n_new
```

If `n_new > self.cap1`, then `n_keep_old` becomes **negative**. Python's slice semantics treat negative indices as offsets from the end of the tensor, so `self.t1_k[:, :, :n_keep_old]` would wrap around and select elements from the *end* of the buffer instead of the *beginning*, causing silent data corruption.

The identical pattern exists in `_evict_to_tier2` (line 148) and `_evict_to_tier3` (line 189), with `n_evict` instead of `n_new`.

**Severity:** **High** — this is an off-by-one-class vulnerability that corrupts memory content without any error signal.

**Mitigation:**

```python
# Cap the incoming write at capacity to prevent negative n_keep_old
n_new = min(n_new, self.cap1)
need_evict = max(0, t1_n + n_new - self.cap1)
```

Same fix for `_evict_to_tier2` and `_evict_to_tier3`.

---

### 3.2 — `PagedAttentionCache.alloc_blocks` silently under-allocates

**File:** `kv_cache_quant.py:52-55`

```python
def alloc_blocks(self, n: int) -> list[int]:
    allocated = self.free_blocks[:n]
    self.free_blocks = self.free_blocks[n:]
    return allocated
```

If `n > len(self.free_blocks)`, Python slicing returns a shorter list silently. The caller receives fewer blocks than requested, and downstream writes may use out-of-range block indices against `self.kv_data` (which is `max_blocks × block_size × 2 × n_heads × head_dim`).

**Severity:** Medium  
**Mitigation:**

```python
def alloc_blocks(self, n: int) -> list[int]:
    if n > len(self.free_blocks):
        raise MemoryError(f"requested {n} blocks, only {len(self.free_blocks)} free")
    ...
```

---

### 3.3 — `PagedAttentionCache.free_blocks` shadows the field name

**File:** `kv_cache_quant.py:57-59`

```python
def free_blocks(self, blocks: list[int]):
    self.free_blocks.extend(blocks)
    self.free_blocks.sort()
```

The method `free_blocks` has the same name as the field `self.free_blocks`. This is legal in Python but highly confusing. It is not exploitable by itself, but could mask future misuses.

**Severity:** Informational  
**Mitigation:** Rename the method to `release_blocks`.

---

### 3.4 — `CpuOffloadManager` replaces parameter data with a scalar tensor

**File:** `memory_optimizer.py:47`

```python
self.offloaded[name] = param.data.to('cpu', non_blocking=True)
param.data = torch.empty(1, device='cpu')
```

This replaces a parameter's `.data` tensor (which may be e.g. `[4096, 4096]`) with a single-element CPU tensor `torch.empty(1)`. The optimizer retains momentum buffers keyed by parameter pointer, which now point to a shape-`(1,)` tensor. When `restore()` copies the original data back, the optimizer's internal state shapes are permanently mismatched, causing silent shape errors or incorrect updates.

**Severity:** **High** — corrupts optimizer state silently  
**Mitigation:** Do not mutate `param.data`. Instead, move the entire parameter to CPU:

```python
param.data = param.data.to('cpu', non_blocking=True)
```

Or implement a proper offload mechanism that preserves shape.

---

### 3.5 — `HierarchicalKVCache.tier_bias` index can exceed bounds

**File:** `hierarchical_kv_cache.py:326`

```python
attn = attn + self.tier_bias[tier_labels].unsqueeze(1).unsqueeze(2)
```

`tier_labels` is generated by `read()` (line 237) and should contain only values 0, 1, 2. But `self.tier_bias` has shape `(3,)`. If `read()` is ever modified to produce labels outside 0..2, or if the tensor is manually constructed by a caller, this indexing will produce an out-of-bounds read.

**Severity:** Low  
**Mitigation:** Add a clamp: `tier_labels.clamp(0, 2)` before indexing, or use `nn.Embedding(3, 1).weight` instead of a raw `nn.Parameter`.

---

## 4. Race Conditions

### 4.1 — `AsyncConsolidationPipeline.running` flag is unsynchronized

**File:** `async_memory.py:33-43, 54`

```python
def start(self):
    self.running = True
    ...
def stop(self):
    self.running = False

def _worker_loop(self, worker_id: int):
    while self.running:  # Unsynchronized read from another thread
```

The `running` flag is written by the main thread and read by worker threads with **no memory barrier**. The Python GIL guarantees atomic reads/writes for simple types, but there is no *visibility* guarantee across threads. A worker thread may cache the old value of `running` indefinitely on some Python implementations or under JIT compilation (e.g. PyPy).

**Severity:** Low (benign on CPython due to GIL + GIL release at I/O boundaries)  
**Mitigation:** Use `threading.Event()`:

```python
self.stop_event = threading.Event()
# workers check: while not self.stop_event.is_set():
# stop: self.stop_event.set()
```

---

### 4.2 — `UnifiedMemoryManager._background_loop` shares mutable state without locks

**File:** `unified_manager.py:55-58`

The background thread runs a periodic loop that accesses `self.paged_lts`, `self.deduplicator`, `self.precision`, etc. These same objects are accessed from the main thread via the public API (e.g. `write`, `read`). None of them use locks.

Concretely, `PagedLTSMemory.gpu_pages` is a `nn.Parameter` backing a CUDA tensor. Concurrent reads/writes from different CPU threads to the same CUDA tensor cause undefined behavior at the CUDA driver level (two CPU threads issuing kernels to the same stream/device without synchronization).

**Severity:** **High**  
**Mitigation:** Add a `threading.Lock` per memory module, or use a single reentrant lock for the entire `UnifiedMemoryManager`. All public methods must acquire the lock before accessing shared state.

---

### 4.3 — `AgentLoopController.episode_buffer` is not thread-safe

**File:** `agent_loop.py:65-68`

```python
def learn(self, episode: AgentEpisode):
    self.episode_buffer.append(episode)
```

If `learn()` is called from multiple threads (e.g. in a parallel actor-critic setup), the list's internal state may be corrupted by concurrent `append` and `pop` operations.

**Severity:** Medium  
**Mitigation:** Use `threading.Lock` or `queue.Queue` for the buffer.

---

### 4.4 — `PagedOptimizerState._states` and `_lru` dictionary accessed without locks

**File:** `paged_optimizer.py:11-14`, `paged_optimizer.py:110-166`

`PagedAdamW.step()` mutates `self._state_manager._states` and `self._state_manager._lru`. If `step()` is called from multiple threads (e.g. with `torch.nn.DataParallel` or manual threading), the dictionary is corrupted. Even single-threaded `step()` from the optimizer is generally assumed thread-hostile, so this is a design note.

**Severity:** Low (standard PyTorch optimizer assumption — not thread-safe by design)  
**Mitigation:** Document as not thread-safe. If multi-threaded access is required, add a lock.

---

## 5. Quantization Safety

### 5.1 — `FP8Compressor.compress` all-zero tensor guard

**File:** `fp8_allreduce.py:38-43`

```python
absmax = tensor.abs().max()
if absmax == 0:
    scale = torch.ones(1, ...)
    quantized = torch.zeros(...)
    return quantized, scale
```

This correctly handles the all-zero case. However, if `tensor` is empty (`numel() == 0`), `tensor.abs().max()` raises a runtime error in PyTorch.

**Severity:** Low (empty tensors are programmer error)  
**Mitigation:** Add `if tensor.numel() == 0: return tensor, torch.ones(1, ...)`.

---

### 5.2 — `AdaptivePrecisionManager._demote_precision` promotes from 16-bit to `float8_e4m3fn` unsafely

**File:** `adaptive_precision.py:47`

```python
cfg['bits'], cfg['dtype'] = 8, torch.float8_e4m3fn
```

`float8_e4m3fn` has a maximum value of 448.0. A tensor with values in the range [-448, 448] will survive conversion, but any value outside this range saturates to ±448 or becomes NaN (PyTorch's `to()` clips to representable range). This is fine for activations but dangerous for gradients: a clipped gradient loses the direction signal for outlier dimensions.

The `auto_tune` path (line 26-33) demotes when `error_rate < 0.005`, but the error is computed *after* quantization, so large errors from previous steps may be undetected by the time demotion triggers.

**Severity:** Medium  
**Mitigation:** When demoting to FP8, compute a histogram of the tensor values. If any absolute value exceeds `fp8_max * 0.95`, abort the demotion. Track the maximum observed value per tier.

---

### 5.3 — `FP8LTSMemory.store` applies no scale per element

**File:** `adaptive_precision.py:67-70`

```python
def store(self, data: torch.Tensor, indices: torch.Tensor):
    with torch.no_grad():
        data_fp8 = data.to(torch.float8_e4m3fn)
        self.fp8_storage.data[0, indices] = data_fp8
```

This stores raw FP8 values with no per-element scale factor. When retrieved, `retrieve` converts the entire storage to FP32 uniformly (line 73). The `FP8LTSMemory` was meant for long-term storage but does not keep the scale, so the FP8 values are permanently degraded. The `_dequantize` helper in `hierarchical_kv_cache.py:96` correctly uses per-element scales — this class should follow the same pattern.

**Severity:** Medium  
**Mitigation:** Add a `self.scale` buffer (already exists at line 61 but is never updated) and write per-row scale factors during `store()`. Use block-wise scaling (e.g. per-64-element groups) to bound the quantization error.

---

## 6. Deserialization

### 6.1 — `torch.save` used but no `torch.load` found

**File:** `train_optimized.py:54`

```python
torch.save(state, path)
```

`torch.save` uses Python's `pickle` internally. While `save` is relatively safe (serialization only), if a `torch.load()` is ever added to load these checkpoints, it would execute arbitrary Python code from a malicious checkpoint file.

No `torch.load` calls exist in the codebase at this time.

**Severity:** None now; **High** if a load path is added without `weights_only=True`  
**Mitigation:** When adding a loading path, always use `torch.load(path, weights_only=True)`. If full deserialization is needed, load with `weights_only=True` first and validate the structure before any fallback to unsafe loading.

---

### 6.2 — `yaml.safe_load` used for configuration

**File:** `train_optimized.py:221`

```python
config = yaml.safe_load(f)
```

`safe_load` does not execute arbitrary Python. This is correct.

**Severity:** None  
**Mitigation:** None required.

---

### 6.3 — `LZ4MemoryCompressor` has no untrusted-input reader

**File:** `deduplication.py:137-159`

The decompress path calls `lz4.frame.decompress(compressed)` which is memory-safe (lz4-frame is a C library, but fuzzed extensively). However, a malicious compressed payload could cause a very large allocation via the decompressed output size. Since the compressed data originates from the same process (internal paging), this is not an external attack surface.

**Severity:** Low  
**Mitigation:** If the compressed pages are ever loaded from disk, cap the decompressed size to the expected tensor size before decompressing.

---

## 7. Summary of Findings

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| 1.1 | **Medium** | Memory safety | Missing `#[repr(C)]` on `CheckpointHeader` in `checkpoint.rs` |
| 1.2 | Low | Memory safety | `transmute` of stack var — sound with correct repr |
| 1.3 | None | Memory safety | `from_raw_parts` tensor cast — correct |
| 1.4 | Low | Memory safety | No overflow check on `usize → u64` cast |
| 2.1 | Medium | Data integrity | Differential checkpoint format has no magic or checksum |
| 2.2 | Low | Binary safety | Name length field is unbounded `u32` |
| **3.1** | **High** | **Buffer overflow** | Negative `n_keep_old` in `HierarchicalKVCache` eviction |
| 3.2 | Medium | Buffer overflow | `PagedAttentionCache.alloc_blocks` silent under-allocation |
| 3.3 | Info | Code quality | Method `free_blocks` shadows field |
| **3.4** | **High** | **Buffer overflow** | `CpuOffloadManager` replaces param with scalar tensor |
| 3.5 | Low | Bounds | `tier_bias` indexing can exceed `(3,)` |
| 4.1 | Low | Race condition | `running` flag unsynchronized (GIL-dependent) |
| **4.2** | **High** | **Race condition** | `UnifiedMemoryManager` shared state accessed without locks |
| 4.3 | Medium | Race condition | `episode_buffer` list not thread-safe |
| 4.4 | Low | Race condition | `PagedOptimizerState` dict unsafe by design |
| 5.1 | Low | Quantization safety | Empty tensor causes crash in `FP8Compressor` |
| 5.2 | Medium | Quantization safety | Unsafe FP8 demotion in `AdaptivePrecisionManager` |
| 5.3 | Medium | Quantization safety | `FP8LTSMemory` lacks per-element scaling |
| 6.1 | None→High | Deserialization | No `torch.load` yet, but future use must use `weights_only=True` |
| 6.2 | None | Deserialization | `yaml.safe_load` is correct |
| 6.3 | Low | Deserialization | LZ4 decompression unbounded if exposed externally |

---

## 8. Recommended Immediate Mitigations

1. **Fix `#[repr(C)]` on `CheckpointHeader`** in `checkpoint.rs` (and deduplicate the definition with `lib.rs`).
2. **Cap incoming writes in `HierarchicalKVCache.write()`** — bound `n_new` by `self.cap1` before computing `n_keep_old`. Apply the same fix to `_evict_to_tier2` and `_evict_to_tier3`.
3. **Add thread locks to `UnifiedMemoryManager`** — all public methods that touch `paged_lts`, `deduplicator`, `prefetcher`, `precision` must acquire a lock.
4. **Fix `CpuOffloadManager._offload_idle_params`** — preserve the shape of the offloaded parameter instead of replacing with a scalar.
5. **Guard `PagedAttentionCache.alloc_blocks`** against exhaustion.
6. **Add per-element scale tracking to `FP8LTSMemory`** or use block-wise quantization with a scale factor.
7. **When adding checkpoint loading**, use `torch.load(path, weights_only=True)` exclusively.
