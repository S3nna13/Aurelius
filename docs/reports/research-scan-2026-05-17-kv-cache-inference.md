# Research Scan: KV Cache Reuse and Speculative Decoding

Date: 2026-05-17

## Scope

Small, implementation-oriented scan for material that can guide the next Aurelius inference improvements without expanding scope too broadly.

## Useful material found

### SPIRe: Boosting LLM Inference Throughput with Speculative Decoding

- arXiv: 2504.06419
- Focus: speculative decoding at larger batch sizes using sparse draft-model KV cache, pruned initialization, and feedback memory.
- Practical takeaway for Aurelius: speculative decoding should be measured by throughput and accepted/reused token volume, not only by request-level success.

### QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache

- arXiv: 2502.10424
- Focus: self-speculative decoding with hierarchical 4-bit quantized KV cache and quantized weights for long-context edge inference.
- Practical takeaway for Aurelius: KV-cache memory footprint must be visible before adding more quantized/cache tiers.

### LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference

- arXiv: 2510.09665
- Focus: external KV-cache storage and reuse across queries and inference engines.
- Practical takeaway for Aurelius: prefix-cache observability should include stored KV bytes and reused-token ratios so future cache offload/reuse decisions are evidence-driven.

## Change made from this scan

Added token-level and byte-level observability to `src/inference/prefix_cache.py`:

- `estimate_kv_cache_bytes(kv_cache)` helper
- `lookup_tokens`
- `reused_tokens`
- `token_reuse_fraction`
- `tokens_stored`
- `avg_prefix_len`
- `bytes_stored`

This keeps the implementation small while supporting future KV-cache optimization work with measurable signals.

## Validation

Targeted test run:

```bash
python3 -m pytest tests/inference/test_prefix_cache.py -q --tb=short
```

Result: 18 passed.
