# Aurelius 1.3B — Model Card

## Model Details

| Field | Value |
|-------|-------|
| **Family** | Aurelius |
| **Variant** | aurelius-1.3b |
| **Backbone class** | AureliusTransformer (decoder-only) |
| **Tokenizer** | AureliusTokenizer (128k BPE) |
| **Vocab size** | 128,000 |
| **Max sequence length** | 8,192 |
| **Parameters** | ~1.3B |
| **Architecture** | GQA (16 query, 8 KV), SwiGLU FFN, RoPE, RMSNorm |
| **Checkpoint format** | SafeTensors (primary), PyTorch (legacy) |
| **Release track** | dev |
| **Compatibility version** | 0.1.0 |
| **License** | MIT — Christien Antonio, 2026 |
| **Checkpoint SHA-256** | (To be computed on release) |
| **Tokenizer SHA-256** | (To be computed on release) |

## Intended Use

Aurelius 1.3B is a frontier-tier agentic coding and instruction-following model designed for:

- Code generation, completion, and explanation
- Multi-turn instruction following
- Agentic tool use and task planning
- Chat and reasoning tasks
- Safety-aligned serving via guardrails

## Training

| Detail | Value |
|--------|-------|
| **Framework** | PyTorch + DeepSpeed ZeRO |
| **Training config** | See `configs/train_1b.yaml` |
| **Data** | Curated multilingual code + instruction data (see Dataset Card) |
| **Alignment** | DPO + constitutional constraints + red-team probes |
| **Compute** | Consumer GPU (2× RTX 3060) reference; scalable to multi-GPU |

## Capabilities

See `src/eval/` for internal scorer implementations. Key benchmarks:

- HumanEval, HumanEval+, MBPP (code)
- GSM8K (math)
- MMLU, ARC, HellaSwag (reasoning)
- TruthfulQA (truthfulness)
- Agent Red Team (safety)

## Limitations

- 1.3B parameters limit complex reasoning depth
- English + code primary; multilingual partial
- May hallucinate or generate incorrect code
- Safety guardrails are defense-in-depth but not infallible

## Safety & Ethics

- Threat model documented in `docs/threat_model.md`
- Safety alignment via constitutional constraints, RLHF/DPO, guardrails
- Canary-token exfiltration detection
- Sandbox execution for code evaluation
- All `torch.load()` calls use `weights_only=True`
- All `yaml.load()` uses `SafeLoader`
- Archive extraction uses safe-extract wrappers
- Serving endpoints enforce rate limiting, auth middleware, input size caps

## Threat Model

See `docs/threat_model.md` for the full STRIDE analysis per surface.

## Evaluation

See `docs/eval_card.md` for benchmark scores and evaluation methodology.

## Citation

```bibtex
@misc{aurelius2026,
  title={Aurelius: A Frontier-Tier Agentic Coding LLM},
  author={Christien Antonio},
  year={2026},
  url={https://github.com/S3nna13/Aurelius}
}
```