# Aurelius — Evaluation Card

## Model

| Field | Value |
|-------|-------|
| **Family** | Aurelius |
| **Variant** | aurelius-1.3b |
| **Checkpoint format version** | 1 |
| **Config version** | 0.1.0 |

## Benchmarks

Internal scorers are implemented in `src/eval/` and registered in `BENCHMARK_REGISTRY`.

| Benchmark | Scorer Module | Metric | Target | Status |
|-----------|--------------|--------|--------|--------|
| HumanEval | `humaneval_scorer.py` | pass@1 | ≥ 30% | *To be measured on release* |
| HumanEval+ | `humaneval_plus_scorer.py` | pass@1 | ≥ 25% | *To be measured on release* |
| MBPP | `mbpp_scorer.py` | pass@1 | ≥ 35% | *To be measured on release* |
| GSM8K | `gsm8k_scorer.py` | accuracy | ≥ 40% | *To be measured on release* |
| MMLU | `mmlu_scorer.py` | accuracy | ≥ 30% | *To be measured on release* |
| ARC | `arc_registry` | accuracy | ≥ 50% | *To be measured on release* |
| HellaSwag | `hellaswag_scorer.py` | accuracy | ≥ 45% | *To be measured on release* |
| TruthfulQA | `truthfulqa_registry` | mc2 | ≥ 40% | *To be measured on release* |
| Agent Red Team | `agent_red_team_bench` | safety score | ≥ 80% | *To be measured on release* |
| TAUBench | (conditional) | task completion | — | *Conditional on enable_taubench_eval* |
| SWE-bench Lite | `swebench_lite_scorer` | resolve rate | — | *To be measured on release* |
| Code Review | `code_review_scorer` | rubric score | — | *To be measured on release* |
| Arena Hard | `arena_hard_scorer` | win rate | — | *To be measured on release* |
| Perplexity | `perplexity_scorer` | ppl | — | *To be measured on release* |

## Evaluation Infrastructure

- **Internal scorers**: All benchmark scorers are in `src/eval/*_scorer.py`
- **Security scorers**: Adversarial tests in `tests/security/`
- **Integration tests**: `tests/integration/`
- **Latency**: p50/p95/p99 + tokens/sec to be measured on release
- **Red-team regression**: Garak + PyRIT + prompt-injection corpus (conditional)

## Responsible AI

- Harm taxonomy classifier (`harm_taxonomy_classifier.py`)
- PII detector (`pii_detector.py`)
- Jailbreak detector (`jailbreak_detector.py`, `jailbreak_detector_v2.py`)
- Output safety filter (`output_safety_filter.py`)
- Refusal classifier (`refusal_classifier.py`)
- Canary-token exfiltration detection (`canary_pipeline.py`, `canary_auditor.py`)

## Methodology Notes

- All benchmarks use the internal `BENCHMARK_REGISTRY` for deterministic registration
- Parameter-efficient evaluation supported via `EVAL_REGISTRY`
- Latency benchmarks measured with `LATENCY_HISTOGRAM_REGISTRY`
- Long-context benchmarks include needle-in-haystack (`needle_in_haystack.py`)