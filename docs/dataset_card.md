# Aurelius — Dataset Card

## Overview

Aurelius uses curated multilingual code and instruction data for pretraining and alignment.

## Pretraining Data

| Property | Value |
|----------|-------|
| **Format** | Tokenized SafeTensors + raw text |
| **Languages** | English (primary), code (Python, JS, Go, Rust, C, etc.), multilingual (partial) |
| **Tokenizer** | AureliusTokenizer (128k BPE) |
| **Data pipeline** | `src/data/` — download, tokenize, pack, augment |

## Alignment Data

| Phase | Data Source | Scorer |
|-------|------------|--------|
| SFT | Instruction-following datasets | `sft.py`, DPO datasets via `dpo_trainer.py` |
| DPO | Preference pairs | `dpo_trainer.py`, `online_dpo.py` |
| Constitutional | 15-dimension constitution | `constitutional_constraints_scorer.py` |
| Red-team | Synthetic + curated adversarial | `red_team_dataset_gen.py`, `synthetic_jailbreak_generator.py` |

## Data Quality

- N-gram Jaccard deduplication (`ngram_deduplication_enabled` flag)
- PII detection and redaction (`pii_detector.py`)
- Canary-token injection for exfiltration detection (`canary_pipeline.py`)
- Adversarial text augmentation for robustness (`adversarial_text_augment.py`)
- Data provenance tracking (`data_provenance.py`)
- CWE synthetic data for security training (`cwe_synthesis.py`, disabled by default)

## Data Governance

- No user secrets in training data
- License provenance tracked in `.aurelius-provenance.log`
- Restrictive licenses (GPL, AGPL) treated as clean-room reference only
- Permissive licenses (MIT, Apache-2.0, BSD) adapted with attribution

## Dataset Integrity

| Artifact | Verification |
|----------|-------------|
| Tokenizer SHA-256 | `tokenizer_contract.py` validates on load |
| Checkpoint SHA-256 | `checkpoint.py` validates on load |
| SafeTensors format | Preferred; no pickle deserialization |
| Dataset hashes | Manifest in `dataset_sha256_manifest` field |