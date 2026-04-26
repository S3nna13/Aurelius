# CONFIDENTIAL — Aurelius Training Materials & Proprietary Resources

**Classification:** INTERNAL USE ONLY  
**Effective Date:** 2025-04-26  
**Last Updated:** 2025-04-26

---

## 1. Purpose

This document defines the confidentiality boundary between the **open-source Aurelius software** (licensed under the Aurelius Open License) and the **proprietary training materials, datasets, and internal resources** maintained by Aurelius Systems, Inc.

The open-source codebase — including the model architecture, inference engine, training framework code, API server, and web UI — remains freely available under the Aurelius Open License. However, the actual training artifacts, proprietary datasets, experiment configurations, and internal documentation are **strictly confidential** and are **not** part of the open-source distribution.

---

## 2. Confidential Materials

The following categories of materials are classified as **CONFIDENTIAL** and are for **internal use only**:

### 2.1 Training Datasets & Corpora
- Raw and processed training data
- Conversation datasets (JSONL, parquet, or any format)
- Reference corpora extracted from proprietary or licensed sources
- Synthetic training data generators and their outputs
- Data mixing recipes and curriculum configurations
- Any data located in `data/`, `training_data/`, or similar directories

### 2.2 Training Configurations
- Pretraining configs (`configs/train_*.yaml`, `configs/curriculum.yaml`, `configs/merge_*.yaml`)
- Fine-tuning configurations (`configs/yarn_finetune.yaml`, etc.)
- DeepSpeed configs (`configs/deepspeed_*.json`)
- Hyperparameter schedules and optimization recipes
- Tokenizer training configurations and vocabularies

### 2.3 Training Scripts & Pipelines
- Data extraction scripts (`scripts/extract_*.py`)
- Data conversion and encoding scripts (`scripts/convert_*.py`, `scripts/encode_*.py`)
- Data preparation pipelines (`scripts/prepare_data.sh`)
- Tokenizer training scripts (`scripts/train_tokenizer.sh`, `scripts/tokenize_*.py`)
- Training launch scripts (`scripts/run_training.sh`, `scripts/run_sft.sh`, `scripts/run_dpo.sh`)
- Standalone training scripts at repository root (`train_*.py`)
- Red teaming and adversarial evaluation scripts (`scripts/red_team.sh`)
- Collection and harvesting scripts (`scripts/collect_training_data.py`, `scripts/prepare_glm5_training_data.py`)

### 2.4 Model Checkpoints & Weights
- All checkpoint directories (`checkpoints/`)
- Trained model weights (`.pt`, `.safetensors`, `.bin`, `.gguf`)
- Intermediate training checkpoints
- Merged or distilled model artifacts
- LoRA / adapter weights

### 2.5 Experiment Logs & Documentation
- `docs/dataset_card.md` — proprietary dataset composition and sourcing
- `docs/model_card.md` — internal training details, experiment history, and evaluation results
- `docs/eval_card.md` — internal benchmark results and comparison data
- `docs/harvest_journal.md` — data harvesting logs and provenance records
- `docs/plans/` — internal roadmaps and strategic planning documents
- Training logs, wandb runs, and experiment trackers
- `extract_corpus.log` and similar log files

### 2.6 Internal Infrastructure
- `scripts/dev.sh` and other internal development helpers
- `deployment/compose.override.yaml` — environment-specific overrides
- Any `.env` files or secrets (already in `.gitignore`)

---

## 3. What Remains Open Source

The following are **explicitly open** under the Aurelius Open License:

| Category | Location | Status |
|---|---|---|
| Model architecture code | `src/model/` | ✅ Open |
| Training framework code | `src/training/` | ✅ Open (framework only) |
| Data processing framework | `src/data/` | ✅ Open (framework only) |
| Alignment algorithms | `src/alignment/` | ✅ Open |
| Inference engine | `src/inference/` | ✅ Open |
| Evaluation framework | `src/eval/` | ✅ Open |
| Security modules | `src/security/` | ✅ Open |
| Agent framework | `src/agent/` | ✅ Open |
| Chat templates | `src/chat/` | ✅ Open |
| Long-context utilities | `src/longcontext/` | ✅ Open |
| Retrieval engine | `src/retrieval/` | ✅ Open |
| Safety modules | `src/safety/` | ✅ Open |
| API server | `src/serving/` | ✅ Open |
| CLI | `src/cli/` | ✅ Open |
| Web UI | `frontend/` | ✅ Open |
| Deployment configs | `deployment/Dockerfile`, `deployment/compose.yaml` | ✅ Open |
| General documentation | `README.md`, `LICENSE`, `EULA.md` | ✅ Open |
| Tests | `tests/` | ✅ Open |

> **Distinction:** `src/training/` and `src/data/` contain the **framework code** — the building blocks for training and data processing. They do not contain proprietary datasets, trained checkpoints, or run-specific configurations. The framework is open; the runs are confidential.

---

## 4. Access & Distribution Policy

### 4.1 Internal Team
- Full access to all confidential materials
- May use, modify, and distribute within the organization
- Must not commit secrets, API keys, or credentials to any branch

### 4.2 External Contributors
- Access only to open-source components
- No access to training datasets, checkpoints, or internal configs
- May contribute to `src/`, `frontend/`, `tests/`, and open documentation
- Pull requests touching confidential materials will be rejected

### 4.3 Public Distribution
- The GitHub repository may contain **markers and references** to confidential materials (e.g., `.gitignore` entries, directory stubs)
- Actual confidential files **must not** be present in public branches
- If confidential data is accidentally committed, follow the incident response procedure in Section 6

---

## 5. Developer Guidelines

### 5.1 Before Committing
Run this checklist before every commit:

```bash
# Check for large files / checkpoints
git diff --cached --name-only | grep -E '\.(pt|pth|bin|safetensors|gguf|parquet|jsonl)$'

# Check for env files / secrets
git diff --cached --name-only | grep -E '\.env|secret|key|credential'

# Check for log files
git diff --cached --name-only | grep -E '\.log$'

# Check for data directories
git diff --cached --name-only | grep -E '^data/|^checkpoints/|^training_data/'
```

### 5.2 .gitignore Enforcement
The repository `.gitignore` already excludes:
- `checkpoints/` (all model weights)
- `data/` (raw and processed data)
- `training_data/` (training artifacts)
- `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.gguf`
- `*.log`, `*.jsonl` (large data files)
- `.env`, `.env.*` (secrets)
- `wandb/`, `mlruns/` (experiment trackers)

**Never** force-add (`git add -f`) any of the above.

### 5.3 Documentation Boundaries
When writing public-facing documentation:
- ✅ DO describe the architecture and framework capabilities
- ✅ DO provide API examples and usage patterns
- ❌ DO NOT list specific datasets used for training
- ❌ DO NOT disclose hyperparameters, learning rates, or batch sizes
- ❌ DO NOT share experiment results, loss curves, or benchmark scores
- ❌ DO NOT reveal data sources, scraping methods, or harvesting techniques

---

## 6. Incident Response

If confidential material is accidentally committed to a public branch:

1. **Do not panic.** Do not draw attention to the commit.
2. **Immediately** remove the files from the working tree.
3. **Rewrite history** to purge the data:
   ```bash
   git filter-repo --path <confidential-file> --invert-paths
   # Or use BFG Repo-Cleaner for large files
   ```
4. **Force-push** the cleaned branch:
   ```bash
   git push --force-with-lease origin main
   ```
5. **Rotate any exposed secrets** (API keys, tokens, passwords).
6. **Notify the security lead** within 24 hours.
7. **Document the incident** in the internal security log.

> **Note:** Once pushed to GitHub, data may be cached by mirrors, CI systems, or third-party integrations. Contact GitHub Support for sensitive data removal if needed.

---

## 7. Legal Basis

This confidentiality policy is established under:

- The **Aurelius Open License** (Section 3: Architecture IP)
- The **Aurelius EULA** (Section 4: Restrictions)
- Applicable trade secret and data protection laws

Violation of this policy may result in:
- Revocation of repository access
- Legal action for misappropriation of trade secrets
- Termination of contributor agreements

---

## 8. Contact

For questions about this policy, or to request access to confidential materials:

- **Security & Compliance:** security@aurelius.systems
- **Legal:** legal@aurelius.systems
- **General Inquiries:** hello@aurelius.systems

---

*This document is a living policy. It will be updated as the project evolves. All team members and contributors are expected to review this document quarterly.*

**© 2025 Aurelius Systems, Inc. All rights reserved.**
