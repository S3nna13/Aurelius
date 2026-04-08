# Aurelius

Aurelius is a Python codebase for training, aligning, evaluating, and serving a decoder-only language model. The repository includes model components, data pipelines, alignment methods, inference utilities, evaluation helpers, and local serving scripts.

## What is in this repo

- `src/model/`: transformer architecture, attention variants, MoE modules, multimodal pieces, and model config.
- `src/data/`: tokenizer training, dataset download helpers, preprocessing, packing, filtering, and data pipelines.
- `src/training/`: pretraining loop, checkpointing, optimization helpers, curriculum learning, distillation, and related utilities.
- `src/alignment/`: SFT, DPO, GRPO-adjacent components, reward modeling, red teaming, and safety tooling.
- `src/eval/`: evaluation harnesses, metrics, probing, causal tracing, and model-card helpers.
- `src/inference/`: decoding, quantization, structured output, RAG, caching, and inference-time controls.
- `src/serving/`: chat client and local serving helpers.
- `configs/`: training, merge, tokenizer, and serving configuration files.
- `scripts/`: convenience wrappers for data prep, training, model merging, GGUF conversion, and local serving.
- `tests/`: unit tests for the repo.

## Requirements

- Python `3.12+`
- A virtual environment such as `.venv`
- PyTorch-compatible hardware for actual training runs
- Optional tools depending on workflow: `Ollama`, `mergekit`, `llama.cpp`, `deepspeed`, `flash-attn`

## Quick start

Create an environment and install the base package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For development and tests:

```bash
pip install -e .[dev]
```

For training-oriented extras:

```bash
pip install -e .[train]
```

For serving utilities:

```bash
pip install -e .[serve]
```

## Running tests

Run the full test suite:

```bash
pytest -q
```

Run a focused module test:

```bash
pytest -q tests/alignment/test_grpo.py
pytest -q tests/training/test_learned_optimizer.py
```

## Main workflows

### 1. Train a tokenizer

Wrapper script:

```bash
bash scripts/train_tokenizer.sh
```

Direct CLI:

```bash
python -m src.data.tokenizer --help
```

### 2. Prepare data

Download a sample and run the preprocessing pipeline:

```bash
bash scripts/prepare_data.sh
```

Useful variants:

```bash
bash scripts/prepare_data.sh --sample-only
bash scripts/prepare_data.sh --full
```

Direct CLIs:

```bash
python -m src.data.download_sample --help
python -m src.data.pipeline --help
```

### 3. Pretrain the model

Default config:

```bash
python -m src.training.trainer --config configs/train_1b.yaml
```

Resume from a checkpoint:

```bash
python -m src.training.trainer \
  --config configs/train_1b.yaml \
  --resume checkpoints/<checkpoint-dir>
```

There is also a wrapper script:

```bash
bash scripts/run_training.sh
```

### 4. Alignment passes

SFT CLI stub:

```bash
python -m src.alignment.sft --help
```

DPO CLI stub:

```bash
python -m src.alignment.dpo --help
```

Convenience wrappers:

```bash
bash scripts/run_sft.sh
bash scripts/run_dpo.sh
```

Note: the current SFT and DPO CLIs expose configuration and expected arguments, but the actual training flow is intended to be wired up programmatically with loaded model/tokenizer objects.

### 5. Evaluate a checkpoint

```bash
python -m src.eval.harness checkpoints/<checkpoint-dir> --results-dir results
```

See available options with:

```bash
python -m src.eval.harness --help
```

### 6. Convert and serve locally

Convert a Hugging Face checkpoint to GGUF:

```bash
bash scripts/convert_to_gguf.sh <hf-model-path> [output-dir]
```

Start local Ollama serving:

```bash
bash scripts/serve_local.sh --model-path models/gguf/aurelius-1.3b-q4_k_m.gguf
```

Send a message through the Python client:

```bash
python -m src.serving.chat_client --model aurelius -m "Hello"
```

## Config files worth knowing

- `configs/train_1b.yaml`: default pretraining config.
- `configs/curriculum.yaml`: curriculum-related settings.
- `configs/merge_slerp.yaml`: model-merge config.
- `configs/ollama.Modelfile`: local Ollama model definition template.
- `configs/tokenizer_config.json`: tokenizer metadata copied into tokenizer outputs.

## Package imports

The repository historically used `src.*` imports internally. It now also exposes a public `aurelius.*` namespace for downstream consumers.

Examples:

```python
from src.model.transformer import AureliusTransformer
from aurelius.model.transformer import AureliusTransformer
```

Both styles currently work.

## Current status

- Test suite currently passes locally with `1604 passed, 2 skipped`.
- A handoff note for concurrent agents lives at `docs/plans/2026-04-08-learned-optimizer-fix-handoff.md`.

## Notes for contributors

- The repo may contain other in-progress local changes in the working tree.
- `__pycache__/` and `.pyc` files are ignored and should not be committed.
- Prefer focused test runs while iterating, then run `pytest -q` before handing off.
