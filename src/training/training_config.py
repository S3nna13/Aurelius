from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingConfig:
    # Model
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    n_kv_heads: int | None = None
    max_seq_len: int = 1024
    head_dim: int = 128
    tie_embeddings: bool = True

    # Training
    max_steps: int = 600_000
    warmup_steps: int = 2_000
    batch_size: int = 12
    gradient_accumulation_steps: int = 1
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Eval
    eval_interval: int = 2_000
    eval_iters: int = 200
    log_interval: int = 1

    # System
    device: str = "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    compile: bool = False

    # Paths
    out_dir: str = "out"
    data_dir: str = "data"

    def effective_tokens_per_step(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps * self.max_seq_len

    def chinchilla_optimal_steps(self, total_tokens: int) -> int:
        return total_tokens // self.effective_tokens_per_step()

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        import dataclasses
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_yaml(cls, path: str) -> TrainingConfig:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)

        model = raw.get("model", {})
        training = raw.get("training", {})
        optimizer = raw.get("optimizer", {})

        return cls(
            vocab_size=model.get("vocab_size", cls.vocab_size),
            n_layers=model.get("n_layers", cls.n_layers),
            n_heads=model.get("n_heads", cls.n_heads),
            d_model=model.get("d_model", cls.d_model),
            d_ff=model.get("d_ff", cls.d_ff),
            n_kv_heads=model.get("n_kv_heads", cls.n_kv_heads),
            max_seq_len=model.get("max_seq_len", cls.max_seq_len),
            head_dim=model.get("head_dim", cls.head_dim),
            tie_embeddings=model.get("tie_embeddings", cls.tie_embeddings),
            max_steps=training.get("total_steps", cls.max_steps),
            learning_rate=optimizer.get("lr", cls.learning_rate),
            min_lr=optimizer.get("min_lr", cls.min_lr),
            weight_decay=optimizer.get("weight_decay", cls.weight_decay),
            beta1=optimizer.get("beta1", cls.beta1),
            beta2=optimizer.get("beta2", cls.beta2),
        )

    @classmethod
    def gpt2_small(cls) -> TrainingConfig:
        return cls(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)

    @classmethod
    def gpt2_medium(cls) -> TrainingConfig:
        return cls(vocab_size=50257, n_layers=24, n_heads=16, d_model=1024)

    @classmethod
    def aurelius_1b(cls) -> TrainingConfig:
        return cls(
            vocab_size=128000, n_layers=24, n_heads=16, d_model=2048,
            n_kv_heads=8, d_ff=5632, max_seq_len=8192,
            batch_size=4, learning_rate=3e-4,
        )


DEFAULT_TRAINING_CONFIG = TrainingConfig()
