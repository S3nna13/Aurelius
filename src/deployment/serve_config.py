from __future__ import annotations

import json
from dataclasses import dataclass, field

_VALID_DTYPES = {"float32", "float16", "bfloat16", "int8", "int4"}


@dataclass
class ServeDeploymentConfig:
    model_name: str
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 32
    max_seq_len: int = 4096
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    engine: str = "aurelius"
    extra: dict = field(default_factory=dict)


class ServeConfigBuilder:
    def __init__(self, config: ServeDeploymentConfig) -> None:
        self.config = config

    def to_dict(self) -> dict:
        c = self.config
        return {
            "model_name": c.model_name,
            "model_path": c.model_path,
            "host": c.host,
            "port": c.port,
            "workers": c.workers,
            "max_batch_size": c.max_batch_size,
            "max_seq_len": c.max_seq_len,
            "dtype": c.dtype,
            "gpu_memory_utilization": c.gpu_memory_utilization,
            "engine": c.engine,
            "extra": c.extra,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_vllm_args(self) -> list[str]:
        c = self.config
        args = [
            "--model", c.model_path,
            "--host", c.host,
            "--port", str(c.port),
            "--dtype", c.dtype,
            "--max-model-len", str(c.max_seq_len),
            "--gpu-memory-utilization", str(c.gpu_memory_utilization),
            "--max-num-seqs", str(c.max_batch_size),
        ]
        return args

    def validate(self) -> list[str]:
        errors: list[str] = []
        c = self.config
        if c.dtype not in _VALID_DTYPES:
            errors.append(f"Invalid dtype '{c.dtype}'; must be one of {sorted(_VALID_DTYPES)}")
        if not (0 < c.gpu_memory_utilization <= 1):
            errors.append(
                f"gpu_memory_utilization must be in (0, 1], got {c.gpu_memory_utilization}"
            )
        if c.max_batch_size <= 0:
            errors.append(f"max_batch_size must be > 0, got {c.max_batch_size}")
        if not (1 <= c.port <= 65535):
            errors.append(f"port must be in 1..65535, got {c.port}")
        return errors


SERVE_CONFIG_REGISTRY: dict[str, type[ServeConfigBuilder]] = {"default": ServeConfigBuilder}
