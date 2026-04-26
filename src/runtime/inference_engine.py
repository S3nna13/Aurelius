from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class InferenceBackend(Enum):
    EAGER = "eager"
    COMPILED = "compiled"
    JIT_CACHED = "jit_cached"
    ONNX = "onnx"


@dataclass
class InferenceConfig:
    backend: InferenceBackend = InferenceBackend.EAGER
    max_batch_size: int = 32
    max_seq_len: int = 2048
    timeout_ms: float = 5000.0
    warmup_steps: int = 3
    use_fp16: bool = False


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    text: str
    latency_ms: float
    tokens_generated: int
    backend_used: InferenceBackend


class InferenceEngine:
    def __init__(self, config: InferenceConfig, generate_fn: Callable | None = None):
        self.config = config
        self.generate_fn = generate_fn

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        t0 = time.perf_counter()
        if self.generate_fn is not None:
            text = self.generate_fn(
                request.prompt,
                request.max_new_tokens,
                request.temperature,
                request.top_p,
            )
        else:
            text = f"Generated: {request.prompt[:50]}..."
        latency_ms = (time.perf_counter() - t0) * 1000.0
        tokens_generated = len(text.split())
        return InferenceResponse(
            request_id=request.request_id,
            text=text,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            backend_used=self.config.backend,
        )

    def batch_generate(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        return [self.generate(req) for req in requests]

    def warmup(self, sample_prompt: str = "Hello") -> list[float]:
        latencies: list[float] = []
        for _ in range(self.config.warmup_steps):
            req = InferenceRequest(prompt=sample_prompt)
            resp = self.generate(req)
            latencies.append(resp.latency_ms)
        return latencies

    def throughput_tokens_per_s(self, responses: list[InferenceResponse]) -> float:
        if not responses:
            return 0.0
        total_tokens = sum(r.tokens_generated for r in responses)
        total_latency_s = sum(r.latency_ms for r in responses) / 1000.0
        if total_latency_s <= 0:
            return 0.0
        return total_tokens / total_latency_s


INFERENCE_ENGINE_REGISTRY: dict[str, type[InferenceEngine]] = {"default": InferenceEngine}
