"""FastAPI serving with OpenAI-compatible endpoints, health checks, and Prometheus metrics."""

import logging
import os
import time
import threading
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from prometheus_client import make_asgi_app, Histogram, Counter, Gauge
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from nn_utils import sample_with_top_p_top_k, validate_input_ids


class ServerState:
    def __init__(self):
        self.model = None
        self.config = None
        self.device = 'cpu'
        self.ready = False
        self.start_time = time.time()
        self.request_count = 0
        self._lock = threading.Lock()


state = ServerState()


REQUEST_LATENCY = Histogram(
    'aurelius_request_duration_seconds',
    'Request latency in seconds',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
TOKENS_GENERATED = Counter('aurelius_tokens_generated_total', 'Total tokens generated')
REQUESTS_TOTAL = Counter('aurelius_requests_total', 'Total requests processed')
ACTIVE_REQUESTS = Gauge('aurelius_requests_in_progress', 'Currently active requests')
MODEL_INFO = Gauge('aurelius_model_info', 'Model metadata', ['d_model', 'n_layers'])


class GenerateRequest(BaseModel):
    prompt: List[int] = Field(..., description="Tokenized prompt as list of ints")
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)


class GenerateResponse(BaseModel):
    tokens: List[int]
    token_count: int
    duration_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    requests_served: int
    cuda_available: bool
    cuda_memory_allocated_mb: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Aurelius API server")
    yield
    logger.info("Shutting down Aurelius API server")


app = FastAPI(
    title="Aurelius AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if HAS_PROMETHEUS:
    app.mount("/metrics", make_asgi_app())


@app.get("/health", response_model=HealthResponse)
async def health():
    cuda_alloc = None
    if torch.cuda.is_available():
        cuda_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
    return HealthResponse(
        status="ready" if state.ready else "loading",
        model_loaded=state.ready,
        device=state.device,
        uptime_seconds=time.time() - state.start_time,
        requests_served=state.request_count,
        cuda_available=torch.cuda.is_available(),
        cuda_memory_allocated_mb=cuda_alloc,
    )


@app.get("/ready")
async def ready():
    if not state.ready or state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not state.ready or state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        input_ids = torch.tensor([req.prompt], dtype=torch.long, device=state.device)
        vocab_size = state.config.get('vocab_size', 32000)
        validate_input_ids(input_ids, vocab_size)

        generated = input_ids.clone()
        for _ in range(req.max_new_tokens):
            with torch.no_grad():
                logits = state.model(generated[:, -2048:])
                if isinstance(logits, dict):
                    logits = logits['logits']
            next_token = sample_with_top_p_top_k(
                logits[:, -1, :], req.temperature, req.top_k, req.top_p,
            )
            generated = torch.cat([generated, next_token], dim=1)

        tokens = generated[0, input_ids.shape[1]:].tolist()
        duration = (time.time() - start) * 1000

        TOKENS_GENERATED.inc(len(tokens))
        REQUESTS_TOTAL.inc()
        REQUEST_LATENCY.observe(duration / 1000)

        with state._lock:
            state.request_count += 1

        return GenerateResponse(
            tokens=tokens,
            token_count=len(tokens),
            duration_ms=round(duration, 2),
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/v1/completions")
async def openai_completions(request: Dict[str, Any]):
    if not state.ready or state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prompt = request.get("prompt", "")
    max_tokens = request.get("max_tokens", 128)
    temperature = request.get("temperature", 0.8)
    top_p = request.get("top_p", 0.9)
    stream = request.get("stream", False)

    if isinstance(prompt, str):
        prompt_ids = [ord(c) for c in prompt[:256]]
    else:
        prompt_ids = prompt if isinstance(prompt, list) else [0]

    gen_req = GenerateRequest(
        prompt=prompt_ids,
        max_new_tokens=min(max_tokens, 512),
        temperature=temperature,
        top_p=top_p,
    )
    result = await generate(gen_req)

    return {
        "id": "cmpl-aurelius",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "aurelius",
        "choices": [{
            "text": " ".join(str(t) for t in result.tokens),
            "index": 0,
            "finish_reason": "length",
        }],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": result.token_count,
            "total_tokens": len(prompt_ids) + result.token_count,
        },
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "aurelius",
            "object": "model",
            "created": int(state.start_time),
            "owned_by": "aurelius",
        }],
    }


def load_model(model_path: str = None, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state.device = device

    if model_path is not None and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        d_model = checkpoint.get('d_model', 768)
        from aurelius_model import AureliusModel
        config = {
            'd_model': d_model, 'n_heads': 12, 'd_ff': d_model * 4,
            'n_layers': 12, 'vocab_size': 50257, 'max_seq_len': 2048,
            'd_mem': 256, 'episodic_slots': 512, 'lts_capacity': 1024,
            'consolidation_freq': 64, 'graph_threshold': 0.65, 'dropout': 0.0,
        }
        state.model = AureliusModel(config)
        state.model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        state.config = config
        logger.info(f"Loaded model from {model_path}")
    else:
        from aurelius_model import AureliusModel
        config = {
            'd_model': 768, 'n_heads': 12, 'd_ff': 3072, 'n_layers': 2,
            'vocab_size': 10000, 'max_seq_len': 256, 'd_mem': 128,
            'episodic_slots': 64, 'lts_capacity': 128,
            'consolidation_freq': 16, 'graph_threshold': 0.65, 'dropout': 0.0,
        }
        state.model = AureliusModel(config)
        state.config = config
        logger.info("Loaded demo model (2-layer)")

    state.model.to(device)
    state.model.eval()
    state.ready = True

    if HAS_PROMETHEUS:
        MODEL_INFO.labels(
            d_model=str(config.get('d_model', '?')),
            n_layers=str(config.get('n_layers', '?')),
        ).set(1)

    logger.info(f"Model ready on {device}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aurelius API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    load_model(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == '__main__':
    main()
