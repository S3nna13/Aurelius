"""Aurelius API Server — FastAPI with WebSocket streaming.

Serves the Agent Cockpit frontend with real-time communication.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from src.serving.api_server import ChatRequest as EngineChatRequest
from src.serving.engine_loader import build_engine, make_mock_generate_fn
from src.serving.metrics_middleware import METRICS
from src.serving.rate_limit import get_rate_limiter

app = FastAPI(
    title="Aurelius API", version="1.0.0", description="Backend API for the Aurelius Agent Cockpit"
)


# ─── Safety input limits ────────────────────────────────────────────────
SAFE_MAX_PROMPT_TOKENS = 32_768  # hard cap; prevents memory exhaustion
SAFE_MAX_TEMPERATURE = 2.0  # sensible range
SAFE_MIN_TEMPERATURE = 0.0
SAFE_MAX_TOP_P = 1.0
SAFE_MIN_TOP_P = 0.01
SAFE_MAX_REP_PENALTY = 2.0
SAFE_MIN_REP_PENALTY = 1.0


def validate_chat_params(body: dict) -> None:
    """Raise HTTPException(400) if any generation parameter is unsafe."""
    msgs = body.get("messages", [])
    if not msgs:
        METRICS.record_validation_failure()
        raise HTTPException(400, detail="messages list is required")

    approx_tokens = sum(len(str(m.get("content", ""))) for m in msgs) // 4
    if approx_tokens > SAFE_MAX_PROMPT_TOKENS:
        METRICS.record_validation_failure()
        raise HTTPException(
            413,
            detail=f"Prompt too large (>{SAFE_MAX_PROMPT_TOKENS} estimated tokens)",
        )

    for msg in msgs:
        _role = msg.get("role", "")
        content_str = str(msg.get("content", ""))
        msg_tokens = len(content_str) // 4
        if msg_tokens > SAFE_MAX_PROMPT_TOKENS:
            METRICS.record_validation_failure()
            raise HTTPException(
                413,
                detail=f"Single message too large ({msg_tokens} estimated tokens)",
            )

    if "temperature" in body:
        t = body["temperature"]
        if not (SAFE_MIN_TEMPERATURE <= t <= SAFE_MAX_TEMPERATURE):
            METRICS.record_validation_failure()
            raise HTTPException(
                400,
                detail=f"temperature must be [{SAFE_MIN_TEMPERATURE}, {SAFE_MAX_TEMPERATURE}]",
            )

    if "top_p" in body:
        tp = body["top_p"]
        if not (SAFE_MIN_TOP_P <= tp <= SAFE_MAX_TOP_P):
            METRICS.record_validation_failure()
            raise HTTPException(
                400,
                detail=f"top_p must be [{SAFE_MIN_TOP_P}, {SAFE_MAX_TOP_P}]",
            )

    if "repetition_penalty" in body:
        rp = body["repetition_penalty"]
        if not (SAFE_MIN_REP_PENALTY <= rp <= SAFE_MAX_REP_PENALTY):
            METRICS.record_validation_failure()
            raise HTTPException(
                400,
                detail=(
                    f"repetition_penalty must be [{SAFE_MIN_REP_PENALTY}, {SAFE_MAX_REP_PENALTY}]"
                ),
            )

    if "max_tokens" in body:
        mt = body["max_tokens"]
        if not (1 <= mt <= SAFE_MAX_PROMPT_TOKENS):
            METRICS.record_validation_failure()
            raise HTTPException(
                400,
                detail=f"max_tokens must be 1–{SAFE_MAX_PROMPT_TOKENS}",
            )


# CORS origins must be configured via CORS_ORIGINS env var (comma-separated).
_cors_origins_str = os.environ.get("CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_origins_str.split(",") if o.strip()]
    if _cors_origins_str.strip()
    else []
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Security Hardening Middlewares ─────────────────────────────


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    csp = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none'; "
        "block-all-mixed-content"
    )
    response.headers["Content-Security-Policy"] = csp
    if os.environ.get("AURELIUS_ENV") == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
    return response


MAX_REQUEST_SIZE = int(os.environ.get("AURELIUS_MAX_REQUEST_SIZE", "1048576"))
MAX_STREAM_SIZE = int(os.environ.get("AURELIUS_MAX_STREAM_SIZE", "10485760"))


@app.middleware("http")
async def limit_request_size(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")
        body_limit = (
            MAX_STREAM_SIZE
            if request.url.path in ("/v1/chat/completions", "/generate")
            else MAX_REQUEST_SIZE
        )
        if size > body_limit:
            raise HTTPException(status_code=413, detail=f"Request too large (>{body_limit} bytes)")
    return await call_next(request)


ALLOWED_HOSTS = {
    h.strip()
    for h in os.environ.get("AURELIUS_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
}


@app.middleware("http")
async def restrict_host(request, call_next):
    host = request.headers.get("host", "")
    hostname = host.split(":")[0] if host else ""
    if hostname and hostname not in ALLOWED_HOSTS:
        raise HTTPException(status_code=400, detail=f"Host '{hostname}' not allowed")
    return await call_next(request)


@app.middleware("http")
async def rate_limit(request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if _rate_limiter is None:
        # Fallback: allow if rate limiter not yet initialized
        return await call_next(request)
    if not _rate_limiter(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return await call_next(request)


@app.middleware("http")
async def require_json_content_type(request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        ct = request.headers.get("content-type", "")
        if not ct.startswith("application/json"):
            raise HTTPException(400, detail="Content-Type must be application/json")
    return await call_next(request)


@app.exception_handler(HTTPException)
async def http_error_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.middleware("http")
async def request_id(request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def record_metrics(request, call_next):
    start = time.time()
    METRICS.connection_opened()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        latency = (time.time() - start) * 1000
        METRICS.record_request(
            method=request.method,
            path=request.url.path,
            status=status,
            latency_ms=latency,
        )
        METRICS.connection_closed()
    return response


class PromptRequest(BaseModel):
    prompt: str
    session_id: str = ""
    mode: str = "coding"


class ActionRequest(BaseModel):
    action_id: str
    approved: bool
    reason: str = ""


class WorkspaceRequest(BaseModel):
    path: str = ""


class BatchRequest(BaseModel):
    """Multiple prompts for static batch inference."""

    prompts: list[str]
    temperature: float = 0.7
    max_tokens: int = 512


sessions: dict[str, dict[str, Any]] = {}
audit_log: list[dict[str, Any]] = []
workspaces: dict[str, str] = {"default": os.getcwd()}


@app.get("/")
async def root():
    return {"service": "Aurelius API", "version": "1.0.0", "status": "ready"}


@app.get("/health")
async def health() -> dict:
    """Liveness probe — service is up."""
    return {
        "status": "ok",
        "uptime": time.time(),
        "sessions": len(sessions),
        "engine_loaded": _engine is not None,
    }


@app.get("/health/ready")
async def readiness() -> dict:
    """Readiness probe — model engine is loaded and ready to serve."""
    if _engine is None:
        raise HTTPException(503, detail="model engine not ready")
    return {"status": "ready", "engine": "loaded"}


@app.get("/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(
        METRICS.prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/workspaces")
async def create_workspace(req: WorkspaceRequest):
    wid = uuid.uuid4().hex[:8]
    workspaces[wid] = req.path or os.getcwd()
    return {"id": wid, "path": workspaces[wid]}


@app.get("/workspaces")
async def list_workspaces():
    return [{"id": k, "path": v} for k, v in workspaces.items()]


@app.post("/sessions")
async def create_session():
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {"id": sid, "events": [], "created_at": time.time()}
    return {"session_id": sid}


@app.post("/sessions/{sid}/prompt")
async def send_prompt(sid: str, req: PromptRequest):
    session = sessions.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")
    event = {
        "type": "response",
        "content": f"Received: {req.prompt[:80]}...",
        "timestamp": time.time(),
    }
    session["events"].append(event)
    return {"event": event}


@app.post("/actions/approve")
async def approve_action(req: ActionRequest):
    audit_log.append(
        {
            "action_id": req.action_id,
            "approved": req.approved,
            "reason": req.reason,
            "timestamp": time.time(),
        }
    )
    return {"status": "approved" if req.approved else "denied"}


@app.get("/audit")
async def get_audit():
    return {"audit_log": audit_log[-100:]}


@app.get("/frontend")
async def get_frontend():
    fp = Path(__file__).parent.parent.parent / "frontend" / "agent_cockpit.html"
    if fp.exists():
        return FileResponse(str(fp))
    return {"error": "Frontend not found"}


connected_clients: dict[str, WebSocket] = {}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = uuid.uuid4().hex[:8]
    connected_clients[client_id] = ws
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "prompt":
                response = {
                    "type": "response",
                    "content": f"Echo: {msg['content'][:60]}",
                    "timestamp": time.time(),
                }
                await ws.send_json(response)
    except WebSocketDisconnect:
        connected_clients.pop(client_id, None)


_engine: Callable[[EngineChatRequest], str] | None = None
_model_id: str = "aurelius-1.3b"
_rate_limiter: Callable[[str], bool] | None = None


@app.on_event("startup")
async def _load_engine() -> None:
    global _engine, _model_id, _engine_obj, _tokenizer
    model_path = os.environ.get("AURELIUS_MODEL_PATH", "checkpoints/aurelius_1.3b")
    backend = os.environ.get("AURELIUS_BACKEND", "vllm")
    tp = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))

    # ── Serving profile ────────────────────────────────────────────────────
    profile = os.environ.get("AURELIUS_SERVING_PROFILE", "production").lower()
    print(f"[profile] AURELIUS_SERVING_PROFILE={profile}")

    if profile == "single-gpu":
        _speculative_default = False
        _batch_default = int(os.environ.get("AURELIUS_BATCH_SIZE_MAX", "16"))
        _mem_util_default = float(os.environ.get("AURELIUS_GPU_MEM_UTIL", "0.85"))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"[profile:single-gpu] speculative=off batch_max={_batch_default} ")
        print(f"                    mem_util={_mem_util_default}")
    else:
        _speculative_default = True
        _batch_default = int(os.environ.get("AURELIUS_BATCH_SIZE_MAX", "32"))
        _mem_util_default = float(os.environ.get("AURELIUS_GPU_MEM_UTIL", "0.90"))

    # Explicit env overrides
    _speculative_env = os.environ.get("AURELIUS_SPECULATIVE_DECODING")
    speculative_decoding = (
        _speculative_env.lower() == "true" if _speculative_env else _speculative_default
    )
    max_num_seqs = _batch_default
    gpu_memory_utilization = _mem_util_default

    try:
        _engine, _model_id, _engine_obj = build_engine(
            backend=backend,
            model_path=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            speculative_decoding=speculative_decoding,
            max_num_seqs=max_num_seqs,
        )
        # Load tokenizer for batch endpoint
        try:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            print("[startup] Tokenizer loaded for batch endpoint")
        except Exception as e:
            print(f"[startup] Warning: tokenizer load failed — batch endpoint disabled: {e}")
            _tokenizer = None
        print(f"[startup] Engine loaded: {_model_id} ({backend})")
    except Exception:
        print("[startup] Engine load failed: {exc}")
        _engine = None
        _engine_obj = None
        _tokenizer = None


@app.on_event("startup")
async def _init_rate_limiter() -> None:
    global _rate_limiter
    try:
        _rate_limiter = get_rate_limiter()
        # Notify metrics about backend choice
        backend = os.environ.get("AURELIUS_RATE_LIMIT_REDIS_URL")
        METRICS.set_rate_limiter_backend("redis" if backend else "memory")
        print("[startup] Rate limiter backend: " + ("redis" if backend else "memory"))
    except Exception as e:
        print(f"[startup] Rate limiter init failed: {e}")
        _rate_limiter = None


def _get_engine() -> Callable[[EngineChatRequest], str]:
    if _engine is None:
        return make_mock_generate_fn()
    return _engine


def _sanitize_completion(text: str) -> str:
    """Clean model output before returning to client."""
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    text = " ".join(text.split())
    return text.strip()


@app.post("/v1/chat/completions")
async def chat_completion(request: dict) -> dict:
    """OpenAI-compatible chat completion endpoint (non-streaming)."""
    validate_chat_params(request)
    messages = request.get("messages")
    if not messages:
        raise HTTPException(400, "field 'messages' is required")

    engine_req = EngineChatRequest(
        model=request.get("model", _model_id),
        messages=messages,
        temperature=float(request.get("temperature", 0.7)),
        max_new_tokens=int(request.get("max_tokens", 512)),
    )
    generate_fn = _get_engine()
    try:
        raw_completion = generate_fn(engine_req)
        completion_text = _sanitize_completion(raw_completion)
    except Exception as exc:
        raise HTTPException(500, f"Generation error: {exc}") from exc

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/v1/batch/completions")
async def batch_completions(req: BatchRequest) -> dict:
    """Batch completion — processes multiple prompts in one forward pass.

    Returns an array of completions in the same order as the input prompts.
    Requires that the engine was initialized with a batch‑capable backend
    (vLLM). The tokenizer is loaded once at startup and cached.
    """
    if not req.prompts:
        raise HTTPException(400, detail="'prompts' list cannot be empty")

    if _tokenizer is None:
        raise HTTPException(503, detail="tokenizer not available — batch endpoint disabled")

    # Tokenize each prompt using the chat template
    input_ids_list = [
        _tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for p in req.prompts
    ]

    if _engine_obj is None or not hasattr(_engine_obj, "generate_batch"):
        raise HTTPException(503, detail="engine not ready or batch unsupported")

    try:
        raw_results = _engine_obj.generate_batch(
            input_ids_list=input_ids_list,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except Exception as exc:
        raise HTTPException(500, f"Batch generation error: {exc}") from exc

    completions = [_sanitize_completion(txt) for txt in raw_results]
    return {"completions": completions, "count": len(completions)}


@app.get("/v1/models")
async def list_models() -> dict:
    """Return a minimal list of available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": _model_id,
                "object": "model",
                "owned_by": "aurelius",
                "permission": [],
            }
        ],
    }


def start(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aurelius FastAPI inference server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")  # default is localhost for security
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    args = parser.parse_args()
    start(host=args.host, port=args.port)
