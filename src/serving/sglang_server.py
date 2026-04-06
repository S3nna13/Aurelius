"""SGLang production server launcher for Aurelius 1.3B.

Provides an OpenAI-compatible API with AWQ quantization,
tensor parallelism, health checks, and graceful lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("aurelius.sglang")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SGLangConfig:
    """Server configuration for SGLang-backed inference."""

    model_path: str = "aurelius/aurelius-1.3b-awq"
    """HuggingFace model ID or local path (AWQ-quantized)."""

    host: str = "0.0.0.0"
    port: int = 8000

    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism."""

    quantization: str = "awq"
    """Quantization method (awq for production)."""

    max_model_len: int = 8192
    """Maximum sequence length."""

    mem_fraction_static: float = 0.85
    """Fraction of GPU memory reserved for KV cache."""

    dtype: str = "float16"
    """Model data type."""

    trust_remote_code: bool = False
    """Whether to trust remote code from HuggingFace."""

    log_level: str = "info"

    @property
    def sglang_launch_args(self) -> list[str]:
        """Build CLI arguments for `python -m sglang.launch_server`."""
        return [
            "--model-path", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tp", str(self.tensor_parallel_size),
            "--quantization", self.quantization,
            "--max-model-len", str(self.max_model_len),
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--dtype", self.dtype,
            "--log-level", self.log_level,
            *(["--trust-remote-code"] if self.trust_remote_code else []),
        ]


# ---------------------------------------------------------------------------
# SGLang process manager
# ---------------------------------------------------------------------------

class SGLangProcess:
    """Manages the SGLang server subprocess lifecycle."""

    def __init__(self, config: SGLangConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._healthy: bool = False

    async def start(self) -> None:
        """Launch the SGLang server subprocess."""
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            *self.config.sglang_launch_args,
        ]
        logger.info("Starting SGLang server: %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for the server to become healthy
        await self._wait_for_health()

    async def _wait_for_health(
        self,
        timeout: float = 300.0,
        interval: float = 2.0,
    ) -> None:
        """Poll the SGLang health endpoint until ready."""
        backend_url = f"http://127.0.0.1:{self.config.port}"
        deadline = time.monotonic() + timeout

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(
                        f"{backend_url}/health", timeout=5.0
                    )
                    if resp.status_code == 200:
                        self._healthy = True
                        logger.info("SGLang server is healthy")
                        return
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(interval)

        msg = f"SGLang server did not become healthy within {timeout}s"
        raise TimeoutError(msg)

    async def stop(self) -> None:
        """Gracefully terminate the SGLang subprocess."""
        if self._process is None:
            return

        logger.info("Shutting down SGLang server (pid=%s)", self._process.pid)
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=30.0)
        except TimeoutError:
            logger.warning("SGLang server did not exit gracefully; killing")
            self._process.kill()
            await self._process.wait()
        self._healthy = False
        self._process = None

    @property
    def is_healthy(self) -> bool:
        return self._healthy and self._process is not None


# ---------------------------------------------------------------------------
# FastAPI gateway
# ---------------------------------------------------------------------------

sglang_process: SGLangProcess | None = None
_config: SGLangConfig = SGLangConfig()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage SGLang subprocess across application lifecycle."""
    global sglang_process
    sglang_process = SGLangProcess(_config)
    await sglang_process.start()
    yield
    await sglang_process.stop()


app = FastAPI(
    title="Aurelius Inference API",
    version="0.1.0",
    description="OpenAI-compatible API gateway for Aurelius 1.3B via SGLang",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for load balancers and orchestrators."""
    if sglang_process is None or not sglang_process.is_healthy:
        raise HTTPException(status_code=503, detail="SGLang backend unavailable")
    return {
        "status": "healthy",
        "model": _config.model_path,
        "quantization": _config.quantization,
        "tensor_parallel": _config.tensor_parallel_size,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """OpenAI-compatible model listing."""
    return {
        "object": "list",
        "data": [
            {
                "id": _config.model_path,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "aurelius",
            }
        ],
    }


async def _proxy_to_sglang(request: Request, path: str) -> Response:
    """Forward requests to the SGLang backend."""
    backend_url = f"http://127.0.0.1:{_config.port}{path}"
    body = await request.body()

    async with httpx.AsyncClient() as client:
        try:
            backend_resp = await client.post(
                backend_url,
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            )
        except httpx.ConnectError as exc:
            raise HTTPException(
                status_code=503, detail="SGLang backend unavailable"
            ) from exc

    # Check if streaming response
    content_type = backend_resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:

        async def stream_events() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient() as sc:
                async with sc.stream(
                    "POST",
                    backend_url,
                    content=body,
                    headers={"Content-Type": "application/json"},
                    timeout=120.0,
                ) as stream:
                    async for chunk in stream.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
        )

    return JSONResponse(
        content=backend_resp.json(),
        status_code=backend_resp.status_code,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """OpenAI-compatible chat completions endpoint."""
    return await _proxy_to_sglang(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    """OpenAI-compatible text completions endpoint."""
    return await _proxy_to_sglang(request, "/v1/completions")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _handle_signal(signum: int, _: Any) -> None:
    """Handle shutdown signals."""
    logger.info("Received signal %s, initiating shutdown", signum)
    raise SystemExit(0)


def serve(config: SGLangConfig | None = None) -> None:
    """Launch the Aurelius inference server.

    Args:
        config: Server configuration. Uses defaults if None.
    """
    global _config
    if config is not None:
        _config = config

    logging.basicConfig(
        level=getattr(logging, _config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    signal.signal(signal.SIGTERM, _handle_signal)

    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port + 1,  # Gateway on port+1, SGLang on configured port
        log_level=_config.log_level,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aurelius SGLang Server")
    parser.add_argument("--model-path", default=SGLangConfig.model_path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    cfg = SGLangConfig(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tp,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
    )
    serve(cfg)
