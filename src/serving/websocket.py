"""WebSocket handler for real-time agent streaming."""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    from fastapi import WebSocket as FastAPIWebSocket  # type: ignore[import-untyped]
except ImportError:
    FastAPIWebSocket = None  # type: ignore[assignment,misc]

logger = logging.getLogger("ark.serving.ws")


async def handle_agent_ws(websocket: Any, memory_manager: Any = None) -> None:
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            task = message.get("task", "")
            mode = message.get("mode", "chat")

            await websocket.send_json({"type": "status", "content": "processing"})

            if mode == "memory" and memory_manager is not None:
                results = memory_manager.contextualize(task, top_k=5)
                for r in results:
                    await websocket.send_json({"type": "memory", "content": r})
            else:
                tokens = f"Processing: {task}".split()
                for token in tokens:
                    await websocket.send_json({"type": "token", "content": token + " "})
                    import asyncio

                    await asyncio.sleep(0.02)

            await websocket.send_json({"type": "done", "content": ""})

    except Exception as exc:
        logger.debug("WebSocket disconnected: %s", exc)
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: S110
            pass
