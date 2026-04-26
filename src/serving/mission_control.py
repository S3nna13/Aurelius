"""Aurelius Mission Control — centralized operations dashboard.

Run: python -m src.serving.mission_control --port 7870

Opens a web dashboard at http://localhost:7870 with:
  - Natural-language command bar
  - Agent status panel
  - Activity timeline
  - Skill / plugin registry view
  - Simple task board

Endpoints:
  GET  /                  — serve the dashboard HTML
  POST /api/command       — accept a natural-language command, return result JSON
  GET  /api/status        — system status JSON (agents, skills, plugins)
  GET  /api/activity      — activity feed JSON
  GET  /health            — health check
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from src.agent.command_dispatcher import (
    CommandDispatcher,
    CommandDispatchError,
)
from src.agent.nl_command_parser import (
    NLCommandParseError,
    NLCommandParser,
)

logger = logging.getLogger(__name__)

#: Maximum allowed request body size (1 MiB).
_MAX_CONTENT_LENGTH = 1_048_576

# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_HTML_TEMPLATE: str = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Aurelius Mission Control</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f0f1a;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    background: #1a1a2e;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #2d2d44;
    flex-shrink: 0;
  }
  header h1 { font-size: 1.1rem; font-weight: 600; color: #4fc3f7; }
  .status-badge {
    background: #16213e;
    color: #4fc3f7;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 12px;
    border: 1px solid #0f3460;
  }
  #main {
    flex: 1;
    display: flex;
    overflow: hidden;
  }
  #sidebar {
    width: 260px;
    background: #16162a;
    border-right: 1px solid #2d2d44;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    flex-shrink: 0;
  }
  .panel {
    padding: 14px 16px;
    border-bottom: 1px solid #2d2d44;
  }
  .panel h2 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9e9eb0;
    margin-bottom: 10px;
  }
  .panel-item {
    background: #1a1a2e;
    border: 1px solid #2d2d44;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 0.82rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .panel-item .id { color: #4fc3f7; font-weight: 500; }
  .panel-item .state { font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; }
  .state-idle { background: #1b5e20; color: #81c784; }
  .state-running { background: #e65100; color: #ffcc80; }
  .state-crashed { background: #b71c1c; color: #ef9a9a; }
  #center {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  #command-bar {
    background: #1a1a2e;
    border-bottom: 1px solid #2d2d44;
    padding: 14px 20px;
    display: flex;
    gap: 10px;
    align-items: center;
    flex-shrink: 0;
  }
  #command-input {
    flex: 1;
    background: #0f0f1a;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    color: #e0e0e0;
    font-size: 0.95rem;
    padding: 10px 14px;
    outline: none;
    font-family: inherit;
  }
  #command-input:focus { border-color: #4fc3f7; }
  #command-btn {
    background: #1565c0;
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.15s;
  }
  #command-btn:hover { background: #1976d2; }
  #command-btn:disabled { background: #374151; color: #6b7280; cursor: not-allowed; }
  #output-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg {
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.5;
    font-size: 0.9rem;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    background: #1565c0;
    color: #fff;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }
  .msg.assistant {
    background: #1a1a2e;
    color: #e0e0e0;
    align-self: flex-start;
    border: 1px solid #2d2d44;
    border-bottom-left-radius: 4px;
  }
  .msg.system {
    background: #16213e;
    color: #9e9eb0;
    align-self: center;
    font-size: 0.8rem;
    border: 1px solid #0f3460;
  }
  .msg.error {
    background: #b71c1c;
    color: #ffcdd2;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
  }
  #board {
    display: flex;
    gap: 12px;
    padding: 0 20px 20px;
  }
  .board-col {
    flex: 1;
    background: #16162a;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    padding: 12px;
    min-height: 140px;
  }
  .board-col h3 {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #9e9eb0;
    margin-bottom: 10px;
    letter-spacing: 0.06em;
  }
  .board-card {
    background: #1a1a2e;
    border: 1px solid #2d2d44;
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 6px;
    font-size: 0.82rem;
  }
  #activity-feed {
    height: 180px;
    overflow-y: auto;
    border-top: 1px solid #2d2d44;
    background: #0f0f1a;
    padding: 10px 20px;
    font-size: 0.8rem;
    color: #9e9eb0;
    flex-shrink: 0;
  }
  .activity-entry {
    padding: 3px 0;
    border-bottom: 1px solid #1a1a2e;
  }
  .activity-entry .time { color: #4fc3f7; margin-right: 6px; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2d2d44; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>🜔 Aurelius Mission Control</h1>
  <span class="status-badge" id="conn-badge">● Connected</span>
</header>
<div id="main">
  <div id="sidebar">
    <div class="panel">
      <h2>Agents</h2>
      <div id="agent-list"><div class="panel-item">"
        "<span class=\"id\">default</span>"
        "<span class=\"state state-idle\">IDLE</span></div></div>
    </div>
    <div class="panel">
      <h2>Skills</h2>
      <div id="skill-list"><div class="panel-item">"
        "<span class=\"id\">code-review</span>"
        "<span class=\"state state-idle\">ACTIVE</span></div></div>
    </div>
    <div class="panel">
      <h2>Plugins</h2>
      <div id="plugin-list"><div class="panel-item">"
        "<span class=\"id\">mcp-core</span>"
        "<span class=\"state state-idle\">LOADED</span></div></div>
    </div>
  </div>
  <div id="center">
    <div id="command-bar">
      <input id="command-input" type="text" "
        "placeholder=\"Type a command...\" />
      <button id="command-btn">Execute</button>
    </div>
    <div id="output-area"></div>
    <div id="board">
      <div class="board-col"><h3>To Do</h3>"
        "<div class="board-card">Review PR #42</div></div>
      <div class="board-col"><h3>In Progress</h3>"
        "<div class="board-card">Agent refactoring</div></div>
      <div class="board-col"><h3>Done</h3>"
        "<div class="board-card">Cycle 207 deploy</div></div>
    </div>
    <div id="activity-feed"></div>
  </div>
</div>
<script>
  const inputEl = document.getElementById('command-input');
  const btnEl = document.getElementById('command-btn');
  const outputEl = document.getElementById('output-area');
  const activityEl = document.getElementById('activity-feed');
  const agentListEl = document.getElementById('agent-list');
  const skillListEl = document.getElementById('skill-list');
  const pluginListEl = document.getElementById('plugin-list');

  function appendMsg(role, text) {
    const div = document.createElement('div');
    div.classList.add('msg', role);
    div.textContent = text;
    outputEl.appendChild(div);
    outputEl.scrollTop = outputEl.scrollHeight;
  }

  function appendActivity(text) {
    const div = document.createElement('div');
    div.classList.add('activity-entry');
    const t = new Date().toLocaleTimeString();
    div.innerHTML = '<span class="time">' + t + '</span>' + escapeHtml(text);
    activityEl.appendChild(div);
    activityEl.scrollTop = activityEl.scrollHeight;
  }

  function escapeHtml(t) {
    return t.replace(/[&<>"']/g, m => ({""
        "'&':'&amp;','<':'&lt;','>':'&gt;',""
        "'"':'&quot;','\\'':'&#39;'}[m]));
  }

  async function sendCommand() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    btnEl.disabled = true;
    appendMsg('user', text);
    appendActivity('Command: ' + text);

    try {
      const res = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: text })
      });
      const data = await res.json();
      if (data.error) {
        appendMsg('error', data.error);
      } else {
        appendMsg('assistant', data.output || '(no output)');
      }
      appendActivity('Result: ' + (data.success ? 'success' : 'failed'));
    } catch (err) {
      appendMsg('error', 'Network error: ' + err.message);
    } finally {
      btnEl.disabled = false;
      inputEl.focus();
    }
  }

  async function refreshStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      agentListEl.innerHTML = (data.agents || []).map(a =>
        '<div class="panel-item"><span class="id">' + escapeHtml(a.id) +
        '</span><span class="state state-' + (a.state || 'idle').toLowerCase() + '">' +
        (a.state || 'IDLE') + '</span></div>'
      ).join('');
      skillListEl.innerHTML = (data.skills || []).map(s =>
        '<div class="panel-item"><span class="id">' + escapeHtml(s.id) +
        '</span><span class="state state-' + (s.active ? 'running' : 'idle') + '">' +
        (s.active ? 'ACTIVE' : 'INACTIVE') + '</span></div>'
      ).join('');
      pluginListEl.innerHTML = (data.plugins || []).map(p =>
        '<div class="panel-item"><span class="id">' + escapeHtml(p.id) +
        '</span><span class="state state-idle">LOADED</span></div>'
      ).join('');
    } catch (e) {
      document.getElementById('conn-badge').textContent = '● Disconnected';
      document.getElementById('conn-badge').style.color = '#ef5350';
    }
  }

  btnEl.addEventListener('click', sendCommand);
  inputEl.addEventListener('keydown', e => { if (e.key === 'Enter') sendCommand(); });
  setInterval(refreshStatus, 5000);
  refreshStatus();
  inputEl.focus();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Activity log
# ---------------------------------------------------------------------------


@dataclass
class ActivityEntry:
    """One entry in the mission-control activity feed."""

    id: str
    timestamp: float
    command: str
    success: bool
    output: str


@dataclass
class ActivityLog:
    """Thread-safe-ish in-memory activity log (GIL protects list ops)."""

    _entries: list[ActivityEntry] = field(default_factory=list)
    _max_entries: int = 1000

    def append(self, command: str, success: bool, output: str) -> ActivityEntry:
        entry = ActivityEntry(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            command=command,
            success=success,
            output=output,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]
        return entry

    def list_entries(self, limit: int = 50) -> list[ActivityEntry]:
        return self._entries[-limit:]

    def to_dicts(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "command": e.command,
                "success": e.success,
                "output": e.output,
            }
            for e in self._entries[-limit:]
        ]


# ---------------------------------------------------------------------------
# Server handler
# ---------------------------------------------------------------------------


class MissionControlHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        if content_length < 0:
            raise ValueError("Negative Content-Length")
        if content_length > _MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Content-Length {content_length} exceeds maximum {_MAX_CONTENT_LENGTH}"
            )
        return self.rfile.read(content_length)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = _HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        if self.path == "/api/status":
            self._handle_status()
            return

        if self.path == "/api/activity":
            self._handle_activity()
            return

        self.send_response(404)
        self.end_headers()

    def _handle_status(self) -> None:
        server = self.server
        status: dict[str, Any] = {
            "agents": [],
            "skills": [],
            "plugins": [],
        }

        # Agents
        supervisor = getattr(server, "supervisor", None)
        if supervisor is not None and hasattr(supervisor, "_agents"):
            for aid, rec in supervisor._agents.items():
                status["agents"].append(
                    {
                        "id": aid,
                        "state": str(getattr(rec, "state", "UNKNOWN")),
                    }
                )
        else:
            status["agents"].append({"id": "default", "state": "IDLE"})

        # Skills
        skill_catalog = getattr(server, "skill_catalog", None)
        if skill_catalog is not None and hasattr(skill_catalog, "list"):
            try:
                for entry in skill_catalog.list():
                    status["skills"].append(
                        {
                            "id": entry.skill_id,
                            "active": getattr(entry, "active", False),
                        }
                    )
            except Exception:
                logger.warning("MissionControl: failed to enumerate skills", exc_info=True)
        else:
            status["skills"].append({"id": "code-review", "active": True})

        # Plugins
        plugin_loader = getattr(server, "plugin_loader", None)
        if plugin_loader is not None and hasattr(plugin_loader, "list_loaded"):
            try:
                for pid in plugin_loader.list_loaded():
                    status["plugins"].append({"id": pid})
            except Exception:
                logger.warning("MissionControl: failed to enumerate plugins", exc_info=True)
        else:
            status["plugins"].append({"id": "mcp-core"})

        self._send_json(200, status)

    def _handle_activity(self) -> None:
        log = getattr(self.server, "activity_log", None)
        if log is None:
            self._send_json(200, {"entries": []})
            return
        self._send_json(200, {"entries": log.to_dicts()})

    def do_POST(self):
        if self.path == "/api/command":
            self._handle_command()
            return
        self.send_response(404)
        self.end_headers()

    def _handle_command(self) -> None:
        try:
            raw = self._read_body()
        except ValueError as exc:
            self._send_json(413, {"error": str(exc)})
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        text = payload.get("command", "")
        if not isinstance(text, str):
            self._send_json(400, {"error": "command must be a string"})
            return

        server = self.server
        parser = getattr(server, "nl_parser", None) or NLCommandParser()
        dispatcher = getattr(server, "command_dispatcher", None) or CommandDispatcher()
        log = getattr(server, "activity_log", None)

        try:
            parsed = parser.parse(text)
        except NLCommandParseError as exc:
            if log is not None:
                log.append(text, False, str(exc))
            self._send_json(400, {"error": str(exc), "success": False})
            return

        try:
            result = dispatcher.dispatch(parsed)
        except CommandDispatchError as exc:
            if log is not None:
                log.append(text, False, str(exc))
            self._send_json(400, {"error": str(exc), "success": False})
            return
        except Exception as exc:
            logger.exception("Unexpected dispatch error")
            if log is not None:
                log.append(text, False, str(exc))
            self._send_json(500, {"error": str(exc), "success": False})
            return

        if log is not None:
            log.append(text, result.success, result.output)

        self._send_json(
            200,
            {
                "success": result.success,
                "output": result.output,
                "action": parsed.action,
                "target": parsed.target,
            },
        )

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class MissionControlServer(HTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        nl_parser: NLCommandParser | None = None,
        command_dispatcher: CommandDispatcher | None = None,
        skill_catalog: Any | None = None,
        plugin_loader: Any | None = None,
        supervisor: Any | None = None,
        bind_and_activate: bool = True,
    ):
        super().__init__(
            (host, port),
            MissionControlHandler,
            bind_and_activate=bind_and_activate,
        )
        self.nl_parser = nl_parser
        self.command_dispatcher = command_dispatcher
        self.skill_catalog = skill_catalog
        self.plugin_loader = plugin_loader
        self.supervisor = supervisor
        self.activity_log = ActivityLog()


def create_mission_control_server(
    host: str,
    port: int,
    *,
    nl_parser: NLCommandParser | None = None,
    command_dispatcher: CommandDispatcher | None = None,
    skill_catalog: Any | None = None,
    plugin_loader: Any | None = None,
    supervisor: Any | None = None,
    bind_and_activate: bool = True,
) -> MissionControlServer:
    return MissionControlServer(
        host,
        port,
        nl_parser=nl_parser,
        command_dispatcher=command_dispatcher,
        skill_catalog=skill_catalog,
        plugin_loader=plugin_loader,
        supervisor=supervisor,
        bind_and_activate=bind_and_activate,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    arg_parser = argparse.ArgumentParser(description="Aurelius Mission Control")
    arg_parser.add_argument(
        "--port", type=int, default=7870, help="Port to listen on (default: 7870)"
    )
    arg_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose)",
    )
    args = arg_parser.parse_args()

    server = create_mission_control_server(args.host, args.port)
    url = f"http://localhost:{args.port}"
    logger.info("Serving Aurelius Mission Control at %s", url)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()


__all__ = [
    "ActivityEntry",
    "ActivityLog",
    "MissionControlHandler",
    "MissionControlServer",
    "create_mission_control_server",
]
