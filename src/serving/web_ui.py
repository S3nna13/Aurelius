"""
Browser-based chat UI for Aurelius.

Run: python -m src.serving.web_ui --port 7860

Opens a browser chat interface at http://localhost:7860
Chat messages are sent to the API server at http://localhost:8080/v1/chat/completions
"""

import argparse
import json
import logging
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, List

logger = logging.getLogger(__name__)

HTML_TEMPLATE: str = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Aurelius Chat</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #16213e;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #0f3460;
    flex-shrink: 0;
  }
  header h1 {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e0e0e0;
  }
  .model-badge {
    background: #0f3460;
    color: #4fc3f7;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .message {
    max-width: 70%;
    padding: 10px 14px;
    border-radius: 16px;
    line-height: 1.5;
    font-size: 0.95rem;
    word-break: break-word;
    white-space: pre-wrap;
  }
  .message.user {
    background: #1565c0;
    color: #fff;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }
  .message.assistant {
    background: #2d2d44;
    color: #e0e0e0;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
  }
  .message.thinking {
    background: #2d2d44;
    color: #9e9eb0;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
    font-style: italic;
  }
  #input-area {
    background: #16213e;
    border-top: 1px solid #0f3460;
    padding: 14px 20px;
    display: flex;
    gap: 10px;
    align-items: flex-end;
    flex-shrink: 0;
  }
  #user-input {
    flex: 1;
    background: #0f3460;
    border: 1px solid #1a5276;
    border-radius: 12px;
    color: #e0e0e0;
    font-size: 0.95rem;
    padding: 10px 14px;
    resize: none;
    min-height: 44px;
    max-height: 160px;
    outline: none;
    font-family: inherit;
    line-height: 1.4;
  }
  #user-input:focus {
    border-color: #4fc3f7;
  }
  #send-btn {
    background: #1565c0;
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.15s;
    height: 44px;
  }
  #send-btn:hover { background: #1976d2; }
  #send-btn:disabled { background: #374151; color: #6b7280; cursor: not-allowed; }
  #messages::-webkit-scrollbar { width: 6px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
  #messages::-webkit-scrollbar-thumb { background: #0f3460; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>Aurelius</h1>
  <span class="model-badge">Aurelius</span>
</header>
<div id="messages"></div>
<div id="input-area">
  <textarea id="user-input" placeholder="Message Aurelius..." rows="1"></textarea>
  <button id="send-btn">Send</button>
</div>
<script>
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  let history = [];

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendMessage(role, text) {
    const div = document.createElement('div');
    div.classList.add('message', role);
    div.textContent = text;
    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendBtn.disabled = true;

    appendMessage('user', text);
    const thinkingDiv = appendMessage('thinking', 'Aurelius is thinking...');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: history })
      });
      const data = await res.json();
      const reply = data.response || '(no response)';

      thinkingDiv.remove();
      appendMessage('assistant', reply);

      history.push({ role: 'user', content: text });
      history.push({ role: 'assistant', content: reply });
    } catch (err) {
      thinkingDiv.remove();
      appendMessage('assistant', 'Error: ' + err.message);
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  sendBtn.addEventListener('click', sendMessage);

  inputEl.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  inputEl.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 160) + 'px';
  });

  inputEl.focus();
</script>
</body>
</html>"""


class WebUIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            body = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
                message = payload.get("message", "")
                history = payload.get("history", [])
                reply = self.server.generate_fn(message, history)
                body = json.dumps({"response": reply}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                logger.debug("Error in /api/chat: %s", exc)
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


class WebUIServer(HTTPServer):
    def __init__(self, host: str, port: int, generate_fn: Callable[[str, List], str]):
        super().__init__((host, port), WebUIHandler)
        self.generate_fn = generate_fn


def make_mock_generate_fn() -> Callable[[str, List], str]:
    def _generate(message: str, history: List) -> str:
        return f"You said: {message}"
    return _generate


def create_ui_server(host: str, port: int, generate_fn: Callable[[str, List], str]) -> WebUIServer:
    return WebUIServer(host, port, generate_fn)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Aurelius browser chat UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    parser.add_argument("--api-url", default="http://localhost:8080/v1/chat/completions",
                        help="Upstream API URL (default: http://localhost:8080/v1/chat/completions)")
    args = parser.parse_args()

    server = create_ui_server("0.0.0.0", args.port, make_mock_generate_fn())
    url = f"http://localhost:{args.port}"
    logger.info("Serving Aurelius UI at %s", url)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()
