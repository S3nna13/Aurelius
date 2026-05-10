# Aurelius — CLI Terminal

Dragon-themed AI coding terminal with lightning blue (#00BFFF) theme.
Inspired by Claude Code, OPENDEV, Terminal-Bench, NL2SH, YAMLE.

## Quick Start
```bash
aurelius                   # Launch interactive CLI (terminal REPL with slash commands)
aurelius-cli               # Same as above
aurelius chat --react --model-path <ckpt>  # Enable ReAct tool-use loop
aurelius serve --engine agentic --model-path <ckpt>  # Start agentic API server
aurelius serve --port 8080 # Start API server
```

## Features
- Dual-agent architecture (planner + executor)
- ReAct tool-use loop for local model chat and the API backend
- NL→Bash translation with functional equivalence
- 7-mode deny-first permission system
- Dragon ASCII art + lightning blue accent color
- Session persistence with save/load
- Session statistics: token counts, latency, tokens/sec (use `/stats` command)
- Rich/ANSI fallback rendering with or without terminal support

## Chat Modes

The CLI runs in **terminal REPL mode** (via `aurelius` or `aurelius chat`).

The **Mission Control web frontend** (`aurelius serve` or `frontend/`) offers two chat modes:

| Mode | Endpoint | Behavior |
|------|----------|----------|
| **Agent Chat** (default) | `/api/chat/agent` | Routes to the best-fit agent (Coding, Research, General) based on message content |
| **Model Chat** | `/api/chat/completions` | Direct model inference; backend selection controls which provider handles the request |

In Model Chat, the **backend selector** lets you choose:
- **Auto** — Uses the **Default Backend** saved in Settings (falls back to `vllm` if unset)
- **Mock** — Fast local stub for offline testing (no upstream call)
- **vLLM** — GPU-backed inference server at the configured vLLM upstream URL
- **Agentic** — ReAct tool-use loop via the Aurelius agent runtime

Explicit choices (`mock`, `vLLM`, `agentic`) override the Settings default; `Auto` omits the backend field so the BFF resolves it from config.

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/quit`, `/exit` | End the session |
| `/reset` | Clear conversation history |
| `/history` | Show conversation history |
| `/system <prompt>` | Set a new system prompt |
| `/save <id>` | Save conversation to disk |
| `/load <id>` | Load a saved conversation |
| `/list` | List saved conversations |
| `/tokens` | Show token usage for current session |
| `/stats` | Show session statistics (turns, tokens, latency) |
| `/model` | Show current model info |
| `/version` | Show CLI version |
| `/clear` | Clear the screen |
