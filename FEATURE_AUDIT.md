# Aurelius Feature Audit — Clean-Room Inspirations

> Zero new dependencies. Zero citations. Pure stdlib + existing Aurelius infrastructure.

## Sources Analyzed (20 repos)

Roo-Code, PowerShell, ml-intern, ds2api, build-your-own-x, awesome-codex-skills, Kronos, FinceptTerminal, Archon, DeepTutor, markitdown, GenericAgent, timesfm, rtk, claude-mem, Open-Generative-AI, hackingtool, multica, DeepGEMM, OpenSRE.

---

## Tier 1: Immediately Implementable (stdlib only)

### 1. Agent Modes (from Roo-Code)
**What:** Behavior presets — Code, Architect, Ask, Debug, Custom. Each mode has a system prompt prefix + allowed toolset.
**File:** `src/agent/agent_mode_registry.py`
**Pattern:** Dataclass registry. `AgentMode(name, prompt_prefix, allowed_tools)`. Switch at runtime.
**Value:** Users can switch agent personality instantly without restarting.

### 2. Workflow Engine (from Archon)
**What:** YAML-free DAG executor. Nodes = AI prompts, bash commands, human gates, loops. Dependencies between nodes. Checkpoints.
**File:** `src/agent/workflow_engine.py`
**Pattern:** `WorkflowNode`, `WorkflowDAG`, `WorkflowExecutor`. Support `depends_on`, `loop_until`, `interactive_gate`.
**Value:** Deterministic, repeatable agent processes. Fire-and-forget task execution.

### 3. Command Output Compressor (from rtk)
**What:** Filter/compress shell command outputs before they hit the agent context. Smart truncation, deduplication, grouping.
**File:** `src/cli/output_compressor.py`
**Pattern:** `OutputCompressor` with strategies: `smart_filter`, `group_by_directory`, `truncate_lines`, `dedup_counts`.
**Value:** 60-90% token savings on `ls`, `git status`, `pytest`, `grep` outputs.

### 4. Layered Memory (from GenericAgent)
**What:** L0 Meta Rules → L1 Insight Index → L2 Global Facts → L3 Task Skills → L4 Session Archive.
**File:** `src/memory/layered_memory.py`
**Pattern:** 5-layer hierarchy with progressive recall. Each layer has different TTL and retrieval cost.
**Value:** Token-efficient long-term memory. Right knowledge at right scope.

### 5. Progressive Search (from claude-mem)
**What:** 3-layer search: (1) compact index with IDs, (2) timeline context around results, (3) full details only for filtered IDs.
**File:** `src/memory/progressive_search.py`
**Pattern:** `search(query)` → index, `timeline(id)` → context, `get_full(ids)` → details.
**Value:** ~10x token savings vs dumping entire memory into context.

### 6. Self-Evolving Skill Tree (from GenericAgent)
**What:** After task completion, crystallize execution path into reusable skill. Skills accumulate over time forming a tree.
**File:** `src/agent/skill_evolver.py`
**Pattern:** `SkillEvolver.crystallize(task_record) -> Skill`. Auto-extract parameters, preconditions, steps.
**Value:** Agent gets faster at repeated tasks. Personal skill tree per deployment.

### 7. Document-to-Markdown Converter (from markitdown)
**What:** Convert files to Markdown using only stdlib. HTML (html.parser), JSON, XML, CSV, plain text.
**File:** `src/tools/document_converter.py`
**Pattern:** `DocumentConverter.convert(path) -> MarkdownResult`. MIME-type detection + dispatch.
**Value:** Agent can ingest any document type as context without external parsers.

### 8. Socratic Tutor (from DeepTutor)
**What:** Socratic questioning mode. Step-by-step hints, knowledge tracing, difficulty adaptation.
**File:** `src/agent/socratic_tutor.py`
**Pattern:** `SocraticTutor.ask(question) -> Hint` (not answer). Tracks mastery per concept.
**Value:** Educational agent mode. Teaches by guiding, not telling.

### 9. Task Scheduler (from Kronos)
**What:** Cron-like scheduling for agent tasks. Recurring jobs, one-shot delayed tasks.
**File:** `src/agent/task_scheduler.py`
**Pattern:** `TaskScheduler.schedule(cron_expr, task_fn)`. Background thread execution.
**Value:** Agents can perform periodic maintenance, monitoring, reporting.

### 10. Multi-Agent Coordination (from multica)
**What:** Coordination patterns for multiple agents — consensus voting, delegation chains, competition.
**File:** `src/agent/multi_agent_coordination.py`
**Pattern:** `ConsensusVoter`, `DelegationChain`, `AgentSwarm`. Message-passing between agents.
**Value:** Teams of agents working together on complex tasks.

### 11. Pipeline Processor (from PowerShell)
**What:** Object-oriented command pipeline. JSON/streamable object transforms.
**File:** `src/cli/pipeline_processor.py`
**Pattern:** `cat data.json | filter "x>5" | sort "name" | head 10`. Chainable transforms.
**Value:** Shell-like composability for agent tool outputs.

### 12. SRE Golden Signals (from OpenSRE)
**What:** Track latency, traffic, errors, saturation. Synthetic incident simulation.
**File:** `src/monitoring/sre_metrics.py`
**Pattern:** `SREMetricsCollector` with counters, histograms, health scores.
**Value:** Agent can monitor its own performance and trigger self-healing.

### 13. Terminal Multi-Pane Dashboard (from FinceptTerminal)
**What:** ANSI-based multi-pane dashboard. No curses dependency.
**File:** `src/serving/terminal_dashboard.py`
**Pattern:** `TerminalDashboard` with panes for logs, metrics, agent status, command input.
**Value:** Power-user terminal UI without heavy TUI frameworks.

### 14. Tutorial Engine (from build-your-own-x)
**What:** Guided step-by-step tutorials / learning paths for building things.
**File:** `src/agent/tutorial_engine.py`
**Pattern:** `Tutorial` = ordered steps with checks. Progress tracking, hint system.
**Value:** Onboard users to Aurelius capabilities interactively.

### 15. Security Scanner Collection (from hackingtool)
**What:** Basic network reconnaissance using stdlib (socket, ssl). Port scanner, subdomain enum.
**File:** `src/security/port_scanner.py`, `src/security/subdomain_enum.py`
**Pattern:** `PortScanner.scan(host, ports)`. Async socket connections.
**Value:** Agent can perform basic security audits.

### 16. Time Series Analyzer (from timesfm)
**What:** Basic time series ops using stdlib math: moving average, trend detection, seasonality.
**File:** `src/analysis/time_series.py`
**Pattern:** `TimeSeriesAnalyzer.moving_average(data, window)`, `.detect_trend(data)`.
**Value:** Agent can analyze temporal data without ML libraries.

---

## Tier 2: Partially Implementable (limited by stdlib)

| Feature | Source | Limitation | Workaround |
|---------|--------|------------|------------|
| PDF/DOCX/PPTX parsing | markitdown | Need external libs | Skip; support only HTML/JSON/XML/CSV/text |
| Browser automation | GenericAgent | Need selenium/playwright | Skip; use `urllib` + `html.parser` for basic scraping |
| Image OCR | markitdown | Need tesseract/vision API | Skip |
| Speech transcription | markitdown | Need whisper/etc | Skip |
| Vector semantic search | claude-mem | Need embeddings | Use keyword + inverted index instead |
| Real trading | FinceptTerminal | Need broker APIs | Skip; simulate with data feeds |
| QuantLib analytics | FinceptTerminal | Need QuantLib | Skip; basic math only |
| Optimized GEMM | DeepGEMM | CUDA kernels | Skip; not relevant to agent layer |

---

## Proposed Roadmap

### Cycle 209 — Agent Core Power-Up
- `agent_mode_registry.py` — Roo-Code modes
- `workflow_engine.py` — Archon-style DAG executor
- `output_compressor.py` — rtk-style command output filtering

### Cycle 210 — Memory & Learning
- `layered_memory.py` — GenericAgent 5-layer memory
- `progressive_search.py` — claude-mem 3-layer search
- `skill_evolver.py` — GenericAgent self-evolving skills

### Cycle 211 — Tools & Tutoring
- `document_converter.py` — markitdown stdlib converter
- `socratic_tutor.py` — DeepTutor Socratic mode
- `task_scheduler.py` — Kronos cron scheduler

### Cycle 212 — Multi-Agent & Coordination
- `multi_agent_coordination.py` — multica coordination patterns
- `pipeline_processor.py` — PowerShell object pipeline
- `tutorial_engine.py` — build-your-own-x guided paths

### Cycle 213 — Monitoring & Security
- `sre_metrics.py` — OpenSRE golden signals
- `terminal_dashboard.py` — FinceptTerminal multi-pane UI
- `port_scanner.py` + `subdomain_enum.py` — hackingtool basics
- `time_series.py` — timesfm basic analysis
