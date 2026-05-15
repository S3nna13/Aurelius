"""Agent training pipeline — collect trajectories and prepare training data."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .persistent_agent import PersistentAgent
from .react_loop import AgentTrace


def _compose_instruction(task: str, system_prompt: str) -> str:
    """Build the instruction text used for supervised training examples."""
    task_text = task.strip()
    system_text = system_prompt.strip()
    if system_text and task_text:
        return f"{system_text}\n\nTask: {task_text}"
    if system_text:
        return system_text
    return task_text


def trace_to_conversation(trace: AgentTrace) -> list[dict]:
    """Convert an AgentTrace to a chat-format conversation."""
    messages: list[dict] = []
    for step in trace.steps:
        if step.role == "assistant":
            msg = {"role": "assistant", "content": step.content}
            if step.tool_name:
                msg["tool_calls"] = [{"name": step.tool_name, "args": step.tool_input or {}}]
            messages.append(msg)
        elif step.role == "tool":
            messages.append({"role": "tool", "content": step.tool_output or step.error or ""})
    if trace.final_answer:
        messages.append({"role": "assistant", "content": trace.final_answer})
    return messages


def trace_to_training_entry(trace: AgentTrace, agent_id: str = "default") -> dict:
    """Format a trace for supervised fine-tuning (SFT format)."""
    conversations = trace_to_conversation(trace)
    instruction = _compose_instruction(trace.task, trace.system_prompt)
    return {
        "agent_id": agent_id,
        "status": trace.status,
        "steps_used": trace.steps_used,
        "task": trace.task,
        "system_prompt": trace.system_prompt,
        "instruction": instruction,
        "final_answer": trace.final_answer,
        "conversations": conversations,
        "timestamp": time.time(),
    }


class AgentTrainingCollector:
    """Collect and export agent trajectories for training."""

    def __init__(self, output_dir: str = ".aurelius/training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []

    def collect(self, agent: PersistentAgent) -> dict | None:
        trace = agent.last_trace
        if trace is None:
            return None
        entry = trace_to_training_entry(trace, agent.agent_id)
        self._entries.append(entry)
        return entry

    def export_jsonl(self, name: str = "agent_traces") -> str:
        fp = self.output_dir / f"{name}.jsonl"
        with open(fp, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry) + "\n")
        return str(fp)

    def export_sft_format(self, name: str = "sft_data") -> str:
        fp = self.output_dir / f"{name}.json"
        data = []
        for entry in self._entries:
            instruction = str(entry.get("instruction", "")).strip()
            output = str(entry.get("final_answer") or "").strip()
            if not output:
                assistant_turns = [
                    conv.get("content", "")
                    for conv in entry.get("conversations", [])
                    if conv.get("role") == "assistant"
                ]
                if assistant_turns:
                    output = str(assistant_turns[-1]).strip()
            if not instruction or not output:
                continue
            record = {
                "instruction": instruction,
                "output": output,
            }
            if entry.get("agent_id"):
                record["agent_id"] = entry["agent_id"]
            data.append(record)
        with open(fp, "w") as f:
            json.dump(data, f, indent=2)
        return str(fp)

    @property
    def count(self) -> int:
        return len(self._entries)
