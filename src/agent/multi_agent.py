"""Multi-agent coordination — supervisor, swarm, debate patterns."""

from __future__ import annotations

from .agent_runtime import AgentRuntime, AgentSpec


def _new_messages(runtime: AgentRuntime, recipient: str, start_index: int) -> list[str]:
    return [
        msg.content
        for msg in runtime.mailbox[start_index:]
        if msg.recipient == recipient
    ]


class SupervisorCoordinator:
    """Supervisor-worker pattern: one supervisor delegates to workers."""

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self.supervisor_id: str | None = None
        self.worker_ids: list[str] = []

    def set_supervisor(self, spec: AgentSpec) -> str:
        self.supervisor_id = self.runtime.register_agent(spec)
        return self.supervisor_id

    def add_worker(self, spec: AgentSpec) -> str:
        wid = self.runtime.register_agent(spec)
        self.worker_ids.append(wid)
        return wid

    def delegate(self, task: str) -> list[str]:
        results: list[str] = []
        for worker_id in self.worker_ids:
            start = len(self.runtime.mailbox)
            self.runtime.route_message(task, self.supervisor_id or "", worker_id)
            results.extend(_new_messages(self.runtime, worker_id, start))
        return results


class SwarmCoordinator:
    """Swarm pattern: all agents broadcast and converge."""

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self.agent_ids: list[str] = []

    def add_agent(self, spec: AgentSpec) -> str:
        aid = self.runtime.register_agent(spec)
        self.agent_ids.append(aid)
        return aid

    def broadcast(self, message: str, sender: str) -> list[str]:
        results: list[str] = []
        for aid in self.agent_ids:
            if aid != sender:
                start = len(self.runtime.mailbox)
                self.runtime.route_message(message, sender, aid)
                for msg in _new_messages(self.runtime, aid, start):
                    results.append(f"{aid}: {msg[:100]}")
        return results


class DebateCoordinator:
    """Debate pattern: agents argue positions and converge."""

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self.debaters: list[str] = []
        self.rounds: int = 3

    def add_debater(self, spec: AgentSpec) -> str:
        did = self.runtime.register_agent(spec)
        self.debaters.append(did)
        return did

    def debate(self, topic: str) -> dict[str, list[str]]:
        transcript: dict[str, list[str]] = {d: [] for d in self.debaters}
        for rnd in range(self.rounds):
            for i, debater in enumerate(self.debaters):
                position = f"Round {rnd + 1}: {topic}"
                if rnd > 0:
                    prev = self.debaters[(i - 1) % len(self.debaters)]
                    prev_msgs = self.runtime.get_messages(prev)
                    if prev_msgs:
                        position += f"\nCountering: {prev_msgs[-1].content[:100]}"
                self.runtime.route_message(position, debater, debater)
                transcript[debater].append(position)
        return transcript
