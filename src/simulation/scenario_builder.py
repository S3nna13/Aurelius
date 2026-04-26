"""Scenario builder for constructing simulation scenarios."""

from __future__ import annotations


class ScenarioBuilder:
    """Build and validate simulation scenarios declaratively."""

    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        self._actors: dict[str, dict] = {}
        self._events: list[dict] = []
        self._environment: dict[str, object] = {}
        self._duplicate_actors: set[str] = set()

    def add_actor(self, name: str, role: str, params: dict | None = None) -> None:
        """Register an actor. Duplicate names are tracked for validation."""
        if name in self._actors:
            self._duplicate_actors.add(name)
        self._actors[name] = {"role": role, "params": params or {}}

    def add_event(
        self,
        timestamp: int,
        actor: str,
        action: str,
        payload: dict | None = None,
    ) -> None:
        """Append an event to the scenario timeline."""
        self._events.append(
            {
                "timestamp": timestamp,
                "actor": actor,
                "action": action,
                "payload": payload or {},
            }
        )

    def set_environment(self, key: str, value: object) -> None:
        """Set an environment variable for the scenario."""
        self._environment[key] = value

    def build(self) -> dict:
        """Return the complete scenario dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "actors": self._actors,
            "events": self._events,
            "environment": self._environment,
        }

    def validate(self) -> list[str]:
        """Return a list of validation errors; empty list means valid."""
        errors: list[str] = []
        if not self.scenario_id:
            errors.append("scenario_id must be non-empty")
        for name in self._duplicate_actors:
            errors.append(f"Duplicate actor: {name}")
        for event in self._events:
            if event["timestamp"] < 0:
                errors.append(f"Negative timestamp: {event['timestamp']}")
            if event["actor"] not in self._actors:
                errors.append(f"Unknown actor: {event['actor']}")
        return errors


SCENARIO_BUILDER_REGISTRY: dict[str, object] = {
    "builder": ScenarioBuilder,
}
