"""Routes incoming requests to multiple model endpoints based on tag-matching rules.

Endpoints are registered with a name, model ID, priority, and an optional set of
tags.  Routing rules map a tag to a specific endpoint.  The first matching rule
wins; when no rule matches the highest-priority endpoint is returned, with
alphabetical name as the tie-breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelEndpoint:
    """A named model endpoint with a priority and searchable tags."""

    name: str
    model_id: str
    priority: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingRule:
    """Maps a single tag to a named endpoint."""

    tag: str
    endpoint_name: str


class ModelMultiplexer:
    """Routes requests to registered endpoints via tag-based rules.

    Args:
        endpoints: Initial list of :class:`ModelEndpoint` objects.
        rules:     Optional ordered list of :class:`RoutingRule` objects.
                   The first rule whose tag appears in *request_tags* wins.

    Raises:
        ValueError: If :meth:`route` is called when no endpoints are registered.
    """

    def __init__(
        self,
        endpoints: list[ModelEndpoint],
        rules: list[RoutingRule] | None = None,
    ) -> None:
        # Keyed by endpoint name for O(1) look-up
        self._endpoints: dict[str, ModelEndpoint] = {ep.name: ep for ep in endpoints}
        self._rules: list[RoutingRule] = list(rules) if rules else []

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, request_tags: list[str]) -> ModelEndpoint:
        """Return the best endpoint for *request_tags*.

        Rule matching:
            Iterate over registered rules in order.  The first rule whose
            ``.tag`` is contained in *request_tags* wins, provided the
            referenced endpoint still exists.  If no rule matches, fall back
            to the highest-priority endpoint (lowest index in a descending
            sort by priority, then ascending name).

        Raises:
            ValueError: When no endpoints have been registered.
        """
        if not self._endpoints:
            raise ValueError("No endpoints registered in ModelMultiplexer.")

        tag_set = set(request_tags)

        for rule in self._rules:
            if rule.tag in tag_set and rule.endpoint_name in self._endpoints:
                return self._endpoints[rule.endpoint_name]

        # Fallback: highest priority, alphabetical name as tie-breaker
        return max(
            self._endpoints.values(),
            key=lambda ep: (ep.priority, [-ord(c) for c in ep.name]),
        )

    # ------------------------------------------------------------------
    # Endpoint management
    # ------------------------------------------------------------------

    def add_endpoint(self, endpoint: ModelEndpoint) -> None:
        """Register a new endpoint (replaces an existing one with the same name)."""
        self._endpoints[endpoint.name] = endpoint

    def remove_endpoint(self, name: str) -> bool:
        """Remove the endpoint named *name*.

        Returns:
            ``True`` if the endpoint existed and was removed, ``False``
            otherwise.
        """
        if name in self._endpoints:
            del self._endpoints[name]
            return True
        return False

    def add_rule(self, rule: RoutingRule) -> None:
        """Append *rule* to the ordered rule list."""
        self._rules.append(rule)

    def list_endpoints(self) -> list[str]:
        """Return a sorted list of registered endpoint names."""
        return sorted(self._endpoints.keys())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_MULTIPLEXER_REGISTRY: dict[str, type] = {
    "default": ModelMultiplexer,
}
