"""Unified SOC-style defensive pipeline (Aurelius cycle-121).

Synthesises patterns from Archie (A/B/dispatcher tiers), BlueGuardian
(consensus + hallucination guard), and Cerebro (evidence-first triage) into
a five-stage pluggable pipeline operating on :class:`SecurityEvent`:

    ingest -> normalize -> enrich -> decide -> route

Each stage is a pure function over a typed event. The module keeps to pure
stdlib (``abc``, ``dataclasses``, ``time``, ``typing``) and exposes a
``DEFAULT_SOC_PIPELINE`` with stubbed enrichers for quick tests.

Design notes
------------
* Stages never mutate inputs destructively beyond the well-known fields on
  :class:`SecurityEvent` (enrichments dict, decisions list, route str).
* Errors raised inside a stage are captured into
  ``event.enrichments["pipeline_error"]`` so downstream stages can see the
  failure without the whole pipeline aborting. The pipeline continues but
  leaves the ``route`` un-set.
* The blast-radius-aware ``evidence_first_gate`` downgrades any
  ``auto_remediate`` decision that also asserts ``requires_human=True`` to
  ``alert_human`` - a direct port of Cerebro's "evidence first, action
  second" guard.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, Iterable, List, Optional


__all__ = [
    "SecurityEvent",
    "Finding",
    "TriageDecision",
    "Rule",
    "PipelineStage",
    "IngestStage",
    "NormalizeStage",
    "EnrichStage",
    "DecideStage",
    "RouteStage",
    "SOCPipeline",
    "DEFAULT_SOC_PIPELINE",
    "default_route_table",
    "evidence_first_gate",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


_VALID_DECISIONS = {
    "auto_remediate",
    "alert_human",
    "suppress",
    "escalate",
    "defer",
}
_VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}


@dataclass
class SecurityEvent:
    """A normalised SOC event flowing through the pipeline."""

    event_id: str
    source: str
    severity: str
    timestamp: str
    raw_payload: Dict[str, Any]
    enrichments: Dict[str, Any] = field(default_factory=dict)
    decisions: List["TriageDecision"] = field(default_factory=list)
    route: Optional[str] = None


@dataclass
class Finding:
    """Cerebro-style finding surfaced by an enricher."""

    finding_id: str
    entity: str
    entity_type: str
    issue_type: str
    severity: str
    evidence: List[Dict[str, Any]]
    source: str
    environment: str
    discovered_at: str


@dataclass
class TriageDecision:
    """Decision emitted by a :class:`DecideStage`."""

    decision: str
    priority: str
    rationale: str
    evidence_refs: List[str]
    recommended_action: str
    requires_human: bool

    def __post_init__(self) -> None:
        if self.decision not in _VALID_DECISIONS:
            raise ValueError(
                f"invalid decision {self.decision!r}; "
                f"must be one of {sorted(_VALID_DECISIONS)}"
            )
        if self.priority not in _VALID_PRIORITIES:
            raise ValueError(
                f"invalid priority {self.priority!r}; "
                f"must be one of {sorted(_VALID_PRIORITIES)}"
            )


@dataclass
class Rule:
    """Simple declarative rule used by :class:`DecideStage`.

    ``predicate`` receives the event; if it returns truthy, ``decision``
    becomes the active :class:`TriageDecision`. Rules are evaluated in
    order and the first match wins.
    """

    name: str
    predicate: Callable[[SecurityEvent], bool]
    decision: TriageDecision


# ---------------------------------------------------------------------------
# Stage base + concrete stages
# ---------------------------------------------------------------------------


class PipelineStage(ABC):
    """Abstract pluggable stage."""

    name: str = "stage"

    @abstractmethod
    def process(self, event: SecurityEvent) -> SecurityEvent:  # pragma: no cover - abstract
        raise NotImplementedError


class IngestStage(PipelineStage):
    """Pass-through stage that records the ingest timestamp."""

    name = "ingest"

    def process(self, event: SecurityEvent) -> SecurityEvent:
        event.enrichments.setdefault("ingested_at", time.time())
        return event


class NormalizeStage(PipelineStage):
    """Normalise severity strings to a canonical lowercase form."""

    name = "normalize"

    _CANONICAL = {
        "critical": "critical",
        "high": "high",
        "medium": "medium",
        "moderate": "medium",
        "low": "low",
        "info": "info",
        "informational": "info",
    }

    def process(self, event: SecurityEvent) -> SecurityEvent:
        sev = (event.severity or "").strip().lower()
        event.severity = self._CANONICAL.get(sev, sev or "info")
        return event


class EnrichStage(PipelineStage):
    """Runs a list of enricher functions; merges their dicts into enrichments."""

    name = "enrich"

    def __init__(
        self,
        enricher_fns: Optional[Iterable[Callable[[SecurityEvent], Dict[str, Any]]]] = None,
    ) -> None:
        self.enricher_fns: List[Callable[[SecurityEvent], Dict[str, Any]]] = list(
            enricher_fns or []
        )

    def process(self, event: SecurityEvent) -> SecurityEvent:
        for fn in self.enricher_fns:
            contrib = fn(event) or {}
            if not isinstance(contrib, dict):
                raise TypeError(
                    f"enricher {getattr(fn, '__name__', fn)!r} must return dict, "
                    f"got {type(contrib).__name__}"
                )
            event.enrichments.update(contrib)
        return event


class DecideStage(PipelineStage):
    """Produces a :class:`TriageDecision` from rules and/or a decider_fn.

    Evaluation order:
      1. ``rules`` (first-match wins).
      2. ``decider_fn`` fallback if no rule matched.
      3. Default ``defer``/``P4`` decision if neither yields a result.
    """

    name = "decide"

    def __init__(
        self,
        decider_fn: Optional[Callable[[SecurityEvent], TriageDecision]] = None,
        rules: Optional[Iterable[Rule]] = None,
    ) -> None:
        self.decider_fn = decider_fn
        self.rules: List[Rule] = list(rules or [])

    def process(self, event: SecurityEvent) -> SecurityEvent:
        chosen: Optional[TriageDecision] = None
        for rule in self.rules:
            try:
                matched = bool(rule.predicate(event))
            except Exception as exc:  # pragma: no cover - defensive
                event.enrichments.setdefault(
                    "rule_errors", []
                ).append({"rule": rule.name, "error": repr(exc)})
                continue
            if matched:
                chosen = rule.decision
                break
        if chosen is None and self.decider_fn is not None:
            chosen = self.decider_fn(event)
        if chosen is None:
            chosen = TriageDecision(
                decision="defer",
                priority="P4",
                rationale="no rule or decider produced a verdict",
                evidence_refs=[],
                recommended_action="await additional signal",
                requires_human=False,
            )
        chosen = evidence_first_gate(chosen)
        event.decisions.append(chosen)
        return event


def default_route_table() -> Dict[str, str]:
    """Built-in mapping from decision kind to destination queue."""

    return {
        "auto_remediate": "worker_queue",
        "alert_human": "human_review",
        "escalate": "oncall_page",
        "suppress": "archive",
        "defer": "backlog",
    }


class RouteStage(PipelineStage):
    """Sets ``event.route`` from the latest decision."""

    name = "route"

    def __init__(self, route_table: Optional[Dict[str, str]] = None) -> None:
        self.route_table: Dict[str, str] = (
            dict(route_table) if route_table is not None else default_route_table()
        )

    def process(self, event: SecurityEvent) -> SecurityEvent:
        if not event.decisions:
            return event
        decision = event.decisions[-1]
        event.route = self.route_table.get(decision.decision, "backlog")
        return event


# ---------------------------------------------------------------------------
# Blast-radius-aware gate
# ---------------------------------------------------------------------------


def evidence_first_gate(decision: TriageDecision) -> TriageDecision:
    """Downgrade ``auto_remediate`` to ``alert_human`` when human review
    is required.

    Returns a new :class:`TriageDecision` if a downgrade occurs; otherwise
    returns the input unchanged.
    """

    if decision.decision == "auto_remediate" and decision.requires_human:
        return TriageDecision(
            decision="alert_human",
            priority=decision.priority,
            rationale=(
                "blast-radius gate: auto_remediate downgraded because "
                "requires_human=True -- " + decision.rationale
            ),
            evidence_refs=list(decision.evidence_refs),
            recommended_action=decision.recommended_action,
            requires_human=True,
        )
    return decision


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def _validate_raw_event(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(
            f"raw_event must be a dict, got {type(raw).__name__}"
        )
    required = ("event_id", "source", "severity", "timestamp")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"raw_event missing required keys: {missing}"
        )
    for k in required:
        if not isinstance(raw[k], str):
            raise ValueError(
                f"raw_event field {k!r} must be str, got {type(raw[k]).__name__}"
            )
    return raw


class SOCPipeline:
    """Ordered list of :class:`PipelineStage` instances."""

    def __init__(self, stages: Iterable[PipelineStage]) -> None:
        self.stages: List[PipelineStage] = list(stages)
        self._metrics: Dict[str, Dict[str, float]] = {}
        for stage in self.stages:
            self._metrics[stage.name] = {"count": 0.0, "total_latency_s": 0.0}

    # --- construction helpers -------------------------------------------------

    def evidence_first_gate(self, decision: TriageDecision) -> TriageDecision:
        """Instance-bound alias for :func:`evidence_first_gate`."""

        return evidence_first_gate(decision)

    # --- execution ------------------------------------------------------------

    def run_one(self, raw_event: Dict[str, Any]) -> SecurityEvent:
        raw = _validate_raw_event(raw_event)
        event = SecurityEvent(
            event_id=raw["event_id"],
            source=raw["source"],
            severity=raw["severity"],
            timestamp=raw["timestamp"],
            raw_payload=dict(raw.get("payload", {})),
            enrichments={},
            decisions=[],
            route=None,
        )
        # Preserve any extra top-level fields inside raw_payload for visibility.
        for k, v in raw.items():
            if k in ("event_id", "source", "severity", "timestamp", "payload"):
                continue
            event.raw_payload.setdefault(k, v)

        for stage in self.stages:
            start = time.perf_counter()
            try:
                event = stage.process(event)
            except Exception as exc:
                event.enrichments["pipeline_error"] = {
                    "stage": stage.name,
                    "error": repr(exc),
                }
                # Record latency then break out; subsequent stages skipped.
                self._record_metric(stage.name, time.perf_counter() - start)
                break
            self._record_metric(stage.name, time.perf_counter() - start)
        return event

    def run_batch(self, raw_events: Iterable[Dict[str, Any]]) -> List[SecurityEvent]:
        return [self.run_one(r) for r in raw_events]

    # --- metrics --------------------------------------------------------------

    def _record_metric(self, stage_name: str, latency_s: float) -> None:
        slot = self._metrics.setdefault(
            stage_name, {"count": 0.0, "total_latency_s": 0.0}
        )
        slot["count"] += 1.0
        slot["total_latency_s"] += latency_s

    def metrics(self) -> Dict[str, Dict[str, float]]:
        """Return a deep-copied snapshot of per-stage counters."""

        return {k: dict(v) for k, v in self._metrics.items()}


# ---------------------------------------------------------------------------
# Default pipeline
# ---------------------------------------------------------------------------


def _stub_geoip_enricher(event: SecurityEvent) -> Dict[str, Any]:
    src_ip = str(event.raw_payload.get("src_ip", "")) if event.raw_payload else ""
    if not src_ip:
        return {}
    # Deterministic pseudo-geo bucket based on leading octet.
    first = src_ip.split(".", 1)[0] if "." in src_ip else src_ip
    bucket = "internal" if first in ("10", "192", "172") else "external"
    return {"geoip_bucket": bucket}


def _stub_reputation_enricher(event: SecurityEvent) -> Dict[str, Any]:
    sev = event.severity
    known_bad = event.severity in ("critical", "high")
    return {"reputation": {"known_bad": known_bad, "observed_severity": sev}}


def _default_decider(event: SecurityEvent) -> TriageDecision:
    sev = event.severity
    rep = event.enrichments.get("reputation", {}) or {}
    known_bad = bool(rep.get("known_bad"))
    if sev == "critical" and known_bad:
        return TriageDecision(
            decision="escalate",
            priority="P1",
            rationale="critical severity + known-bad reputation",
            evidence_refs=[event.event_id],
            recommended_action="page oncall",
            requires_human=True,
        )
    if sev == "high":
        return TriageDecision(
            decision="alert_human",
            priority="P2",
            rationale="high severity event",
            evidence_refs=[event.event_id],
            recommended_action="analyst review",
            requires_human=True,
        )
    if sev == "medium":
        return TriageDecision(
            decision="defer",
            priority="P3",
            rationale="medium severity; queue for triage",
            evidence_refs=[event.event_id],
            recommended_action="triage queue",
            requires_human=False,
        )
    if sev in ("low", "info"):
        return TriageDecision(
            decision="suppress",
            priority="P4",
            rationale="informational; suppress",
            evidence_refs=[event.event_id],
            recommended_action="archive",
            requires_human=False,
        )
    return TriageDecision(
        decision="defer",
        priority="P4",
        rationale=f"unknown severity {sev!r}",
        evidence_refs=[event.event_id],
        recommended_action="await classification",
        requires_human=False,
    )


def _build_default_pipeline() -> SOCPipeline:
    return SOCPipeline(
        [
            IngestStage(),
            NormalizeStage(),
            EnrichStage([_stub_geoip_enricher, _stub_reputation_enricher]),
            DecideStage(decider_fn=_default_decider),
            RouteStage(),
        ]
    )


DEFAULT_SOC_PIPELINE = _build_default_pipeline()
