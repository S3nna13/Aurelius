"""
AgentReputation: A Decentralized Agentic AI Reputation Framework

Paper: arXiv:2605.00073

Three-Layer Architectural Framework:
    - Functional layer: task execution (task owners, agents, verifiers)
    - Reputation services layer: evidence collection, reputation cards, policy engine
    - Blockchain/storage layer: hybrid on-chain/off-chain persistence
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class VerificationStrength(Enum):
    """Ordinal strength measure for evidence verification."""

    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


class VerificationMethod(Enum):
    """Methods for verifying agent behavior."""

    MANUAL_REVIEW = auto()
    AUTOMATED_TESTING = auto()
    STAKE_BASED_VOTING = auto()
    CRYPTOGRAPHIC_PROOF = auto()
    AUDIT_LOG_ANALYSIS = auto()
    ORACLE_REPORT = auto()
    PEER_REVIEW = auto()


@dataclass(frozen=True)
class VerificationRegime:
    """
    Defines verification method, success conditions, and strength level for evidence collection.

    Attributes:
        regime_id: Unique identifier for this verification regime
        method: The verification method used
        success_conditions: JSON-serializable conditions that define success
        strength: Verification strength ordinal
        min_stake_required: Minimum collateral stake required
        escalation_timeout: Time before escalating verification
    """

    regime_id: str
    method: VerificationMethod
    success_conditions: dict[str, Any]
    strength: VerificationStrength
    min_stake_required: float = 0.0
    escalation_timeout_seconds: float = 300.0

    def _threshold(self, key: str, default: float) -> float:
        try:
            value = float(self.success_conditions.get(key, default))
        except (TypeError, ValueError):
            value = default
        return min(1.0, max(0.0, value))

    def evaluate_success(self, evidence_data: dict[str, Any]) -> tuple[bool, float]:
        """
        Evaluate if evidence meets success conditions.

        Returns:
            Tuple of (success: bool, confidence: float 0-1)
        """
        if self.method == VerificationMethod.CRYPTOGRAPHIC_PROOF:
            return self._evaluate_cryptographic(evidence_data)
        elif self.method == VerificationMethod.AUTOMATED_TESTING:
            return self._evaluate_automated(evidence_data)
        elif self.method == VerificationMethod.STAKE_BASED_VOTING:
            return self._evaluate_voting(evidence_data)
        else:
            return self._evaluate_manual(evidence_data)

    def _evaluate_cryptographic(self, data: dict[str, Any]) -> tuple[bool, float]:
        required_fields = {"proof", "challenge", "response"}
        if not required_fields.issubset(data.keys()):
            return False, 0.0
        if data.get("verified", False):
            return True, 1.0
        return False, 0.0

    def _evaluate_automated(self, data: dict[str, Any]) -> tuple[bool, float]:
        passed = data.get("tests_passed", False)
        coverage = data.get("coverage_percent", 0.0) / 100.0
        confidence = (0.7 * int(passed) + 0.3 * coverage)
        coverage_threshold = self._threshold("coverage_threshold", 0.0)
        required_passed = bool(self.success_conditions.get("tests_passed", True))
        success = bool(passed) == required_passed and coverage >= coverage_threshold
        return success, confidence

    def _evaluate_voting(self, data: dict[str, Any]) -> tuple[bool, float]:
        votes_for = data.get("votes_for", 0)
        votes_against = data.get("votes_against", 0)
        total = votes_for + votes_against
        if total == 0:
            return False, 0.0
        approval_ratio = votes_for / total
        approval_threshold = self._threshold("approval_ratio", 0.5)
        return approval_ratio >= approval_threshold, approval_ratio

    def _evaluate_manual(self, data: dict[str, Any]) -> tuple[bool, float]:
        outcome = data.get("outcome", "unknown")
        if outcome == "success":
            confidence = 0.8
        elif outcome == "partial":
            confidence = 0.5
        else:
            return False, 0.3

        expected_outcome = self.success_conditions.get("outcome")
        outcome_ok = expected_outcome is None or outcome == expected_outcome
        confidence_threshold = self._threshold("confidence_threshold", 0.5)
        return outcome_ok and confidence >= confidence_threshold, confidence


@dataclass(frozen=True)
class EvidenceEvent:
    """
    Standardized evidence record produced by verification regimes.

    Attributes:
        event_id: Unique identifier for this evidence event
        agent_id: Agent this evidence pertains to
        regime_id: Verification regime that produced this evidence
        context: Task context/domain for this evidence
        timestamp: When evidence was recorded
        raw_data: Original evidence data
        processed_data: Processed evidence after evaluation
        success: Whether verification succeeded
        confidence: Confidence score 0-1
        strength: Verification strength ordinal
        ipfs_cid: Off-chain storage reference (if stored)
    """

    event_id: str
    agent_id: str
    regime_id: str
    context: str
    timestamp: datetime
    raw_data: dict[str, Any]
    processed_data: dict[str, Any]
    success: bool
    confidence: float
    strength: VerificationStrength
    ipfs_cid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize evidence event to dictionary."""
        return {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "regime_id": self.regime_id,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "raw_data": self.raw_data,
            "processed_data": self.processed_data,
            "success": self.success,
            "confidence": self.confidence,
            "strength": self.strength.value,
            "ipfs_cid": self.ipfs_cid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceEvent:
        """Deserialize evidence event from dictionary."""
        return cls(
            event_id=data["event_id"],
            agent_id=data["agent_id"],
            regime_id=data["regime_id"],
            context=data["context"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            raw_data=data["raw_data"],
            processed_data=data["processed_data"],
            success=data["success"],
            confidence=data["confidence"],
            strength=VerificationStrength(data["strength"]),
            ipfs_cid=data.get("ipfs_cid"),
        )


@dataclass
class ReputationCard:
    """
    Context-conditioned reputation assessment.

    Aggregates evidence weighted by verification strength and recency.
    Prevents reputation conflation across domains.

    Attributes:
        agent_id: Agent this card belongs to
        context: Specific context/domain this assessment applies to
        score: Computed reputation score 0-1
        evidence_count: Number of evidence events considered
        last_updated: Timestamp of last update
        evidence_ids: References to aggregated evidence
        confidence_level: Confidence in the assessment
        is_cold_start: Whether this is a newly initialized agent
        recovery_mode: Whether agent is in reputation recovery
    """

    agent_id: str
    context: str
    score: float = 0.0
    evidence_count: int = 0
    last_updated: datetime = field(default_factory=_utcnow)
    evidence_ids: list[str] = field(default_factory=list)
    confidence_level: float = 0.0
    is_cold_start: bool = True
    recovery_mode: bool = False

    def compute_score(self, evidence_events: list[EvidenceEvent]) -> float:
        """
        Compute reputation score from evidence events using weighted aggregation.

        Weight = strength * recency_factor * confidence
        """
        if not evidence_events:
            return 0.0

        self.evidence_count = len(evidence_events)
        self.evidence_ids = [e.event_id for e in evidence_events]

        now = _utcnow()
        total_weight = 0.0
        weighted_sum = 0.0

        for event in evidence_events:
            age_days = max(0.0, float((now - _as_utc(event.timestamp)).days))
            recency_factor = min(1.0, max(0.0, 1.0 - (age_days / 365.0)))

            weight = (
                event.strength.value
                * recency_factor
                * event.confidence
            )

            base_value = 1.0 if event.success else 0.0
            weighted_sum += base_value * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        raw_score = weighted_sum / total_weight

        if self.is_cold_start and self.evidence_count < 5:
            cold_start_boost = 1.0 - (self.evidence_count / 5.0)
            raw_score = raw_score * (1.0 - cold_start_boost * 0.3)

        self.score = min(1.0, max(0.0, raw_score))
        self.confidence_level = min(1.0, total_weight / 10.0)
        self.last_updated = now

        if self.evidence_count >= 5:
            self.is_cold_start = False

        return self.score

    def to_dict(self) -> dict[str, Any]:
        """Serialize reputation card to dictionary."""
        return {
            "agent_id": self.agent_id,
            "context": self.context,
            "score": self.score,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated.isoformat(),
            "evidence_ids": self.evidence_ids,
            "confidence_level": self.confidence_level,
            "is_cold_start": self.is_cold_start,
            "recovery_mode": self.recovery_mode,
        }


@dataclass
class PolicyRule:
    """Policy rule for decision facing engine."""

    rule_id: str
    name: str
    min_reputation: float
    min_confidence: float
    context: str
    requires_stake: bool = False
    min_stake: float = 0.0
    allow_cold_start: bool = False
    max_task_value: float = float("inf")


@dataclass
class TaskAllocation:
    """Task allocation decision from policy engine."""

    task_id: str
    agent_id: str
    approved: bool
    reason: str
    required_stake: float = 0.0
    verification_regime_id: str | None = None


class PolicyEngine:
    """
    Decision-facing engine for task allocation, access control, and verification escalation.

    Governs:
        - Task allocation based on reputation
        - Access control with reputation-based gating
        - Adaptive verification escalation based on risk and uncertainty
        - Collateral mechanisms
    """

    def __init__(self):
        self.rules: dict[str, PolicyRule] = {}
        self.risk_thresholds: dict[str, float] = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
        }

    def add_rule(self, rule: PolicyRule) -> None:
        """Register a policy rule."""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        """Remove a policy rule."""
        self.rules.pop(rule_id, None)

    def evaluate_task_allocation(
        self,
        agent_id: str,
        reputation_card: ReputationCard,
        task_value: float,
        task_context: str,
    ) -> TaskAllocation:
        """
        Evaluate whether to allocate a task to an agent.

        Returns:
            TaskAllocation with approval decision and reasoning.
        """
        applicable_rules = [
            r for r in self.rules.values()
            if r.context == task_context or r.context == "*"
        ]

        if not applicable_rules:
            return TaskAllocation(
                task_id="",
                agent_id=agent_id,
                approved=True,
                reason="No specific rules; default allow",
            )

        best_rule = max(applicable_rules, key=lambda r: r.min_reputation)

        if reputation_card.is_cold_start and not best_rule.allow_cold_start:
            return TaskAllocation(
                task_id="",
                agent_id=agent_id,
                approved=False,
                reason="Agent in cold-start period; rule requires established reputation",
            )

        if reputation_card.score < best_rule.min_reputation:
            return TaskAllocation(
                task_id="",
                agent_id=agent_id,
                approved=False,
                reason=(
                    f"Reputation {reputation_card.score:.2f} below "
                    f"required {best_rule.min_reputation:.2f}"
                ),
            )

        if reputation_card.confidence_level < best_rule.min_confidence:
            return TaskAllocation(
                task_id="",
                agent_id=agent_id,
                approved=False,
                reason=(
                    f"Confidence {reputation_card.confidence_level:.2f} "
                    f"below required {best_rule.min_confidence:.2f}"
                ),
            )

        if task_value > best_rule.max_task_value:
            return TaskAllocation(
                task_id="",
                agent_id=agent_id,
                approved=False,
                reason=f"Task value {task_value} exceeds max {best_rule.max_task_value}",
            )

        required_stake = best_rule.min_stake if best_rule.requires_stake else 0.0

        return TaskAllocation(
            task_id="",
            agent_id=agent_id,
            approved=True,
            reason=f"Approved under rule '{best_rule.name}'",
            required_stake=required_stake,
        )

    def determine_verification_escalation(
        self,
        risk_level: float,
        agent_reputation: ReputationCard,
        task_value: float,
    ) -> VerificationRegime:
        """
        Determine appropriate verification regime based on risk and agent standing.

        Higher risk and lower reputation trigger stronger verification.
        """
        base_regimes = {
            "low": VerificationRegime(
                regime_id="verify_manual",
                method=VerificationMethod.MANUAL_REVIEW,
                success_conditions={"outcome": "success"},
                strength=VerificationStrength.MODERATE,
            ),
            "medium": VerificationRegime(
                regime_id="verify_automated",
                method=VerificationMethod.AUTOMATED_TESTING,
                success_conditions={"tests_passed": True},
                strength=VerificationStrength.STRONG,
            ),
            "high": VerificationRegime(
                regime_id="verify_stake_vote",
                method=VerificationMethod.STAKE_BASED_VOTING,
                success_conditions={"approval_ratio": 0.7},
                strength=VerificationStrength.VERY_STRONG,
            ),
        }

        if risk_level < self.risk_thresholds["low"]:
            return base_regimes["low"]
        elif risk_level < self.risk_thresholds["medium"]:
            return base_regimes["medium"]
        else:
            return base_regimes["high"]

    def compute_risk_level(
        self,
        agent_reputation: ReputationCard,
        task_value: float,
        uncertainty: float,
    ) -> float:
        """Compute risk level (0-1) for a task-agent combination."""
        rep_factor = 1.0 - agent_reputation.score
        value_factor = min(1.0, task_value / 10000.0)
        uncertainty_factor = uncertainty

        risk = (rep_factor * 0.4) + (value_factor * 0.4) + (uncertainty_factor * 0.2)
        return min(1.0, max(0.0, risk))


class OffChainStorage(ABC):
    """Abstract interface for off-chain storage (IPFS-compatible)."""

    @abstractmethod
    def store(self, data: dict[str, Any]) -> str:
        """Store data and return content identifier."""
        pass

    @abstractmethod
    def retrieve(self, cid: str) -> dict[str, Any] | None:
        """Retrieve data by content identifier."""
        pass


class InMemoryOffChainStorage(OffChainStorage):
    """In-memory implementation of off-chain storage for development/testing."""

    def __init__(self):
        self._store: dict[str, dict[str, Any]] = {}

    def store(self, data: dict[str, Any]) -> str:
        content = json.dumps(data, sort_keys=True)
        cid = hashlib.sha256(content.encode()).hexdigest()[:32]
        self._store[cid] = data
        return cid

    def retrieve(self, cid: str) -> dict[str, Any] | None:
        return self._store.get(cid)


class OnChainStorage(ABC):
    """Abstract interface for on-chain storage (cryptographic commitments)."""

    @abstractmethod
    def commit(self, key: str, value: str) -> str:
        """Commit value and return transaction/reference hash."""
        pass

    @abstractmethod
    def verify(self, key: str, proof: dict[str, Any]) -> bool:
        """Verify a proof against committed data."""
        pass


class InMemoryOnChainStorage(OnChainStorage):
    """In-memory implementation of on-chain storage for development/testing."""

    def __init__(self):
        self._chain: dict[str, str] = {}
        self._blocks: list[dict[str, Any]] = []

    def commit(self, key: str, value: str) -> str:
        block_data = {
            "key": key,
            "value": value,
            "timestamp": time.time(),
            "block_hash": self._compute_hash(),
        }
        self._blocks.append(block_data)
        self._chain[key] = value
        return block_data["block_hash"]

    def _compute_hash(self) -> str:
        content = json.dumps(self._blocks, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify(self, key: str, proof: dict[str, Any]) -> bool:
        if key not in self._chain:
            return False
        return self._chain[key] == proof.get("value")


@dataclass
class AgentRegistration:
    """Agent registration record."""

    agent_id: str
    registered_at: datetime
    is_active: bool = True
    cold_start_until: datetime | None = None
    recovery_until: datetime | None = None


class ReputationService:
    """
    Manages reputation cards, evidence collection, and policy evaluation.

    Coordinates with policy engine for decision-making and manages
    evidence aggregation for reputation cards.
    """

    def __init__(self, policy_engine: PolicyEngine):
        self.policy_engine = policy_engine
        self.reputation_cards: dict[str, dict[str, ReputationCard]] = {}
        self.evidence_store: dict[str, EvidenceEvent] = {}
        self.pending_evidence: dict[str, list[EvidenceEvent]] = {}

    def register_agent(self, agent_id: str, context: str) -> ReputationCard:
        """Register a new agent and initialize reputation card."""
        if agent_id not in self.reputation_cards:
            self.reputation_cards[agent_id] = {}

        card = ReputationCard(
            agent_id=agent_id,
            context=context,
            is_cold_start=True,
            score=0.0,
        )
        self.reputation_cards[agent_id][context] = card
        return card

    def record_evidence(self, event: EvidenceEvent) -> None:
        """Record evidence event for an agent."""
        self.evidence_store[event.event_id] = event

        if event.agent_id not in self.pending_evidence:
            self.pending_evidence[event.agent_id] = []
        self.pending_evidence[event.agent_id].append(event)

    def finalize_evidence(self, agent_id: str, context: str) -> EvidenceEvent | None:
        """Finalize pending evidence and update reputation card."""
        if agent_id not in self.pending_evidence:
            return None

        pending = self.pending_evidence.get(agent_id, [])
        if not pending:
            return None

        context_evidence = [
            e for e in pending
            if e.context == context
        ]

        if not context_evidence:
            return None

        latest = context_evidence[-1]

        if agent_id in self.reputation_cards:
            cards = self.reputation_cards[agent_id]
            if context in cards:
                card = cards[context]
                all_evidence = [
                    self.evidence_store[eid]
                    for eid in card.evidence_ids
                    if eid in self.evidence_store
                ]
                all_evidence.append(latest)
                card.compute_score(all_evidence)

        self.pending_evidence[agent_id] = [
            e for e in pending if e.event_id != latest.event_id
        ]

        return latest

    def get_reputation_card(
        self,
        agent_id: str,
        context: str,
    ) -> ReputationCard | None:
        """Retrieve reputation card for agent in given context."""
        return self.reputation_cards.get(agent_id, {}).get(context)

    def get_agent_reputation(
        self,
        agent_id: str,
        contexts: list[str] | None = None,
    ) -> dict[str, ReputationCard]:
        """Get all reputation cards for an agent, optionally filtered by context."""
        cards = self.reputation_cards.get(agent_id, {})
        if contexts:
            return {k: v for k, v in cards.items() if k in contexts}
        return cards

    def aggregate_cross_context_reputation(
        self,
        agent_id: str,
        target_context: str,
    ) -> float:
        """
        Aggregate reputation from similar contexts into target context.

        Only aggregates context-matching evidence weighted by verification
        strength and recency.
        """
        cards = self.reputation_cards.get(agent_id, {})
        if target_context in cards:
            return cards[target_context].score

        weighted_evidence: list[tuple[EvidenceEvent, float]] = []
        for ctx, card in cards.items():
            if ctx == target_context:
                continue
            similarity = self._compute_context_similarity(target_context, ctx)
            if similarity > 0:
                for eid in card.evidence_ids:
                    if eid in self.evidence_store:
                        weighted_evidence.append((self.evidence_store[eid], similarity))

        if not weighted_evidence:
            return 0.0

        weighted_sum = sum(
            (1.0 if evidence.success else 0.0) * context_weight * evidence.strength.value
            for evidence, context_weight in weighted_evidence
        )
        total_weight = sum(
            context_weight * evidence.strength.value
            for evidence, context_weight in weighted_evidence
        )

        if total_weight == 0:
            return 0.0

        return min(1.0, weighted_sum / total_weight)

    def _compute_context_similarity(self, ctx1: str, ctx2: str) -> float:
        """Compute similarity between two contexts (simple token overlap)."""
        tokens1 = set(ctx1.lower().split())
        tokens2 = set(ctx2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0


class DecentralizedReputationFramework:
    """
    Main class coordinating all three layers of the AgentReputation framework.

    Handles:
        - Agent registration
        - Evidence storage (hybrid on-chain/off-chain)
        - Reputation computation
        - Policy enforcement
        - Graduated trust building (cold-start, reputation recovery)
    """

    def __init__(
        self,
        off_chain_storage: OffChainStorage | None = None,
        on_chain_storage: OnChainStorage | None = None,
        policy_engine: PolicyEngine | None = None,
    ):
        self.off_chain: OffChainStorage = off_chain_storage or InMemoryOffChainStorage()
        self.on_chain: OnChainStorage = on_chain_storage or InMemoryOnChainStorage()
        self.policy_engine = policy_engine or PolicyEngine()
        self.reputation_service = ReputationService(self.policy_engine)

        self.registrations: dict[str, AgentRegistration] = {}
        self.verification_regimes: dict[str, VerificationRegime] = {}

        self._setup_default_regimes()

    def _setup_default_regimes(self) -> None:
        """Initialize default verification regimes."""
        default_regimes = [
            VerificationRegime(
                regime_id="manual_review",
                method=VerificationMethod.MANUAL_REVIEW,
                success_conditions={"outcome": "success"},
                strength=VerificationStrength.MODERATE,
            ),
            VerificationRegime(
                regime_id="automated_test",
                method=VerificationMethod.AUTOMATED_TESTING,
                success_conditions={"tests_passed": True},
                strength=VerificationStrength.STRONG,
            ),
            VerificationRegime(
                regime_id="crypto_proof",
                method=VerificationMethod.CRYPTOGRAPHIC_PROOF,
                success_conditions={"verified": True},
                strength=VerificationStrength.VERY_STRONG,
                min_stake_required=100.0,
            ),
            VerificationRegime(
                regime_id="stake_vote",
                method=VerificationMethod.STAKE_BASED_VOTING,
                success_conditions={"approval_ratio": 0.6},
                strength=VerificationStrength.STRONG,
                min_stake_required=50.0,
            ),
        ]
        for regime in default_regimes:
            self.verification_regimes[regime.regime_id] = regime

    def register_agent(
        self,
        agent_id: str,
        cold_start_period_days: int = 30,
    ) -> AgentRegistration:
        """
        Register a new agent in the framework.

        Args:
            agent_id: Unique identifier for the agent
            cold_start_period_days: Duration of cold-start period

        Returns:
            AgentRegistration record
        """
        registration = AgentRegistration(
            agent_id=agent_id,
            registered_at=_utcnow(),
            cold_start_until=_utcnow() + timedelta(days=cold_start_period_days),
        )
        self.registrations[agent_id] = registration
        self.reputation_service.register_agent(agent_id, "default")
        return registration

    def get_registration(self, agent_id: str) -> AgentRegistration | None:
        """Get agent registration record."""
        return self.registrations.get(agent_id)

    def is_cold_start(self, agent_id: str) -> bool:
        """Check if agent is still in cold-start period."""
        reg = self.registrations.get(agent_id)
        if not reg or not reg.cold_start_until:
            return False
        return _utcnow() < _as_utc(reg.cold_start_until)

    def complete_cold_start(self, agent_id: str) -> bool:
        """Mark cold-start period as complete for an agent."""
        reg = self.registrations.get(agent_id)
        if not reg:
            return False
        reg.cold_start_until = None
        if agent_id in self.reputation_service.reputation_cards:
            for card in self.reputation_service.reputation_cards[agent_id].values():
                card.is_cold_start = False
        return True

    def enter_recovery_mode(self, agent_id: str, duration_days: int = 90) -> bool:
        """Put agent into reputation recovery mode."""
        reg = self.registrations.get(agent_id)
        if not reg:
            return False
        reg.recovery_until = _utcnow() + timedelta(days=duration_days)
        if agent_id in self.reputation_service.reputation_cards:
            for card in self.reputation_service.reputation_cards[agent_id].values():
                card.recovery_mode = True
        return True

    def exit_recovery_mode(self, agent_id: str) -> bool:
        """Remove agent from reputation recovery mode."""
        reg = self.registrations.get(agent_id)
        if not reg:
            return False
        reg.recovery_until = None
        if agent_id in self.reputation_service.reputation_cards:
            for card in self.reputation_service.reputation_cards[agent_id].values():
                card.recovery_mode = False
        return True

    def add_verification_regime(self, regime: VerificationRegime) -> None:
        """Add a custom verification regime."""
        self.verification_regimes[regime.regime_id] = regime

    def collect_evidence(
        self,
        agent_id: str,
        regime_id: str,
        context: str,
        raw_data: dict[str, Any],
    ) -> EvidenceEvent:
        """
        Collect evidence for an agent using specified verification regime.

        Args:
            agent_id: Agent to collect evidence for
            regime_id: Verification regime to use
            context: Task context/domain
            raw_data: Raw evidence data

        Returns:
            EvidenceEvent with processed evidence
        """
        regime = self.verification_regimes.get(regime_id)
        if not regime:
            raise ValueError(f"Unknown verification regime: {regime_id}")

        success, confidence = regime.evaluate_success(raw_data)

        processed_data = {
            "verified": success,
            "confidence": confidence,
            "regime_method": regime.method.name,
            "evaluation_timestamp": _utcnow().isoformat(),
        }

        event_id = self._generate_event_id(agent_id, regime_id, context, raw_data)

        event = EvidenceEvent(
            event_id=event_id,
            agent_id=agent_id,
            regime_id=regime_id,
            context=context,
            timestamp=_utcnow(),
            raw_data=raw_data,
            processed_data=processed_data,
            success=success,
            confidence=confidence,
            strength=regime.strength,
        )

        if regime.strength.value >= VerificationStrength.STRONG.value:
            off_chain_cid = self.off_chain.store(event.to_dict())
            event = EvidenceEvent(
                event_id=event.event_id,
                agent_id=event.agent_id,
                regime_id=event.regime_id,
                context=event.context,
                timestamp=event.timestamp,
                raw_data=event.raw_data,
                processed_data=event.processed_data,
                success=event.success,
                confidence=event.confidence,
                strength=event.strength,
                ipfs_cid=off_chain_cid,
            )
            self.on_chain.commit(f"evidence:{event_id}", event_id)

        self.reputation_service.record_evidence(event)
        return event

    def _generate_event_id(
        self,
        agent_id: str,
        regime_id: str,
        context: str,
        raw_data: dict[str, Any],
    ) -> str:
        """Generate unique event ID from components."""
        content = (
            f"{agent_id}:{regime_id}:{context}:"
            f"{json.dumps(raw_data, sort_keys=True)}:{time.time()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:24]

    def compute_reputation(
        self,
        agent_id: str,
        context: str,
    ) -> ReputationCard | None:
        """
        Compute and return reputation card for agent in context.

        Args:
            agent_id: Agent to compute reputation for
            context: Context/domain for reputation

        Returns:
            Updated ReputationCard or None if agent not found
        """
        self.reputation_service.finalize_evidence(agent_id, context)
        return self.reputation_service.get_reputation_card(agent_id, context)

    def evaluate_task_allocation(
        self,
        agent_id: str,
        task_value: float,
        task_context: str = "default",
    ) -> TaskAllocation:
        """
        Evaluate whether to allocate a task to an agent.

        Args:
            agent_id: Agent to evaluate
            task_value: Value of the task
            task_context: Context of the task

        Returns:
            TaskAllocation decision
        """
        card = self.compute_reputation(agent_id, task_context)
        if not card:
            card = self.reputation_service.register_agent(agent_id, task_context)

        uncertainty = 1.0 - card.confidence_level
        risk_level = self.policy_engine.compute_risk_level(card, task_value, uncertainty)

        if card.is_cold_start:
            risk_level = min(1.0, risk_level + 0.2)

        regime = self.policy_engine.determine_verification_escalation(
            risk_level, card, task_value
        )

        allocation = self.policy_engine.evaluate_task_allocation(
            agent_id, card, task_value, task_context
        )

        if allocation.approved:
            allocation.verification_regime_id = regime.regime_id

        return allocation

    def get_reputation_snapshot(
        self,
        agent_id: str,
    ) -> dict[str, Any]:
        """
        Get complete reputation snapshot for an agent across all contexts.

        Returns:
            Dictionary with all reputation cards and metadata
        """
        registration = self.get_registration(agent_id)
        cards = self.reputation_service.get_agent_reputation(agent_id)

        return {
            "agent_id": agent_id,
            "registered_at": registration.registered_at.isoformat() if registration else None,
            "is_cold_start": self.is_cold_start(agent_id),
            "recovery_mode": (
                _as_utc(registration.recovery_until) > _utcnow()
                if registration and registration.recovery_until
                else False
            ),
            "contexts": {ctx: card.to_dict() for ctx, card in cards.items()},
        }

    def verify_evidence_integrity(self, event_id: str) -> bool:
        """
        Verify integrity of an evidence event using on-chain commitments.

        Args:
            event_id: ID of evidence event to verify

        Returns:
            True if evidence is intact and unmodified
        """
        event = self.reputation_service.evidence_store.get(event_id)
        if not event:
            return False

        if not event.ipfs_cid:
            return True

        stored_data = self.off_chain.retrieve(event.ipfs_cid)
        if not stored_data:
            return False

        expected_data = event.to_dict()
        stored_data_without_cid = dict(stored_data)
        expected_data.pop("ipfs_cid", None)
        stored_data_without_cid.pop("ipfs_cid", None)
        if stored_data_without_cid != expected_data:
            return False

        proof = {"value": event_id}
        return self.on_chain.verify(f"evidence:{event_id}", proof)

    def export_reputation_proof(
        self,
        agent_id: str,
        context: str,
    ) -> dict[str, Any]:
        """
        Export cryptographic proof of reputation for external verification.

        Returns:
            Proof document with reputation claim and verification data
        """
        card = self.compute_reputation(agent_id, context)
        if not card:
            return {"error": "Agent or context not found"}

        proof_id = hashlib.sha256(
            f"{agent_id}:{context}:{card.score}:{time.time()}".encode()
        ).hexdigest()[:24]

        commitment = self.on_chain.commit(
            f"reputation:{agent_id}:{context}",
            f"{card.score}:{card.evidence_count}:{card.confidence_level}"
        )

        return {
            "proof_id": proof_id,
            "agent_id": agent_id,
            "context": context,
            "score": card.score,
            "evidence_count": card.evidence_count,
            "confidence": card.confidence_level,
            "timestamp": _utcnow().isoformat(),
            "commitment": commitment,
            "is_cold_start": card.is_cold_start,
            "recovery_mode": card.recovery_mode,
        }
