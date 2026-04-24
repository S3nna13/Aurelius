"""Multi-agent negotiation with offer / counter-offer protocol."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class NegotiationState(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    AGREED = "agreed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class Offer:
    agent_id: str
    value: float
    offer_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    terms: dict[str, str] = field(default_factory=dict)
    round_num: int = 0
    timestamp_s: float = field(default_factory=time.monotonic)


@dataclass
class NegotiationSession:
    participants: list[str]
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: NegotiationState = NegotiationState.OPEN
    offers: list[Offer] = field(default_factory=list)
    agreed_offer: Offer | None = None
    max_rounds: int = 10


class NegotiationProtocol:
    def create_session(
        self,
        participants: list[str],
        max_rounds: int = 10,
    ) -> NegotiationSession:
        return NegotiationSession(participants=list(participants), max_rounds=max_rounds)

    def current_round(self, session: NegotiationSession) -> int:
        if not session.participants:
            return 0
        return len(session.offers) // len(session.participants)

    def submit_offer(
        self,
        session: NegotiationSession,
        agent_id: str,
        value: float,
        terms: dict | None = None,
    ) -> Offer:
        round_num = self.current_round(session)
        offer = Offer(
            agent_id=agent_id,
            value=value,
            terms=dict(terms) if terms else {},
            round_num=round_num,
        )
        session.offers.append(offer)
        if session.state == NegotiationState.OPEN:
            session.state = NegotiationState.IN_PROGRESS
        return offer

    def latest_offers(self, session: NegotiationSession) -> list[Offer]:
        n = len(session.participants)
        if n == 0:
            return []
        current = self.current_round(session)
        # If we just completed round r, current points to r+1 but offers exist at round r.
        # latest_offers returns the most recent fully-submitted or in-progress round's offers.
        if len(session.offers) % n == 0 and session.offers:
            target_round = current - 1
        else:
            target_round = current
        return [o for o in session.offers if o.round_num == target_round]

    def evaluate_offers(self, session: NegotiationSession) -> Offer | None:
        n = len(session.participants)
        if n == 0:
            return None
        # Check if the latest round has one offer from each participant.
        if len(session.offers) == 0:
            return None
        if len(session.offers) % n != 0:
            # Current round incomplete.
            if self.current_round(session) >= session.max_rounds:
                session.state = NegotiationState.FAILED
            return None

        last_round = self.current_round(session) - 1
        round_offers = [o for o in session.offers if o.round_num == last_round]
        if len(round_offers) != n:
            return None

        agents_in_round = {o.agent_id for o in round_offers}
        if agents_in_round != set(session.participants):
            return None

        values = [o.value for o in round_offers]
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            within = all(abs(v) < 1e-9 for v in values)
        else:
            within = all(abs(v - mean_val) / abs(mean_val) <= 0.05 for v in values)

        if within:
            best = min(round_offers, key=lambda o: abs(o.value - mean_val))
            session.agreed_offer = best
            session.state = NegotiationState.AGREED
            return best

        if self.current_round(session) >= session.max_rounds:
            session.state = NegotiationState.FAILED
        return None


NEGOTIATION_PROTOCOL_REGISTRY: dict[str, type[NegotiationProtocol]] = {
    "default": NegotiationProtocol,
}
