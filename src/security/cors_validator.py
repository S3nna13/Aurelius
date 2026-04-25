"""CORS policy validator and configuration manager."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CORSPolicy:
    allowed_origins: list[str] | None = None
    allowed_methods: list[str] | None = None
    allowed_headers: list[str] | None = None
    allow_credentials: bool = False
    max_age: int = 3600

    def __post_init__(self) -> None:
        if self.allowed_origins is None:
            self.allowed_origins = []
        if self.allowed_methods is None:
            self.allowed_methods = ["GET", "POST"]
        if self.allowed_headers is None:
            self.allowed_headers = ["Content-Type", "Authorization"]


@dataclass
class CORSValidator:
    """Validate CORS requests against configured policy."""

    policies: dict[str, CORSPolicy] = field(default_factory=dict, repr=False)
    default: CORSPolicy = field(default_factory=CORSPolicy)

    def add_policy(self, path: str, policy: CORSPolicy) -> None:
        self.policies[path] = policy

    def check_origin(self, origin: str, path: str = "/") -> bool:
        policy = self.policies.get(path, self.default)
        if not policy.allowed_origins:
            return True  # allow all
        return origin in policy.allowed_origins

    def check_method(self, method: str, path: str = "/") -> bool:
        policy = self.policies.get(path, self.default)
        return method.upper() in [m.upper() for m in (policy.allowed_methods or [])]

    def to_headers(self, path: str = "/") -> dict[str, str]:
        policy = self.policies.get(path, self.default)
        headers = {}
        if policy.allowed_origins:
            headers["Access-Control-Allow-Origin"] = ",".join(policy.allowed_origins)
        if policy.allowed_methods:
            headers["Access-Control-Allow-Methods"] = ",".join(policy.allowed_methods)
        if policy.allowed_headers:
            headers["Access-Control-Allow-Headers"] = ",".join(policy.allowed_headers)
        if policy.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Max-Age"] = str(policy.max_age)
        return headers


CORS_VALIDATOR = CORSValidator()