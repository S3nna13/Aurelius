"""Protocol negotiation and version compatibility checker."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProtocolVersion:
    major: int
    minor: int
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def compatible_with(self, other: ProtocolVersion) -> bool:
        return self.major == other.major and self.minor >= other.minor

    @classmethod
    def parse(cls, version_str: str) -> ProtocolVersion:
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return cls(major=major, minor=minor, patch=patch)


@dataclass
class ProtocolNegotiator:
    versions: dict[str, ProtocolVersion] = field(default_factory=dict, repr=False)

    def register(self, service: str, version: ProtocolVersion) -> None:
        self.versions[service] = version

    def negotiate(self, service: str, client_version: ProtocolVersion) -> tuple[bool, str]:
        server_version = self.versions.get(service)
        if server_version is None:
            return False, f"unknown service: {service}"
        if client_version.compatible_with(server_version):
            return True, f"compatible: client {client_version} server {server_version}"
        return False, f"incompatible: client {client_version} server {server_version}"


PROTOCOL_NEGOTIATOR = ProtocolNegotiator()