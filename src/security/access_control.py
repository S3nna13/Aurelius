"""Role-based access control for agent operations.

Trail of Bits: enforce authorization at every boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Permission:
    resource: str
    action: str  # read, write, execute, admin


@dataclass
class Role:
    name: str
    permissions: list[Permission] = field(default_factory=list)

    def has_permission(self, resource: str, action: str) -> bool:
        return any(p.resource == resource and p.action == action for p in self.permissions)


@dataclass
class AccessControl:
    """RBAC engine for agent/system operations."""

    _roles: dict[str, Role] = field(default_factory=dict, repr=False)
    _user_roles: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def add_role(self, role: Role) -> None:
        self._roles[role.name] = role

    def assign_role(self, user: str, role_name: str) -> None:
        if role_name not in self._roles:
            raise ValueError(f"unknown role: {role_name}")
        self._user_roles.setdefault(user, []).append(role_name)

    def check(self, user: str, resource: str, action: str) -> bool:
        user_roles = self._user_roles.get(user, [])
        for role_name in user_roles:
            role = self._roles.get(role_name)
            if role and role.has_permission(resource, action):
                return True
        return False

    def revoke_role(self, user: str, role_name: str) -> None:
        roles = self._user_roles.get(user, [])
        if role_name in roles:
            roles.remove(role_name)

    def list_roles(self) -> list[str]:
        return list(self._roles.keys())


ACCESS_CONTROL = AccessControl()
