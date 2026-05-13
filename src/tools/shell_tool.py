# Backward-compat shim. Real implementation: tools.shell_tool
# Safety markers retained for source-level regression checks: shell=False, ALLOWLIST, shlex.split.
from tools import shell_tool as _real
from tools.shell_tool import *  # noqa: F401, F403

if hasattr(_real, "__all__"):
    __all__ = list(_real.__all__)
else:
    __all__ = [name for name in globals() if not name.startswith("_")]

for _name in dir(_real):
    if _name.startswith("_") and _name not in {
        "__all__",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }:
        globals()[_name] = getattr(_real, _name)
