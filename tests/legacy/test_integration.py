# ruff: noqa
"""Legacy integration tests — skipped.

These tests reference an old workspace at /Users/christienantonio/aurelius
which is a different path from the current Aurelius repo location
(/Users/christienantonio/Desktop/Aurelius/). All imports in this file use bare
module names (brain_layer, hierarchical_kv_cache, etc.) that are not importable
in the current environment.

These tests are excluded from the active test suite.
"""


import pytest

pytest.skip("legacy integration tests reference unavailable modules", allow_module_level=True)