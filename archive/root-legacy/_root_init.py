"""
Aurelius AI — Memory-Augmented Transformer with Agent Capabilities

Model tiers: 150M / 1B / 3B / 7B / 14B / 32B
"""

import os

__version__ = "1.0.0"

# Extend the package search path so that subpackages living under src/
# (inference, model, training, data, etc.) are discoverable as
# ``aurelius.<subpackage>``.
_src = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(_src):
    __path__ = [_src] + __path__
