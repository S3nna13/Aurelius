"""Synthetic code generation for SFT pretraining."""
import json, os, random
from pathlib import Path
from typing import Any, Dict, List, Optional

LANGUAGE_WEIGHTS = {"python":0.35,"javascript":0.20,"typescript":0.10,"go":0.10,"rust":0.10,"cpp":0.08,"java":0.07}
