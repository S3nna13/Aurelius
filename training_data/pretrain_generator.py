"""Synthetic pretraining data generator for Aurelius 1.3B LLM.

Generates diverse code and text documents across multiple languages,
tokenizes them with AureliusTokenizer, and saves uint16 .npy shards.

Each document is randomly sampled from a large pool of unique templates
with randomized identifiers, types, and algorithmic details.
"""

from __future__ import annotations

import json
import logging
import random
import string
from pathlib import Path
from typing import Any

import numpy as np

from src.data.tokenizer import AureliusTokenizer

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _p(tag: str, body: str, indent: int = 0) -> str:
    """Wrap *body* in a tag (for docstring XML-like snippets)."""
    pad = " " * indent
    return f"{pad}<{tag}>{body}</{tag}>"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class PretrainDataGenerator:
    """Generate synthetic pretraining data and save tokenized shards.

    Parameters
    ----------
    config : dict
        Configuration dictionary.  Expected keys:

        - **seed** (*int*) -- RNG seed (default 42).
        - **tokenizer** -- sub-dict with ``vocab_size`` (used for RNG
          diversification).
        - **pretrain** -- sub-dict with ``shard_size`` (default 16384).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        base_seed = config.get("seed", 42)
        vocab_size = config.get("tokenizer", {}).get("vocab_size", 128000)
        self._rng = random.Random(base_seed + vocab_size)

        self._python_templates: list[callable] = self._build_python_templates()
        self._js_templates: list[callable] = self._build_js_templates()
        self._go_templates: list[callable] = self._build_go_templates()
        self._rust_templates: list[callable] = self._build_rust_templates()
        self._text_templates: list[callable] = self._build_text_templates()

        self._shard_size = config.get("pretrain", {}).get("shard_size", 16384)

    # -- random helpers -----------------------------------------------------

    def _word(self, min_len: int = 3, max_len: int = 10) -> str:
        length = self._rng.randint(min_len, max_len)
        return self._rng.choice(string.ascii_lowercase) + "".join(
            self._rng.choices(string.ascii_lowercase, k=length - 1)
        )

    def _snake(self, *prefixes: str) -> str:
        parts = list(prefixes)
        for _ in range(self._rng.randint(1, 3)):
            parts.append(self._word(3, 9))
        return "_".join(parts)

    def _camel(self, *prefixes: str) -> str:
        parts = list(prefixes)
        for _ in range(self._rng.randint(1, 3)):
            w = self._rng.choice(string.ascii_uppercase) + self._word(3, 10)
            parts.append(w)
        return "".join(parts)

    def _py_type(self) -> str:
        return self._rng.choice([
            "int", "str", "float", "bool", "bytes",
            "list[str]", "dict[str, Any]", "Optional[int]",
            "tuple[int, ...]", "set[str]", "frozenset",
            "Sequence[float]", "Mapping[str, int]", "Any",
            "Path", "Iterator[str]", "Callable[[int], str]",
        ])

    def _go_type(self) -> str:
        return self._rng.choice([
            "int", "int64", "string", "float64", "bool",
            "[]byte", "error", "time.Time", "uint32",
            "int32", "rune", "complex128", "any",
        ])

    def _rust_type(self) -> str:
        return self._rng.choice([
            "i32", "u64", "String", "bool", "f64",
            "Vec<u8>", "HashMap<String, i32>",
            "Option<usize>", "Result<(), Box<dyn Error>>",
            "Duration", "PathBuf", "Arc<Mutex<T>>",
            "BTreeMap<u64, String>", "HashSet<i32>",
        ])

    def _param(self) -> str:
        return self._snake()

    def _py_docstring(self, topic: str) -> str:
        lines = [
            f'    """{topic}',
            "",
        ]
        for _ in range(self._rng.randint(1, 3)):
            lines.append(f"    {self._word(4, 12).capitalize()} "
                         f"{' '.join(self._word(3, 8) for _ in range(self._rng.randint(3, 10)))}.")
        lines.append('    """')
        return "\n".join(lines)

    def _indent(self, code: str, level: int = 1) -> str:
        pad = "    " * level
        return "\n".join(pad + line if line.strip() else line for line in code.split("\n"))

    # ------------------------------------------------------------------
    # Python templates  (≥20)
    # ------------------------------------------------------------------

    def _build_python_templates(self) -> list[callable]:

        def t01() -> str:
            """Data processing pipeline with dataclasses."""
            cls = self._camel("Data")
            field_a = self._snake()
            field_b = self._snake()
            method = self._snake("transform")
            return (
                "from dataclasses import dataclass, field\n"
                "from typing import Optional\n"
                "from pathlib import Path\n\n"
                f"@dataclass\n"
                f"class {cls}:\n"
                f"    {field_a}: str\n"
                f"    {field_b}: Optional[Path] = None\n"
                f'    _cache: dict[str, list[float]] = field(default_factory=dict)\n\n'
                f"    def {method}(self, value: float, /) -> list[float]:\n"
                f'        """Apply the normalizing transform to *value*."""\n'
                f"        result = [v * 1.5 for v in self._cache.get(self.{field_a}, [])]\n"
                f"        result.append(value ** 2)\n"
                f"        return result\n\n"
                f"    def __post_init__(self) -> None:\n"
                f"        if self.{field_b} is not None and not self.{field_b}.exists():\n"
                f"            raise FileNotFoundError(f'{{self.{field_b}}} not found')\n"
            )

        def t02() -> str:
            """Async HTTP client."""
            fn = self._snake("fetch")
            url_p = self._param()
            hdr = self._snake("headers")
            return (
                "import asyncio\n"
                "import aiohttp\n"
                "from typing import Optional\n\n"
                f"async def {fn}({url_p}: str, timeout: float = 10.0) -> Optional[dict[str, Any]]:\n"
                f'    """Retrieve JSON data from *{url_p}* with a timeout."""\n'
                f"    {hdr} = {{'Accept': 'application/json', 'User-Agent': 'Aurelius/1.0'}}\n"
                f"    async with aiohttp.ClientSession(headers={hdr}) as session:\n"
                f"        try:\n"
                f"            async with session.get({url_p}, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:\n"
                f"                if resp.status == 200:\n"
                f"                    return await resp.json()\n"
                f"                resp.raise_for_status()\n"
                f"        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:\n"
                f"            print(f'Request failed: {{exc}}')\n"
                f"            return None\n"
            )

        def t03() -> str:
            """NumPy array manipulation."""
            fn = self._snake("normalize", "tensor")
            eps = self._snake("eps")
            return (
                "import numpy as np\n\n"
                f"def {fn}(arr: np.ndarray, axis: int = -1) -> np.ndarray:\n"
                f'    """L2-normalize *arr* along *axis*, avoiding division by zero."""\n'
                f"    {eps} = np.finfo(arr.dtype).eps if arr.dtype.kind == 'f' else 1e-12\n"
                f"    norm = np.linalg.norm(arr, axis=axis, keepdims=True)\n"
                f"    return arr / np.maximum(norm, {eps})\n\n\n"
                f"def {self._snake('batch', 'normalize')}(batch: list[np.ndarray]) -> list[np.ndarray]:\n"
                f'    """Normalize a batch of arrays."""\n'
                f"    return [{fn}(arr) for arr in batch]\n"
            )

        def t04() -> str:
            """Simple transformer-style MLP block."""
            cls = self._camel("MLP", "Block")
            dim = self._snake("hidden", "dim")
            return (
                "import torch\n"
                "import torch.nn as nn\n\n"
                f"class {cls}(nn.Module):\n"
                f'    """Two-layer MLP with GELU activation."""\n\n'
                f"    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1) -> None:\n"
                f"        super().__init__()\n"
                f"        d_ff = d_ff or d_model * 4\n"
                f"        self.gate = nn.Linear(d_model, d_ff, bias=False)\n"
                f"        self.proj = nn.Linear(d_ff, d_model, bias=False)\n"
                f"        self.drop = nn.Dropout(dropout)\n\n"
                f"    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
                f"        x = self.gate(x)\n"
                f"        x = torch.nn.functional.gelu(x)\n"
                f"        x = self.proj(x)\n"
                f"        return self.drop(x)\n"
            )

        def t05() -> str:
            """CSV parser with generator."""
            fn = self._snake("stream", "csv")
            delim = self._snake("delim")
            return (
                "import csv\n"
                "from pathlib import Path\n"
                "from typing import Iterator\n\n"
                f"def {fn}(path: Path, {delim}: str = ',', skip_header: bool = True) -> Iterator[dict[str, str]]:\n"
                f'    """Stream rows from a CSV file as dictionaries."""\n'
                f"    with open(path, newline='', encoding='utf-8') as fh:\n"
                f"        reader = csv.DictReader(fh, delimiter={delim})\n"
                f"        if skip_header:\n"
                f"            next(reader, None)\n"
                f"        for row in reader:\n"
                f"            yield { {k: v.strip() if v else '' for k, v in row.items()} }\n"
            )

        def t06() -> str:
            """Argparse-based CLI."""
            fn = self._snake("main")
            ap = self._snake("parser")
            arg = self._snake("input", "file")
            return (
                "import argparse\n"
                "import sys\n\n"
                f"def {fn}() -> None:\n"
                f'    """Entry-point for the data processing CLI."""\n'
                f"    {ap} = argparse.ArgumentParser(description='Process data files.')\n"
                f"    {ap}.add_argument('{arg}', type=str, help='Path to input file')\n"
                f"    {ap}.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')\n"
                f"    {ap}.add_argument('--limit', '-n', type=int, default=None, help='Row limit')\n"
                f"    args = {ap}.parse_args()\n"
                f"    if args.verbose:\n"
                f"        print(f'Processing {{args.{arg}}}...', file=sys.stderr)\n"
                f"    # processing logic here\n"
                f"    print(f'Done: {{args.{arg}}}')\n\n\n"
                f"if __name__ == '__main__':\n"
                f"    {fn}()\n"
            )

        def t07() -> str:
            """Quicksort implementation."""
            fn = self._snake("quicksort")
            return (
                "from typing import Protocol, TypeVar\n\n"
                "T = TypeVar('T', bound='Comparable')\n\n\n"
                "class Comparable(Protocol):\n"
                '    """Protocol for comparable types."""\n'
                "    def __lt__(self, other: Any) -> bool: ...\n\n\n"
                f"def {fn}(items: list[T], low: int = 0, high: int | None = None) -> list[T]:\n"
                f'    """In-place quicksort with Hoare partition scheme."""\n'
                f"    if high is None:\n"
                f"        high = len(items) - 1\n"
                f"    if low < high:\n"
                f"        pivot = items[(low + high) // 2]\n"
                f"        i, j = low - 1, high + 1\n"
                f"        while True:\n"
                f"            i += 1\n"
                f"            while items[i] < pivot:\n"
                f"                i += 1\n"
                f"            j -= 1\n"
                f"            while items[j] > pivot:\n"
                f"                j -= 1\n"
                f"            if i >= j:\n"
                f"                break\n"
                f"            items[i], items[j] = items[j], items[i]\n"
                f"        {fn}(items, low, j)\n"
                f"        {fn}(items, j + 1, high)\n"
                f"    return items\n"
            )

        def t08() -> str:
            """Binary search tree node."""
            cls = self._camel("BST", "Node")
            val = self._snake("value")
            return (
                "from __future__ import annotations\n"
                "from typing import Optional, TypeVar\n\n"
                "K = TypeVar('K')\n"
                "V = TypeVar('V')\n\n\n"
                f"class {cls}:\n"
                f'    """Binary search tree node mapping keys to values."""\n\n'
                f"    def __init__(self, key: K, {val}: V) -> None:\n"
                f"        self.key = key\n"
                f"        self.value = {val}\n"
                f"        self.left: Optional[{cls}] = None\n"
                f"        self.right: Optional[{cls}] = None\n\n"
                f"    def insert(self, key: K, {val}: V) -> None:\n"
                f"        if key < self.key:\n"
                f"            if self.left is None:\n"
                f"                self.left = {cls}(key, {val})\n"
                f"            else:\n"
                f"                self.left.insert(key, {val})\n"
                f"        elif key > self.key:\n"
                f"            if self.right is None:\n"
                f"                self.right = {cls}(key, {val})\n"
                f"            else:\n"
                f"                self.right.insert(key, {val})\n"
                f"        else:\n"
                f"            self.value = {val}\n\n"
                f"    def search(self, key: K) -> Optional[V]:\n"
                f"        if key == self.key:\n"
                f"            return self.value\n"
                f"        if key < self.key and self.left:\n"
                f"            return self.left.search(key)\n"
                f"        if key > self.key and self.right:\n"
                f"            return self.right.search(key)\n"
                f"        return None\n"
            )

        def t09() -> str:
            """LRU cache decorator."""
            fn = self._snake("lru")
            cap = self._snake("capacity")
            return (
                "import functools\n"
                "from collections import OrderedDict\n"
                "from typing import Callable, TypeVar\n\n"
                "F = TypeVar('F', bound=Callable[..., Any])\n\n\n"
                f"def {fn}({cap}: int = 128) -> Callable[[F], F]:\n"
                f'    """Decorator: LRU-cache the wrapped function with *{cap}* entries."""\n'
                f"    def decorator(func: F) -> F:\n"
                f"        cache: OrderedDict[tuple, Any] = OrderedDict()\n\n"
                f"        @functools.wraps(func)\n"
                f"        def wrapper(*args: Any, **kwargs: Any) -> Any:\n"
                f"            key = (args, tuple(sorted(kwargs.items())))\n"
                f"            if key in cache:\n"
                f"                cache.move_to_end(key)\n"
                f"                return cache[key]\n"
                f"            result = func(*args, **kwargs)\n"
                f"            cache[key] = result\n"
                f"            if len(cache) > {cap}:\n"
                f"                cache.popitem(last=False)\n"
                f"            return result\n"
                f"        return wrapper  # type: ignore\n"
                f"    return decorator\n"
            )

        def t10() -> str:
            """YAML config loader."""
            fn = self._snake("load", "config")
            return (
                "from pathlib import Path\n"
                "from typing import Any\n"
                "import yaml\n\n\n"
                f"def {fn}(path: str | Path, env_prefix: str = 'APP_') -> dict[str, Any]:\n"
                f'    """Load a YAML config and overlay environment variables prefixed by *env_prefix*."""\n'
                f"    path = Path(path)\n"
                f"    if not path.exists():\n"
                f"        raise FileNotFoundError(f'Config not found: {{path}}')\n"
                f"    with open(path, 'r') as fh:\n"
                f"        cfg: dict[str, Any] = yaml.safe_load(fh) or {{}}\n"
                f"    import os\n"
                f"    for key, val in os.environ.items():\n"
                f"        if key.startswith(env_prefix):\n"
                f"            cfg[key[len(env_prefix):].lower()] = val\n"
                f"    return cfg\n"
            )

        def t11() -> str:
            """Structured logger setup."""
            fn = self._snake("setup", "logger")
            return (
                "import logging\n"
                "import sys\n\n\n"
                f"def {fn}(name: str, level: int = logging.INFO, fmt: str | None = None) -> logging.Logger:\n"
                f'    """Configure a structured logger with JSON-friendly formatting."""\n'
                f"    if fmt is None:\n"
                f"        fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'\n"
                f"    logger = logging.getLogger(name)\n"
                f"    logger.setLevel(level)\n"
                f"    handler = logging.StreamHandler(sys.stdout)\n"
                f"    handler.setFormatter(logging.Formatter(fmt))\n"
                f"    if not logger.handlers:\n"
                f"        logger.addHandler(handler)\n"
                f"    return logger\n"
            )

        def t12() -> str:
            """Thread pool worker."""
            cls = self._camel("Task", "Worker")
            return (
                "import queue\n"
                "import threading\n"
                "from typing import Callable\n\n\n"
                f"class {cls}:\n"
                f'    """Simple thread-pool worker consuming tasks from a queue."""\n\n'
                f"    def __init__(self, num_workers: int = 4) -> None:\n"
                f"        self._queue: queue.Queue = queue.Queue()\n"
                f"        self._workers = [\n"
                f"            threading.Thread(target=self._run, daemon=True)\n"
                f"            for _ in range(num_workers)\n"
                f"        ]\n\n"
                f"    def start(self) -> None:\n"
                f"        for w in self._workers:\n"
                f"            w.start()\n\n"
                f"    def submit(self, fn: Callable, /, *args: Any, **kwargs: Any) -> None:\n"
                f"        self._queue.put((fn, args, kwargs))\n\n"
                f"    def _run(self) -> None:\n"
                f"        while True:\n"
                f"            fn, args, kwargs = self._queue.get()\n"
                f"            try:\n"
                f"                fn(*args, **kwargs)\n"
                f"            except Exception as exc:\n"
                f"                print(f'Worker error: {{exc}}')\n"
                f"            finally:\n"
                f"                self._queue.task_done()\n\n"
                f"    def join(self) -> None:\n"
                f"        self._queue.join()\n"
            )

        def t13() -> str:
            """REST API client."""
            cls = self._camel("API", "Client")
            base = self._snake("base", "url")
            return (
                "import requests\n"
                "from typing import Optional\n\n\n"
                f"class {cls}:\n"
                f'    """Lightweight REST API client with bearer auth."""\n\n'
                f"    def __init__(self, {base}: str, token: Optional[str] = None) -> None:\n"
                f"        self.{base} = {base}.rstrip('/')\n"
                f"        self._session = requests.Session()\n"
                f"        if token:\n"
                f"            self._session.headers['Authorization'] = f'Bearer {{token}}'\n"
                f"        self._session.headers['Content-Type'] = 'application/json'\n\n"
                f"    def get(self, endpoint: str, params: dict | None = None) -> dict:\n"
                f"        resp = self._session.get(f'{{self.{base}}}/{{endpoint.lstrip(\"/\")}}', params=params)\n"
                f"        resp.raise_for_status()\n"
                f"        return resp.json()\n\n"
                f"    def post(self, endpoint: str, data: dict | None = None) -> dict:\n"
                f"        resp = self._session.post(f'{{self.{base}}}/{{endpoint.lstrip(\"/\")}}', json=data)\n"
                f"        resp.raise_for_status()\n"
                f"        return resp.json()\n"
            )

        def t14() -> str:
            """Simple SQLAlchemy-style model."""
            cls = self._camel("Base", "Model")
            tbl = self._snake()
            return (
                "from datetime import datetime\n"
                "from typing import Optional\n"
                "from sqlalchemy import Column, Integer, String, DateTime, create_engine\n"
                "from sqlalchemy.orm import declarative_base, Session\n\n"
                "Base = declarative_base()\n\n\n"
                f"class {cls}(Base):\n"
                f'    """Generic ORM model with timestamps."""\n'
                f"    __tablename__ = '{tbl}'\n\n"
                f"    id = Column(Integer, primary_key=True)\n"
                f"    name = Column(String(255), nullable=False)\n"
                f"    created_at = Column(DateTime, default=datetime.utcnow)\n"
                f"    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)\n\n"
                f"    def to_dict(self) -> dict:\n"
                f"        return {{c.name: getattr(self, c.name) for c in self.__table__.columns}}\n"
            )

        def t15() -> str:
            """Regex-based log parser."""
            fn = self._snake("parse", "log", "line")
            pat = self._snake("pattern")
            return (
                "import re\n"
                "from typing import Optional\n\n\n"
                f"def {fn}(line: str) -> Optional[dict[str, str]]:\n"
                f'    """Extract structured fields from a log line using a named regex."""\n'
                f"    {pat} = re.compile(\n"
                f"        r'^(?P<timestamp>\\d{{4}}-\\d{{2}}-\\d{{2}}T\\d{{2}}:\\d{{2}}:\\d{{2}})'\n"
                f"        r'\\s+(?P<level>INFO|WARN|ERROR|DEBUG)'\n"
                f"        r'\\s+(?P<module>[\\w.]+)'\n"
                f"        r'\\s+(?P<message>.+)$'\n"
                f"    )\n"
                f"    match = {pat}.match(line.strip())\n"
                f"    if match:\n"
                f"        return match.groupdict()\n"
                f"    return None\n"
            )

        def t16() -> str:
            """Pytest-style test suite."""
            fn = self._snake("test", "compute")
            cls = self._camel("Test", "Math")
            return (
                "import pytest\n\n\n"
                f"class {cls}:\n"
                f'    """Test suite for mathematical utilities."""\n\n'
                f"    def {fn}_basic(self) -> None:\n"
                f"        assert 1 + 1 == 2\n"
                f"        assert 2 * 3 == 6\n\n"
                f"    def {fn}_edge(self) -> None:\n"
                f"        assert 0 ** 0 == 1  # by convention\n\n"
                f"    @pytest.mark.parametrize('a,b,expected', [\n"
                f"        (1, 2, 3),\n"
                f"        (-1, 1, 0),\n"
                f"        (0, 0, 0),\n"
                f"        (100, -100, 0),\n"
                f"    ])\n"
                f"    def {fn}_param(self, a: int, b: int, expected: int) -> None:\n"
                f"        assert a + b == expected\n"
            )

        def t17() -> str:
            """Heap priority queue."""
            cls = self._camel("Priority", "Queue")
            return (
                "import heapq\n"
                "from typing import Generic, TypeVar, List, Optional\n\n"
                "P = TypeVar('P')\n"
                "V = TypeVar('V')\n\n\n"
                f"class {cls}(Generic[P, V]):\n"
                f'    """Min-heap priority queue with (priority, value) semantics."""\n\n'
                f"    def __init__(self) -> None:\n"
                f"        self._heap: list[tuple[P, V]] = []\n\n"
                f"    def push(self, priority: P, value: V) -> None:\n"
                f"        heapq.heappush(self._heap, (priority, value))\n\n"
                f"    def pop(self) -> tuple[P, V]:\n"
                f"        if not self._heap:\n"
                f"            raise IndexError('pop from empty priority queue')\n"
                f"        return heapq.heappop(self._heap)\n\n"
                f"    def peek(self) -> tuple[P, V]:\n"
                f"        if not self._heap:\n"
                f"            raise IndexError('peek from empty priority queue')\n"
                f"        return self._heap[0]\n\n"
                f"    def __len__(self) -> int:\n"
                f"        return len(self._heap)\n"
            )

        def t18() -> str:
            """Date/time formatting utility."""
            fn = self._snake("format", "timestamp")
            return (
                "from datetime import datetime, timezone\n\n\n"
                f"def {fn}(dt: datetime | None = None, fmt: str = '%Y-%m-%dT%H:%M:%SZ') -> str:\n"
                f'    """Format a datetime as an ISO-8601 string (default: UTC)."""\n'
                f"    if dt is None:\n"
                f"        dt = datetime.now(timezone.utc)\n"
                f"    return dt.astimezone(timezone.utc).strftime(fmt)\n\n\n"
                f"def {self._snake('parse', 'timestamp')}(s: str) -> datetime:\n"
                f'    """Parse an ISO-8601 string into a datetime."""\n'
                f"    return datetime.fromisoformat(s.replace('Z', '+00:00'))\n"
            )

        def t19() -> str:
            """SHA-256 hashing utility."""
            fn = self._snake("hash", "content")
            return (
                "import hashlib\n\n\n"
                f"def {fn}(data: bytes | str, algorithm: str = 'sha256') -> str:\n"
                f'    """Return the hex digest of *data* using the specified *algorithm*."""\n'
                f"    if isinstance(data, str):\n"
                f"        data = data.encode('utf-8')\n"
                f"    h = hashlib.new(algorithm)\n"
                f"    h.update(data)\n"
                f"    return h.hexdigest()\n"
            )

        def t20() -> str:
            """JSON serialization helpers."""
            fn = self._snake("serialize", "compact")
            return (
                "import json\n"
                "from pathlib import Path\n"
                "from typing import Any\n\n\n"
                f"def {fn}(obj: Any, path: str | Path, sort_keys: bool = True) -> None:\n"
                f'    """Serialize *obj* as compact JSON with sorted keys."""\n'
                f"    with open(Path(path), 'w', encoding='utf-8') as fh:\n"
                f"        json.dump(obj, fh, ensure_ascii=False, separators=(',', ':'), sort_keys=sort_keys)\n\n\n"
                f"def {self._snake('deserialize')}(path: str | Path) -> Any:\n"
                f'    """Load and deserialize a JSON file."""\n'
                f"    with open(Path(path), 'r', encoding='utf-8') as fh:\n"
                f"        return json.load(fh)\n"
            )

        def t21() -> str:
            """Progress reporter context manager."""
            cls = self._camel("Progress", "Reporter")
            return (
                "import time\n"
                "from contextlib import contextmanager\n"
                "from typing import Iterator\n\n\n"
                f"class {cls}:\n"
                f'    """Simple context manager reporting elapsed time for a block."""\n\n'
                f"    def __init__(self, label: str = 'task') -> None:\n"
                f"        self.label = label\n"
                f"        self.elapsed: float = 0.0\n\n"
                f"    def __enter__(self) -> '{cls}':\n"
                f"        self._start = time.perf_counter()\n"
                f"        return self\n\n"
                f"    def __exit__(self, *args: Any) -> None:\n"
                f"        self.elapsed = time.perf_counter() - self._start\n"
                f"        print(f'[{{self.label}}] took {{self.elapsed:.3f}}s')\n"
            )

        def t22() -> str:
            """Graph adjacency list."""
            cls = self._camel("Directed", "Graph")
            return (
                "from collections import defaultdict\n"
                "from typing import Hashable, list\n\n\n"
                f"class {cls}:\n"
                f'    """Adjacency-list directed graph."""\n\n'
                f"    def __init__(self) -> None:\n"
                f"        self._adj: dict[Hashable, list[Hashable]] = defaultdict(list)\n\n"
                f"    def add_edge(self, u: Hashable, v: Hashable) -> None:\n"
                f"        self._adj[u].append(v)\n\n"
                f"    def topological_sort(self) -> list[Hashable]:\n"
                f"        visited: set[Hashable] = set()\n"
                f"        result: list[Hashable] = []\n\n"
                f"        def dfs(node: Hashable) -> None:\n"
                f"            visited.add(node)\n"
                f"            for neighbor in self._adj[node]:\n"
                f"                if neighbor not in visited:\n"
                f"                    dfs(neighbor)\n"
                f"            result.append(node)\n\n"
                f"        for node in list(self._adj):\n"
                f"            if node not in visited:\n"
                f"                dfs(node)\n"
                f"        return result[::-1]\n"
            )

        return [
            t01, t02, t03, t04, t05, t06, t07, t08, t09, t10,
            t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22,
        ]

    # ------------------------------------------------------------------
    # JavaScript / TypeScript templates  (≥15)
    # ------------------------------------------------------------------

    def _build_js_templates(self) -> list[callable]:

        def t01() -> str:
            """React functional component with hooks."""
            comp = self._camel()
            hook = self._snake("use")
            return (
                "import React, { useState, useEffect, useCallback } from 'react';\n\n"
                f"interface {comp}Props {{\n"
                f"  initial{self._camel('Count')}: number;\n"
                f"  on{self._camel('Update')}?: (val: number) => void;\n"
                f"}}\n\n"
                f"const {comp}: React.FC<{comp}Props> = ({{ initial{self._camel('Count')}, on{self._camel('Update')} }}) => {{\n"
                f"  const [{hook}Val, set{hook}Val] = useState<number>(initial{self._camel('Count')});\n\n"
                f"  useEffect(() => {{\n"
                f"    console.log(`{comp} mounted with ${{initial{self._camel('Count')}}}`);\n"
                f"    return () => console.log('{comp} unmounting');\n"
                f"  }}, []);\n\n"
                f"  const handleClick = useCallback(() => {{\n"
                f"    set{hook}Val(prev => prev + 1);\n"
                f"    on{self._camel('Update')}?.(hookVal + 1);\n"
                f"  }}, [on{self._camel('Update')}, hookVal]);\n\n"
                f"  return (\n"
                f"    <div className=\"{self._snake('widget', 'container')}\">\n"
                f"      <p>Current value: {hookVal}</p>\n"
                f"      <button onClick={{handleClick}}>Increment</button>\n"
                f"    </div>\n"
                f"  );\n"
                f"}};\n\n"
                f"export default {comp};\n"
            )

        def t02() -> str:
            """Express route handler."""
            fn = self._snake()
            route = self._snake()
            return (
                "import express, { Request, Response, NextFunction } from 'express';\n\n"
                "const router = express.Router();\n\n"
                f"interface {self._camel('List', 'Params')} {{\n"
                f"  page?: number;\n"
                f"  limit?: number;\n"
                f"  sort?: 'asc' | 'desc';\n"
                f"}}\n\n"
                f"router.get('/{route}', async (req: Request<{{}}, {{}}, {{}}, {self._camel('List', 'Params')}>, res: Response, next: NextFunction) => {{\n"
                f"  try {{\n"
                f"    const {{ page = 1, limit = 20, sort = 'desc' }} = req.query;\n"
                f"    const offset = (page - 1) * limit;\n"
                f"    const results = await {fn}(offset, limit, sort);\n"
                f"    res.json({{ data: results, page, limit }});\n"
                f"  }} catch (err) {{\n"
                f"    next(err);\n"
                f"  }}\n"
                f"}});\n\n"
                f"async function {fn}(offset: number, limit: number, sort: string): Promise<any[]> {{\n"
                f"  // simulated database query\n"
                f"  return Array.from({{ length: limit }}, (_, i) => ({{"
                f" id: offset + i + 1, name: `item-${{offset + i}}` }}));\n"
                f"}}\n\n"
                "export default router;\n"
            )

        def t03() -> str:
            """Async data fetcher with retry."""
            fn = self._snake("fetch", "with", "retry")
            max_r = self._snake("max", "retries")
            return (
                "async function "
                f"{fn}<T>(url: string, {{ {max_r} = 3, delay = 1000 }}: {{ {max_r}?: number; delay?: number }} = {{}}): Promise<T> {{\n"
                f"  for (let attempt = 1; attempt <= {max_r}; attempt++) {{\n"
                f"    try {{\n"
                f"      const resp = await fetch(url);\n"
                f"      if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);\n"
                f"      return (await resp.json()) as T;\n"
                f"    }} catch (err) {{\n"
                f"      if (attempt === {max_r}) throw err;\n"
                f"      console.warn(`Attempt ${{attempt}} failed, retrying...`);\n"
                f"      await new Promise(r => setTimeout(r, delay * attempt));\n"
                f"    }}\n"
                f"  }}\n"
                f"  throw new Error('Unreachable');\n"
                f"}}\n\n"
                "export { fetchWithRetry };\n"
            )

        def t04() -> str:
            """Promise-based rate limiter."""
            cls = self._camel("Rate", "Limiter")
            return (
                "export class "
                f"{cls} {{\n"
                f"  private queue: Array<{{ execute: () => Promise<any>; resolve: (v: any) => void; reject: (e: any) => void }}> = [];\n"
                f"  private active = 0;\n\n"
                f"  constructor(private readonly limit: number, private readonly interval: number) {{}}\n\n"
                f"  async enqueue<T>(fn: () => Promise<T>): Promise<T> {{\n"
                f"    return new Promise<T>((resolve, reject) => {{\n"
                f"      this.queue.push({{ execute: fn, resolve, reject }});\n"
                f"      this.process();\n"
                f"    }});\n"
                f"  }}\n\n"
                f"  private process(): void {{\n"
                f"    if (this.active >= this.limit || this.queue.length === 0) return;\n"
                f"    this.active++;\n"
                f"    const item = this.queue.shift()!;\n"
                f"    item\n"
                f"      .execute()\n"
                f"      .then(item.resolve)\n"
                f"      .catch(item.reject)\n"
                f"      .finally(() => {{\n"
                f"        setTimeout(() => {{\n"
                f"          this.active--;\n"
                f"          this.process();\n"
                f"        }}, this.interval);\n"
                f"      }});\n"
                f"  }}\n"
                f"}}\n"
            )

        def t05() -> str:
            """Generic array utility (TypeScript)."""
            fn = self._snake("chunk")
            return (
                "export function "
                f"{fn}<T>(arr: readonly T[], size: number): T[][] {{\n"
                f"  if (size <= 0) throw new Error('Chunk size must be positive');\n"
                f"  const result: T[][] = [];\n"
                f"  for (let i = 0; i < arr.length; i += size) {{\n"
                f"    result.push(arr.slice(i, i + size));\n"
                f"  }}\n"
                f"  return result;\n"
                f"}}\n\n"
                "export function "
                f"{self._snake('shuffle')}<T>(arr: readonly T[]): T[] {{\n"
                f"  const copy = [...arr];\n"
                f"  for (let i = copy.length - 1; i > 0; i--) {{\n"
                f"    const j = Math.floor(Math.random() * (i + 1));\n"
                f"    [copy[i], copy[j]] = [copy[j], copy[i]];\n"
                f"  }}\n"
                f"  return copy;\n"
                f"}}\n"
            )

        def t06() -> str:
            """Node.js file processor with streams."""
            fn = self._snake("process", "file")
            return (
                "import fs from 'fs';\n"
                "import { Transform } from 'stream';\n\n"
                f"function {fn}(inputPath: string, outputPath: string): Promise<void> {{\n"
                f"  return new Promise((resolve, reject) => {{\n"
                f"    const transform = new Transform({{\n"
                f"      transform(chunk: Buffer, _encoding: string, callback: Function) {{\n"
                f"        const upper = chunk.toString('utf-8').toUpperCase();\n"
                f"        callback(null, Buffer.from(upper, 'utf-8'));\n"
                f"      }},\n"
                f"    }});\n\n"
                f"    fs.createReadStream(inputPath)\n"
                f"      .pipe(transform)\n"
                f"      .pipe(fs.createWriteStream(outputPath))\n"
                f"      .on('finish', resolve)\n"
                f"      .on('error', reject);\n"
                f"  }});\n"
                f"}}\n\n"
                "export { processFile };\n"
            )

        def t07() -> str:
            """Service class with dependency injection pattern."""
            cls = self._camel("Data", "Service")
            repo = self._camel("Repository")
            return (
                f"interface {repo} {{\n"
                f"  findById(id: string): Promise<any | null>;\n"
                f"  save(entity: any): Promise<void>;\n"
                f"}}\n\n"
                f"export class {cls} {{\n"
                f"  constructor(private readonly repository: {repo}) {{}}\n\n"
                f"  async getById(id: string): Promise<any | null> {{\n"
                f"    const entity = await this.repository.findById(id);\n"
                f"    if (!entity) throw new Error(`Entity ${{id}} not found`);\n"
                f"    return entity;\n"
                f"  }}\n\n"
                f"  async create(data: Record<string, unknown>): Promise<void> {{\n"
                f"    const entity = {{ ...data, id: crypto.randomUUID(), createdAt: new Date().toISOString() }};\n"
                f"    await this.repository.save(entity);\n"
                f"  }}\n"
                f"}}\n"
            )

        def t08() -> str:
            """Event emitter pattern."""
            cls = self._camel("Typed", "Emitter")
            return (
                "type EventMap = Record<string, (...args: any[]) => void>;\n\n"
                f"export class {cls}<T extends EventMap> {{\n"
                f"  private listeners: Map<keyof T, Set<T[keyof T]>> = new Map();\n\n"
                f"  on<K extends keyof T>(event: K, listener: T[K]): void {{\n"
                f"    if (!this.listeners.has(event)) {{\n"
                f"      this.listeners.set(event, new Set());\n"
                f"    }}\n"
                f"    this.listeners.get(event)!.add(listener);\n"
                f"  }}\n\n"
                f"  emit<K extends keyof T>(event: K, ...args: Parameters<T[K]>): void {{\n"
                f"    const set = this.listeners.get(event);\n"
                f"    if (set) {{\n"
                f"      for (const listener of set) {{\n"
                f"        listener(...args);\n"
                f"      }}\n"
                f"    }}\n"
                f"  }}\n\n"
                f"  off<K extends keyof T>(event: K, listener: T[K]): void {{\n"
                f"    this.listeners.get(event)?.delete(listener);\n"
                f"  }}\n"
                f"}}\n"
            )

        def t09() -> str:
            """Express error-handling middleware."""
            fn = self._snake("error", "handler")
            return (
                "import { Request, Response, NextFunction } from 'express';\n\n"
                f"function {fn}(err: any, _req: Request, res: Response, _next: NextFunction): void {{\n"
                f"  const status = err.status || err.statusCode || 500;\n"
                f"  const message = err.message || 'Internal Server Error';\n"
                f"  const requestId = _req.headers['x-request-id'] || 'unknown';\n\n"
                f"  console.error(`[${{requestId}}] ${{status}} - ${{message}}`);\n\n"
                f"  res.status(status).json({{\n"
                f"    error: {{\n"
                f"      status,\n"
                f"      message,\n"
                f"      requestId,\n"
                f"      timestamp: new Date().toISOString(),\n"
                f"    }},\n"
                f"  }});\n"
                f"}}\n\n"
                "export default errorHandler;\n"
            )

        def t10() -> str:
            """React custom hook for local storage."""
            hook = self._snake("use", "persisted", "state")
            return (
                "import { useState, useEffect, useCallback } from 'react';\n\n"
                f"function {hook}<T>(key: string, initialValue: T): [T, (val: T | ((prev: T) => T)) => void] {{\n"
                f"  const [storedValue, setStoredValue] = useState<T>(() => {{\n"
                f"    try {{\n"
                f"      const item = window.localStorage.getItem(key);\n"
                f"      return item ? (JSON.parse(item) as T) : initialValue;\n"
                f"    }} catch {{\n"
                f"      return initialValue;\n"
                f"    }}\n"
                f"  }});\n\n"
                f"  const setValue = useCallback((value: T | ((prev: T) => T)) => {{\n"
                f"    setStoredValue(prev => {{\n"
                f"      const next = value instanceof Function ? value(prev) : value;\n"
                f"      try {{\n"
                f"        window.localStorage.setItem(key, JSON.stringify(next));\n"
                f"      }} catch (e) {{\n"
                f"        console.warn('localStorage write failed:', e);\n"
                f"      }}\n"
                f"      return next;\n"
                f"    }});\n"
                f"  }}, [key]);\n\n"
                f"  return [storedValue, setValue];\n"
                f"}}\n\n"
                "export { usePersistedState };\n"
            )

        def t11() -> str:
            """Lodash-style deep clone."""
            fn = self._snake("deep", "clone")
            return (
                f"function {fn}<T>(obj: T, seen: WeakMap<object, any> = new WeakMap()): T {{\n"
                f"  if (obj === null || typeof obj !== 'object') return obj;\n"
                f"  if (seen.has(obj as object)) return seen.get(obj as object);\n\n"
                f"  let clone: any;\n"
                f"  if (Array.isArray(obj)) {{\n"
                f"    clone = [];\n"
                f"    seen.set(obj, clone);\n"
                f"    for (const item of obj) clone.push({fn}(item, seen));\n"
                f"  }} else {{\n"
                f"    clone = {{}};\n"
                f"    seen.set(obj, clone);\n"
                f"    for (const [k, v] of Object.entries(obj)) {{\n"
                f"      clone[k] = {fn}(v, seen);\n"
                f"    }}\n"
                f"  }}\n"
                f"  return clone;\n"
                f"}}\n\n"
                "export { deepClone };\n"
            )

        def t12() -> str:
            """TypeScript enum + interface pattern."""
            name = self._camel("Status")
            iface = self._camel("Task", "Record")
            return (
                f"export enum {name} {{\n"
                f"  Pending = 'PENDING',\n"
                f"  Running = 'RUNNING',\n"
                f"  Completed = 'COMPLETED',\n"
                f"  Failed = 'FAILED',\n"
                f"  Cancelled = 'CANCELLED',\n"
                f"}}\n\n"
                f"export interface {iface} {{\n"
                f"  id: string;\n"
                f"  title: string;\n"
                f"  status: {name};\n"
                f"  priority: number;\n"
                f"  tags: string[];\n"
                f"  createdAt: string;\n"
                f"  updatedAt?: string;\n"
                f"}}\n\n"
                f"export function {self._snake('is', 'completed')}(task: {iface}): boolean {{\n"
                f"  return task.status === {name}.Completed;\n"
                f"}}\n"
            )

        def t13() -> str:
            """Async generator / pagination helper."""
            fn = self._snake("paginate")
            return (
                "async function* "
                f"{fn}<T>(fetchPage: (cursor?: string) => Promise<{{ data: T[]; nextCursor?: string }}>): AsyncGenerator<T, void, undefined> {{\n"
                f"  let cursor: string | undefined;\n"
                f"  while (true) {{\n"
                f"    const {{ data, nextCursor }} = await fetchPage(cursor);\n"
                f"    for (const item of data) {{\n"
                f"      yield item;\n"
                f"    }}\n"
                f"    if (!nextCursor) break;\n"
                f"    cursor = nextCursor;\n"
                f"  }}\n"
                f"}}\n\n"
                "export { paginate };\n"
            )

        def t14() -> str:
            """WebSocket chat handler."""
            cls = self._camel("Chat", "Handler")
            return (
                "import WebSocket from 'ws';\n\n"
                f"export class {cls} {{\n"
                f"  private clients: Map<string, WebSocket> = new Map();\n\n"
                f"  handleConnection(ws: WebSocket): void {{\n"
                f"    const clientId = crypto.randomUUID();\n"
                f"    this.clients.set(clientId, ws);\n"
                f"    console.log(`Client ${{clientId}} connected`);\n\n"
                f"    ws.on('message', (raw: Buffer) => {{\n"
                f"      try {{\n"
                f"        const msg = JSON.parse(raw.toString());\n"
                f"        this.broadcast({{ ...msg, sender: clientId, timestamp: Date.now() }});\n"
                f"      }} catch {{\n"
                f"        ws.send(JSON.stringify({{ error: 'invalid message' }}));\n"
                f"      }}\n"
                f"    }});\n\n"
                f"    ws.on('close', () => {{\n"
                f"      this.clients.delete(clientId);\n"
                f"      console.log(`Client ${{clientId}} disconnected`);\n"
                f"    }});\n"
                f"  }}\n\n"
                f"  private broadcast(msg: object): void {{\n"
                f"    const payload = JSON.stringify(msg);\n"
                f"    for (const [, ws] of this.clients) {{\n"
                f"      if (ws.readyState === WebSocket.OPEN) {{\n"
                f"        ws.send(payload);\n"
                f"      }}\n"
                f"    }}\n"
                f"  }}\n"
                f"}}\n"
            )

        def t15() -> str:
            """Debounce utility."""
            fn = self._snake("debounce")
            return (
                f"function {fn}<T extends (...args: any[]) => any>(\n"
                f"  fn: T,\n"
                f"  delay: number,\n"
                f"  options: {{ leading?: boolean; trailing?: boolean }} = {{}}\n"
                f"): (...args: Parameters<T>) => void {{\n"
                f"  let timer: ReturnType<typeof setTimeout> | null = null;\n"
                f"  let lastCallTime = 0;\n\n"
                f"  return function (this: any, ...args: Parameters<T>): void {{\n"
                f"    const now = Date.now();\n"
                f"    const remaining = delay - (now - lastCallTime);\n\n"
                f"    if (options.leading && remaining <= 0) {{\n"
                f"      lastCallTime = now;\n"
                f"      fn.apply(this, args);\n"
                f"      return;\n"
                f"    }}\n\n"
                f"    if (timer) clearTimeout(timer);\n"
                f"    timer = setTimeout(() => {{\n"
                f"      lastCallTime = Date.now();\n"
                f"      if (options.trailing !== false) fn.apply(this, args);\n"
                f"    }}, delay);\n"
                f"  }};\n"
                f"}}\n\n"
                "export { debounce };\n"
            )

        return [t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15]

    # ------------------------------------------------------------------
    # Go templates  (≥10)
    # ------------------------------------------------------------------

    def _build_go_templates(self) -> list[callable]:

        def t01() -> str:
            """HTTP server handler."""
            fn = self._camel("Handle", self._camel())
            return (
                "package main\n\n"
                "import (\n"
                '    "encoding/json"\n'
                '    "log"\n'
                '    "net/http"\n'
                '    "time"\n'
                ")\n\n"
                f"type {self._camel('Server', 'Config')} struct {{\n"
                f"    Addr         string        `json:\"addr\"`\n"
                f"    ReadTimeout  time.Duration `json:\"read_timeout\"`\n"
                f"    WriteTimeout time.Duration `json:\"write_timeout\"`\n"
                f"}}\n\n"
                f"func {fn}(w http.ResponseWriter, r *http.Request) {{\n"
                f'    w.Header().Set("Content-Type", "application/json")\n'
                f'    resp := map[string]string{{"message": "ok", "time": time.Now().Format(time.RFC3339)}}\n'
                f"    json.NewEncoder(w).Encode(resp)\n"
                f"}}\n\n"
                "func main() {\n"
                f"    cfg := {self._camel('Server', 'Config')}{{\n"
                '        Addr:         ":8080",\n'
                "        ReadTimeout:  10 * time.Second,\n"
                "        WriteTimeout: 10 * time.Second,\n"
                "    }}\n"
                f"    http.HandleFunc(\"/api/v1/{self._snake()}\", {fn})\n"
                f"    log.Printf(\"listening on %s\", cfg.Addr)\n"
                f"    log.Fatal(http.ListenAndServe(cfg.Addr, nil))\n"
                "}\n"
            )

        def t02() -> str:
            """Concurrent worker pool."""
            fn = self._camel("Worker", "Pool")
            return (
                "package main\n\n"
                "import (\n"
                '    "fmt"\n'
                '    "sync"\n'
                ")\n\n"
                f"type {fn} struct {{\n"
                f"    jobs    chan func()\n"
                f"    wg      sync.WaitGroup\n"
                f"    quit    chan struct{{}}\n"
                f"}}\n\n"
                f"func New{fn}(numWorkers int) *{fn} {{\n"
                f"    wp := &{fn}{{\n"
                f"        jobs: make(chan func(), 100),\n"
                f"        quit: make(chan struct{{}}),\n"
                f"    }}\n"
                f"    for i := 0; i < numWorkers; i++ {{\n"
                f"        wp.wg.Add(1)\n"
                f"        go wp.worker()\n"
                f"    }}\n"
                f"    return wp\n"
                f"}}\n\n"
                f"func (wp *{fn}) Submit(fn func()) {{\n"
                f"    wp.jobs <- fn\n"
                f"}}\n\n"
                f"func (wp *{fn}) Stop() {{\n"
                f"    close(wp.quit)\n"
                f"    wp.wg.Wait()\n"
                f"}}\n\n"
                f"func (wp *{fn}) worker() {{\n"
                f"    defer wp.wg.Done()\n"
                f"    for {{\n"
                f"        select {{\n"
                f"        case fn := <-wp.jobs:\n"
                f"            fn()\n"
                f"        case <-wp.quit:\n"
                f"            return\n"
                f"        }}\n"
                f"    }}\n"
                f"}}\n"
            )

        def t03() -> str:
            """Interface + struct pattern."""
            iface = self._camel("Storage", "Backend")
            impl = self._camel("Disk", "Store")
            return (
                "package store\n\n"
                "import (\n"
                '    "context"\n'
                '    "errors"\n'
                ")\n\n"
                f"type {iface} interface {{\n"
                f"    Get(ctx context.Context, key string) ([]byte, error)\n"
                f"    Put(ctx context.Context, key string, data []byte) error\n"
                f"    Delete(ctx context.Context, key string) error\n"
                f"}}\n\n"
                f"type {impl} struct {{\n"
                f"    basePath string\n"
                f"}}\n\n"
                f"func New{impl}(path string) *{impl} {{\n"
                f"    return &{impl}{{basePath: path}}\n"
                f"}}\n\n"
                f"func (s *{impl}) Get(ctx context.Context, key string) ([]byte, error) {{\n"
                f"    if key == \"\" {{\n"
                f"        return nil, errors.New(\"empty key\")\n"
                f"    }}\n"
                f"    // simulated disk read\n"
                f"    return []byte(key), nil\n"
                f"}}\n\n"
                f"func (s *{impl}) Put(ctx context.Context, key string, data []byte) error {{\n"
                f"    if len(data) == 0 {{\n"
                f"        return errors.New(\"empty data\")\n"
                f"    }}\n"
                f"    return nil\n"
                f"}}\n\n"
                f"func (s *{impl}) Delete(ctx context.Context, key string) error {{\n"
                f"    return nil\n"
                f"}}\n"
            )

        def t04() -> str:
            """File processing utility."""
            fn = self._camel("Process", "File")
            return (
                "package main\n\n"
                "import (\n"
                '    "bufio"\n'
                '    "fmt"\n'
                '    "os"\n'
                ")\n\n"
                f"func {fn}(path string) (int, error) {{\n"
                f"    f, err := os.Open(path)\n"
                f"    if err != nil {{\n"
                f"        return 0, fmt.Errorf(\"open: %w\", err)\n"
                f"    }}\n"
                f"    defer f.Close()\n\n"
                f"    scanner := bufio.NewScanner(f)\n"
                f"    lines := 0\n"
                f"    for scanner.Scan() {{\n"
                f"        lines++\n"
                f"    }}\n"
                f"    return lines, scanner.Err()\n"
                f"}}\n\n"
                "func main() {\n"
                f"    count, err := {fn}(\"data.txt\")\n"
                f"    if err != nil {{\n"
                f"        fmt.Fprintf(os.Stderr, \"error: %v\\n\", err)\n"
                f"        os.Exit(1)\n"
                f"    }}\n"
                f"    fmt.Printf(\"Lines: %d\\n\", count)\n"
                "}\n"
            )

        def t05() -> str:
            """HTTP middleware chain."""
            fn = self._camel("Middleware")
            return (
                "package main\n\n"
                "import (\n"
                '    "log"\n'
                '    "net/http"\n'
                '    "time"\n'
                ")\n\n"
                f"type {fn} func(http.Handler) http.Handler\n\n"
                f"func {self._camel('Logging')}() {fn} {{\n"
                f"    return func(next http.Handler) http.Handler {{\n"
                f"        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {{\n"
                f"            start := time.Now()\n"
                f"            next.ServeHTTP(w, r)\n"
                f"            log.Printf(\"%s %s took %v\", r.Method, r.URL.Path, time.Since(start))\n"
                f"        }})\n"
                f"    }}\n"
                f"}}\n\n"
                f"func {self._camel('Recovery')}() {fn} {{\n"
                f"    return func(next http.Handler) http.Handler {{\n"
                f"        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {{\n"
                f"            defer func() {{\n"
                f"                if err := recover(); err != nil {{\n"
                f"                    http.Error(w, \"Internal Error\", http.StatusInternalServerError)\n"
                f"                    log.Printf(\"panic: %v\", err)\n"
                f"                }}\n"
                f"            }}()\n"
                f"            next.ServeHTTP(w, r)\n"
                f"        }})\n"
                f"    }}\n"
                f"}}\n\n"
                f"func {self._camel('Chain')}(h http.Handler, middlewares ...{fn}) http.Handler {{\n"
                f"    for i := len(middlewares) - 1; i >= 0; i-- {{\n"
                f"        h = middlewares[i](h)\n"
                f"    }}\n"
                f"    return h\n"
                f"}}\n"
            )

        def t06() -> str:
            """Table-driven tests."""
            fn = self._camel("Test", self._camel("Math"))
            return (
                "package main\n\n"
                'import "testing"\n\n'
                f"func {fn}(t *testing.T) {{\n"
                f"    tests := []struct {{\n"
                f"        name     string\n"
                f"        a, b     int\n"
                f"        expected int\n"
                f"    }}{{\n"
                f"        {{\"positive\", 3, 5, 8}},\n"
                f"        {{\"negative\", -1, 1, 0}},\n"
                f"        {{\"zero\", 0, 0, 0}},\n"
                f"        {{\"large\", 1000, 2000, 3000}},\n"
                f"    }}\n\n"
                f"    for _, tc := range tests {{\n"
                f"        t.Run(tc.name, func(t *testing.T) {{\n"
                f"            got := tc.a + tc.b\n"
                f"            if got != tc.expected {{\n"
                f"                t.Errorf(\"add(%d, %d) = %d; want %d\", tc.a, tc.b, got, tc.expected)\n"
                f"            }}\n"
                f"        }})\n"
                f"    }}\n"
                f"}}\n"
            )

        def t07() -> str:
            """JSON API client."""
            fn = self._camel("Fetch", "JSON")
            return (
                "package client\n\n"
                "import (\n"
                '    "encoding/json"\n'
                '    "fmt"\n'
                '    "io"\n'
                '    "net/http"\n'
                '    "time"\n'
                ")\n\n"
                f"type {self._camel('Client')} struct {{\n"
                f"    baseURL    string\n"
                f"    httpClient *http.Client\n"
                f"}}\n\n"
                f"func New{self._camel('Client')}(baseURL string) *{self._camel('Client')} {{\n"
                f"    return &{self._camel('Client')}{{\n"
                f"        baseURL: baseURL,\n"
                f"        httpClient: &http.Client{{\n"
                f"            Timeout: 30 * time.Second,\n"
                f"        }},\n"
                f"    }}\n"
                f"}}\n\n"
                f"func (c *{self._camel('Client')}) {fn}(path string, target any) error {{\n"
                f"    url := fmt.Sprintf(\"%s/%s\", c.baseURL, path)\n"
                f"    resp, err := c.httpClient.Get(url)\n"
                f"    if err != nil {{\n"
                f"        return fmt.Errorf(\"get: %w\", err)\n"
                f"    }}\n"
                f"    defer resp.Body.Close()\n\n"
                f"    if resp.StatusCode != http.StatusOK {{\n"
                f"        body, _ := io.ReadAll(resp.Body)\n"
                f"        return fmt.Errorf(\"status %d: %s\", resp.StatusCode, string(body))\n"
                f"    }}\n"
                f"    return json.NewDecoder(resp.Body).Decode(target)\n"
                f"}}\n"
            )

        def t08() -> str:
            """Mutex-protected cache."""
            cls = self._camel("Thread", "Safe", "Cache")
            return (
                "package cache\n\n"
                "import (\n"
                '    "sync"\n'
                '    "time"\n'
                ")\n\n"
                f"type item[T any] struct {{\n"
                f"    value     T\n"
                f"    expiresAt time.Time\n"
                f"}}\n\n"
                f"type {cls}[T any] struct {{\n"
                f"    mu    sync.RWMutex\n"
                f"    items map[string]item[T]\n"
                f"    ttl   time.Duration\n"
                f"}}\n\n"
                f"func New{cls}[T any](ttl time.Duration) *{cls}[T] {{\n"
                f"    return &{cls}[T]{{\n"
                f"        items: make(map[string]item[T]),\n"
                f"        ttl:   ttl,\n"
                f"    }}\n"
                f"}}\n\n"
                f"func (c *{cls}[T]) Get(key string) (T, bool) {{\n"
                f"    c.mu.RLock()\n"
                f"    defer c.mu.RUnlock()\n"
                f"    it, ok := c.items[key]\n"
                f"    if !ok || time.Now().After(it.expiresAt) {{\n"
                f"        var zero T\n"
                f"        return zero, false\n"
                f"    }}\n"
                f"    return it.value, true\n"
                f"}}\n\n"
                f"func (c *{cls}[T]) Set(key string, value T) {{\n"
                f"    c.mu.Lock()\n"
                f"    defer c.mu.Unlock()\n"
                f"    c.items[key] = item[T]{{\n"
                f"        value:     value,\n"
                f"        expiresAt: time.Now().Add(c.ttl),\n"
                f"    }}\n"
                f"}}\n"
            )

        def t09() -> str:
            """Channel-based pipeline."""
            fn = self._camel("Pipeline")
            return (
                "package main\n\n"
                "import \"fmt\"\n\n"
                f"func {fn}(nums []int) <-chan int {{\n"
                f"    out := make(chan int)\n"
                f"    go func() {{\n"
                f"        defer close(out)\n"
                f"        for _, n := range nums {{\n"
                f"            out <- n * n\n"
                f"        }}\n"
                f"    }}()\n"
                f"    return out\n"
                f"}}\n\n"
                f"func {self._camel('Filter', 'Odd')}(in <-chan int) <-chan int {{\n"
                f"    out := make(chan int)\n"
                f"    go func() {{\n"
                f"        defer close(out)\n"
                f"        for n := range in {{\n"
                f"            if n%2 != 0 {{\n"
                f"                out <- n\n"
                f"            }}\n"
                f"        }}\n"
                f"    }}()\n"
                f"    return out\n"
                f"}}\n\n"
                "func main() {\n"
                f"    nums := []int{{1, 2, 3, 4, 5, 6}}\n"
                f"    for n := range {self._camel('Filter', 'Odd')}({fn}(nums)) {{\n"
                f"        fmt.Println(n)\n"
                f"    }}\n"
                "}\n"
            )

        def t10() -> str:
            """CLI tool with flags."""
            fn = self._camel("Run")
            return (
                "package main\n\n"
                "import (\n"
                '    "flag"\n'
                '    "fmt"\n'
                '    "os"\n'
                ")\n\n"
                "func main() {\n"
                f"    var (\n"
                f"        input  = flag.String(\"input\", \"\", \"input file path\")\n"
                f"        output = flag.String(\"output\", \"output.txt\", \"output file path\")\n"
                f"        verbose = flag.Bool(\"verbose\", false, \"enable verbose logging\")\n"
                f"        threads = flag.Int(\"threads\", 4, \"number of worker threads\")\n"
                f"    )\n"
                f"    flag.Parse()\n\n"
                f"    if *input == \"\" {{\n"
                f'        fmt.Fprintln(os.Stderr, "error: --input is required")\n'
                f"        os.Exit(1)\n"
                f"    }}\n\n"
                f"    if *verbose {{\n"
                f'        fmt.Printf("Processing %s -> %s with %d workers\\n", *input, *output, *threads)\n'
                f"    }}\n"
                f"    // processing logic\n"
                f'    fmt.Println("Done")\n'
                "}\n"
            )

        return [t01, t02, t03, t04, t05, t06, t07, t08, t09, t10]

    # ------------------------------------------------------------------
    # Rust templates  (≥10)
    # ------------------------------------------------------------------

    def _build_rust_templates(self) -> list[callable]:

        def t01() -> str:
            """Struct with impl block."""
            cls = self._camel("Config")
            return (
                "use std::path::PathBuf;\n\n"
                f"#[derive(Debug, Clone)]\n"
                f"pub struct {cls} {{\n"
                f"    pub {self._snake()}: String,\n"
                f"    pub {self._snake()}: Option<PathBuf>,\n"
                f"    pub {self._snake()}: u64,\n"
                f"    pub {self._snake()}: Vec<String>,\n"
                f"}}\n\n"
                f"impl {cls} {{\n"
                f"    pub fn new(name: &str) -> Self {{\n"
                f"        Self {{\n"
                f"            name: name.to_string(),\n"
                f"            path: None,\n"
                f"            timeout: 30,\n"
                f"            tags: Vec::new(),\n"
                f"        }}\n"
                f"    }}\n\n"
                f"    pub fn with_path(mut self, path: PathBuf) -> Self {{\n"
                f"        self.path = Some(path);\n"
                f"        self\n"
                f"    }}\n\n"
                f"    pub fn is_valid(&self) -> bool {{\n"
                f"        !self.name.is_empty()\n"
                f"    }}\n"
                f"}}\n"
            )

        def t02() -> str:
            """Enum with match."""
            name = self._camel("Command")
            return (
                "#[derive(Debug, PartialEq)]\n"
                f"pub enum {name} {{\n"
                f"    {self._camel()},\n"
                f"    {self._camel()},\n"
                f"    {self._camel()}(String),\n"
                f"    {self._camel()}{{ {self._snake()}: u32, {self._snake()}: String }},\n"
                f"}}\n\n"
                f"impl {name} {{\n"
                f"    pub fn describe(&self) -> String {{\n"
                f"        match self {{\n"
                f"            Self::{self._camel()} => \"init\".into(),\n"
                f"            Self::{self._camel()} => \"reset\".into(),\n"
                f"            Self::{self._camel()}(msg) => format!(\"echo: {{msg}}\"),\n"
                f"            Self::{self._camel()}{{ {self._snake()}, .. }} => format!(\"set key={{}}\", {self._snake()}),\n"
                f"        }}\n"
                f"    }}\n"
                f"}}\n"
            )

        def t03() -> str:
            """Trait + impl."""
            trait = self._camel("Serializer")
            impl = self._camel("JSON", "Serializer")
            return (
                "use std::collections::HashMap;\n\n"
                f"pub trait {trait} {{\n"
                f"    fn serialize(&self) -> Result<String, String>;\n"
                f"    fn deserialize(input: &str) -> Result<Self, String>\n"
                f"    where\n"
                f"        Self: Sized;\n"
                f"}}\n\n"
                f"pub struct {impl};\n\n"
                f"impl {trait} for HashMap<String, String> {{\n"
                f"    fn serialize(&self) -> Result<String, String> {{\n"
                f"        serde_json::to_string(self).map_err(|e| e.to_string())\n"
                f"    }}\n\n"
                f"    fn deserialize(input: &str) -> Result<Self, String> {{\n"
                f"        serde_json::from_str(input).map_err(|e| e.to_string())\n"
                f"    }}\n"
                f"}}\n"
            )

        def t04() -> str:
            """Result/Option combinators."""
            fn = self._snake("parse", "config")
            return (
                "use std::path::Path;\n\n"
                "#[derive(Debug, thiserror::Error)]\n"
                f"pub enum {self._camel('Config', 'Error')} {{\n"
                f"    #[error(\"file not found: {{0}}\")]\n"
                f"    NotFound(String),\n"
                f"    #[error(\"parse error: {{0}}\")]\n"
                f"    ParseError(String),\n"
                f"}}\n\n"
                f"pub fn {fn}(path: &Path) -> Result<String, {self._camel('Config', 'Error')}> {{\n"
                f"    let content = std::fs::read_to_string(path)\n"
                f"        .map_err(|_| {self._camel('Config', 'Error')}::NotFound(path.display().to_string()))?;\n\n"
                f"    if content.is_empty() {{\n"
                f"        return Err({self._camel('Config', 'Error')}::ParseError(\"empty file\".into()));\n"
                f"    }}\n\n"
                f"    content.lines()\n"
                f"        .find(|l| l.starts_with(\"key=\"))\n"
                f"        .map(|l| l.trim_start_matches(\"key=\").to_string())\n"
                f"        .ok_or_else(|| {self._camel('Config', 'Error')}::ParseError(\"missing key\".into()))\n"
                f"}}\n"
            )

        def t05() -> str:
            """Iterator adapter."""
            fn = self._snake("unique", "windows")
            return (
                "/// Return an iterator over contiguous windows of size `n`\n"
                "/// where all elements in the window are unique.\n"
                f"pub fn {fn}<'a, T: Eq + std::hash::Hash>(\n"
                f"    items: &'a [T],\n"
                f"    n: usize,\n"
                f") -> impl Iterator<Item = &'a [T]> + 'a {{\n"
                f"    items.windows(n).filter(move |win| {{\n"
                f"        let mut seen = std::collections::HashSet::new();\n"
                f"        win.iter().all(move |x| seen.insert(x))\n"
                f"    }})\n"
                f"}}\n\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                f"    use super::*;\n\n"
                "    #[test]\n"
                "    fn test_unique_windows() {\n"
                "        let data = vec![1, 2, 2, 3, 4];\n"
                "        let result: Vec<_> = unique_windows(&data, 3).collect();\n"
                "        assert_eq!(result.len(), 1);\n"
                "        assert_eq!(result[0], &[2, 3, 4]);\n"
                "    }\n"
                "}\n"
            )

        def t06() -> str:
            """Generic function."""
            fn = self._snake("merge", "sorted")
            return (
                "/// Merge two sorted iterators into a single sorted `Vec`.\n"
                f"pub fn {fn}<T: Ord + Clone>(left: &[T], right: &[T]) -> Vec<T> {{\n"
                f"    let mut result = Vec::with_capacity(left.len() + right.len());\n"
                f"    let (mut i, mut j) = (0, 0);\n\n"
                f"    while i < left.len() && j < right.len() {{\n"
                f"        if left[i] <= right[j] {{\n"
                f"            result.push(left[i].clone());\n"
                f"            i += 1;\n"
                f"        }} else {{\n"
                f"            result.push(right[j].clone());\n"
                f"            j += 1;\n"
                f"        }}\n"
                f"    }}\n\n"
                f"    result.extend_from_slice(&left[i..]);\n"
                f"    result.extend_from_slice(&right[j..]);\n"
                f"    result\n"
                f"}}\n"
            )

        def t07() -> str:
            """Async function with tokio."""
            fn = self._snake("fetch", "parallel")
            return (
                "use tokio::task;\n\n"
                f"pub async fn {fn}(urls: &[String]) -> Vec<Result<String, reqwest::Error>> {{\n"
                f"    let handles: Vec<_> = urls\n"
                f"        .iter()\n"
                f"        .map(|url| {{\n"
                f"            let url = url.clone();\n"
                f"            task::spawn(async move {{\n"
                f"                let resp = reqwest::get(&url).await?;\n"
                f"                resp.text().await\n"
                f"            }})\n"
                f"        }})\n"
                f"        .collect();\n\n"
                f"    let mut results = Vec::with_capacity(handles.len());\n"
                f"    for handle in handles {{\n"
                f"        results.push(handle.await.unwrap());\n"
                f"    }}\n"
                f"    results\n"
                f"}}\n"
            )

        def t08() -> str:
            """Error type with thiserror."""
            cls = self._camel("Parse", "Error")
            return (
                "use std::num::ParseIntError;\n\n"
                "#[derive(Debug, thiserror::Error)]\n"
                f"pub enum {cls} {{\n"
                f"    #[error(\"invalid token: {{0}}\")]\n"
                f"    InvalidToken(String),\n\n"
                f"    #[error(\"unexpected end of input\")]\n"
                f"    UnexpectedEof,\n\n"
                f"    #[error(\"integer overflow at line {{0}}\")]\n"
                f"    Overflow(usize),\n\n"
                f"    #[error(transparent)]\n"
                f"    Io(#[from] std::io::Error),\n"
                f"}}\n\n"
                f"pub fn {self._snake('parse', 'number')}(s: &str) -> Result<i32, {cls}> {{\n"
                f"    let trimmed = s.trim();\n"
                f"    if trimmed.is_empty() {{\n"
                f"        return Err({cls}::UnexpectedEof);\n"
                f"    }}\n"
                f"    trimmed.parse::<i32>().map_err(|_| {cls}::InvalidToken(trimmed.into()))\n"
                f"}}\n"
            )

        def t09() -> str:
            """Smart pointer usage (Arc<Mutex<>>)."""
            cls = self._camel("Shared", "Counter")
            return (
                "use std::sync::{{Arc, Mutex}};\n\n"
                f"#[derive(Clone)]\n"
                f"pub struct {cls} {{\n"
                f"    inner: Arc<Mutex<u64>>,\n"
                f"}}\n\n"
                f"impl {cls} {{\n"
                f"    pub fn new() -> Self {{\n"
                f"        Self {{\n"
                f"            inner: Arc::new(Mutex::new(0)),\n"
                f"        }}\n"
                f"    }}\n\n"
                f"    pub fn increment(&self) -> u64 {{\n"
                f"        let mut guard = self.inner.lock().unwrap();\n"
                f"        *guard += 1;\n"
                f"        *guard\n"
                f"    }}\n\n"
                f"    pub fn value(&self) -> u64 {{\n"
                f"        *self.inner.lock().unwrap()\n"
                f"    }}\n"
                f"}}\n\n"
                "unsafe impl Send for SharedCounter {{}}\n"
                "unsafe impl Sync for SharedCounter {{}}\n"
            )

        def t10() -> str:
            """Collection utility (group by)."""
            fn = self._snake("group", "by")
            return (
                "use std::collections::HashMap;\n\n"
                "/// Group items by a key extracted via `f`.\n"
                f"pub fn {fn}<T, K, F>(items: impl IntoIterator<Item = T>, f: F) -> HashMap<K, Vec<T>>\n"
                f"where\n"
                f"    K: Eq + std::hash::Hash,\n"
                f"    F: Fn(&T) -> K,\n"
                f"{{\n"
                f"    let mut map = HashMap::new();\n"
                f"    for item in items {{\n"
                f"        let key = f(&item);\n"
                f"        map.entry(key).or_default().push(item);\n"
                f"    }}\n"
                f"    map\n"
                f"}}\n\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    use super::*;\n\n"
                "    #[test]\n"
                "    fn test_group_by_modulo() {\n"
                "        let nums = vec![0, 1, 2, 3, 4, 5];\n"
                "        let groups = group_by(nums, |x| x % 2);\n"
                "        assert_eq!(groups[&0], vec![0, 2, 4]);\n"
                "        assert_eq!(groups[&1], vec![1, 3, 5]);\n"
                "    }\n"
                "}\n"
            )

        return [t01, t02, t03, t04, t05, t06, t07, t08, t09, t10]

    # ------------------------------------------------------------------
    # Text / prose templates  (≥20)
    # ------------------------------------------------------------------

    def _build_text_templates(self) -> list[callable]:

        def t01() -> str:
            """Transformer attention explanation."""
            return (
                "Attention mechanisms form the backbone of modern transformer architectures. "
                "The scaled dot-product attention computes a weighted sum over values, where "
                "weights are derived from the compatibility between queries and keys. "
                "Formally, Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. "
                "The scaling factor sqrt(d_k) prevents the dot products from growing large "
                "in magnitude, which would push the softmax into regions of extremely small gradients. "
                "Multi-head attention extends this by projecting queries, keys, and values h times "
                "with different learned linear projections, allowing the model to jointly attend to "
                "information from different representation subspaces at different positions."
            )

        def t02() -> str:
            """Rust ownership model."""
            return (
                "Rust's ownership model enforces memory safety without a garbage collector. "
                "Every value in Rust has exactly one owner at any given time. When the owner "
                "goes out of scope, the value is dropped. References allow borrowing without "
                "taking ownership: immutable references (&T) are shared and unlimited, while "
                "mutable references (&mut T) are exclusive. The borrow checker enforces these "
                "rules at compile time, preventing data races and dangling pointers. "
                "This zero-cost abstraction enables systems programmers to write safe concurrent "
                "code without sacrificing performance."
            )

        def t03() -> str:
            """Data pipeline best practices."""
            return (
                "Building efficient data pipelines requires careful consideration of "
                "throughput, fault tolerance, and observability. A common pattern is to "
                "separate the pipeline into three stages: extraction, transformation, and "
                "loading (ETL). Extraction sources data from various inputs, transformation "
                "applies cleaning and normalization, and loading writes the result to a "
                "destination. For large-scale workloads, batch processing with Apache Spark "
                "or streaming with Apache Kafka provides horizontal scalability. "
                "Key design principles include idempotency (re-running a stage produces the "
                "same result), checkpointing (saving intermediate state for recovery), and "
                "schema validation early in the pipeline to fail fast on malformed data."
            )

        def t04() -> str:
            """Kernel methods in ML."""
            return (
                "Kernel methods allow learning algorithms to operate in a high-dimensional, "
                "implicit feature space without ever computing the coordinates of the data "
                "in that space. Instead, they compute inner products between the images of "
                "all pairs of data points via a kernel function. The radial basis function "
                "(RBF) kernel k(x, y) = exp(-gamma ||x - y||^2) is one of the most widely "
                "used kernels due to its universal approximation property. Kernel methods "
                "are particularly effective for problems where the decision boundary is "
                "highly nonlinear in the input space but becomes linear after the "
                "transformation. The main limitation is scalability: computing the full "
                "kernel matrix costs O(n^2) memory, which becomes prohibitive for datasets "
                "exceeding tens of thousands of examples."
            )

        def t05() -> str:
            """Consensus protocols."""
            return (
                "Distributed consensus protocols enable a group of nodes to agree on a "
                "single value despite failures. Paxos was the first practical consensus "
                "algorithm, using a two-phase protocol with prepare and accept rounds to "
                "achieve safety under asynchronous network conditions. Raft simplifies Paxos "
                "by decomposing consensus into leader election, log replication, and safety. "
                "In Raft, a leader is elected and all log entries flow through it. Entries "
                "are committed once a majority of nodes have replicated them. "
                "The CAP theorem states that a distributed system can guarantee at most two "
                "of three properties: consistency, availability, and partition tolerance. "
                "Most production systems choose availability and partition tolerance (AP) or "
                "consistency and partition tolerance (CP)."
            )

        def t06() -> str:
            """Memory management."""
            return (
                "Memory management in systems programming involves tracking which regions "
                "of memory are in use, allocating new regions when needed, and deallocating "
                "them when no longer required. Manual memory management, as in C, gives the "
                "programmer full control but is error-prone. Garbage collection, as in Java "
                "or Go, automates reclamation but introduces pause times. Reference counting "
                "provives deterministic deallocation at the cost of cyclic reference handling. "
                "Rust's ownership system offers a middle path: compile-time checks enforce "
                "that every resource has exactly one owner, and the drop glue is inserted "
                "automatically. Modern allocators like mimalloc and jemalloc use thread-local "
                "caching and size-class segregation to achieve near-zero allocation overhead."
            )

        def t07() -> str:
            """Graph traversal algorithms."""
            return (
                "Graph traversal algorithms systematically visit vertices in a graph. "
                "Breadth-first search (BFS) explores a vertex's neighbors before moving "
                "to the next level, making it ideal for finding shortest paths in unweighted "
                "graphs. Depth-first search (DFS) explores as far as possible along each "
                "branch before backtracking, making it useful for topological sorting and "
                "cycle detection. Both have O(V + E) time complexity when implemented with "
                "adjacency lists. Dijkstra's algorithm extends BFS to weighted graphs with "
                "non-negative edge weights, using a priority queue to always expand the "
                "least-cost frontier node. For graphs with negative edges, the Bellman-Ford "
                "algorithm is preferred, though it runs in O(VE) time."
            )

        def t08() -> str:
            """Matrix multiplication optimization."""
            return (
                "Optimizing matrix multiplication is critical for deep learning performance. "
                "The naive O(n^3) triple-loop implementation is memory-bandwidth bound because "
                "it does not exploit cache locality. Tiling (or blocking) is the primary "
                "optimization: by breaking the matrices into sub-blocks that fit in L1/L2 "
                "cache, we reduce cache misses dramatically. Strassen's algorithm achieves "
                "O(n^2.807) by recursively decomposing matrices into 7 multiplications "
                "instead of 8. In practice, BLAS libraries like cuBLAS and MKL use a "
                "combination of tiling, SIMD vectorization, and prefetching. For very large "
                "matrices, distributed approaches like Cannon's algorithm partition the "
                "computation across a processor grid."
            )

        def t09() -> str:
            """Introduction to type theory."""
            return (
                "Type theory provides a formal foundation for reasoning about program "
                "correctness. Simply typed lambda calculus extends untyped lambda calculus "
                "with type annotations on function abstractions. The Curry-Howard "
                "correspondence reveals a deep isomorphisms between types and logical "
                "propositions: a program of type A -> B corresponds to a proof of "
                "implication A implies B. Dependent types, as found in languages like "
                "Idris and Agda, allow types to depend on values, enabling the encoding "
                "of precise program specifications. Algebraic data types, comprising "
                "product types (structs/tuples) and sum types (enums), provide a "
                "structured way to model data."
            )

        def t10() -> str:
            """Concurrency models."""
            return (
                "Concurrency models determine how different parts of a program execute "
                "independently while coordinating access to shared state. The shared-memory "
                "model uses mutexes and condition variables to protect critical sections. "
                "The actor model, popularized by Erlang, treats each concurrent entity as "
                "an actor that communicates exclusively through message passing. "
                "Communicating sequential processes (CSP), as implemented in Go with channels, "
                "combines the simplicity of message passing with first-class synchronization. "
                "Software transactional memory (STM) applies database-style transactions to "
                "memory operations, allowing optimistic concurrent execution with rollback "
                "on conflicts. Each model makes different trade-offs between performance, "
                "correctness guarantees, and programmer ergonomics."
            )

        def t11() -> str:
            """Cache-friendly data structures."""
            return (
                "Cache-friendly data structures minimize cache misses by laying out data "
                "in a way that exploits spatial and temporal locality. Arrays are the most "
                "cache-friendly data structure because elements are contiguous in memory. "
                "Linked lists, by contrast, scatter nodes across memory, causing pointer "
                "chasing and frequent cache misses. A cache-oblivious data structure "
                "recursively partitions data to achieve good cache behavior at any cache "
                "level. The B-tree is a classic example: its high fanout produces shallow "
                "trees that require few traversals. For hash tables, open addressing with "
                "linear probing outperforms separate chaining because probing accesses "
                "consecutive memory locations."
            )

        def t12() -> str:
            """Numerical stability."""
            return (
                "Numerical stability is essential for training deep learning models reliably. "
                "Float16 (half precision) has limited range and precision compared to float32, "
                "which can cause underflow during gradient computation. Mixed-precision training "
                "addresses this by maintaining a float32 master copy of weights while performing "
                "forward/backward passes in float16, using loss scaling to prevent underflow. "
                "Kahan summation reduces floating-point error by tracking a running compensation "
                "term. In softmax computations, subtracting the maximum logit before exponentiation "
                "prevents overflow. Layer normalization and batch normalization inherently improve "
                "numerical conditioning by keeping activations in a stable range."
            )

        def t13() -> str:
            """Network protocol design."""
            return (
                "Network protocol design involves defining message formats, state machines, "
                "and error-handling strategies for communication between distributed components. "
                "Binary protocols like Protocol Buffers and FlatBuffers offer compact encoding "
                "and fast parsing compared to text-based formats. gRPC uses HTTP/2 as its "
                "transport, providing multiplexed streams, flow control, and header compression. "
                "For request-response patterns, a simple framing protocol with a length prefix "
                "and message type byte is often sufficient. Idempotency keys and retry budgets "
                "are important for handling transient failures in distributed systems. "
                "TLS mutual authentication provides transport-layer security, while application-level "
                "signatures add defense in depth."
            )

        def t14() -> str:
            """Compilation phases."""
            return (
                "A modern compiler transforms source code through several phases. "
                "The front end handles lexical analysis (tokenization), syntax analysis "
                "(parsing into an abstract syntax tree), and semantic analysis (type "
                "checking and name resolution). The intermediate representation (IR) "
                "bridges the front end and back end: LLVM IR, for example, is a "
                "low-level language with explicit control flow and type information. "
                "Optimization passes operate on the IR, performing constant folding, "
                "dead code elimination, loop unrolling, and inlining. The back end "
                "lowers the optimized IR to target machine code through instruction "
                "selection, register allocation, and instruction scheduling. "
                "Just-in-time (JIT) compilers combine interpretation with dynamic "
                "compilation for adaptive optimization."
            )

        def t15() -> str:
            """Database indexing."""
            return (
                "Database indexing accelerates query performance by reducing the number "
                "of disk blocks that must be examined. B+ trees are the most common index "
                "structure in relational databases, offering O(log n) search, insert, and "
                "delete operations while maintaining sorted order. Hash indexes provide "
                "O(1) lookups for equality predicates but do not support range queries. "
                "Bitmap indexes are efficient for columns with low cardinality. Covering "
                "indexes include all columns needed by a query, allowing the database to "
                "answer the query entirely from the index without touching the table. "
                "The query optimizer chooses between index scans and full table scans based "
                "on selectivity estimates and I/O cost models."
            )

        def t16() -> str:
            """Probability distributions for ML."""
            return (
                "Understanding probability distributions is fundamental to machine learning. "
                "The normal (Gaussian) distribution N(mu, sigma^2) is ubiquitous due to the "
                "central limit theorem and its convenient analytic properties. The Bernoulli "
                "distribution models binary outcomes and is the basis for logistic regression. "
                "The categorical distribution generalizes Bernoulli to multiple classes. "
                "The Dirichlet distribution is the conjugate prior for categorical "
                "distributions in Bayesian statistics. For modeling counts, the Poisson "
                "distribution is the natural choice. The exponential family encompasses "
                "many common distributions and enables elegant generalization in generalized "
                "linear models. Maximum likelihood estimation provides a principled way to "
                "fit distribution parameters to data."
            )

        def t17() -> str:
            """Functional programming principles."""
            return (
                "Functional programming emphasizes pure functions, immutability, and "
                "declarative code. Pure functions have no side effects and always produce "
                "the same output for the same input, making them referentially transparent "
                "and easy to test. Higher-order functions take functions as arguments or "
                "return them as results, enabling powerful abstraction patterns like map, "
                "filter, and reduce. Immutability prevents accidental mutation and simplifies "
                "concurrent programming. Algebraic data types model data as combinations of "
                "product and sum types. Pattern matching provides a concise way to destructure "
                "and dispatch on data. Monads abstract computation context such as optionality "
                "(Maybe), error handling (Either), or asynchronous execution (IO)."
            )

        def t18() -> str:
            """Container orchestration basics."""
            return (
                "Container orchestration platforms manage the lifecycle of containerized "
                "applications across a cluster of machines. Kubernetes, the dominant "
                "orchestrator, uses a declarative model where users specify desired state "
                "and the control plane reconciles actual state. Pods are the smallest "
                "deployable unit, encapsulating one or more containers with shared storage "
                "and networking. Deployments manage replica sets and support rolling updates "
                "with zero downtime. Services provide stable networking endpoints via label "
                "selectors. Horizontal Pod Autoscaling adjusts replica counts based on CPU, "
                "memory, or custom metrics. The scheduler places pods onto nodes based on "
                "resource requirements, affinity rules, and taints/tolerations."
            )

        def t19() -> str:
            """Cryptographic hash functions."""
            return (
                "Cryptographic hash functions map arbitrary-length input to a fixed-length "
                "output with three essential properties: preimage resistance (given a hash, "
                "finding an input that produces it is infeasible), second-preimage resistance "
                "(given an input, finding a different input with the same hash is infeasible), "
                "and collision resistance (finding any two inputs with the same hash is "
                "infeasible). SHA-256, part of the SHA-2 family, produces a 256-bit digest "
                "and is widely used in blockchain technology and digital signatures. "
                "SHA-3, based on the Keccak sponge construction, offers a different internal "
                "structure as a backup. BLAKE3 is a modern alternative that is substantially "
                "faster than SHA-256 while maintaining a high security margin."
            )

        def t20() -> str:
            """Static analysis techniques."""
            return (
                "Static analysis examines program source code without executing it to find "
                "bugs, enforce style, and prove correctness. Linting tools like Ruff and ESLint "
                "use pattern matching and abstract syntax tree traversal to detect common "
                "code smells. Type checking verifies that operations are applied to "
                "compatible types, preventing type errors at runtime. Data-flow analysis "
                "tracks how values propagate through a program to detect issues like "
                "uninitialized variables or unreachable code. Abstract interpretation "
                "approximates program behavior by mapping concrete values to an abstract "
                "domain. Formal verification tools like Dafny or Coq go further, allowing "
                "\"programmers to specify preconditions, postconditions, and invariants "
                "that are mechanically verified."
            )

        def t21() -> str:
            """Gradient descent variants."""
            return (
                "Gradient descent minimizes an objective function by iteratively moving "
                "parameters in the direction of steepest descent. Vanilla gradient descent "
                "uses the entire dataset per step, which is expensive for large datasets. "
                "Stochastic gradient descent (SGD) samples a single example per step, "
                "introducing noise that can help escape local minima. Mini-batch SGD "
                "strikes a balance, using a small random batch per step. Momentum "
                "accelerates convergence by accumulating a velocity vector in the "
                "direction of persistent gradient. Adam combines momentum with per-parameter "
                "adaptive learning rates computed from the first and second moments of "
                "gradients. Learning rate schedules (cosine decay, warmup, step decay) "
                "further improve convergence quality."
            )

        def t22() -> str:
            """System design: key-value store."""
            return (
                "Building a distributed key-value store requires partitioning data across "
                "nodes, replicating for fault tolerance, and maintaining consistency. "
                "Consistent hashing distributes keys across nodes with minimal rehashing "
                "when the cluster size changes. Each key maps to a position on a hash ring, "
                "and the nearest clockwise node stores that key. Replication factor r means "
                "the next r nodes on the ring also hold copies. For reads and writes, "
                "quorum consistency ensures that a majority of replicas agree before "
                "acknowledging. Hinted handoff allows nodes that are temporarily down to "
                "receive missed writes when they recover. Anti-entropy via Merkle trees "
                "detects and repairs silent data corruption across replicas."
            )

        return [t01, t02, t03, t04, t05, t06, t07, t08, t09, t10,
                t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
                t21, t22]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_python_doc(self) -> str:
        """Return a synthetic Python source document."""
        template = self._rng.choice(self._python_templates)
        return template()

    def generate_js_doc(self) -> str:
        """Return a synthetic JavaScript / TypeScript source document."""
        template = self._rng.choice(self._js_templates)
        return template()

    def generate_go_doc(self) -> str:
        """Return a synthetic Go source document."""
        template = self._rng.choice(self._go_templates)
        return template()

    def generate_rust_doc(self) -> str:
        """Return a synthetic Rust source document."""
        template = self._rng.choice(self._rust_templates)
        return template()

    def generate_text_doc(self) -> str:
        """Return a synthetic technical blog post or documentation text."""
        template = self._rng.choice(self._text_templates)
        return template()

    def generate_document(self, language: str) -> str:
        """Dispatch to the appropriate generator by language name.

        Parameters
        ----------
        language : str
            One of ``"python"``, ``"javascript"``, ``"typescript"``,
            ``"js"``, ``"go"``, ``"rust"``, or ``"text"`` (or any alias).

        Returns
        -------
        str
            The generated document text.
        """
        lang_map = {
            "python": self.generate_python_doc,
            "javascript": self.generate_js_doc,
            "typescript": self.generate_js_doc,
            "js": self.generate_js_doc,
            "go": self.generate_go_doc,
            "rust": self.generate_rust_doc,
            "text": self.generate_text_doc,
            "english_text": self.generate_text_doc,
        }
        generator = lang_map.get(language)
        if generator is None:
            msg = f"Unknown language: {language!r}.  Known: {list(lang_map)}"
            raise ValueError(msg)
        return generator()

    # ------------------------------------------------------------------
    # Run: generate → tokenize → shard
    # ------------------------------------------------------------------

    def run(self, num_samples: int, output_dir: str | Path) -> None:
        """Generate *num_samples* synthetic documents, tokenize them, and
        save uint16 .npy shards to ``output_dir/pretrain/``.

        Parameters
        ----------
        num_samples : int
            Number of documents to generate.
        output_dir : str or Path
            Root output directory; shards go under ``<dir>/pretrain/``.
        """
        output_dir = Path(output_dir)
        shard_dir = output_dir / "pretrain"
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Language weights match the default config.yaml distribution.
        languages = ["python", "javascript", "go", "rust", "text"]
        weights = [0.30, 0.25, 0.10, 0.10, 0.25]

        # -- tokenizer -------------------------------------------------------
        tokenizer_dir = self.config.get("tokenizer", {}).get("path", "checkpoints/tokenizer")
        try:
            tokenizer = AureliusTokenizer.load(tokenizer_dir)
        except (FileNotFoundError, OSError):
            logger.warning(
                "Tokenizer not found at %s. Using identity tokenizer (single token per char).",
                tokenizer_dir,
            )
            tokenizer = None  # will use identity fallback below

        # -- generate & tokenize ---------------------------------------------
        all_ids: list[int] = []
        manifest: list[dict[str, Any]] = []
        doc_index = 0
        shard_index = 0

        iterator: range = range(num_samples)
        if _tqdm is not None:
            iterator = _tqdm(iterator, desc="Generating      ", unit="doc")

        for _ in iterator:
            lang = self._rng.choices(languages, weights=weights, k=1)[0]
            doc = self.generate_document(lang)

            if tokenizer is not None:
                ids = tokenizer.encode(doc, add_eos=True)
            else:
                # identity fallback
                ids = [ord(c) for c in doc] + [0]

            all_ids.extend(ids)

            # flush completed shards
            while len(all_ids) >= self._shard_size:
                shard = all_ids[: self._shard_size]
                all_ids = all_ids[self._shard_size :]

                shard_path = shard_dir / f"shard_{shard_index:06d}.npy"
                np.save(str(shard_path), np.array(shard, dtype=np.uint16))

                doc_end = doc_index
                token_end = (shard_index + 1) * self._shard_size
                manifest.append({
                    "shard": f"shard_{shard_index:06d}.npy",
                    "path": str(shard_path.relative_to(output_dir)),
                    "tokens": self._shard_size,
                    "doc_start": doc_index,
                    "doc_end": doc_end,
                    "token_start": shard_index * self._shard_size,
                    "token_end": token_end,
                })

                tokenizer_dir_str = str(tokenizer_dir)
                label = f"Shard {shard_index:06d}"
                if _tqdm is not None:
                    iterator.set_description(f"Writing {label:24s}")
                shard_index += 1
                doc_index = doc_end

        # -- write any remaining tokens as a partial shard -------------------
        if all_ids:
            remaining = self._shard_size - len(all_ids)
            all_ids.extend([0] * remaining)  # pad with null id
            shard_path = shard_dir / f"shard_{shard_index:06d}.npy"
            np.save(str(shard_path), np.array(all_ids, dtype=np.uint16))
            manifest.append({
                "shard": f"shard_{shard_index:06d}.npy",
                "path": str(shard_path.relative_to(output_dir)),
                "tokens": len(all_ids),
                "padded": remaining,
                "doc_start": doc_index,
                "doc_end": doc_index,
                "token_start": shard_index * self._shard_size,
                "token_end": shard_index * self._shard_size + len(all_ids),
            })
            shard_index += 1

        # -- save manifest ---------------------------------------------------
        manifest_path = shard_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({
                "num_samples": num_samples,
                "shard_size": self._shard_size,
                "num_shards": shard_index,
                "shards": manifest,
            }, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Generated %d documents → %d shards in %s",
            num_samples,
            shard_index,
            shard_dir,
        )
