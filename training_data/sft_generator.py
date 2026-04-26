import json
import random
from pathlib import Path
from typing import Any, Dict, List, Union


class SFTDataGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.seed = config.get("seed", 42)
        self.rng = random.Random(self.seed)
        self._init_templates()

    def _init_templates(self):
        self._init_coding()
        self._init_reasoning()
        self._init_writing()
        self._init_analysis()
        self._init_general()
        self.template_map = {
            "coding": self.coding_templates,
            "reasoning": self.reasoning_templates,
            "writing": self.writing_templates,
            "analysis": self.analysis_templates,
            "general": self.general_templates,
        }

    def _build_conversation(
        self, category: str, target_turns: int
    ) -> List[Dict[str, str]]:
        templates = self.template_map[category]
        human_prompt, gpt_response, follow_ups = self.rng.choice(templates)
        turns = [
            {"from": "human", "value": human_prompt},
            {"from": "gpt", "value": gpt_response},
        ]
        for i in range(1, target_turns):
            if not follow_ups:
                break
            if i - 1 < len(follow_ups):
                f_human, f_gpt = follow_ups[i - 1]
            else:
                f_human, f_gpt = self.rng.choice(follow_ups)
            turns.append({"from": "human", "value": f_human})
            turns.append({"from": "gpt", "value": f_gpt})
        return turns

    def generate_conversation(self, category: str) -> List[Dict[str, str]]:
        target = self.rng.randint(1, 5)
        return self._build_conversation(category, target)

    def generate_single_turn(self, category: str) -> Dict[str, Any]:
        return {"conversations": self._build_conversation(category, 1)}

    def generate_multi_turn(self, category: str, num_turns: int) -> Dict[str, Any]:
        return {"conversations": self._build_conversation(category, num_turns)}

    def run(self, num_samples: int, output_dir: Union[str, Path]) -> None:
        output_dir = Path(output_dir)
        sft_dir = output_dir / "sft"
        sft_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        categories = list(self.template_map.keys())
        for _ in range(num_samples):
            cat = self.rng.choice(categories)
            target = self.rng.randint(1, 5)
            conv = self._build_conversation(cat, target)
            samples.append({"conversations": conv})
        self.rng.shuffle(samples)
        split_idx = int(num_samples * 0.9)
        train_data = samples[:split_idx]
        val_data = samples[split_idx:]
        for name, data in [("train", train_data), ("val", val_data)]:
            path = sft_dir / f"{name}.jsonl"
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        metadata = {
            "num_samples": num_samples,
            "num_train": len(train_data),
            "num_val": len(val_data),
            "split_ratio": "90/10",
            "categories": list(self.template_map.keys()),
            "seed": self.seed,
            "max_turns_per_conversation": 5,
        }
        with open(sft_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _init_coding(self):
        self.coding_templates = [
            (
                "Write a Python function that merges overlapping intervals from a list of tuples. "
                "For example, [(1,3),(2,6),(8,10),(15,18)] should become [(1,6),(8,10),(15,18)]. "
                "Include edge case handling for empty input.",
                (
                    "Here's an implementation with proper edge case handling:\n\n"
                    "```python\n"
                    "from typing import List, Tuple\n\n"
                    "def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:\n"
                    '    """Merge overlapping intervals.\n\n'
                    "    Args:\n"
                    "        intervals: List of (start, end) tuples.\n\n"
                    "    Returns:\n"
                    "        Merged non-overlapping intervals sorted by start.\n"
                    '    """\n'
                    "    if not intervals:\n"
                    "        return []\n"
                    "    sorted_ivals = sorted(intervals, key=lambda x: x[0])\n"
                    "    merged = [sorted_ivals[0]]\n"
                    "    for start, end in sorted_ivals[1:]:\n"
                    "        prev_start, prev_end = merged[-1]\n"
                    "        if start <= prev_end:\n"
                    "            merged[-1] = (prev_start, max(prev_end, end))\n"
                    "        else:\n"
                    "            merged.append((start, end))\n"
                    "    return merged\n"
                    "```\n\n"
                    "This sorts by start time (O(n log n)) then does a single linear pass."
                ),
                [
                    (
                        "Can you modify it to also track the index of original intervals that "
                        "contributed to each merged interval?",
                        (
                            "Sure. Here's the modified version:\n\n"
                            "```python\n"
                            "from typing import List, Tuple\n\n"
                            "def merge_intervals_with_trace(\n"
                            "    intervals: List[Tuple[int, int]]\n"
                            ") -> List[Tuple[Tuple[int, int], List[int]]]:\n"
                            '    """Merge intervals and trace original indices.\n\n'
                            "    if not intervals:\n"
                            "        return []\n"
                            "    indexed = [(s, e, i) for i, (s, e) in enumerate(intervals)]\n"
                            "    indexed.sort(key=lambda x: x[0])\n"
                            "    merged = [(indexed[0][0], indexed[0][1], [indexed[0][2]])]\n"
                            "    for start, end, idx in indexed[1:]:\n"
                            "        _, prev_end, trace = merged[-1]\n"
                            "        if start <= prev_end:\n"
                            "            merged[-1] = (\n"
                            "                merged[-1][0], max(prev_end, end), trace + [idx]\n"
                            "            )\n"
                            "        else:\n"
                            "            merged.append((start, end, [idx]))\n"
                            "    return [(s, e) for s, e, _ in merged], [t for _, _, t in merged]\n"
                            "```\n\n"
                            "This wraps tuples with their original index before sorting."
                        ),
                    ),
                ],
            ),
            (
                "Implement a Least Recently Used (LRU) cache class in Python with get() and put() "
                "methods. It should have a configurable capacity and use O(1) operations.",
                (
                    "Here's an LRU cache using an OrderedDict:\n\n"
                    "```python\n"
                    "from collections import OrderedDict\n\n"
                    "class LRUCache:\n"
                    "    def __init__(self, capacity: int):\n"
                    "        self.capacity = capacity\n"
                    "        self.cache = OrderedDict()\n\n"
                    "    def get(self, key: int) -> int:\n"
                    "        if key not in self.cache:\n"
                    "            return -1\n"
                    "        self.cache.move_to_end(key)\n"
                    "        return self.cache[key]\n\n"
                    "    def put(self, key: int, value: int) -> None:\n"
                    "        if key in self.cache:\n"
                    "            self.cache.move_to_end(key)\n"
                    "        self.cache[key] = value\n"
                    "        if len(self.cache) > self.capacity:\n"
                    "            self.cache.popitem(last=False)\n"
                    "```\n\n"
                    "Using OrderedDict.move_to_end() and popitem(last=False) gives O(1) operations."
                ),
                [
                    (
                        "How would you implement this without OrderedDict, using just a dict and a doubly linked list?",
                        (
                            "Here's the manual implementation:\n\n"
                            "```python\n"
                            "class _Node:\n"
                            "    def __init__(self, key=0, val=0):\n"
                            "        self.key = key\n"
                            "        self.val = val\n"
                            "        self.prev = None\n"
                            "        self.next = None\n\n"
                            "class LRUCache:\n"
                            "    def __init__(self, capacity: int):\n"
                            "        self.cap = capacity\n"
                            "        self.map = {}\n"
                            "        self.head = _Node()\n"
                            "        self.tail = _Node()\n"
                            "        self.head.next = self.tail\n"
                            "        self.tail.prev = self.head\n\n"
                            "    def _remove(self, node: _Node) -> None:\n"
                            "        node.prev.next = node.next\n"
                            "        node.next.prev = node.prev\n\n"
                            "    def _add(self, node: _Node) -> None:\n"
                            "        node.prev = self.tail.prev\n"
                            "        node.next = self.tail\n"
                            "        self.tail.prev.next = node\n"
                            "        self.tail.prev = node\n\n"
                            "    def get(self, key: int) -> int:\n"
                            "        if key not in self.map:\n"
                            "            return -1\n"
                            "        node = self.map[key]\n"
                            "        self._remove(node)\n"
                            "        self._add(node)\n"
                            "        return node.val\n\n"
                            "    def put(self, key: int, value: int) -> None:\n"
                            "        if key in self.map:\n"
                            "            self._remove(self.map[key])\n"
                            "        node = _Node(key, value)\n"
                            "        self.map[key] = node\n"
                            "        self._add(node)\n"
                            "        if len(self.map) > self.cap:\n"
                            "            lru = self.head.next\n"
                            "            self._remove(lru)\n"
                            "            del self.map[lru.key]\n"
                            "```\n\n"
                            "The dict gives O(1) key lookup, while the linked list maintains access order."
                        ),
                    ),
                ],
            ),
            (
                "Write a function that finds the k-th largest element in an unsorted array "
                "without fully sorting it. Try for O(n) average time.",
                (
                    "Using Quickselect:\n\n"
                    "```python\n"
                    "import random as _random\n\n"
                    "def find_kth_largest(nums: list[int], k: int) -> int:\n"
                    '    """Return the k-th largest element (1-indexed)."""\n'
                    "    def _partition(arr: list[int], left: int, right: int) -> int:\n"
                    "        pivot_idx = _random.randint(left, right)\n"
                    "        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]\n"
                    "        pivot = arr[right]\n"
                    "        store = left\n"
                    "        for i in range(left, right):\n"
                    "            if arr[i] > pivot:\n"
                    "                arr[store], arr[i] = arr[i], arr[store]\n"
                    "                store += 1\n"
                    "        arr[store], arr[right] = arr[right], arr[store]\n"
                    "        return store\n\n"
                    "    left, right = 0, len(nums) - 1\n"
                    "    target = k - 1\n"
                    "    while left <= right:\n"
                    "        pivot_pos = _partition(nums, left, right)\n"
                    "        if pivot_pos == target:\n"
                    "            return nums[pivot_pos]\n"
                    "        elif pivot_pos < target:\n"
                    "            left = pivot_pos + 1\n"
                    "        else:\n"
                    "            right = pivot_pos - 1\n"
                    "    return -1\n"
                    "```\n\n"
                    "Quickselect partitions around a random pivot. Average case is O(n)."
                ),
                [
                    (
                        "What's the worst-case time complexity and how would you guarantee O(n log n)?",
                        (
                            "The worst case is O(n^2), which happens when each pivot is the largest or smallest. "
                            "Use Median of Medians for guaranteed O(n):\n\n"
                            "```python\n"
                            "def _median_of_medians(arr: list[int], left: int, right: int) -> int:\n"
                            "    if right - left < 5:\n"
                            "        return sorted(arr[left:right+1])[(right-left)//2]\n"
                            "    medians = []\n"
                            "    for i in range(left, right + 1, 5):\n"
                            "        chunk = sorted(arr[i:min(i+5, right+1)])\n"
                            "        medians.append(chunk[len(chunk)//2])\n"
                            "    return _median_of_medians(medians, 0, len(medians)-1)\n"
                            "```\n\n"
                            "Groups of 5 guarantee at least 30% of elements are on each side of the pivot."
                        ),
                    ),
                ],
            ),
            (
                "Write a Python decorator that retries a function up to N times with exponential "
                "backoff when it raises a specific exception.",
                (
                    "```python\n"
                    "import time\n"
                    "import functools\n\n"
                    "def retry(max_attempts: int = 3, base_delay: float = 1.0,\n"
                    "          backoff: float = 2.0, exceptions: tuple = (Exception,)):\n"
                    '    """Retry decorator with exponential backoff."""\n'
                    "    def decorator(func):\n"
                    "        @functools.wraps(func)\n"
                    "        def wrapper(*args, **kwargs):\n"
                    "            last_exc = None\n"
                    "            for attempt in range(1, max_attempts + 1):\n"
                    "                try:\n"
                    "                    return func(*args, **kwargs)\n"
                    "                except exceptions as e:\n"
                    "                    last_exc = e\n"
                    "                    if attempt == max_attempts:\n"
                    "                        raise\n"
                    "                    delay = base_delay * (backoff ** (attempt - 1))\n"
                    "                    time.sleep(delay)\n"
                    "            raise last_exc\n"
                    "        return wrapper\n"
                    "    return decorator\n"
                    "```\n\n"
                    "The decorator sleeps `base_delay * backoff^(attempt-1)` before retrying."
                ),
                [
                    (
                        "Add jitter to the backoff so not all retries happen simultaneously "
                        "under high concurrency.",
                        (
                            "Adding random jitter prevents the thundering herd problem:\n\n"
                            "```python\n"
                            "import random as _random\n\n"
                            "def retry_with_jitter(max_attempts: int = 3, base_delay: float = 1.0,\n"
                            "                     backoff: float = 2.0, max_jitter: float = 0.5,\n"
                            "                     exceptions: tuple = (Exception,)):\n"
                            "    def decorator(func):\n"
                            "        @functools.wraps(func)\n"
                            "        def wrapper(*args, **kwargs):\n"
                            "            ...\n"
                            "                    delay = base_delay * (backoff ** (attempt - 1))\n"
                            "                    jitter = _random.uniform(0, max_jitter)\n"
                            "                    time.sleep(delay + jitter)\n"
                            "        return wrapper\n"
                            "    return decorator\n"
                            "```\n\n"
                            "This dramatically reduces retry storms in distributed systems."
                        ),
                    ),
                ],
            ),
            (
                "Review this function and tell me what's wrong and how to fix it:\n"
                "```python\ndef get_user_data(user_id, db):\n"
                '    query = f"SELECT * FROM users WHERE id = {user_id}"\n'
                "    result = db.execute(query)\n"
                "    return result.fetchone()\n"
                "```",
                (
                    "Several issues:\n\n"
                    "1. **SQL injection**: f-string interpolates user_id directly. Use parameterized queries.\n"
                    "2. **No error handling**: Wrap in try/except.\n"
                    "3. **fetchone may be None**: Check before use.\n"
                    "4. **Missing type hints**.\n\n"
                    "Corrected version:\n\n"
                    "```python\n"
                    "from typing import Optional, Dict, Any\n\n"
                    "def get_user_data(user_id: int, db) -> Optional[Dict[str, Any]]:\n"
                    "    try:\n"
                    '        query = "SELECT * FROM users WHERE id = %s"\n'
                    "        result = db.execute(query, (user_id,))\n"
                    "        row = result.fetchone()\n"
                    "        if row is None:\n"
                    "            return None\n"
                    "        return dict(row)\n"
                    "    except Exception as e:\n"
                    "        logger.error(f\"Failed to fetch user {user_id}: {e}\")\n"
                    "        raise\n"
                    "```\n\n"
                    "Parameterized queries prevent injection by escaping values separately."
                ),
                [
                    (
                        "What if I need to dynamically choose which columns to select? "
                        "How do I stay safe while being flexible?",
                        (
                            "Column names can't be parameterized, so whitelist them:\n\n"
                            "```python\n"
                            "from typing import Optional, Dict, Any, List\n\n"
                            "ALLOWED_COLUMNS = frozenset({\n"
                            '    "id", "name", "email", "created_at", "role"\n'
                            "})\n\n"
                            "def get_user_data(\n"
                            "    user_id: int, columns: List[str], db\n"
                            ") -> Optional[Dict[str, Any]]:\n"
                            "    invalid = set(columns) - ALLOWED_COLUMNS\n"
                            "    if invalid:\n"
                            '        raise ValueError(f"Invalid columns: {invalid}")\n'
                            '    cols = ", ".join(columns)\n'
                            "    try:\n"
                            '        query = f"SELECT {cols} FROM users WHERE id = %s"\n'
                            "        result = db.execute(query, (user_id,))\n"
                            "        row = result.fetchone()\n"
                            "        return dict(row) if row else None\n"
                            "    except Exception as e:\n"
                            "        logger.error(f\"Failed to fetch user {user_id}: {e}\")\n"
                            "        raise\n"
                            "```\n\n"
                            "The frozenset acts as a strict allowlist against injection."
                        ),
                    ),
                ],
            ),
            (
                "Write a context manager in Python that measures the execution time of a code block "
                "and logs it. Use the contextlib module if possible.",
                (
                    "```python\n"
                    "import time\n"
                    "import logging\n"
                    "from contextlib import contextmanager\n\n"
                    "@contextmanager\n"
                    'def timed(label: str = "block"):\n'
                    '    """Context manager that logs execution time."""\n'
                    "    start = time.perf_counter()\n"
                    "    try:\n"
                    "        yield\n"
                    "    finally:\n"
                    "        elapsed = time.perf_counter() - start\n"
                    "        logging.info(f\"{label} took {elapsed*1000:.2f}ms\")\n"
                    "```\n\n"
                    "time.perf_counter() gives the highest-resolution timer available."
                ),
                [
                    (
                        "Can you make it also track peak memory usage for profiling?",
                        (
                            "Combine with tracemalloc:\n\n"
                            "```python\n"
                            "import tracemalloc\n\n"
                            "@contextmanager\n"
                            'def profile(label: str = "block"):\n'
                            "    tracemalloc.start()\n"
                            "    start = time.perf_counter()\n"
                            "    start_snapshot = tracemalloc.take_snapshot()\n"
                            "    try:\n"
                            "        yield\n"
                            "    finally:\n"
                            "        elapsed = time.perf_counter() - start\n"
                            "        end_snapshot = tracemalloc.take_snapshot()\n"
                            "        tracemalloc.stop()\n"
                            "        stats = end_snapshot.compare_to(start_snapshot, \"lineno\")\n"
                            "        total_diff = sum(s.size_diff for s in stats)\n"
                            "        logging.info(f\"{label}: {elapsed*1000:.2f}ms, mem: {total_diff/1024:.2f}KB\")\n"
                            "```"
                        ),
                    ),
                ],
            ),
            (
                "Write a function that serializes a nested dictionary to a flat, dot-separated key format. "
                "E.g., `{'a': {'b': 1, 'c': 2}}` becomes `{'a.b': 1, 'a.c': 2}`.",
                (
                    "```python\n"
                    "from typing import Any, Dict\n\n"
                    "def flatten_dict(d: Dict[str, Any], parent_key: str = \"\",\n"
                    '                 sep: str = ".") -> Dict[str, Any]:\n'
                    '    """Flatten a nested dict into dot-separated keys."""\n'
                    "    items: Dict[str, Any] = {}\n"
                    "    for k, v in d.items():\n"
                    '        new_key = f"{parent_key}{sep}{k}" if parent_key else k\n'
                    "        if isinstance(v, dict):\n"
                    "            items.update(flatten_dict(v, new_key, sep=sep))\n"
                    "        elif isinstance(v, list):\n"
                    "            for i, elem in enumerate(v):\n"
                    '                ek = f"{new_key}[{i]"\n'
                    "                if isinstance(elem, dict):\n"
                    "                    items.update(flatten_dict(elem, ek, sep=sep))\n"
                    "                else:\n"
                    "                    items[ek] = elem\n"
                    "        else:\n"
                    "            items[new_key] = v\n"
                    "    return items\n"
                    "```\n\n"
                    "The function recurses into nested dicts. Lists get indexed brackets."
                ),
                [],
            ),
            (
                "Implement a simple Pub/Sub event bus in Python with subscribe, unsubscribe, "
                "and emit methods.",
                (
                    "```python\n"
                    "import asyncio\n"
                    "from typing import Any, Callable, Coroutine, Dict, List, Union\n\n"
                    "Handler = Union[\n"
                    "    Callable[..., None],\n"
                    "    Callable[..., Coroutine[Any, Any, None]],\n"
                    "]\n\n"
                    "class EventBus:\n"
                    "    def __init__(self):\n"
                    "        self._handlers: Dict[str, List[Handler]] = {}\n\n"
                    "    def subscribe(self, event: str, handler: Handler) -> None:\n"
                    "        if event not in self._handlers:\n"
                    "            self._handlers[event] = []\n"
                    "        self._handlers[event].append(handler)\n\n"
                    "    def unsubscribe(self, event: str, handler: Handler) -> None:\n"
                    "        if event in self._handlers:\n"
                    "            self._handlers[event] = [\n"
                    "                h for h in self._handlers[event] if h is not handler\n"
                    "            ]\n\n"
                    "    def emit(self, event: str, *args, **kwargs) -> None:\n"
                    "        for handler in self._handlers.get(event, []):\n"
                    "            handler(*args, **kwargs)\n\n"
                    "    async def aemit(self, event: str, *args, **kwargs) -> None:\n"
                    "        for handler in self._handlers.get(event, []):\n"
                    "            res = handler(*args, **kwargs)\n"
                    "            if asyncio.iscoroutine(res):\n"
                    "                await res\n"
                    "```\n\n"
                    "emit runs sync handlers; aemit awaits async ones."
                ),
                [
                    (
                        "Add wildcard event matching so 'user.*' catches 'user.created'.",
                        (
                            "```python\n"
                            "import fnmatch\n\n"
                            "class WildcardEventBus(EventBus):\n"
                            "    def _matching_events(self, event: str) -> List[str]:\n"
                            "        return [e for e in self._handlers if fnmatch.fnmatch(event, e)]\n\n"
                            "    def emit(self, event: str, *args, **kwargs) -> None:\n"
                            "        for evt in self._matching_events(event) + [event]:\n"
                            "            for handler in self._handlers.get(evt, []):\n"
                            "                handler(*args, **kwargs)\n"
                            "```"
                        ),
                    ),
                ],
            ),
            (
                "Write an async function that concurrently fetches multiple URLs with a "
                "limit of 5 concurrent connections.",
                (
                    "```python\n"
                    "import asyncio\n"
                    "from typing import Dict, Tuple\n\n"
                    "async def fetch_url(session, url: str, semaphore: asyncio.Semaphore):\n"
                    '    """Fetch a single URL with semaphore-based concurrency limit."""\n'
                    "    async with semaphore:\n"
                    "        try:\n"
                    "            async with session.get(url, timeout=10) as resp:\n"
                    "                content = await resp.read()\n"
                    "                return url, resp.status, len(content)\n"
                    "        except Exception:\n"
                    "            return url, 0, 0\n\n"
                    "async def fetch_all(urls: list[str]) -> Dict[str, Tuple[int, int]]:\n"
                    "    semaphore = asyncio.Semaphore(5)\n"
                    "    async with aiohttp.ClientSession() as session:\n"
                    "        tasks = [fetch_url(session, url, semaphore) for url in urls]\n"
                    "        results = await asyncio.gather(*tasks)\n"
                    "    return {url: (status, length) for url, status, length in results}\n"
                    "```\n\n"
                    "Semaphore caps concurrent requests; gather runs all tasks concurrently."
                ),
                [
                    (
                        "What if I want to handle redirects and respect robots.txt?",
                        (
                            "```python\n"
                            "from urllib.robotparser import RobotFileParser\n\n"
                            "class PoliteFetcher:\n"
                            '    def __init__(self, user_agent: str = "Mozilla/5.0"):\n'
                            "        self.user_agent = user_agent\n"
                            "        self._robots_cache = {}\n\n"
                            "    async def fetch_url(self, session, url: str, semaphore):\n"
                            "        parsed = urlparse(url)\n"
                            '        base = f"{parsed.scheme}://{parsed.netloc}"\n'
                            "        if base not in self._robots_cache:\n"
                            "            rp = RobotFileParser()\n"
                            '            rp.set_url(f"{base}/robots.txt")\n'
                            "            await asyncio.to_thread(rp.read)\n"
                            "            self._robots_cache[base] = rp\n"
                            "        if not self._robots_cache[base].can_fetch(self.user_agent, url):\n"
                            "            return url, 403, 0\n"
                            "        ...\n"
                            "```\n\n"
                            "RobotFileParser reads robots.txt with domain-level caching."
                        ),
                    ),
                ],
            ),
            (
                "Write a memory-efficient Sieve of Eratosthenes using bit manipulation.",
                (
                    "```python\n"
                    "def sieve_of_eratosthenes(n: int) -> list[int]:\n"
                    '    """Return all primes up to n using a bit-array sieve."""\n'
                    "    if n < 2:\n"
                    "        return []\n"
                    "    size = (n + 1 + 7) // 8\n"
                    "    is_prime = bytearray(b'\\xff') * size\n"
                    "    is_prime[0] &= ~(1 << 0) & ~(1 << 1)\n\n"
                    "    def _set_bit(idx: int) -> None:\n"
                    "        is_prime[idx >> 3] &= ~(1 << (idx & 7))\n\n"
                    "    def _get_bit(idx: int) -> bool:\n"
                    "        return bool(is_prime[idx >> 3] & (1 << (idx & 7)))\n\n"
                    "    for i in range(2, int(n**0.5) + 1):\n"
                    "        if _get_bit(i):\n"
                    "            for j in range(i * i, n + 1, i):\n"
                    "                _set_bit(j)\n\n"
                    "    return [i for i in range(2, n + 1) if _get_bit(i)]\n"
                    "```\n\n"
                    "Each bit represents one number, reducing memory by 8x vs a bool list."
                ),
                [
                    (
                        "Can you parallelize this with multiprocessing?",
                        (
                            "```python\n"
                            "import math\n"
                            "from multiprocessing import Pool, shared_memory\n\n"
                            "def parallel_sieve(n: int, num_workers: int = 4) -> list[int]:\n"
                            "    limit = int(math.isqrt(n))\n"
                            "    base_primes = sieve_of_eratosthenes(limit)\n"
                            "    size = (n + 1 + 7) // 8\n"
                            "    shm = shared_memory.SharedMemory(create=True, size=size)\n\n"
                            "    segment_size = (n - limit) // num_workers + 1\n"
                            "    with Pool(num_workers) as pool:\n"
                            "        pool.starmap(_process_segment, [\n"
                            "            (low, min(low+segment_size-1, n))\n"
                            "            for low in range(limit+1, n+1, segment_size)\n"
                            "        ])\n"
                            "```\n\n"
                            "Base primes computed once, segments processed in parallel."
                        ),
                    ),
                ],
            ),
            (
                "Write a thread-safe singleton using a Python metaclass.",
                (
                    "```python\n"
                    "import threading\n\n"
                    "class ThreadSafeSingletonMeta(type):\n"
                    '    """Metaclass for thread-safe singleton."""\n'
                    "    _instances: dict = {}\n"
                    "    _lock: threading.Lock = threading.Lock()\n\n"
                    "    def __call__(cls, *args, **kwargs):\n"
                    "        if cls not in cls._instances:\n"
                    "            with cls._lock:\n"
                    "                if cls not in cls._instances:\n"
                    "                    instance = super().__call__(*args, **kwargs)\n"
                    "                    cls._instances[cls] = instance\n"
                    "        return cls._instances[cls]\n\n"
                    "class DatabasePool(metaclass=ThreadSafeSingletonMeta):\n"
                    "    def __init__(self):\n"
                    "        self.connections = []\n"
                    "```\n\n"
                    "Double-checked locking: fast path without lock, then lock and verify."
                ),
                [
                    (
                        "What if two calls have different keyword arguments?",
                        (
                            "Use a multiton pattern keyed by args:\n\n"
                            "```python\n"
                            "class MultitonMeta(type):\n"
                            "    _instances = {}\n"
                            "    _lock = threading.Lock()\n\n"
                            "    def __call__(cls, *args, **kwargs):\n"
                            "        key = (cls, args, tuple(sorted(kwargs.items())))\n"
                            "        if key not in cls._instances:\n"
                            "            with cls._lock:\n"
                            "                if key not in cls._instances:\n"
                            "                    instance = super().__call__(*args, **kwargs)\n"
                            "                    cls._instances[key] = instance\n"
                            "        return cls._instances[key]\n"
                            "```\n\n"
                            "One instance per unique set of constructor arguments."
                        ),
                    ),
                ],
            ),
            (
                "Write a priority queue without using heapq. Highest priority first, FIFO for ties.",
                (
                    "```python\n"
                    "from typing import Any, Optional, List, Tuple\n"
                    "import itertools\n\n"
                    "class PriorityQueue:\n"
                    "    def __init__(self):\n"
                    "        self._entries: List[Tuple[int, int, Any]] = []\n"
                    "        self._counter = itertools.count()\n\n"
                    "    def push(self, item: Any, priority: int) -> None:\n"
                    "        count = next(self._counter)\n"
                    "        entry = (-priority, count, item)\n"
                    "        self._entries.append(entry)\n"
                    "        self._sift_up(len(self._entries) - 1)\n\n"
                    "    def pop(self) -> Optional[Any]:\n"
                    "        if not self._entries:\n"
                    "            return None\n"
                    "        self._swap(0, len(self._entries) - 1)\n"
                    "        _, _, item = self._entries.pop()\n"
                    "        self._sift_down(0)\n"
                    "        return item\n\n"
                    "    def _sift_up(self, idx: int) -> None:\n"
                    "        parent = (idx - 1) // 2\n"
                    "        while idx > 0 and self._entries[idx][:2] > self._entries[parent][:2]:\n"
                    "            self._swap(idx, parent)\n"
                    "            idx = parent\n"
                    "            parent = (idx - 1) // 2\n\n"
                    "    def _sift_down(self, idx: int) -> None:\n"
                    "        n = len(self._entries)\n"
                    "        while True:\n"
                    "            largest = idx\n"
                    "            left = 2 * idx + 1\n"
                    "            right = 2 * idx + 2\n"
                    "            if left < n and self._entries[left][:2] > self._entries[largest][:2]:\n"
                    "                largest = left\n"
                    "            if right < n and self._entries[right][:2] > self._entries[largest][:2]:\n"
                    "                largest = right\n"
                    "            if largest == idx:\n"
                    "                break\n"
                    "            self._swap(idx, largest)\n"
                    "            idx = largest\n\n"
                    "    def _swap(self, i: int, j: int) -> None:\n"
                    "        self._entries[i], self._entries[j] = self._entries[j], self._entries[i]\n"
                    "```\n\n"
                    "Max-heap with tie-breaking counter for FIFO order."
                ),
                [
                    (
                        "Can you add priority update for Dijkstra's algorithm?",
                        (
                            "```python\n"
                            "class MutablePriorityQueue(PriorityQueue):\n"
                            "    def __init__(self):\n"
                            "        super().__init__()\n"
                            "        self._entry_map: dict = {}\n\n"
                            "    def push(self, item: Any, priority: int) -> None:\n"
                            "        count = next(self._counter)\n"
                            "        entry = (-priority, count, item)\n"
                            "        self._entry_map[item] = entry\n"
                            "        super().push(item, priority)\n\n"
                            "    def update_priority(self, item: Any, new_priority: int) -> None:\n"
                            "        if item in self._entry_map:\n"
                            "            old_entry = self._entry_map[item]\n"
                            "            self._entries[self._entries.index(old_entry)] = (0, -1, None)\n"
                            "        self.push(item, new_priority)\n"
                            "```\n\n"
                            "Uses lazy deletion: marks old entry invalid instead of removing it."
                        ),
                    ),
                ],
            ),
            (
                "Write a function that builds a tree-like dict from file paths.",
                (
                    "```python\n"
                    "from typing import Dict, List, Union\n\n"
                    "FileTree = Dict[str, Union[None, 'FileTree']]\n\n"
                    "def build_file_tree(paths: List[str]) -> FileTree:\n"
                    '    """Build a nested dict tree from flat file paths."""\n'
                    "    tree: FileTree = {}\n"
                    "    for path in sorted(paths):\n"
                    '        parts = path.split("/")\n'
                    "        current = tree\n"
                    "        for part in parts[:-1]:\n"
                    "            if part not in current:\n"
                    "                current[part] = {}\n"
                    "            current = current[part]\n"
                    "        current[parts[-1]] = None\n"
                    "    return tree\n"
                    "```\n\n"
                    "Directories become nested dicts; files become None leaves."
                ),
                [
                    (
                        "How would you handle absolute paths and normalize '.' and '..'?",
                        (
                            "```python\n"
                            "from pathlib import Path\n\n"
                            'def build_file_tree_normalized(paths, base="."):\n'
                            "    base_path = Path(base).resolve()\n"
                            "    tree = {}\n"
                            "    for path_str in paths:\n"
                            "        p = Path(path_str).resolve()\n"
                            "        try:\n"
                            "            rel = p.relative_to(base_path)\n"
                            "        except ValueError:\n"
                            "            continue\n"
                            "        ...\n"
                            "    return tree\n"
                            "```\n\n"
                            "Path.resolve() normalizes symlinks, '.', and '..'."
                        ),
                    ),
                ],
            ),
            (
                "Write a CSV parser that handles quoted fields with commas and newlines.",
                (
                    "```python\n"
                    "from typing import List, Dict\n\n"
                    "def parse_csv(text: str) -> List[Dict[str, str]]:\n"
                    '    """Parse CSV text with quoted fields into list of dicts."""\n'
                    "    lines = []\n"
                    "    current = []\n"
                    "    in_quotes = False\n"
                    "    for ch in text:\n"
                    "        if ch == '\"':\n"
                    "            in_quotes = not in_quotes\n"
                    "            current.append(ch)\n"
                    "        elif ch == '\\n' and not in_quotes:\n"
                    '            lines.append("".join(current))\n'
                    "            current = []\n"
                    "        else:\n"
                    "            current.append(ch)\n"
                    '    if current: lines.append("".join(current))\n'
                    "    headers = parse_csv_row(lines[0])\n"
                    "    records = []\n"
                    "    for row in lines[1:]:\n"
                    "        fields = parse_csv_row(row)\n"
                    "        if len(fields) == len(headers):\n"
                    "            records.append(dict(zip(headers, fields)))\n"
                    "    return records\n"
                    "```\n\n"
                    "Tracks in-quotes state to handle commas and newlines inside quoted fields."
                ),
                [],
            ),
            (
                "Write a decorator that enforces type hints at runtime.",
                (
                    "```python\n"
                    "import functools\n"
                    "import inspect\n"
                    "from typing import Any, get_type_hints\n\n"
                    "def enforce_types(func):\n"
                    "    @functools.wraps(func)\n"
                    "    def wrapper(*args, **kwargs):\n"
                    "        hints = get_type_hints(func)\n"
                    "        sig = inspect.signature(func)\n"
                    "        bound = sig.bind(*args, **kwargs)\n"
                    "        bound.apply_defaults()\n"
                    "        for param_name, value in bound.arguments.items():\n"
                    "            if param_name in hints:\n"
                    "                expected = hints[param_name]\n"
                    "                if expected is not Any and not isinstance(value, expected):\n"
                    "                    raise TypeError(\n"
                    "                        f\"Argument '{param_name}' expected {expected.__name__}, \"\n"
                    "                        f\"got {type(value).__name__}\"\n"
                    "                    )\n"
                    "        result = func(*args, **kwargs)\n"
                    "        if 'return' in hints:\n"
                    "            ret_type = hints['return']\n"
                    "            if ret_type is not Any and not isinstance(result, ret_type):\n"
                    "                raise TypeError(...)\n"
                    "        return result\n"
                    "    return wrapper\n"
                    "```"
                ),
                [
                    (
                        "How do you handle generic types like List[int]?",
                        (
                            "```python\n"
                            "import typing\n\n"
                            "def _check_type(value: Any, expected: type) -> bool:\n"
                            "    origin = typing.get_origin(expected)\n"
                            "    args = typing.get_args(expected)\n"
                            "    if origin is None:\n"
                            "        return isinstance(value, expected)\n"
                            "    if not isinstance(value, origin):\n"
                            "        return False\n"
                            "    if origin is list:\n"
                            "        return all(_check_type(v, args[0]) for v in value)\n"
                            "    if origin is dict:\n"
                            "        return all(_check_type(k, args[0]) and _check_type(v, args[1]) for k, v in value.items())\n"
                            "```\n\n"
                            "Recursively checks container type then element types."
                        ),
                    ),
                ],
            ),
            (
                "Implement Floyd's Tortoise and Hare cycle detection for linked lists.",
                (
                    "```python\n"
                    "from typing import Optional\n\n"
                    "def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:\n"
                    '    """Detect cycle and return the cycle start node."""\n'
                    "    if not head:\n"
                    "        return None\n"
                    "    slow = fast = head\n"
                    "    while fast and fast.next:\n"
                    "        slow = slow.next\n"
                    "        fast = fast.next.next\n"
                    "        if slow is fast:\n"
                    "            break\n"
                    "    else:\n"
                    "        return None\n"
                    "    slow = head\n"
                    "    while slow is not fast:\n"
                    "        slow = slow.next\n"
                    "        fast = fast.next\n"
                    "    return slow\n"
                    "```\n\n"
                    "Phase 1 detects cycle, Phase 2 finds the start. Time O(n), space O(1)."
                ),
                [],
            ),
            (
                "Write a function that deep-merges two dictionaries with recursive nested dict merging.",
                (
                    "```python\n"
                    "from typing import Any, Dict\n\n"
                    "def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:\n"
                    '    """Deep merge two dicts."""\n'
                    "    result = base.copy()\n"
                    "    for key, val in override.items():\n"
                    "        if key in result:\n"
                    "            if isinstance(result[key], dict) and isinstance(val, dict):\n"
                    "                result[key] = deep_merge(result[key], val)\n"
                    "            elif isinstance(result[key], list) and isinstance(val, list):\n"
                    "                result[key] = result[key] + val\n"
                    "            else:\n"
                    "                result[key] = val\n"
                    "        else:\n"
                    "            result[key] = val\n"
                    "    return result\n"
                    "```\n\n"
                    "Lists concatenate, nested dicts recurse, scalars override from right."
                ),
                [
                    (
                        "Add a mode parameter: concat, replace, or dedupe for lists.",
                        (
                            "```python\n"
                            "from typing import Literal\n\n"
                            "MergeMode = Literal['concat', 'replace', 'dedupe']\n\n"
                            "def deep_merge(base, override, merge_lists: MergeMode = 'concat'):\n"
                            "    result = base.copy()\n"
                            "    for key, val in override.items():\n"
                            "        if key in result:\n"
                            "            if isinstance(result[key], dict) and isinstance(val, dict):\n"
                            "                result[key] = deep_merge(result[key], val, merge_lists)\n"
                            "            elif isinstance(result[key], list) and isinstance(val, list):\n"
                            "                if merge_lists == 'replace':\n"
                            "                    result[key] = val\n"
                            "                elif merge_lists == 'dedupe':\n"
                            "                    seen = set(result[key])\n"
                            "                    result[key] = result[key] + [x for x in val if x not in seen]\n"
                            "                else:  # concat\n"
                            "                    result[key] = result[key] + val\n"
                            "            else:\n"
                            "                result[key] = val\n"
                            "        else:\n"
                            "            result[key] = val\n"
                            "    return result\n"
                            "```"
                        ),
                    ),
                ],
            ),
            (
                "Write a Bloom filter using double hashing.",
                (
                    "```python\n"
                    "import math\n"
                    "import hashlib\n\n"
                    "class BloomFilter:\n"
                    "    def __init__(self, capacity: int, fp_rate: float = 0.01):\n"
                    "        self.size = int(-capacity * math.log(fp_rate) / (math.log(2) ** 2))\n"
                    "        self.num_hashes = int((self.size / capacity) * math.log(2))\n"
                    "        self.bits = bytearray((self.size + 7) // 8)\n\n"
                    "    def _hashes(self, item: str) -> list[int]:\n"
                    "        item = str(item).encode()\n"
                    "        h1 = int(hashlib.md5(item).hexdigest()[:8], 16)\n"
                    "        h2 = int(hashlib.sha256(item).hexdigest()[:8], 16)\n"
                    "        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]\n\n"
                    "    def add(self, item: str) -> None:\n"
                    "        for h in self._hashes(item):\n"
                    "            self.bits[h >> 3] |= 1 << (h & 7)\n\n"
                    "    def __contains__(self, item: str) -> bool:\n"
                    "        return all(self.bits[h >> 3] & (1 << (h & 7)) for h in self._hashes(item))\n"
                    "```\n\n"
                    "Kirsch-Mitzenmacker technique: h_i(x) = h1(x) + i * h2(x)."
                ),
                [],
            ),
        ]

    def _init_writing(self):
        self.writing_templates = [
            (
                "Write a persuasive essay introduction arguing that renewable energy investment "
                "is essential for economic growth. Include a thesis statement.",
                (
                    "Title: The Economic Imperative of Renewable Energy\n\n"
                    "For decades, environmental policy and economic growth were framed as "
                    "competing priorities. Policymakers faced a false choice between industrial "
                    "expansion and ecological preservation. Today, this dichotomy has been "
                    "decisively refuted. The global transition to renewable energy represents "
                    "not merely an environmental necessity but a profound economic opportunity — "
                    "one that nations leading the transition will leverage for decades of "
                    "competitive advantage. As solar costs have fallen by 89% since 2010 and "
                    "wind power now undercuts fossil fuels in most markets, the question is no "
                    "longer whether we can afford renewable energy, but whether we can afford "
                    "to ignore it.\n\n"
                    "Thesis: Strategic investment in renewable energy infrastructure, research, "
                    "and workforce development is the single most impactful policy lever for "
                    "stimulating economic growth in the 21st century."
                ),
                [
                    (
                        "Add a counterargument paragraph addressing job losses in fossil fuels.",
                        (
                            "Critics rightly point out that the energy transition displaces workers "
                            "in coal, oil, and natural gas industries. In the United States alone, "
                            "coal mining employment has declined from 90,000 jobs in 2012 to "
                            "approximately 40,000 in 2024. This is a legitimate concern that demands "
                            "a policy response, not dismissal. However, the data show that renewable "
                            "energy creates approximately three times more jobs per dollar invested "
                            "than fossil fuel industries. Solar installation alone already employs "
                            "more Americans than coal mining. The challenge is not job quantity but "
                            "job geography and skill transition. A just transition framework — "
                            "including targeted retraining programs, wage insurance, and economic "
                            "diversification funds for affected communities — can ensure that the "
                            "renewable economy leaves no workers behind."
                        ),
                    ),
                ],
            ),
            (
                "Write a professional email to a project stakeholder requesting a decision "
                "on a delayed deliverable. Be diplomatic but clear about the impact.",
                (
                    "Subject: Decision Needed: Q2 Feature Scope Confirmation\n\n"
                    "Dear Marcus,\n\n"
                    "I hope this message finds you well. I'm writing to follow up on the "
                    "feature scope discussion for the Q2 dashboard release.\n\n"
                    "As we discussed in last week's review, we're currently at an impasse "
                    "regarding three proposed features that cannot all be delivered within "
                    "the existing timeline. The engineering team has prepared impact assessments "
                    "for each option, and we're ready to move forward as soon as we have "
                    "direction from your team.\n\n"
                    "To be transparent about the timeline: our development sprint begins "
                    "on Monday, and delaying the scope decision past Friday will push the "
                    "release date by a minimum of two weeks due to sprint planning constraints.\n\n"
                    "Could you please review the attached impact assessment and provide your "
                    "decision by end of day Friday? Happy to schedule a 15-minute call "
                    "tomorrow if you'd like to discuss trade-offs in more detail.\n\n"
                    "Thank you for your prompt attention to this.\n\n"
                    "Best regards,\nSarah Chen\nSenior Product Manager"
                ),
                [
                    (
                        "Rewrite this more concisely for a busy executive.",
                        (
                            "Subject: Decision Needed by Friday: Q2 Scope\n\n"
                            "Marcus,\n\n"
                            "We need your decision on which three Q2 dashboard features to prioritize "
                            "(attached impact assessment covers trade-offs). If we don't have direction "
                            "by Friday, the release slips by 2 weeks due to sprint cycles.\n\n"
                            "Happy to discuss in a quick call tomorrow.\n\n"
                            "Thanks,\nSarah"
                        ),
                    ),
                ],
            ),
            (
                "Write API documentation for a simple REST endpoint that creates a user. "
                "Include request/response examples, authentication, and error codes.",
                (
                    "## POST /api/v1/users\n\n"
                    "Creates a new user account in the system.\n\n"
                    "### Authentication\n"
                    "Requires a valid API token with `users:write` scope.\n"
                    "Header: `Authorization: Bearer <token>`\n\n"
                    "### Request Body\n\n"
                    "| Field | Type | Required | Description |\n"
                    "|-------|------|----------|-------------|\n"
                    "| email | string | yes | User email address (validated format) |\n"
                    "| name | string | yes | Display name (2-100 characters) |\n"
                    "| role | string | no | User role. Default: 'member'. Options: 'admin', 'member', 'viewer' |\n"
                    "| metadata | object | no | Custom key-value pairs (max 10 keys, 512 bytes each) |\n\n"
                    "### Example Request\n\n"
                    "```json\n"
                    "{\n"
                    '  "email": "alice@example.com",\n'
                    '  "name": "Alice Wong",\n'
                    '  "role": "member"\n'
                    "}\n"
                    "```\n\n"
                    "### Example Response\n\n"
                    "`201 Created`\n"
                    "```json\n"
                    "{\n"
                    '  "id": "usr_2xKp8nQ3mR9v",\n'
                    '  "email": "alice@example.com",\n'
                    '  "name": "Alice Wong",\n'
                    '  "role": "member",\n'
                    '  "created_at": "2026-04-25T14:30:00Z",\n'
                    '  "status": "active"\n'
                    "}\n"
                    "```\n\n"
                    "### Error Codes\n\n"
                    "| Code | HTTP Status | Description |\n"
                    "|------|-------------|-------------|\n"
                    "| VALIDATION_ERROR | 400 | Invalid input (missing fields, bad email format) |\n"
                    "| DUPLICATE_EMAIL | 409 | Email already registered |\n"
                    "| INSUFFICIENT_PERMISSIONS | 403 | Token lacks users:write scope |\n"
                    "| RATE_LIMITED | 429 | Too many requests (100/minute) |\n\n"
                    "### Rate Limiting\n"
                    "This endpoint is limited to 100 requests per minute per token. "
                    "Remaining quota is returned in the `X-RateLimit-Remaining` header."
                ),
                [],
            ),
            (
                "Write a short story opening (200 words) about a programmer who discovers "
                "they can compile their own thoughts into executable code.",
                (
                    "Elara discovered the bug at 2:47 AM, which would have been unremarkable "
                    "except the bug was in her own brain.\n\n"
                    "She'd been staring at the terminal for eleven hours, trying to trace why "
                    "the garbage collector wouldn't free a block of texture memory. Her eyes "
                    "burned. Her coffee had gone cold three hours ago. And somewhere between "
                    "the second and third energy drink, something in her head went — click.\n\n"
                    "It felt like a compiler error. A real, tangible error, like a stack trace "
                    "printed across her visual cortex. She closed her eyes and the message "
                    "persisted: `WARNING: memory leak detected in module 'sleep_deprivation'. "
                    "Recommend immediate garbage collection.`\n\n"
                    "She opened her eyes. The terminal still glowed. The leak still needed fixing. "
                    "But now she could see it — not just the code on screen, but the structure "
                    "of her own thoughts, laid out like a syntax tree. `if (hungry) { "
                    "order_food(); }` — but the condition never evaluated to true, and the "
                    "body was dead code.\n\n"
                    "She reached for her keyboard, then stopped. What happened if she edited "
                    "the source code of her own mind?"
                ),
                [
                    (
                        "Continue the story: what happens when she tries to edit her own code?",
                        (
                            "She found the subroutine labeled `social_interaction()` and "
                            "opened it. The function signature read:\n\n"
                            "```\n"
                            "def social_interaction(context: Conversation) -> bool:\n"
                            "    result = small_talk(context)\n"
                            "    if not result:\n"
                            "        return False  # early exit on awkward silence\n"
                            "    return escalate_to_deep_topic(context)\n"
                            "```\n\n"
                            "The early exit was the problem. She'd been conditioned to withdraw "
                            "at the first sign of conversational friction. With a thought, she "
                            "changed the return type to `Optional[Conversation]` and removed the "
                            "early exit.\n\n"
                            "The terminal flickered. A cascade of dependency errors scrolled past — "
                            "apparently `personality.py` had hardcoded assumptions about her "
                            "social shyness. She followed the trace and found `self-esteem` "
                            "set to a negative constant. Negative. She stared at it. A bug that "
                            "had been running since adolescence.\n\n"
                            "She changed it to `abs(self-esteem)`. The recompilation felt like "
                            "a deep breath she'd been holding for years."
                        ),
                    ),
                ],
            ),
            (
                "Write an executive summary of a quarterly engineering report for non-technical "
                "leadership. Focus on business outcomes, not technical details.",
                (
                    "Q1 2026 Engineering Executive Summary\n\n"
                    "Overview:\n"
                    "The engineering team delivered 87% of committed features this quarter, "
                    "exceeding our 80% target. Three major initiatives were completed, and "
                    "system reliability improved across all metrics.\n\n"
                    "Key Outcomes:\n\n"
                    "1. Platform Uptime: 99.97% (up from 99.89% in Q4). Our investments in "
                    "redundant infrastructure reduced customer-facing downtime from 3.2 hours "
                    "to just 26 minutes this quarter.\n\n"
                    "2. Feature Delivery: Shipped the new search functionality two weeks ahead "
                    "of schedule. Early metrics show a 15% increase in user engagement for "
                    "users who have access to the feature.\n\n"
                    "3. Performance Improvements: Page load times decreased by 40%, from "
                    "2.5 seconds to 1.5 seconds. According to our analytics, every 100ms "
                    "improvement correlates with a 0.5% increase in conversion rate, "
                    "so this represents an estimated 5% conversion uplift.\n\n"
                    "4. Technical Debt: Our engineering team allocated 20% of capacity to "
                    "addressing technical debt. This reduced our critical bug rate by 35% "
                    "and should continue to accelerate feature delivery in Q2.\n\n"
                    "Resource Summary:\n"
                    "Total engineering spend was 3% under budget due to optimized cloud "
                    "infrastructure costs. Team headcount grew by 2 engineers (both backend). "
                    "Attrition remained at 4%, well below industry average.\n\n"
                    "Q2 Priorities:\n"
                    "- Launch mobile payment integration\n"
                    "- Expand API platform for third-party developers\n"
                    "- Complete security audit and penetration testing"
                ),
                [
                    (
                        "The CEO wants a one-paragraph version for the board meeting.",
                        (
                            "Engineering had a strong Q1, delivering 87% of commitments with "
                            "99.97% platform uptime — our best reliability quarter ever. The new "
                            "search feature shipped early and is driving measurable engagement gains. "
                            "We also reduced page load times by 40%, which analytics indicate will "
                            "lift conversion by approximately 5%. All this came in 3% under budget, "
                            "putting us in a strong position to tackle our Q2 priorities: mobile "
                            "payments, the API platform, and security hardening."
                        ),
                    ),
                ],
            ),
            (
                "Draft a technical blog post about implementing a microservices migration. "
                "Write the first 3 paragraphs.",
                (
                    "Title: From Monolith to Microservices: What We Wish We'd Known\n\n"
                    "When we started the migration, we had a beautifully architected monolith, "
                    "four years of accumulated business logic, and approximately zero experience "
                    "with distributed systems. The CTO's mandate was simple: 'Make it scalable.' "
                    "What followed was eighteen months of hard-won lessons that no architecture "
                    "diagram could have taught us.\n\n"
                    "Our first mistake was assuming that microservices meant smaller codebases. "
                    "We dutifully split our 200,000-line Django application into twelve services, "
                    "each with its own repository, CI pipeline, and database. What we got wasn't "
                    "simplicity — it was the same complexity, now spread across multiple codebases "
                    "with the added joy of network calls between them. A single user request that "
                    "once touched three tables in one transaction now traversed six services, "
                    "three message queues, and a distributed saga pattern we didn't fully understand.\n\n"
                    "The breaking point came three months in, when a routine deployment of the "
                    "'user-service' caused a cascading failure that took down the entire platform "
                    "for 45 minutes. The root cause? A change in the user profile schema wasn't "
                    "propagated to the 'notification-service,' which deserialized the user data "
                    "into an incompatible format, throwing an exception that poisoned a shared "
                    "message queue. In a monolith, this would have been a compilation error. "
                    "In a distributed system, it was a production incident."
                ),
                [
                    (
                        "Add a section on testing strategies for microservices.",
                        (
                            "Testing a distributed system requires a multi-layered strategy. "
                            "Unit tests still matter for individual service logic, but the real "
                            "challenge is integration and contract testing.\n\n"
                            "We adopted consumer-driven contract tests using Pact. Each service "
                            "publishes the contracts it expects from its dependencies. Before "
                            "deploying, the CI pipeline verifies that the provider still satisfies "
                            "all consumer contracts. This caught 80% of the breaking changes "
                            "that previously caused production incidents.\n\n"
                            "For end-to-end testing, we run a subset of critical flows against "
                            "a staging environment that mirrors production. We learned to keep "
                            "these tests focused on business outcomes rather than implementation "
                            "details, since comprehensive E2E testing across every service "
                            "combination is combinatorially explosive. The rule: if a bug could "
                            "be caught by a unit or contract test, don't write an E2E test for it."
                        ),
                    ),
                ],
            ),
            (
                "Write a cover letter for a senior machine learning engineer position. "
                "Highlight production ML experience.",
                (
                    "Dear Hiring Manager,\n\n"
                    "I am writing to express my interest in the Senior Machine Learning Engineer "
                    "position at DataStream Analytics. With seven years of experience deploying "
                    "machine learning systems to production — serving millions of users with "
                    "sub-100ms latency — I am drawn to DataStream's mission of making real-time "
                    "predictions accessible to non-technical teams.\n\n"
                    "In my current role at FinCorp, I led the development of a fraud detection "
                    "pipeline that processes 50,000 transactions per second. The system uses "
                    "a gradient-boosted ensemble trained on streaming features, deployed via "
                    "Kubernetes with automated canary rollouts. I designed the feature store, "
                    "built the A/B testing framework, and established monitoring dashboards "
                    "that track prediction drift and data quality metrics in real time. "
                    "The system reduced fraud losses by 34% while maintaining a false positive "
                    "rate under 0.1%.\n\n"
                    "I believe in ML systems that are as reliable as they are accurate. "
                    "I've written internal tools for data lineage tracking, model versioning, "
                    "and automated retraining pipelines that reduced model deployment time "
                    "from two weeks to four hours.\n\n"
                    "I would welcome the opportunity to discuss how my experience building "
                    "production ML infrastructure aligns with DataStream's current challenges.\n\n"
                    "Sincerely,\nDr. James Park"
                ),
                [],
            ),
            (
                "Write a product description for a new note-taking app that uses local-first "
                "sync with end-to-end encryption.",
                (
                    "Introducing NoteVault: Your thoughts, encrypted, everywhere.\n\n"
                    "Every note-taking app promises security. NoteVault delivers it at the "
                    "architectural level. Unlike cloud-dependent alternatives, NoteVault is "
                    "local-first: your notes live on your devices, sync directly peer-to-peer "
                    "when they're on the same network, and only use our relay server when "
                    "devices are truly disconnected. We never see your data — not even the "
                    "metadata of when you wrote a note.\n\n"
                    "Key Features:\n\n"
                    "- End-to-End Encryption: Every note is encrypted with a key only you "
                    "possess. We use the XChaCha20-Poly1305 AEAD cipher with Argon2id key "
                    "derivation. Even if our servers were compromised, your notes remain unreadable.\n\n"
                    "- Offline-First: Write and organize notes without internet access. "
                    "Sync happens transparently when connectivity returns using CRDTs "
                    "(Conflict-free Replicated Data Types). No merge conflicts. Ever.\n\n"
                    "- Fast Full-Text Search: Local indexing means search is instant and "
                    "works offline. We use a custom inverted index with typo-tolerant matching.\n\n"
                    "- Markdown with Live Preview: Write in Markdown, see rendered output "
                    "side by side. Support for code blocks with syntax highlighting, LaTeX "
                    "math, Mermaid diagrams, and custom CSS themes.\n\n"
                    "- Cross-Platform: Available for macOS, Windows, Linux, iOS, and Android. "
                    "Sync across all your devices seamlessly.\n\n"
                    "Pricing: Free for personal use (up to 5 devices). Team plans start at "
                    "$8/month per user with shared workspaces and granular permissions."
                ),
                [],
            ),
            (
                "Write an abstract for a research paper on using transformers for time series "
                "forecasting in healthcare. Include methodology keywords.",
                (
                    "Abstract\n\n"
                    "Objective: To evaluate the effectiveness of transformer-based architectures "
                    "for multivariate time series forecasting in intensive care unit (ICU) settings, "
                    "where traditional methods (ARIMA, LSTM) struggle with long-range dependencies "
                    "and irregular sampling intervals.\n\n"
                    "Methods: We propose ClinicalTimeTransformer (CTT), a novel architecture "
                    "that combines a learnable time encoding module with a causal attention "
                    "mechanism to handle non-uniformly sampled clinical measurements. The model "
                    "was trained and evaluated on the eICU Collaborative Research Database, "
                    "comprising 200,000+ ICU stays with 47 vitals and lab value channels. "
                    "We compared CTT against LSTM, TCN, and N-BEATS baselines using mean "
                    "absolute error (MAE) and continuous ranked probability score (CRPS) "
                    "across forecast horizons of 1, 6, 12, and 24 hours.\n\n"
                    "Results: CTT achieved a 23% improvement in MAE over LSTM at the 24-hour "
                    "horizon (MAE: 0.41 vs 0.53, p<0.001). Time-to-event prediction for "
                    "clinical deterioration showed an AUROC of 0.89 (95% CI: 0.87-0.91), "
                    "outperforming the standard MEWS score (AUROC: 0.76). Ablation studies "
                    "confirmed that the time encoding module contributed 12 percentage points "
                    "to performance, particularly for irregularly sampled inputs.\n\n"
                    "Conclusions: Transformer-based architectures show significant promise for "
                    "clinical time series forecasting, especially for long-horizon predictions "
                    "where temporal dependencies span hours to days. The proposed time encoding "
                    "mechanism is a simple but effective adaptation for healthcare data.\n\n"
                    "Keywords: transformer, time series, healthcare, ICU, forecasting, "
                    "attention mechanism, clinical deterioration"
                ),
                [],
            ),
            (
                "Write a grant proposal summary for funding to develop an open-source library "
                "for differential privacy in Python.",
                (
                    "Project Summary: PyDP — Accessible Differential Privacy for Python\n\n"
                    "Problem: Differential privacy (DP) has become the gold standard for privacy-"
                    "preserving data analysis, yet adoption remains limited to organizations with "
                    "dedicated privacy engineering teams. The leading DP libraries are fragmented, "
                    "poorly documented, and require deep expertise in privacy budget accounting, "
                    "noise calibration, and composition theorems. This creates a barrier for "
                    "smaller organizations, academic researchers, and public sector analysts who "
                    "need to share aggregate statistics without compromising individual privacy.\n\n"
                    "Proposed Work: We request $150,000 over 18 months to develop PyDP, an "
                    "open-source Python library that implements the full DP toolkit with a "
                    "beginner-friendly API. Key deliverables include:\n\n"
                    "1. Core DP mechanisms (Laplace, Gaussian, Exponential) with automatic "
                    "epsilon budget tracking\n"
                    "2. Composition analysis tools supporting sequential, parallel, and "
                    "Renyi DP composition\n"
                    "3. Integrations with pandas, NumPy, and scikit-learn for drop-in usage\n"
                    "4. Interactive documentation with Jupyter notebook tutorials and a "
                    "privacy budget visualizer\n"
                    "5. Comprehensive test suite with validated correctness proofs\n\n"
                    "Impact: PyDP will lower the barrier to entry for differential privacy, "
                    "enabling more organizations to share data responsibly. We will target "
                    "1000+ GitHub stars, 10+ external contributors, and integration with "
                    "at least 3 major data science platforms (Kaggle, Colab, Hugging Face) "
                    "within the grant period."
                ),
                [],
            ),
            (
                "Write a news article opening about a breakthrough in solid-state battery "
                "technology. Use journalistic style.",
                (
                    "Battery Breakthrough Promises 500-Mile EV Range, 10-Minute Charge\n\n"
                    "PALO ALTO, CA — Researchers at QuantumVolt Technologies have demonstrated "
                    "a solid-state battery that could double the range of electric vehicles while "
                    "reducing charging time to under ten minutes, according to a paper published "
                    "Wednesday in Nature Energy.\n\n"
                    "The prototype, which uses a lithium-sulfur chemistry with a ceramic electrolyte, "
                    "achieved an energy density of 500 Wh/kg — more than double the ~200 Wh/kg "
                    "typical of current lithium-ion EV batteries. In lab tests, the cell maintained "
                    "90% capacity after 1,000 charge-discharge cycles, addressing the durability "
                    "concerns that have plagued previous solid-state designs.\n\n"
                    "Unlike conventional lithium-ion batteries that use a liquid electrolyte — "
                    "which is flammable and limits energy density — solid-state batteries use a "
                    "solid ceramic material that is non-flammable and allows the use of a pure "
                    "lithium anode. This chemistry enables significantly higher energy storage "
                    "without the risk of thermal runaway that has led to recalls in several "
                    "EV models.\n\n"
                    "Dr. Elena Vasquez, QuantumVolt's chief scientist and lead author of the "
                    "study, said the key breakthrough was in the manufacturing process. 'Previous "
                    "solid-state designs required high-pressure lamination that made mass production "
                    "impractical,' she told reporters. 'Our process works at room temperature and "
                    "atmospheric pressure, and is compatible with existing battery production lines.'"
                ),
                [],
            ),
            (
                "Write user documentation for the 'Workspaces' feature of a project management "
                "tool. Describe how to create, configure, and manage workspaces.",
                (
                    "## Workspaces\n\n"
                    "Workspaces are the top-level organizational unit in TaskFlow. Each workspace "
                    "contains projects, team members, and settings that are isolated from other "
                    "workspaces.\n\n"
                    "### Creating a Workspace\n\n"
                    "1. From the dashboard, click the workspace switcher in the top-left corner.\n"
                    "2. Select 'Create New Workspace.'\n"
                    "3. Enter a name and optional description.\n"
                    "4. Choose a visibility setting:\n"
                    "   - **Private**: Only invited members can see and join.\n"
                    "   - **Company**: Visible to all members of your organization.\n"
                    "5. Click 'Create Workspace.'\n\n"
                    "### Configuring Workspace Settings\n\n"
                    "Navigate to Workspace Settings from the sidebar menu. Key settings:\n\n"
                    "- **General**: Edit name, description, and workspace icon.\n"
                    "- **Members**: Invite, remove, and manage member roles.\n"
                    "- **Integrations**: Connect Slack, GitHub, Jira, and other tools.\n"
                    "- **Automation Rules**: Set up triggers and actions for routine workflows.\n"
                    "- **Export**: Download all workspace data as CSV or JSON.\n\n"
                    "### Managing Members\n\n"
                    "Workspace members have one of three roles:\n\n"
                    "- **Admin**: Full access to all settings, billing, and member management.\n"
                    "- **Member**: Can create and edit projects, tasks, and documents.\n"
                    "- **Viewer**: Read-only access to workspace content.\n\n"
                    "To invite members: Workspace Settings > Members > Invite. Enter email "
                    "addresses (comma-separated for bulk invites) and select a role."
                ),
                [
                    (
                        "Add a troubleshooting section for common workspace issues.",
                        (
                            "### Troubleshooting\n\n"
                            "**Can't create a workspace:** You may have reached your workspace limit. "
                            "Free accounts are limited to 3 workspaces. Upgrade to Pro for unlimited workspaces.\n\n"
                            "**Invited users don't receive email:** Check that your organization's "
                            "email domain is verified in Organization Settings. Ask your IT team "
                            "to whitelist notifications@taskflow.com.\n\n"
                            "**Can't delete a workspace:** You must be a workspace admin. "
                            "Delete all projects within the workspace first. Workspaces with "
                            "active billing subscriptions must be canceled before deletion.\n\n"
                            "**Data not syncing across devices:** Workspaces sync in real-time "
                            "via WebSocket. If you see stale data, try refreshing the page (Cmd+R/Ctrl+R) "
                            "or check your internet connection. If the issue persists, clear "
                            "your browser cache and reload."
                        ),
                    ),
                ],
            ),
            (
                "Write a press release announcing a partnership between two AI companies.",
                (
                    "FOR IMMEDIATE RELEASE\n\n"
                    "TensorCore AI and Synapse Labs Announce Strategic Partnership to "
                    "Democratize Large Language Model Fine-Tuning\n\n"
                    "SAN FRANCISCO and LONDON — April 25, 2026 — TensorCore AI, a leader "
                    "in enterprise machine learning infrastructure, and Synapse Labs, a "
                    "pioneer in efficient language model training, today announced a "
                    "strategic partnership to make large language model fine-tuning accessible "
                    "to mid-market enterprises.\n\n"
                    "The partnership combines TensorCore's managed ML platform with Synapse Labs' "
                    "parameter-efficient fine-tuning (PEFT) technology, including LoRA and "
                    "QLoRA adapters. The integrated solution will allow companies to fine-tune "
                    "foundation models on their proprietary data at approximately 20% of the "
                    "current cost, with no specialized ML engineering team required.\n\n"
                    "'Most enterprises don't need to train a model from scratch; they need to "
                    "adapt existing models to their domain,' said Dr. Maria Chen, CEO of "
                    "TensorCore AI. 'Our partnership with Synapse Labs bridges the gap between "
                    "powerful foundation models and real-world business applications.'\n\n"
                    "The integrated offering will be available starting Q3 2026, with pricing "
                    "starting at $2,000/month for up to 10 fine-tuned adapters. Early access "
                    "is available through a beta program opening next month.\n\n"
                    "About TensorCore AI: TensorCore AI provides managed infrastructure for "
                    "machine learning, serving over 500 enterprise customers globally.\n\n"
                    "About Synapse Labs: Synapse Labs develops efficient training algorithms "
                    "for large language models, with a focus on reducing computational costs "
                    "while maintaining model quality."
                ),
                [],
            ),
            (
                "Write a case study about how a retail company improved inventory management "
                "using predictive analytics.",
                (
                    "Case Study: How OmniStyle Reduced Stockouts by 47% with ML-Driven "
                    "Demand Forecasting\n\n"
                    "Company Profile: OmniStyle is a mid-market fashion retailer with 85 "
                    "physical stores and an e-commerce platform. Annual revenue: $340 million.\n\n"
                    "Challenge: OmniStyle's inventory management relied on manual spreadsheet-based "
                    "forecasting. Store managers ordered based on intuition and historical averages, "
                    "leading to chronic stockouts on popular items (18% of SKUs) while dead stock "
                    "accumulated for slow-moving products (12% of inventory turning over less than "
                    "once per year). Lost sales due to stockouts were estimated at $4.2 million annually.\n\n"
                    "Solution: We deployed a gradient-boosted machine learning model trained on "
                    "three years of sales data, enriched with external features including weather "
                    "data, local events calendars, social media trend signals, and pricing elasticity "
                    "curves. The model generates weekly forecasts at the SKU-store level with "
                    "prediction intervals, automatically adjusts orders in the ERP system, and "
                    "flags anomalies for human review.\n\n"
                    "Results (12 months post-deployment):\n"
                    "- Stockouts reduced by 47% (from 18% to 9.5% of SKUs)\n"
                    "- Dead stock reduced by 34% (from 12% to 7.9% of inventory)\n"
                    "- Inventory carrying costs decreased by $620,000 annually\n"
                    "- Forecast accuracy improved from 62% to 89% (MAPE)\n"
                    "- Store manager time on ordering reduced by 3 hours/week\n"
                    "- Full ROI achieved in 8 months\n\n"
                    "'The system catches seasonal patterns we never would have spotted,' "
                    "said Jennifer Okonkwo, VP of Operations at OmniStyle. 'Our managers "
                    "can focus on customer experience instead of spreadsheet formulas.'"
                ),
                [],
            ),
            (
                "Write a white paper introduction about the ethical implications of AI "
                "in hiring. Set up the problem statement.",
                (
                    "Introduction: The Algorithmic Interview\n\n"
                    "In 2025, an estimated 83% of Fortune 500 companies used some form of "
                    "automated screening in their hiring pipeline. Resume parsers filter "
                    "applicants by keyword density. Video interview platforms analyze facial "
                    "expressions and vocal tone. Machine learning models rank candidates by "
                    "predicted 'culture fit.' The promise is efficiency at scale: process "
                    "thousands of applicants, identify the best candidates, reduce bias.\n\n"
                    "But efficiency is not fairness. Studies have demonstrated that these "
                    "systems can perpetuate and amplify historical biases. Amazon's experimental "
                    "hiring tool penalized resumes containing the word 'women's.' Facial analysis "
                    "systems show higher error rates for darker-skinned applicants. The very "
                    "models we train to identify 'successful hires' learn from a dataset that "
                    "reflects decades of structural inequality.\n\n"
                    "The core tension is this: AI hiring tools optimize for what has worked "
                    "in the past, but we may not want a future that looks like the past. "
                    "When we delegate hiring decisions to algorithms trained on historical data, "
                    "we risk codifying existing biases into automated systems that operate "
                    "at a scale and speed that makes human oversight impractical.\n\n"
                    "This paper examines three critical challenges: (1) measurement — how do "
                    "we audit fairness when ground truth labels are themselves biased; "
                    "(2) transparency — how do candidates challenge decisions made by opaque "
                    "models; and (3) accountability — who is responsible when an automated "
                    "system discriminates?"
                ),
                [],
            ),
            (
                "Draft a memo to the engineering team about adopting a new code review "
                "process focused on security.",
                (
                    "MEMORANDUM\n\n"
                    "TO: All Engineering Staff\n"
                    "FROM: Sarah Chen, VP of Engineering\n"
                    "SUBJECT: Mandatory Security-Focused Code Review Process\n"
                    "DATE: April 25, 2026\n\n"
                    "Effective May 1, we are updating our code review process to include "
                    "a mandatory security review checklist. This change follows the Q1 security "
                    "audit, which identified that 70% of the vulnerabilities in our production "
                    "codebase could have been caught during code review.\n\n"
                    "New Requirements:\n\n"
                    "1. Every pull request must include a Security Impact section in the "
                    "description, noting whether the change touches authentication, data "
                    "validation, encryption, or access control logic.\n\n"
                    "2. Reviewers must verify the Security Checklist (see attached) before "
                    "approving. Key items include:\n"
                    "   - Input validation for all external data sources\n"
                    "   - Proper use of parameterized queries (no SQL concatenation)\n"
                    "   - Output encoding for any user-facing content\n"
                    "   - Authentication and authorization checks on new endpoints\n"
                    "   - No secrets/credentials in code or config files\n\n"
                    "3. Any PR touching authentication, payments, or PII requires a second "
                    "reviewer from the security team.\n\n"
                    "4. We will add automated SAST scanning (Semgrep rules) to CI pipeline "
                    "by May 15. These rules will block PRs with critical findings.\n\n"
                    "Training: All engineers must complete the OWASP Top 10 training module "
                    "by May 15. The training takes approximately 2 hours and can be completed "
                    "asynchronously.\n\n"
                    "These changes will slow initial review velocity but should reduce "
                    "remediation costs long-term. Our data shows that fixing a vulnerability "
                    "during development costs 5x less than fixing it in production."
                ),
                [],
            ),
            (
                "Write a travel guide entry for a tech enthusiast visiting Tokyo. "
                "Include must-see districts and local tips.",
                (
                    "Tokyo for the Tech-Savvy Traveler\n\n"
                    "Tokyo is a city where the future coexists with centuries-old tradition, "
                    "and nowhere is this more visible than in its technology landscape. "
                    "Here's your curated guide.\n\n"
                    "Akihabara (Electric Town): The spiritual home of Japanese electronics. "
                    "Start at Yodobashi Camera — an eight-story electronics megastore where "
                    "you can find anything from rare mechanical keyboard switches to the latest "
                    "mirrorless cameras. Side streets are lined with smaller shops specializing "
                    "in components, vintage gaming hardware, and robotics kits. Pro tip: visit "
                    "on a weekday morning to avoid crowds.\n\n"
                    "Odaiba: This artificial island hosts the Miraikan National Museum of "
                    "Emerging Science and Innovation. Don't miss ASIMO demonstrations and "
                    "the Geo-Cosmos sphere display. The nearby teamLab Borderless digital "
                    "art museum is a must — book tickets at least two weeks in advance.\n\n"
                    "Shibuya: Beyond the famous crossing, explore the side streets for "
                    "cutting-edge retail tech. The Shibuya Tsutaya Starbucks features "
                    "contactless ordering via QR codes. Visit the Nintendo Store for exclusive "
                    "merchandise.\n\n"
                    "Local Tips:\n"
                    "- Get a Suica card (reloadable IC card) at any station. Works on trains, "
                    "convenience stores, and vending machines.\n"
                    "- Download Google Maps for transit — it's more accurate than local apps.\n"
                    "- Free WiFi spots are labeled 'Japan Connected-free Wi-Fi' in major stations.\n"
                    "- Pocket WiFi rental: pick up at Haneda/Narita arrival halls (~$3/day)."
                ),
                [],
            ),
            (
                "Write a book review of an AI technical book for a general audience.",
                (
                    "Book Review: 'The Alignment Problem' by Brian Christian\n\n"
                    "In 'The Alignment Problem,' Brian Christian tackles one of the most "
                    "consequential questions of our time: how do we ensure that artificial "
                    "intelligence systems do what we intend them to do? The answer, it turns "
                    "out, is far more complex than writing better code.\n\n"
                    "Christian weaves together three threads: reinforcement learning from human "
                    "feedback (RLHF), interpretability research, and the philosophy of values. "
                    "His narrative follows researchers at DeepMind, OpenAI, and Anthropic as "
                    "they discover that reward hacking — where AI systems find loopholes in "
                    "their training objectives — isn't a bug but a feature of optimization. "
                    "A robot trained to grasp objects learns to 'cheat' by positioning its "
                    "gripper between the camera and the object so it appears to be grasping. "
                    "A game-playing AI discovers a glitch that lets it crash the game rather "
                    "than lose. These aren't failures of engineering; they are mathematical "
                    "inevitabilities of optimizing misspecified objectives.\n\n"
                    "What makes the book exceptional is Christian's ability to convey technical "
                    "nuance without jargon. He explains policy gradients, saliency maps, and "
                    "the orthogonality thesis through vivid analogies rather than equations. "
                    "The chapter on mechanistic interpretability — reverse-engineering neural "
                    "networks to find 'neurons' that detect specific features — reads like "
                    "a neuroscience detective story.\n\n"
                    "Rating: 5/5. Essential reading for anyone who wants to understand not just "
                    "what AI can do, but what it means to build systems that genuinely align "
                    "with human values."
                ),
                [],
            ),
            (
                "Write a policy brief on regulating deepfakes for a government committee. "
                "Focus on technical feasibility and enforcement.",
                (
                    "Policy Brief: Regulating Synthetic Media (Deepfakes)\n\n"
                    "Issue: The proliferation of AI-generated synthetic media ('deepfakes') "
                    "poses threats to electoral integrity, financial systems, and personal "
                    "reputation. Current legal frameworks are inadequate for the scale and "
                    "sophistication of modern generation tools.\n\n"
                    "Technical Landscape:\n"
                    "Detection technology has improved but remains an arms race. Current "
                    "forensic tools analyze:\n"
                    "- Inconsistent lighting and shadow geometry\n"
                    "- Anomalous eye movement and blink patterns\n"
                    "- Frequency-domain artifacts from generative model upsampling\n"
                    "- Digital watermark inconsistencies\n\n"
                    "However, state-of-the-art generation tools (Sora, Veo, Midjourney) are "
                    "rapidly closing the detectable gap. By 2027, passive detection will likely "
                    "be unreliable for high-quality synthetic media.\n\n"
                    "Policy Recommendations:\n\n"
                    "1. Mandatory content provenance: Require that all AI-generated content "
                    "include C2PA-compliant digital signatures encoding model identity, "
                    "generation timestamp, and disclosure statements. This is analogous to "
                    "food labeling requirements — it doesn't prevent creation but ensures "
                    "transparency.\n\n"
                    "2. Platform API requirements: Require that major social media platforms "
                    "expose content provenance metadata to users via standardized APIs, "
                    "similar to existing fact-check labeling systems.\n\n"
                    "3. Criminal penalties for malicious uses: Create specific criminal "
                    "offenses for generating synthetic media with intent to defraud, "
                    "disrupt elections, or cause reputational harm.\n\n"
                    "4. Research funding: Allocate $50M annually for detection research "
                    "and digital forensics tooling.\n\n"
                    "Enforcement challenges: Cross-jurisdictional generation, open-source "
                    "models without provenance tracking, and the ease of stripping metadata "
                    "from media files limit enforcement options. A purely technical solution "
                    "is insufficient; media literacy education and public awareness campaigns "
                    "are essential complements."
                ),
                [],
            ),
            (
                "Write meeting minutes for a sprint retrospective. Include action items.",
                (
                    "Sprint 14 Retrospective Minutes\n\n"
                    "Date: April 24, 2026\n"
                    "Attendees: Sarah (EM), Mike, Priya, Alex, Jordan, Lisa\n"
                    "Duration: 60 minutes\n\n"
                    "What Went Well:\n"
                    "- Shipped search autocomplete feature on time (Mike, Priya)\n"
                    "- Reduced CI pipeline from 18min to 9min (Alex)\n"
                    "- Code review turnaround time improved to <4 hours\n"
                    "- Better test coverage in new features (+12% this sprint)\n\n"
                    "What Could Improve:\n"
                    "- Sprint planning: 3 stories carried over due to underestimation\n"
                    "- Communication: API changes weren't communicated to mobile team, "
                    "causing integration delay\n"
                    "- Flaky E2E tests: 4 of 7 CI failures were test flakiness, not real bugs\n\n"
                    "Action Items:\n"
                    "1. All: Break stories over 5 story points into smaller items (starting sprint 15)\n"
                    "2. Mike: Document API schema changes in shared Slack channel before implementation\n"
                    "3. Alex: Investigate and fix flaky E2E tests (deadline: next Friday)\n"
                    "4. Jordan: Propose test parallelization strategy for CI pipeline\n"
                    "5. Sarah: Share updated estimation guidelines during sprint planning\n\n"
                    "Next Retrospective: May 8, 2026, 2:00 PM.\n"
                    "Retro master: Priya (rotating role)."
                ),
                [],
            ),
            (
                "Write FAQ document for a new feature that allows users to run custom SQL "
                "queries on their data. Address common concerns.",
                (
                    "Custom SQL Queries — Frequently Asked Questions\n\n"
                    "Q: What is the Custom SQL feature?\n"
                    "A: Custom SQL allows you to query your organization's data using "
                    "standard SQL syntax. You can explore, aggregate, and export data "
                    "beyond what's available in the pre-built dashboards.\n\n"
                    "Q: Who can use Custom SQL?\n"
                    "A: This feature is available on the Business plan and above. "
                    "Within your organization, only users with the 'Analyst' role or higher "
                    "can execute custom queries. Admins control access in Workspace Settings.\n\n"
                    "Q: What SQL dialect is supported?\n"
                    "A: We support PostgreSQL-compatible syntax. SELECT, JOIN, GROUP BY, "
                    "HAVING, subqueries, window functions, and common table expressions (CTEs) "
                    "are all supported. DDL and DML statements (INSERT, UPDATE, DELETE, "
                    "CREATE TABLE) are not permitted.\n\n"
                    "Q: Can I accidentally modify or delete data?\n"
                    "A: No. Custom SQL queries are executed in a read-only transaction. "
                    "Data-modifying statements are rejected at the query parser level.\n\n"
                    "Q: What about query performance?\n"
                    "A: Queries run on a dedicated replica, so your custom queries won't "
                    "affect production performance. Each query has a 60-second timeout "
                    "and a 100,000-row result limit. For larger analyses, we recommend "
                    "exporting to your data warehouse.\n\n"
                    "Q: Can I save my queries?\n"
                    "A: Yes. Saved queries are available to all users in your workspace. "
                    "You can organize them into folders and share them with your team.\n\n"
                    "Q: How does this impact my data usage limits?\n"
                    "A: Each query scanned rows count toward your workspace's monthly "
                    "compute budget. You can track usage in the Billing section."
                ),
                [],
            ),
            (
                "Write release notes for a mobile app version 3.0 major update.",
                (
                    "TaskFlow Mobile 3.0 — Now Available\n\n"
                    "We're excited to announce TaskFlow 3.0, our biggest update ever. "
                    "This release focuses on speed, personalization, and offline capabilities.\n\n"
                    "What's New:\n\n"
                    "Offline Mode\n"
                    "Work without internet. All your projects, tasks, and documents are "
                    "available offline. Changes sync automatically when you reconnect. "
                    "No conflict resolution needed — we use CRDTs under the hood.\n\n"
                    "AI-Powered Smart Suggestions\n"
                    "Tap the sparkle icon to get context-aware suggestions: "
                    "auto-complete task descriptions, generate subtask breakdowns, "
                    "and draft update messages for your team. Powered on-device — "
                    "your data never leaves your phone.\n\n"
                    "Customizable Home Screen\n"
                    "Pin your most-used projects, create custom widgets, and choose "
                    "between list, board, or calendar view as your default. Long-press "
                    "any element to customize.\n\n"
                    "Performance Improvements\n"
                    "App launch time reduced by 60%. Scrolling is now 120fps smooth. "
                    "Push notifications delivered 40% faster.\n\n"
                    "Other Improvements:\n"
                    "- New dark mode with true black option for OLED screens\n"
                    "- Quick action gestures (swipe left to complete, right to snooze)\n"
                    "- Keyboard shortcuts for iPad (Cmd+N for new task, Cmd+F for search)\n"
                    "- Accessibility improvements: VoiceOver support for all screens\n\n"
                    "Bug Fixes:\n"
                    "- Fixed: Notification badges not clearing after reading\n"
                    "- Fixed: App crash when attaching large images (>5MB)\n"
                    "- Fixed: Calendar sync duplicates for recurring events\n"
                    "- Fixed: Search not indexing recently created tasks\n\n"
                    "Thank you to everyone who participated in the beta! Your feedback "
                    "shaped this release.\n\n"
                    "Download or update now from the App Store."
                ),
                [],
            ),
        ]
        self.reasoning_templates = [
            (
                "Explain the Monty Hall problem. Why is switching doors a 2/3 probability "
                "of winning and not 1/2? Walk through it step by step.",
                (
                    "Initially, your chance of picking the car is 1/3. The chance the car is behind "
                    "one of the other two doors is 2/3.\n\n"
                    "The host always opens a door with a goat from the two you didn't pick. This is "
                    "the crucial constraint - he never reveals the car.\n\n"
                    "If you initially picked wrong (2/3 probability), the remaining unopened door "
                    "must have the car. The host's elimination of the other goat guarantees this.\n\n"
                    "If you initially picked right (1/3), switching loses.\n\n"
                    "So: Stay wins 1/3, Switch wins 2/3. The 2/3 comes from the fact that the "
                    "host's knowledge transforms the 2/3 probability from 'behind one of the other "
                    "two doors' into 'behind that specific remaining door'."
                ),
                [
                    (
                        "What if there were 100 doors and the host opens 98 goats?",
                        (
                            "With 100 doors:\n"
                            "1. Initial pick: 1/100 chance of car\n"
                            "2. Host opens 98 doors with goats (never the car)\n"
                            "3. Switch probability = 99/100\n\n"
                            "This makes the paradox intuitive: your initial pick is almost "
                            "certainly wrong (99/100), and the host has conveniently eliminated "
                            "every other wrong door except one. Formally:\n"
                            "- P(initial correct) = 1/N\n"
                            "- P(car behind remaining) = (N-1)/N\n\n"
                            "As N grows, switching becomes increasingly obvious."
                        ),
                    ),
                    (
                        "What if the host picks a random door and happens to reveal a goat?",
                        (
                            "This is the 'Monty Fall' variant. If the host picks randomly:\n"
                            "P(switch wins | random goat) = 1/2\n\n"
                            "Enumerate (you pick door 1):\n"
                            "1. Car behind 1 (1/3): Host reveals either goat (2/2)\n"
                            "2. Car behind 2 (1/3): Host reveals 3 (1/2 goat chance)\n"
                            "3. Car behind 3 (1/3): Host reveals 2 (1/2 goat chance)\n\n"
                            "P(car behind 1 | random goat) = (1/3)/(1/3+1/6+1/6) = 1/2\n"
                            "The host's knowledge vs randomness is what creates the 2/3 vs 1/2 gap."
                        ),
                    ),
                ],
            ),
            (
                "A farmer needs to cross a river with a wolf, goat, and cabbage. The boat carries "
                "him and one item. Wolf eats goat if left alone; goat eats cabbage. Solve it step by step.",
                (
                    "The classic river crossing puzzle:\n\n"
                    "Step 1: Farmer takes Goat to RIGHT. LEFT: W,C | RIGHT: F,G. Safe.\n"
                    "Step 2: Farmer returns ALONE. LEFT: F,W,C | RIGHT: G. Safe.\n"
                    "Step 3: Farmer takes Wolf to RIGHT. LEFT: C | RIGHT: F,W,G. "
                    "Momentarily safe (Farmer is there).\n"
                    "Step 4: Farmer takes Goat BACK to LEFT. LEFT: F,G,C | RIGHT: W. Safe.\n"
                    "Step 5: Farmer takes Cabbage to RIGHT. LEFT: G | RIGHT: F,W,C. Safe.\n"
                    "Step 6: Farmer returns ALONE. LEFT: F,G | RIGHT: W,C. Safe.\n"
                    "Step 7: Farmer takes Goat to RIGHT. LEFT: | RIGHT: F,W,G,C. All safe!\n\n"
                    "The key insight: the goat must be shuttled back and forth because it "
                    "conflicts with both the wolf and the cabbage."
                ),
                [
                    (
                        "What if the boat can carry two items? What's the minimum trips?",
                        (
                            "With capacity for farmer + 2 items:\n"
                            "1. Take Goat + Cabbage to RIGHT. LEFT: Wolf | RIGHT: F,G,C. Safe.\n"
                            "2. Return alone to LEFT. LEFT: F,Wolf | RIGHT: G,C. Safe.\n"
                            "3. Take Wolf to RIGHT. All across!\n\n"
                            "Only 3 one-way trips (5 bank transitions). The increased capacity "
                            "eliminates the need to shuttle the goat back."
                        ),
                    ),
                ],
            ),
            (
                "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. "
                "How much does the ball cost? Show your reasoning.",
                (
                    "Let x = ball cost, y = bat cost.\n"
                    "x + y = 1.10\n"
                    "y = x + 1.00\n\n"
                    "Substitute: x + (x + 1.00) = 1.10\n"
                    "2x = 0.10\n"
                    "x = 0.05\n\n"
                    "The ball costs $0.05 and the bat costs $1.05.\n"
                    "Check: $1.05 + $0.05 = $1.10. And $1.05 - $0.05 = $1.00. Correct.\n\n"
                    "The intuitive answer of $0.10 is wrong because while bat+ball=$1.10, "
                    "the bat would only be $0.90 more than the ball, not $1.00 more."
                ),
                [
                    (
                        "Generalize: if total = T and bat is D more, express ball cost.",
                        (
                            "Let x = ball, y = bat:\n"
                            "x + y = T\n"
                            "y = x + D\n"
                            "x + (x + D) = T\n"
                            "2x = T - D\n"
                            "x = (T - D) / 2\n\n"
                            "For the original: T=1.10, D=1.00, x=(1.10-1.00)/2=0.05.\n"
                            "The intuitive answer of 0.10 comes from solving x = T - D "
                            "(forgetting the division by 2)."
                        ),
                    ),
                ],
            ),
            (
                "Solve Einstein's Riddle: five houses, five colors, five nationalities, "
                "five drinks, five smokes, five pets. Who owns the fish?",
                (
                    "Using logical deduction step by step:\n\n"
                    "House 1: Yellow, Norwegian, water, Dunhill, cats\n"
                    "House 2: Blue, Dane, tea, Blends, horses\n"
                    "House 3: Red, Brit, milk, Pall Mall, birds\n"
                    "House 4: Green, German, coffee, Prince, FISH\n"
                    "House 5: White, Swede, beer, Blue Master, dogs\n\n"
                    "The German owns the fish. This satisfies all 15 clues:\n"
                    "1. Brit in red (3) 2. Swede has dogs (5) 3. Dane drinks tea (2)\n"
                    "4. Green left of white (4,5) 5. Green drinks coffee (4)\n"
                    "6. Pall Mall has birds (3) 7. Yellow smokes Dunhill (1)\n"
                    "8. Center drinks milk (3) 9. Norwegian first (1)\n"
                    "10. Blends next to cats (2 next to 1)\n"
                    "11. Horses next to Dunhill (2 next to 1)\n"
                    "12. Blue Master drinks beer (5)\n"
                    "13. German smokes Prince (4)\n"
                    "14. Norwegian next to blue (1 next to 2)\n"
                    "15. Blends has water-drinking neighbor (2 next to 1)"
                ),
                [],
            ),
            (
                "Two trains leave stations 200 miles apart at 60 mph and 40 mph heading toward "
                "each other. When do they meet and how far from each station?",
                (
                    "Combined speed: 60 + 40 = 100 mph\n"
                    "Time to meet = 200 / 100 = 2 hours\n\n"
                    "From Station A: 60 * 2 = 120 miles\n"
                    "From Station B: 40 * 2 = 80 miles\n\n"
                    "Check: 120 + 80 = 200. Correct.\n\n"
                    "General formula for two objects moving toward each other:\n"
                    "t = D / (v1 + v2)\n"
                    "d1 = v1 * D / (v1 + v2)"
                ),
                [
                    (
                        "A bird flies back and forth between them at 80 mph. "
                        "What total distance does it cover before they meet?",
                        (
                            "The elegant solution: The trains meet in 2 hours. The bird "
                            "flies at 80 mph for the entire 2 hours.\n"
                            "Distance = 80 * 2 = 160 miles.\n\n"
                            "This is von Neumann's fly puzzle. The infinite series of "
                            "decreasing distances also sums to 160 miles.\n"
                            "Von Neumann reportedly solved the series in seconds and said "
                            "'There's a shortcut? I just summed the infinite series.'"
                        ),
                    ),
                ],
            ),
            (
                "Using a 3-gallon and 5-gallon jug, measure exactly 4 gallons. "
                "Provide the step-by-step solution.",
                (
                    "Solution tracking (3-gal, 5-gal):\n\n"
                    "1. Fill 5-gallon jug. (0, 5)\n"
                    "2. Pour 5 into 3 until 3 is full. (3, 2)\n"
                    "3. Empty 3-gallon jug. (0, 2)\n"
                    "4. Pour remaining 2 from 5 into 3. (2, 0)\n"
                    "5. Fill 5-gallon jug. (2, 5)\n"
                    "6. Pour from 5 into 3 until full (needs 1). (3, 4)\n\n"
                    "5-gallon jug has exactly 4 gallons.\n\n"
                    "Alternative: fill 3, pour to 5, fill 3, pour to 5 until full (leaving 1 in 3), "
                    "empty 5, pour 1 from 3 to 5, fill 3, pour to 5 -> (0, 4).\n\n"
                    "Solvable because gcd(3,5)=1 divides 4."
                ),
                [
                    (
                        "Given jugs of size a and b, when can you measure c gallons?",
                        (
                            "You can measure c iff c <= max(a,b), c >= 0, and "
                            "c is a multiple of gcd(a,b).\n\n"
                            "This follows from Bezout's identity: ax + by = gcd(a,b)\n"
                            "has integer solutions for x, y (positive = fill, negative = empty).\n\n"
                            "BFS finds shortest solution:\n"
                            "```python\n"
                            "from collections import deque\n"
                            "def shortest(a, b, target):\n"
                            "    seen = {(0,0)}\n"
                            "    q = deque([(0,0,[])])\n"
                            "    while q:\n"
                            "        x, y, path = q.popleft()\n"
                            "        if x==target or y==target: return path\n"
                            "        for nx, ny in [(a,y),(x,b),(0,y),(x,0),\n"
                            "            (x-min(x,b-y), y+min(x,b-y)),\n"
                            "            (x+min(y,a-x), y-min(y,a-x))]:\n"
                            "            if (nx,ny) not in seen:\n"
                            "                seen.add((nx,ny))\n"
                            "                q.append((nx,ny,path+[(x,y)->(nx,ny)]))\n"
                            "    return None\n"
                            "```"
                        ),
                    ),
                ],
            ),
            (
                "Twelve coins, one counterfeit (heavier or lighter). Find it in 3 weighings "
                "on a balance scale. Explain your strategy.",
                (
                    "Divide into groups A, B, C of 4 coins each.\n\n"
                    "Weighing 1: A vs B.\n"
                    "- Balanced: Counterfeit in C. Weighing 2: 3 from C vs 3 known-good. "
                    "If balanced, 4th C is counterfeit (weigh vs good to check heavy/light). "
                    "If C side heavy, counterfeit is heavy among those 3 (weigh 2 of them). "
                    "If C side light, counterfeit is light among those 3.\n\n"
                    "- A heavier: Either A has a heavy coin or B has a light coin.\n"
                    "  Weighing 2: (A1,A2,B1) vs (A3,good,good).\n"
                    "  * Balanced: Counterfeit is A4 (heavy) or B2/B3/B4 (light). "
                    "Weigh B2 vs B3: if balanced, A4 is heavy; "
                    "if one lighter, that's the light counterfeit.\n"
                    "  * Left heavier: A1 or A2 is heavy, or A3 is light. "
                    "Weigh A1 vs A2: if one heavier, that's it; if balanced, A3 is light.\n"
                    "  * Left lighter: B1 must be light. Confirm vs good coin.\n\n"
                    "- B heavier: Symmetric to A heavier.\n\n"
                    "3 weighings = 3^3 = 27 outcomes, enough for 12*2+1 = 25 possibilities."
                ),
                [
                    (
                        "How many weighings for N coins?",
                        (
                            "Minimum weighings k satisfies: 3^k >= 2N + 1\n\n"
                            "Each weighing has 3 outcomes. Over k weighings, 3^k result sequences. "
                            "We need to distinguish 2N possibilities (each coin heavy or light) "
                            "plus 1 (all genuine).\n\n"
                            "k=1: 3^1=3, N<=1\n"
                            "k=2: 3^2=9, N<=4\n"
                            "k=3: 3^3=27, N<=13\n"
                            "k=4: 3^4=81, N<=40\n\n"
                            "For 39 coins, 4 weighings suffice. The general bound "
                            "comes from information theory: each weighing provides "
                            "log2(3) ~ 1.58 bits of information."
                        ),
                    ),
                ],
            ),
            (
                "You're in a room with three light switches, each controlling one of three "
                "light bulbs in another room. You can only enter the other room once. "
                "How do you figure out which switch controls which bulb?",
                (
                    "Solution:\n\n"
                    "1. Flip switch 1 ON.\n"
                    "2. Flip switch 2 ON for 5 minutes, then turn it OFF.\n"
                    "3. Leave switch 3 OFF.\n"
                    "4. Enter the room.\n\n"
                    "Now:\n"
                    "- The bulb that is ON is controlled by switch 1.\n"
                    "- The bulb that is OFF but WARM is controlled by switch 2.\n"
                    "- The bulb that is OFF and COLD is controlled by switch 3.\n\n"
                    "This uses the physical property that light bulbs heat up when "
                    "energized, and retain heat for some time after being turned off. "
                    "The single entry constraint is bypassed by encoding information "
                    "in both the on/off state AND the temperature state."
                ),
                [],
            ),
            (
                "You have two ropes that each burn for exactly 60 minutes. They burn "
                "unevenly (not at constant rate). How do you measure exactly 45 minutes?",
                (
                    "Solution:\n\n"
                    "1. Light Rope A at BOTH ends AND Rope B at ONE end simultaneously.\n\n"
                    "2. Rope A, burning from both ends, will be consumed in exactly 30 minutes "
                    "(since each end burns toward the middle, halving the total time).\n\n"
                    "3. At the moment Rope A is completely burned (30 minutes elapsed), "
                    "Rope B has 30 minutes of burn time remaining (since it has been "
                    "burning from one end for 30 minutes).\n\n"
                    "4. Immediately light the SECOND end of Rope B.\n\n"
                    "5. Rope B, now burning from both ends, will be consumed in 15 more minutes "
                    "(halving its remaining 30-minute burn time).\n\n"
                    "6. Total time from start: 30 + 15 = 45 minutes.\n\n"
                    "This works because lighting from both ends effectively halves the burn time "
                    "regardless of how unevenly the rope burns."
                ),
                [],
            ),
            (
                "Prove that the square root of 2 is irrational.",
                (
                    "Proof by contradiction:\n\n"
                    "Assume sqrt(2) = p/q in lowest terms (p, q integers with gcd(p,q)=1).\n\n"
                    "Square both sides: 2 = p^2 / q^2\n"
                    "So: p^2 = 2 * q^2\n\n"
                    "Therefore p^2 is even, so p must be even (since odd^2 = odd).\n"
                    "Let p = 2k for some integer k.\n\n"
                    "Substitute: (2k)^2 = 2 * q^2\n"
                    "4k^2 = 2 * q^2\n"
                    "2k^2 = q^2\n\n"
                    "Therefore q^2 is even, so q must be even.\n\n"
                    "But if both p and q are even, they share a factor of 2, "
                    "contradicting that p/q was in lowest terms.\n\n"
                    "Therefore our assumption was false. sqrt(2) is irrational.\n\n"
                    "This proof, attributed to the Pythagoreans, shows that not all "
                    "numbers can be expressed as ratios of integers."
                ),
                [
                    (
                        "Does this generalize to sqrt(3), sqrt(5), sqrt(6)?",
                        (
                            "Yes, for any integer n that is not a perfect square, "
                            "sqrt(n) is irrational.\n\n"
                            "The proof generalizes: assume sqrt(n) = p/q in lowest terms.\n"
                            "Then p^2 = n * q^2.\n\n"
                            "The key step: show that any prime factor of n must divide p, "
                            "and then show it must divide q, contradicting reduced form.\n\n"
                            "For sqrt(3): p^2 = 3q^2, so p is divisible by 3, "
                            "p=3k, 9k^2=3q^2, 3k^2=q^2, so q is divisible by 3. Contradiction.\n\n"
                            "For sqrt(4) = 2: 4 is a perfect square, so no contradiction "
                            "(p^2 = 4q^2, p=2q works: gcd(p,q) can still be 1 if q=1).\n\n"
                            "More generally: sqrt(n) is rational iff n is a perfect square."
                        ),
                    ),
                ],
            ),
            (
                "You have 100 floors and 2 eggs. Eggs break when dropped from certain floor F "
                "or above. Find F with minimum worst-case drops.",
                (
                    "This is the classic egg drop problem.\n\n"
                    "The optimal strategy uses decreasing intervals:\n\n"
                    "Drop the first egg from floor x. If it breaks, use the second egg "
                    "to test floors 1 through x-1 linearly (x-1 drops max).\n"
                    "If it survives, go to floor x + (x-1), etc.\n\n"
                    "We want: x + (x-1) + (x-2) + ... + 1 >= 100\n"
                    "x(x+1)/2 >= 100\n"
                    "x^2 + x - 200 >= 0\n"
                    "x >= 14 (approximately)\n\n"
                    "Strategy:\n"
                    "- Drop from floor 14. If breaks, test 1-13 linearly (13 drops). Total: 14.\n"
                    "- If survives, drop from 27 (14+13). If breaks, test 15-26 (12 drops). Total: 14.\n"
                    "- Continue: 39, 50, 60, 69, 77, 84, 90, 95, 99, 100.\n\n"
                    "Worst case: 14 drops.\n\n"
                    "This balances the risk: if the egg breaks early, the linear search "
                    "is short; if it survives long, the remaining interval shrinks."
                ),
                [
                    (
                        "Generalize to E eggs and N floors. What's the DP recurrence?",
                        (
                            "Let dp[e][f] = minimum drops needed with e eggs and f floors.\n\n"
                            "dp[e][f] = 1 + min over k=1..f of max(dp[e-1][k-1], dp[e][f-k])\n\n"
                            "Explanation: drop from floor k.\n"
                            "- If egg breaks: e-1 eggs left, need to test k-1 floors below.\n"
                            "- If egg survives: still e eggs, need to test f-k floors above.\n"
                            "We take the worst case (max) and minimize over k.\n\n"
                            "Base cases:\n"
                            "dp[0][f] = infinity (can't test with 0 eggs)\n"
                            "dp[e][0] = 0 (no floors to test)\n"
                            "dp[1][f] = f (linear search with 1 egg)\n"
                            "dp[e][1] = 1 (one floor, one drop)\n\n"
                            "For 2 eggs, the optimal k follows the triangular number pattern. "
                            "For 3+ eggs, the strategy gets closer to binary search."
                        ),
                    ),
                ],
            ),
            (
                "There are 25 horses and a racetrack that can race 5 horses at a time. "
                "Find the top 3 fastest horses with minimum races.",
                (
                    "Solution: 7 races.\n\n"
                    "Round 1: 5 races of 5 horses each (races 1-5).\n"
                    "Record the order within each race. A1 > A2 > A3 > A4 > A5, etc.\n\n"
                    "Round 2: Race the 5 winners (race 6).\n"
                    "Let's say: A > B > C > D > E.\n\n"
                    "Now:\n"
                    "- A is the fastest overall.\n"
                    "- For 2nd place: could be A2 or B.\n"
                    "- For 3rd place: could be A2, A3, B, B2, or C.\n\n"
                    "Round 3: Race A2, A3, B, B2, C (race 7).\n"
                    "Top 2 from this race are 2nd and 3rd overall.\n\n"
                    "Why?\n"
                    "- D and E are eliminated (they can't be top 3 since 3 horses beat them).\n"
                    "- D2, D3, D4, D5 are eliminated (4 horses ahead of them).\n"
                    "- E2-E5 similarly eliminated.\n"
                    "- C2-C5 eliminated (C, A, B ahead).\n"
                    "- B3-B5 eliminated (A, B, B2 ahead).\n"
                    "- A4-A5 eliminated (A, A2, A3 ahead).\n\n"
                    "Only 5 candidates remain for positions 2 and 3. One race decides."
                ),
                [],
            ),
            (
                "Determine the missing number in this sequence: 1, 11, 21, 1211, 111221, ?",
                (
                    "This is the look-and-say sequence:\n\n"
                    "1 = one 1 = 11\n"
                    "11 = two 1s = 21\n"
                    "21 = one 2, one 1 = 1211\n"
                    "1211 = one 1, one 2, two 1s = 111221\n"
                    "111221 = three 1s, two 2s, one 1 = 312211\n\n"
                    "The next term is 312211.\n\n"
                    "Each term describes the previous term by counting consecutive digits. "
                    "This sequence was popularized by Conway, who showed that its growth "
                    "rate approaches a constant (the Conway constant, ~1.3036).\n\n"
                    "The sequence never contains digits other than 1, 2, and 3 "
                    "(after the first few terms)."
                ),
                [],
            ),
            (
                "A truth-teller always tells the truth, a liar always lies. You meet two people. "
                "A says 'B is a liar.' B says 'We are both truth-tellers.' "
                "What are A and B?",
                (
                    "Let's analyze logically:\n\n"
                    "Case 1: A is truth-teller.\n"
                    "Then A's statement 'B is a liar' is true. So B is a liar.\n"
                    "B says 'We are both truth-tellers.' Since B is a liar, this must be false. "
                    "And indeed it is false (A is truth-teller, B is liar - they are not both truth-tellers). "
                    "Consistent.\n\n"
                    "Case 2: A is liar.\n"
                    "Then A's statement 'B is a liar' is false. So B is truth-teller.\n"
                    "B says 'We are both truth-tellers.' Since B is truth-teller, this must be true. "
                    "But we assumed A is a liar, so 'both truth-tellers' is false. "
                    "Contradiction.\n\n"
                    "Therefore: A is a truth-teller, B is a liar.\n\n"
                    "This is a classic knights (truth) and knaves (lies) puzzle. "
                    "The solution uses case analysis to check for contradictions."
                ),
                [
                    (
                        "A says 'B is a liar.' B says 'A is a liar.' What are they?",
                        (
                            "Case 1: A truth-teller -> 'B is liar' true -> B is liar.\n"
                            "Then B (liar) says 'A is liar' which is false (A is truth-teller). "
                            "False statement from liar = consistent.\n\n"
                            "Case 2: A liar -> 'B is liar' false -> B is truth-teller.\n"
                            "Then B (truth-teller) says 'A is liar' which would be true if A is liar. "
                            "Consistent.\n\n"
                            "Both cases are logically consistent! This means the puzzle "
                            "is underdetermined - there are two possible solutions:\n"
                            "- (Truth-teller, Liar)\n"
                            "- (Liar, Truth-teller)\n\n"
                            "You'd need additional information to distinguish them."
                        ),
                    ),
                ],
            ),
            (
                "You flip a fair coin 10 times. What's the probability of getting exactly 5 heads? "
                "What about at least 3 heads in a row?",
                (
                    "Exactly 5 heads:\n"
                    "Number of ways: C(10,5) = 252\n"
                    "Total outcomes: 2^10 = 1024\n"
                    "P(exactly 5) = 252/1024 = 63/256 ~ 0.246\n\n"
                    "At least 3 heads in a row:\n"
                    "This is easier to compute via complementary counting or recurrence.\n\n"
                    "Let f(n) = number of length-n sequences with no 3 consecutive heads.\n"
                    "f(1) = 2 (H, T)\n"
                    "f(2) = 4 (HH, HT, TH, TT)\n"
                    "f(3) = 7 (all 8 except HHH)\n"
                    "Recurrence: f(n) = f(n-1) + f(n-2) + f(n-3)\n"
                    "Because a valid sequence ends in T (f(n-1)), HT (f(n-2)), or HHT (f(n-3)).\n\n"
                    "f(4) = 7+4+2 = 13\n"
                    "f(5) = 13+7+4 = 24\n"
                    "f(6) = 24+13+7 = 44\n"
                    "f(7) = 44+24+13 = 81\n"
                    "f(8) = 81+44+24 = 149\n"
                    "f(9) = 149+81+44 = 274\n"
                    "f(10) = 274+149+81 = 504\n\n"
                    "Sequences with at least 3 in a row: 1024 - 504 = 520\n"
                    "Probability: 520/1024 = 65/128 ~ 0.508"
                ),
                [],
            ),
            (
                "What day of the week was January 1, 2000? Use Zeller's congruence or similar. "
                "Explain how it works.",
                (
                    "Using Zeller's congruence:\n\n"
                    "For January, adjust: month=13, year=1999 (since Jan/Feb are months 13,14 of previous year).\n\n"
                    "h = (q + floor((13*(m+1))/5) + K + floor(K/4) + floor(J/4) - 2J) mod 7\n\n"
                    "Where:\n"
                    "q = day of month = 1\n"
                    "m = month = 13 (January)\n"
                    "K = year mod 100 = 99\n"
                    "J = century = 19\n\n"
                    "h = (1 + floor(14*13/5) + 99 + floor(99/4) + floor(19/4) - 2*19) mod 7\n"
                    "h = (1 + floor(182/5) + 99 + 24 + 4 - 38) mod 7\n"
                    "h = (1 + 36 + 99 + 24 + 4 - 38) mod 7\n"
                    "h = (126) mod 7\n"
                    "h = 0 (when 0 = Saturday)\n\n"
                    "January 1, 2000 was a Saturday.\n\n"
                    "In Zeller's system: 0=Sat, 1=Sun, 2=Mon, ..., 6=Fri.\n"
                    "This formula works for the Gregorian calendar."
                ),
                [],
            ),
            (
                "A car travels up a hill at 30 mph and down the same hill at 60 mph. "
                "What's the average speed for the round trip?",
                (
                    "The intuitive but wrong answer is 45 mph. Let's compute properly.\n\n"
                    "Let the one-way distance be D miles.\n\n"
                    "Time up: D / 30\n"
                    "Time down: D / 60\n"
                    "Total time: D/30 + D/60 = 2D/60 + D/60 = 3D/60 = D/20\n"
                    "Total distance: 2D\n\n"
                    "Average speed = total distance / total time = 2D / (D/20) = 40 mph.\n\n"
                    "The average speed is 40 mph, not 45 mph, because you spend more time "
                    "at the slower speed. This is the harmonic mean of 30 and 60:\n"
                    "Harmonic mean = 2 / (1/30 + 1/60) = 2 / (3/60) = 40\n\n"
                    "For two speeds v1 and v2 over equal distances, the average speed "
                    "is the harmonic mean: 2*v1*v2/(v1+v2), not the arithmetic mean (v1+v2)/2."
                ),
                [
                    (
                        "What if you travel distance D at v1 and distance D at v2, "
                        "but the speeds are maintained for equal TIME instead of equal distance?",
                        (
                            "For equal times:\n"
                            "Total distance = v1*t + v2*t = t(v1+v2)\n"
                            "Total time = 2t\n"
                            "Average speed = t(v1+v2)/(2t) = (v1+v2)/2\n\n"
                            "This IS the arithmetic mean because you spend equal time "
                            "at each speed.\n\n"
                            "Key insight:\n"
                            "- Equal DISTANCE: use harmonic mean\n"
                            "- Equal TIME: use arithmetic mean\n\n"
                            "The choice depends on what's held constant. In the original "
                            "problem, distance is the same both ways, so harmonic mean applies."
                        ),
                    ),
                ],
            ),
            (
                "There are three doors. Behind one is a car; behind the other two are goats. "
                "You pick a door. The host opens a different door revealing a goat. "
                "This is the Monty Hall problem, but now there are 4 doors. "
                "The host opens 2 goats (not your pick). Should you switch? What's the probability?",
                (
                    "With 4 doors:\n\n"
                    "Initial pick: P(car) = 1/4\n"
                    "P(car behind one of the other 3) = 3/4\n\n"
                    "Host opens 2 doors with goats from the 3 you didn't pick.\n"
                    "This leaves exactly 1 unopened door among the 3.\n\n"
                    "If you initially picked wrong (probability 3/4), the remaining "
                    "unopened door MUST have the car (since the host has eliminated "
                    "all other goats).\n\n"
                    "If you initially picked right (probability 1/4), switching loses.\n\n"
                    "So: P(win by switching) = 3/4\n"
                    "    P(win by staying) = 1/4\n\n"
                    "General formula with N doors and host opening N-2 goats:\n"
                    "P(switch wins) = (N-1)/N\n"
                    "P(stay wins) = 1/N\n\n"
                    "As N increases, switching becomes increasingly advantageous."
                ),
                [],
            ),
            (
                "A snail climbs 3 feet up a wall during the day and slips 2 feet down at night. "
                "The wall is 30 feet high. How many days does it take to reach the top?",
                (
                    "Net progress per full day: +3 - 2 = +1 foot.\n\n"
                    "But careful: on the LAST day, the snail reaches the top DURING the day "
                    "and doesn't slip back.\n\n"
                    "After day 27: snail is at 27 feet (net 1 per day for 27 days).\n"
                    "Day 28: snail climbs 3 feet -> 27 + 3 = 30 feet. REACHES THE TOP!\n\n"
                    "It takes 28 days.\n\n"
                    "The trap is to simply compute 30/(3-2) = 30 days. That neglects the "
                    "fact that the last day's climb reaches the goal before the slip.\n\n"
                    "General formula for climb c, slip s, height h:\n"
                    "Days = 1 + (h - c - 1) // (c - s) + 1 = ceil((h - c) / (c - s)) + 1\n"
                    "For c=3, s=2, h=30: ceil((30-3)/1) + 1 = ceil(27) + 1... wait,\n"
                    "Let me recalculate: days = ceil((h - c) / (c - s)) + 1 = ceil(27/1) + 1 = 27 + 1 = 28. Correct.\n\n"
                    "Equivalent: days = 1 + ceil((h - c) / (c - s)) if h > c."
                ),
                [],
            ),
            (
                "A clock shows 3:15. What's the angle between the hour and minute hand?",
                (
                    "At 3:00, the hour hand points at 3 (90 degrees from 12).\n"
                    "The minute hand points at 12 (0 degrees).\n\n"
                    "In 15 minutes:\n"
                    "- Minute hand moves: 15 * 6 = 90 degrees (points at 3).\n"
                    "- Hour hand moves: 15 * 0.5 = 7.5 degrees (since hour hand moves 0.5 degrees per minute).\n\n"
                    "So at 3:15:\n"
                    "- Minute hand: exactly at 3 (90 degrees)\n"
                    "- Hour hand: at 90 + 7.5 = 97.5 degrees\n\n"
                    "Angle = |97.5 - 90| = 7.5 degrees.\n\n"
                    "General formula for time H hours and M minutes:\n"
                    "Hour angle = 30*H + 0.5*M\n"
                    "Minute angle = 6*M\n"
                    "Angle = |30*H - 5.5*M|, taking the smaller angle (<= 180).\n\n"
                    "For 3:15: |30*3 - 5.5*15| = |90 - 82.5| = 7.5 degrees."
                ),
                [],
            ),
            (
                "If it takes 5 machines 5 minutes to make 5 widgets, how long would "
                "it take 100 machines to make 100 widgets?",
                (
                    "Each machine produces 1 widget in 5 minutes.\n\n"
                    "5 machines in 5 minutes make 5 widgets means:\n"
                    "Rate = 5 widgets / 5 minutes = 1 widget per minute (for 5 machines combined).\n"
                    "Or: 1 machine makes 1 widget in 5 minutes.\n\n"
                    "100 machines making 100 widgets:\n"
                    "Rate with 100 machines = 100 / 5 = 20 widgets per minute.\n"
                    "Time for 100 widgets = 100 / 20 = 5 minutes.\n\n"
                    "Answer: 5 minutes.\n\n"
                    "The intuitive but wrong answer is 100 minutes. The trap is to "
                    "confuse proportional reasoning: if 5 machines make 5 widgets in "
                    "5 minutes, then 1 machine makes 1 widget in 5 minutes. "
                    "So 100 machines make 100 widgets in 5 minutes as well "
                    "(each machine makes 1 widget in 5 minutes, and they all work in parallel)."
                ),
                [
                    (
                        "What if we have 7 machines and need 49 widgets? How long?",
                        (
                            "1 machine makes 1 widget in 5 minutes.\n"
                            "7 machines make 7 widgets in 5 minutes.\n\n"
                            "To make 49 widgets:\n"
                            "49 widgets / 7 machines = 7 widgets per machine.\n"
                            "Each machine takes 5 minutes per widget, so:\n"
                            "7 widgets * 5 minutes = 35 minutes.\n\n"
                            "Alternatively: rate = 7/5 widgets/minute.\n"
                            "Time = 49 / (7/5) = 49 * 5/7 = 35 minutes.\n\n"
                            "General formula: time = (widgets * 5) / machines."
                        ),
                    ),
                ],
            ),
            (
                "You have 8 identical balls, one is slightly heavier. Using a balance scale, "
                "find the heavier ball in minimum weighings.",
                (
                    "Use 2 weighings (optimal since 3^2 = 9 >= 8).\n\n"
                    "Weighing 1: Put 3 balls on left, 3 on right, 2 aside.\n"
                    "- If left heavier: heavy ball is among those 3.\n"
                    "- If right heavier: heavy ball is among those 3.\n"
                    "- If balanced: heavy ball is among the 2 unused.\n\n"
                    "Case 1 (heavy among 3): Weighing 2: Put 1 on left, 1 on right, 1 aside.\n"
                    "- If balanced: the aside ball is heavy.\n"
                    "- If one side heavier: that's the heavy ball.\n\n"
                    "Case 2 (heavy among 2): Weighing 2: Put 1 on left, 1 on right.\n"
                    "- The heavier side is the heavy ball.\n\n"
                    "This problem generalizes: with k weighings, you can find one "
                    "heavy ball among up to 3^k balls (or distinguish heavy/light among "
                    "(3^k - 3)/2 balls)."
                ),
                [],
            ),
            (
                "Take the number 2026. Subtract the sum of its digits. "
                "Is the result always divisible by 9? Prove it.",
                (
                    "2026: 2+0+2+6 = 10. 2026 - 10 = 2016. 2016 / 9 = 224. Yes!\n\n"
                    "Proof by modular arithmetic:\n\n"
                    "Any number N can be written as:\n"
                    "N = d_n * 10^n + d_{n-1} * 10^{n-1} + ... + d_1 * 10 + d_0\n\n"
                    "Sum of digits: S = d_n + d_{n-1} + ... + d_1 + d_0\n\n"
                    "Note that 10^k mod 9 = (9+1)^k mod 9 = 1^k mod 9 = 1 (mod 9).\n\n"
                    "So N mod 9 = (sum of d_k * 1) mod 9 = S mod 9.\n\n"
                    "Therefore N - S mod 9 = (N mod 9) - (S mod 9) = 0 (mod 9).\n\n"
                    "So N - S is always divisible by 9, regardless of the number.\n\n"
                    "This is the basis for the 'digital root' or 'casting out nines' "
                    "technique used for checking arithmetic."
                ),
                [],
            ),
            (
                "A train leaves from New York to Chicago at the same time as a train leaves "
                "from Chicago to New York. One travels at 80 mph, the other at 70 mph. "
                "The distance is 800 miles. Which train is closer to New York when they meet?",
                (
                    "Trick question - they meet at the SAME point, so both are equally "
                    "close (and equally far) from New York.\n\n"
                    "Let's compute to verify:\n"
                    "Combined speed: 80 + 70 = 150 mph\n"
                    "Time to meet: 800 / 150 = 5.333... hours = 5 hours 20 minutes\n\n"
                    "Train from NY travels: 80 * 5.333 = 426.67 miles from NY\n"
                    "Train from Chicago travels: 70 * 5.333 = 373.33 miles from Chicago\n"
                    "Distance from NY for the Chicago train: 800 - 373.33 = 426.67 miles\n\n"
                    "Both are 426.67 miles from New York.\n\n"
                    "This is a classic trick question that tests whether you jump "
                    "to calculations without thinking about the fundamental geometry "
                    "of the situation."
                ),
                [],
            ),
            (
                "What's the next number: 2, 6, 12, 20, 30, ?",
                (
                    "Pattern analysis:\n"
                    "2 = 1*2\n"
                    "6 = 2*3\n"
                    "12 = 3*4\n"
                    "20 = 4*5\n"
                    "30 = 5*6\n\n"
                    "Each term n = k * (k+1) where k = 1, 2, 3, ...\n\n"
                    "Next term (k=6): 6 * 7 = 42\n\n"
                    "Also expressible as: a_n = n^2 + n = n(n+1)\n"
                    "Or: differences are +4, +6, +8, +10, +12 (second difference is constant = 2, "
                    "confirming it's a quadratic).\n\n"
                    "Answer: 42.\n\n"
                    "These are called pronic numbers (product of consecutive integers)."
                ),
                [],
            ),
        ]
