from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any


class AgentDataGenerator:
    TOOL_TYPES = [
        "code_execution",
        "web_search",
        "file_ops",
        "shell_commands",
        "api_calls",
        "database",
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        seed = config.get("seed", 42)
        self.rng = random.Random(seed)

    def _get_templates(self, tool_type: str) -> list[callable]:
        dispatch = {
            "code_execution": self._code_templates,
            "web_search": self._search_templates,
            "file_ops": self._file_templates,
            "shell_commands": self._shell_templates,
            "api_calls": self._api_templates,
            "database": self._db_templates,
        }
        fn = dispatch.get(tool_type)
        if fn is None:
            msg = f"Unknown tool_type: {tool_type}. Choose from {self.TOOL_TYPES}"
            raise ValueError(msg)
        return fn()

    def _code_templates(self) -> list[callable]:
        def t01():
            return [
                {
                    "from": "human",
                    "value": "Can you analyze this Python script for performance bottlenecks and suggest improvements?",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll start by reading the script to analyze it.\n<tool_call>\n{"name": "read_file", "args": {"path": "script.py"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "def process(data):\\n    result = []\\n    for i in range(len(data)):\\n        for j in range(len(data)):\\n            if data[i] < data[j]:\\n                result.append(data[i] * data[j])\\n    return sum(result)\\n\\n# usage\\nimport random\\nnums = [random.randint(1, 100) for _ in range(1000)]\\nprint(process(nums))"}',
                },
                {
                    "from": "gpt",
                    "value": 'I can see several performance issues. Let me benchmark the current implementation.\n<tool_call>\n{"name": "execute_code", "args": {"language": "python", "code": "import random, time\\nnums = [random.randint(1, 100) for _ in range(1000)]\\ndef process(data):\\n    result = []\\n    for i in range(len(data)):\\n        for j in range(len(data)):\\n            if data[i] < data[j]:\\n                result.append(data[i] * data[j])\\n    return sum(result)\\nstart = time.perf_counter()\\nprint(process(nums))\\nprint(f\'Time: {time.perf_counter() - start:.3f}s\')"}}\n</tool_call>',
                },
                {"from": "tool", "value": '{"result": "32728134\\nTime: 0.184s"}'},
                {
                    "from": "gpt",
                    "value": "The O(n\u00b2) nested loop is the main bottleneck. Here's an optimized version using sorting and prefix sums to get O(n log n). The key insight is that sorting first lets us avoid the quadratic inner loop.",
                },
            ]

        def t02():
            return [
                {"from": "human", "value": "Write a unit test suite for my calculator module."},
                {
                    "from": "gpt",
                    "value": 'Let me first look at the module to understand what needs testing.\n<tool_call>\n{"name": "read_file", "args": {"path": "calculator.py"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "def add(a, b): return a + b\\ndef subtract(a, b): return a - b\\ndef multiply(a, b): return a * b\\ndef divide(a, b):\\n    if b == 0:\\n        raise ValueError(\'Division by zero\')\\n    return a / b\\ndef power(a, b): return a ** b"}',
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll write pytest tests covering normal cases, edge cases, and error conditions.\n<tool_call>\n{"name": "execute_code", "args": {"language": "python", "code": "import pytest\\nfrom calculator import add, subtract, multiply, divide, power\\n\\ndef test_add():\\n    assert add(2, 3) == 5\\n    assert add(-1, 1) == 0\\n    assert add(0, 0) == 0\\n\\ndef test_divide_by_zero():\\n    with pytest.raises(ValueError):\\n        divide(1, 0)\\n\\ndef test_power_zero():\\n    assert power(0, 0) == 1\\n\\nimport sys\\nsys.exit(pytest.main([\'-v\', \'--tb=short\']))"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "============================= test session starts ==============================\\ncollected 3 items\\n\\ntest_add PASSED\\ntest_divide_by_zero PASSED\\ntest_power_zero PASSED\\n============================== 3 passed in 0.02s ==============================="}',
                },
                {
                    "from": "gpt",
                    "value": "All 3 tests pass. The test suite covers addition, division by zero error handling, and edge case for power with zero exponent.",
                },
            ]

        def t03():
            return [
                {
                    "from": "human",
                    "value": "Refactor this Python class to use dataclasses instead of manual __init__.",
                },
                {
                    "from": "gpt",
                    "value": 'Let me read the current implementation.\n<tool_call>\n{"name": "read_file", "args": {"path": "user_model.py"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "class User:\\n    def __init__(self, name, email, age):\\n        self.name = name\\n        self.email = email\\n        self.age = age\\n    def is_adult(self):\\n        return self.age >= 18"}',
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll refactor this to use dataclasses.\n<tool_call>\n{"name": "execute_code", "args": {"language": "python", "code": "from dataclasses import dataclass\\n@dataclass\\nclass User:\\n    name: str\\n    email: str\\n    age: int\\n    def is_adult(self) -> bool:\\n        return self.age >= 18\\nu = User(name=\'Alice\', email=\'alice@example.com\', age=25)\\nprint(u)\\nprint(f\'Adult: {u.is_adult()}\')"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": "{\"result\": \"User(name='Alice', email='alice@example.com', age=25)\\nAdult: True\"}",
                },
                {
                    "from": "gpt",
                    "value": "Refactored to use @dataclass. The class now has automatic __init__, __repr__, and __eq__ methods. The type annotations also improve readability and enable static type checking.",
                },
            ]

        return [t01, t02, t03]

    def _search_templates(self) -> list[callable]:
        def t01():
            return [
                {
                    "from": "human",
                    "value": "Can you research the latest advancements in solid-state battery technology and summarize the key breakthroughs from 2024-2025?",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll search for recent developments in solid-state batteries.\n<tool_call>\n{"name": "web_search", "args": {"query": "solid-state battery breakthroughs 2024 2025 latest research"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "Found 15 results. Top hits:\\n1. Toyota solid-state battery 745-mile range announcement (2024)\\n2. Samsung SDI solid-state battery production line 2025\\n3. QuantumScape solid-state battery partnership with VW\\n4. MIT sulfide-based solid electrolyte breakthrough\\n5. CATL all-solid-state battery roadmap"}',
                },
                {
                    "from": "gpt",
                    "value": 'Let me get more details on the most significant developments.\n<tool_call>\n{"name": "web_search", "args": {"query": "QuantumScape solid-state battery 2025 production details"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "QuantumScape achieved 15-minute fast charging, 80% capacity retention after 1000 cycles. Partnership with VW aims for 2026 production vehicles. Key innovation: ceramic separator that prevents dendrite formation."}',
                },
                {
                    "from": "gpt",
                    "value": "Here's my summary of the key breakthroughs: (1) Toyota demonstrated 745-mile range prototypes using sulfide-based solid electrolytes. (2) Samsung SDI began pilot production lines in 2025. (3) QuantumScape's ceramic separator technology achieved 15-min fast charging with 1000-cycle durability. (4) MIT published a sulfide electrolyte with 10x higher ionic conductivity. (5) CATL announced mass production target for 2027.",
                },
            ]

        def t02():
            return [
                {
                    "from": "human",
                    "value": "Find me the current interest rates for 30-year fixed mortgages in California and recent trends.",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll search for current mortgage rates.\n<tool_call>\n{"name": "web_search", "args": {"query": "California 30-year fixed mortgage rate April 2026"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "Current average 30-year fixed mortgage rate in California: 6.75%. Up from 6.5% in January 2026. National average: 6.625%. Source: Freddie Mac Primary Mortgage Market Survey."}',
                },
                {
                    "from": "gpt",
                    "value": "Here are the current rates and trends for California: The average 30-year fixed mortgage rate in California is approximately 6.75% as of April 2026, slightly above the national average of 6.625%. Rates have increased about 25 basis points since January 2026. This follows the Fed's recent signals about maintaining higher rates to manage inflation.",
                },
            ]

        def t03():
            return [
                {
                    "from": "human",
                    "value": "Compare the specifications of the latest MacBook Pro and Dell XPS 15 for machine learning work.",
                },
                {
                    "from": "gpt",
                    "value": 'Let me search for both laptops\' latest specs.\n<tool_call>\n{"name": "web_search", "args": {"query": "MacBook Pro M4 specs RAM GPU 2025"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "MacBook Pro M4 Max: up to 128GB unified memory, 16-core CPU, 40-core GPU, up to 8TB SSD. Neural Engine for ML. $2,499 base."}',
                },
                {
                    "from": "gpt",
                    "value": 'Got MacBook specs. Now looking up Dell.\n<tool_call>\n{"name": "web_search", "args": {"query": "Dell XPS 15 2025 specs RAM GPU"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "Dell XPS 15 (2025): Intel Core Ultra 9, up to 64GB DDR5, NVIDIA RTX 4070 8GB, up to 4TB SSD. $2,199 base."}',
                },
                {
                    "from": "gpt",
                    "value": "Comparison for ML work: MacBook Pro offers 128GB unified memory (good for large models) but limited GPU compute vs NVIDIA's CUDA ecosystem on the Dell. The Dell's RTX 4070 supports CUDA and cuDNN which most ML frameworks use. For training, Dell wins. For large model inference with limited batch sizes, MacBook's unified memory is advantageous.",
                },
            ]

        return [t01, t02, t03]

    def _file_templates(self) -> list[callable]:
        def t01():
            return [
                {
                    "from": "human",
                    "value": "Organize my Downloads folder. Categorize all files by type and move them into appropriate subfolders.",
                },
                {
                    "from": "gpt",
                    "value": 'Let me first see what\'s in the Downloads folder.\n<tool_call>\n{"name": "list_directory", "args": {"path": "/Users/user/Downloads"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "report.pdf, photo.jpg, notes.txt, script.py, data.csv, image.png, song.mp3, archive.zip, video.mp4, presentation.pptx, README.md, config.json, invoice.pdf"}',
                },
                {
                    "from": "gpt",
                    "value": "I'll create categorized folders and move files accordingly.\n<tool_call>\n{\"name\": \"execute_code\", \"args\": {\"language\": \"python\", \"code\": \"import os, shutil, json\\ndownloads = '/Users/user/Downloads'\\ncategories = {\\n    'Documents': ['.pdf', '.docx', '.txt', '.md'],\\n    'Images': ['.jpg', '.jpeg', '.png', '.gif', '.svg'],\\n    'Code': ['.py', '.js', '.ts', '.json', '.yaml'],\\n    'Data': ['.csv', '.xlsx', '.sql'],\\n    'Media': ['.mp3', '.mp4', '.wav', '.mov'],\\n    'Archives': ['.zip', '.tar', '.gz'],\\n}\\nmoved = {}\\nfor f in os.listdir(downloads):\\n    path = os.path.join(downloads, f)\\n    if os.path.isfile(path):\\n        ext = os.path.splitext(f)[1].lower()\\n        for cat, exts in categories.items():\\n            if ext in exts:\\n                dest = os.path.join(downloads, cat)\\n                os.makedirs(dest, exist_ok=True)\\n                shutil.move(path, os.path.join(dest, f))\\n                moved[f] = cat\\n                break\\nprint(json.dumps(moved, indent=2))\"}}\n</tool_call>",
                },
                {
                    "from": "tool",
                    "value": '{"result": "{\\n  \\"report.pdf\\": \\"Documents\\",\\n  \\"photo.jpg\\": \\"Images\\",\\n  \\"notes.txt\\": \\"Documents\\",\\n  \\"script.py\\": \\"Code\\",\\n  \\"data.csv\\": \\"Data\\",\\n  \\"image.png\\": \\"Images\\",\\n  \\"song.mp3\\": \\"Media\\",\\n  \\"archive.zip\\": \\"Archives\\",\\n  \\"video.mp4\\": \\"Media\\",\\n  \\"presentation.pptx\\": \\"Documents\\",\\n  \\"README.md\\": \\"Documents\\",\\n  \\"config.json\\": \\"Code\\",\\n  \\"invoice.pdf\\": \\"Documents\\"\\n}"}',
                },
                {
                    "from": "gpt",
                    "value": "Done. All 13 files have been organized into 6 categories: Documents (5 files), Images (2), Code (2), Data (1), Media (2), and Archives (1).",
                },
            ]

        def t02():
            return [
                {
                    "from": "human",
                    "value": "Find all files in the project that import from deprecated libraries and list them.",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll search through the project source files.\n<tool_call>\n{"name": "glob", "args": {"pattern": "**/*.py", "path": "."}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": "{\"result\": \"['src/main.py', 'src/utils.py', 'tests/test_main.py', 'scripts/analyze.py']\"}",
                },
                {
                    "from": "gpt",
                    "value": 'Let me check each file for deprecated imports.\n<tool_call>\n{"name": "grep", "args": {"pattern": "^(import|from)", "path": "src/main.py"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "import os\\nimport sys\\nfrom collections import OrderedDict\\nimport distutils"}',
                },
                {
                    "from": "gpt",
                    "value": 'Found \'distutils\' which was deprecated in Python 3.10 and removed in 3.12. Let me check the other files too.\n<tool_call>\n{"name": "grep", "args": {"pattern": "distutils|imp|pyxc|optparse", "path": "."}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "src/main.py: import distutils\\nsrc/utils.py: from distutils import util\\nsrc/old_script.py: import imp"}',
                },
                {
                    "from": "gpt",
                    "value": "Found 3 files using deprecated modules: src/main.py (distutils), src/utils.py (distutils.util), and src/old_script.py (imp). distutils should be replaced with setuptools or packaging. imp is replaced by importlib. I recommend updating these to avoid compatibility issues with Python 3.12+.",
                },
            ]

        return [t01, t02]

    def _shell_templates(self) -> list[callable]:
        def t01():
            return [
                {
                    "from": "human",
                    "value": "Check disk usage and find the top 10 largest directories in /var/log",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll check the disk usage in /var/log.\n<tool_call>\n{"name": "shell_command", "args": {"command": "du -sh /var/log/* | sort -rh | head -10"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "2.3G\\t/var/log/system.log\\n1.1G\\t/var/log/install.log\\n890M\\t/var/log/apache2\\n450M\\t/var/log/mail.log\\n320M\\t/var/log/kernel.log\\n210M\\t/var/log/nginx\\n180M\\t/var/log/audit\\n95M\\t/var/log/docker\\n67M\\t/var/log/postgresql\\n34M\\t/var/log/cron.log"}',
                },
                {
                    "from": "gpt",
                    "value": "Here are the top 10 largest log directories/files. system.log is the largest at 2.3GB. Total log size is approximately 5.6GB. I recommend archiving logs older than 30 days and setting up log rotation if not already configured.",
                },
            ]

        def t02():
            return [
                {
                    "from": "human",
                    "value": "Find all processes currently using more than 500MB of memory.",
                },
                {
                    "from": "gpt",
                    "value": 'Let me check running processes with their memory usage.\n<tool_call>\n{"name": "shell_command", "args": {"command": "ps aux --sort=-%mem | awk \'$6 > 512000 {print $11, $6/1024 \\" MB\\"}\'"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "/Applications/Firefox.app 1245.6 MB\\n/Applications/Chrome.app 892.3 MB\\n/Applications/Slack.app 678.9 MB\\n/usr/local/bin/node 543.2 MB\\n/Applications/VS Code.app 512.1 MB"}',
                },
                {
                    "from": "gpt",
                    "value": "Found 5 processes using over 500MB. Firefox is the most memory-intensive at 1.2GB. If your system is running low on memory, consider closing unused browser tabs or restarting these applications.",
                },
            ]

        return [t01, t02]

    def _api_templates(self) -> list[callable]:
        def t01():
            return [
                {
                    "from": "human",
                    "value": "Fetch the latest GitHub repository information for the aurelius project and summarize the activity.",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll fetch the repo data from the GitHub API.\n<tool_call>\n{"name": "api_call", "args": {"method": "GET", "url": "https://api.github.com/repos/user/aurelius", "headers": {"Accept": "application/vnd.github.v3+json"}}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"status": 200, "result": {"name": "aurelius", "description": "Aurelius 1.3B LLM training framework", "stars": 42, "forks": 8, "open_issues": 3, "language": "Python", "default_branch": "main", "updated_at": "2026-04-24T10:30:00Z", "pushed_at": "2026-04-24T15:00:00Z"}}',
                },
                {
                    "from": "gpt",
                    "value": 'The Aurelius repository has 42 stars, 8 forks, and 3 open issues. It was last pushed to today. Python is the primary language. Let me check recent commit activity.\n<tool_call>\n{"name": "api_call", "args": {"method": "GET", "url": "https://api.github.com/repos/user/aurelius/commits?per_page=5", "headers": {"Accept": "application/vnd.github.v3+json"}}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"status": 200, "result": [{"sha": "abc123", "commit": {"message": "Add safety data generator", "author": {"date": "2026-04-24T14:00:00Z"}}}, {"sha": "def456", "commit": {"message": "Fix tokenizer edge case", "author": {"date": "2026-04-23T11:00:00Z"}}}, {"sha": "ghi789", "commit": {"message": "Update training config", "author": {"date": "2026-04-22T09:00:00Z"}}}]}',
                },
                {
                    "from": "gpt",
                    "value": "Recent activity: 3 commits in the last 3 days. The latest commit added a safety data generator. The project is actively maintained.",
                },
            ]

        def t02():
            return [
                {
                    "from": "human",
                    "value": "Check the status of our CI pipeline and list any failing workflows.",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll query the GitHub Actions API for workflow runs.\n<tool_call>\n{"name": "api_call", "args": {"method": "GET", "url": "https://api.github.com/repos/user/aurelius/actions/runs?status=failure&per_page=5", "headers": {"Accept": "application/vnd.github.v3+json"}}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"status": 200, "result": {"total_count": 0, "workflow_runs": []}}',
                },
                {
                    "from": "gpt",
                    "value": 'No failing workflows found. Let me also check the overall status.\n<tool_call>\n{"name": "api_call", "args": {"method": "GET", "url": "https://api.github.com/repos/user/aurelius/actions/runs?per_page=3&status=completed", "headers": {"Accept": "application/vnd.github.v3+json"}}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"status": 200, "result": {"total_count": 15, "workflow_runs": [{"name": "CI", "conclusion": "success", "created_at": "2026-04-24T12:00:00Z"}, {"name": "Lint", "conclusion": "success", "created_at": "2026-04-24T11:00:00Z"}]}}',
                },
                {
                    "from": "gpt",
                    "value": "All CI pipelines are passing. The last 2 workflow runs (CI and Lint) completed successfully today. No failures to report.",
                },
            ]

        return [t01, t02]

    def _db_templates(self) -> list[callable]:
        def t01():
            return [
                {
                    "from": "human",
                    "value": "Query the database for the top 5 customers by total purchase amount this month.",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll run the query to find top customers.\n<tool_call>\n{"name": "execute_sql", "args": {"connection": "production", "query": "SELECT c.id, c.name, SUM(o.total) as total_spent\\nFROM customers c\\nJOIN orders o ON c.id = o.customer_id\\nWHERE o.created_at >= date_trunc(\'month\', CURRENT_DATE)\\nGROUP BY c.id, c.name\\nORDER BY total_spent DESC\\nLIMIT 5;"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "[{\\"id\\": 101, \\"name\\": \\"Acme Corp\\", \\"total_spent\\": \\"45230.00\\"}, {\\"id\\": 204, \\"name\\": \\"Globex Inc\\", \\"total_spent\\": \\"38100.00\\"}, {\\"id\\": 89, \\"name\\": \\"Initech\\", \\"total_spent\\": \\"29500.00\\"}, {\\"id\\": 156, \\"name\\": \\"Hooli\\", \\"total_spent\\": \\"22300.00\\"}, {\\"id\\": 312, \\"name\\": \\"Stark Industries\\", \\"total_spent\\": \\"18900.00\\"}]"}',
                },
                {
                    "from": "gpt",
                    "value": "The top 5 customers this month are: (1) Acme Corp at $45,230, (2) Globex Inc at $38,100, (3) Initech at $29,500, (4) Hooli at $22,300, and (5) Stark Industries at $18,900. Total from top 5: $153,930.",
                },
            ]

        def t02():
            return [
                {
                    "from": "human",
                    "value": "Find all orders that have been pending for more than 7 days and need manual review.",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll query the database for stale pending orders.\n<tool_call>\n{"name": "execute_sql", "args": {"connection": "production", "query": "SELECT o.id, o.customer_id, c.name, o.total, o.created_at\\nFROM orders o\\nJOIN customers c ON o.customer_id = c.id\\nWHERE o.status = \'pending\'\\n  AND o.created_at < CURRENT_DATE - INTERVAL \'7 days\'\\nORDER BY o.created_at ASC;"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "[{\\"id\\": 4501, \\"customer_id\\": 89, \\"name\\": \\"Initech\\", \\"total\\": \\"1500.00\\", \\"created_at\\": \\"2026-04-10\\"}, {\\"id\\": 4503, \\"customer_id\\": 156, \\"name\\": \\"Hooli\\", \\"total\\": \\"8900.00\\", \\"created_at\\": \\"2026-04-12\\"}]"}',
                },
                {
                    "from": "gpt",
                    "value": "Found 2 orders pending longer than 7 days: Order #4501 from Initech ($1,500, pending since April 10) and Order #4503 from Hooli ($8,900, pending since April 12). These should be flagged for manual review to contact the customers and resolve the pending status.",
                },
            ]

        def t03():
            return [
                {
                    "from": "human",
                    "value": "Show me the monthly revenue trend for the past 6 months.",
                },
                {
                    "from": "gpt",
                    "value": 'I\'ll query the monthly revenue data.\n<tool_call>\n{"name": "execute_sql", "args": {"connection": "analytics", "query": "SELECT DATE_TRUNC(\'month\', o.created_at) as month, SUM(o.total) as revenue\\nFROM orders o\\nWHERE o.status = \'completed\'\\n  AND o.created_at >= CURRENT_DATE - INTERVAL \'6 months\'\\nGROUP BY month\\nORDER BY month ASC;"}}\n</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '{"result": "[{\\"month\\": \\"2024-11-01\\", \\"revenue\\": \\"125000.00\\"}, {\\"month\\": \\"2024-12-01\\", \\"revenue\\": \\"142000.00\\"}, {\\"month\\": \\"2025-01-01\\", \\"revenue\\": \\"98000.00\\"}, {\\"month\\": \\"2025-02-01\\", \\"revenue\\": \\"115000.00\\"}, {\\"month\\": \\"2025-03-01\\", \\"revenue\\": \\"134000.00\\"}, {\\"month\\": \\"2025-04-01\\", \\"revenue\\": \\"156000.00\\"}]"}',
                },
                {
                    "from": "gpt",
                    "value": "Monthly revenue trend (last 6 months): Nov $125K, Dec $142K, Jan $98K, Feb $115K, Mar $134K, Apr $156K. Revenue dipped in January but has been steadily increasing since, with April being the strongest month at $156K (+16% month-over-month).",
                },
            ]

        return [t01, t02, t03]

    def generate_trajectory(self, tool_type: str) -> dict[str, Any]:
        templates = self._get_templates(tool_type)
        template_fn = self.rng.choice(templates)
        conversations = template_fn()

        tools_used = set()
        for msg in conversations:
            if msg["from"] == "gpt" and "<tool_call>" in msg["value"]:
                for match in re.finditer(r'"name":\s*"([^"]+)"', msg["value"]):
                    tools_used.add(match.group(1))

        steps = len([m for m in conversations if m["from"] == "gpt"])
        success = self.rng.random() < 0.9

        return {
            "conversations": conversations,
            "tools_used": sorted(tools_used),
            "steps": steps,
            "success": success,
        }

    def run(self, num_samples: int, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        agent_dir = output_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        weights = {
            "code_execution": 0.25,
            "web_search": 0.20,
            "file_ops": 0.15,
            "shell_commands": 0.15,
            "api_calls": 0.15,
            "database": 0.10,
        }
        types = list(weights.keys())
        type_weights = [weights[t] for t in types]

        records: list[dict[str, Any]] = []
        for _ in range(num_samples):
            tt = self.rng.choices(types, weights=type_weights, k=1)[0]
            records.append(self.generate_trajectory(tt))

        split = int(len(records) * 0.9)
        train_records = records[:split]
        val_records = records[split:]

        with open(agent_dir / "train.jsonl", "w") as f:
            for rec in train_records:
                f.write(json.dumps(rec) + "\n")

        with open(agent_dir / "val.jsonl", "w") as f:
            for rec in val_records:
                f.write(json.dumps(rec) + "\n")
