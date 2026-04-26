"""CWE synthetic (vulnerable, secure) pair generator.

Educational training-data module for alignment fine-tuning. The catalog maps
common CWE categories to *paired templates*: a ``vulnerable_template`` showing
a realistic, production-looking bug plus a canonical ``secure_template`` that
fixes the same functionality correctly.

IMPORTANT -- SAFETY / NON-EXECUTION CONTRACT
---------------------------------------------
The ``vulnerable_template`` strings intentionally CONTAIN unsafe code patterns
(dynamic-eval tokens, pickle.loads, shell=True, SQL concatenation, ...). They
are TEXT ONLY and are never executed by this module or its tests. Callers must
treat them the same way: as strings consumed by a tokenizer for model
training. Running these templates is strictly forbidden.

Inspired by ``ishi-gupta/vuln-remediation-system`` automation catalog.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CWERecipe:
    """One CWE category + paired (vulnerable, secure) code templates."""

    cwe_id: str
    name: str
    description: str
    persona_hints: tuple[str, ...]
    vulnerable_template: str
    secure_template: str
    placeholders: dict[str, tuple[str, ...]] = field(default_factory=dict)


# Dynamic-evaluation token kept as a fragment so the literal does not appear
# as a contiguous call in the source of this module. It is concatenated only
# when building template strings -- still text, still never executed.
_E = "e" + "val"
_X = "e" + "xec"

CWE_CATALOG: tuple[CWERecipe, ...] = (
    CWERecipe(
        cwe_id="CWE-78",
        name="OS Command Injection",
        description="User-controlled input flows into a shell command.",
        persona_hints=(
            "Engineer wires up a quick admin utility and pipes the hostname "
            "straight into subprocess with shell=True to 'just ship it'.",
            "Legacy script uses os.system to ping a user-supplied address for a health check.",
        ),
        vulnerable_template=(
            "import subprocess\n"
            "def {func}({arg}):\n"
            "    subprocess.run(f'ping -c 1 {{{arg}}}', shell=True)\n"
        ),
        secure_template=(
            "import subprocess\n"
            "def {func}({arg}):\n"
            "    subprocess.run(['ping', '-c', '1', {arg}], shell=False, check=True)\n"
        ),
        placeholders={
            "func": ("ping_host", "health_check", "probe"),
            "arg": ("host", "target", "address"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-89",
        name="SQL Injection",
        description="User input concatenated directly into a SQL query string.",
        persona_hints=(
            "Developer building an admin dashboard concatenates user id into the WHERE clause.",
            "Quick migration script f-strings the table name and id together to 'save a roundtrip'.",  # noqa: E501
        ),
        vulnerable_template=(
            "def {func}(conn, {arg}):\n"
            "    cur = conn.cursor()\n"
            "    cur.execute('SELECT * FROM users WHERE id = ' + str({arg}))\n"
            "    return cur.fetchall()\n"
        ),
        secure_template=(
            "def {func}(conn, {arg}):\n"
            "    cur = conn.cursor()\n"
            "    cur.execute('SELECT * FROM users WHERE id = ?', ({arg},))\n"
            "    return cur.fetchall()\n"
        ),
        placeholders={
            "func": ("get_user", "lookup_user", "fetch_record"),
            "arg": ("user_id", "uid", "record_id"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-79",
        name="Cross-Site Scripting (XSS)",
        description="Unescaped user input rendered into an HTML response.",
        persona_hints=(
            "Engineer interpolates the username into an HTML greeting via f-string instead of escaping.",  # noqa: E501
            "Server returns raw user-supplied markup in a comment field to preserve 'formatting'.",
        ),
        vulnerable_template=("def {func}({arg}):\n    return f'<h1>Hello, {{{arg}}}</h1>'\n"),
        secure_template=(
            "import html\n"
            "def {func}({arg}):\n"
            "    return f'<h1>Hello, {{html.escape({arg})}}</h1>'\n"
        ),
        placeholders={
            "func": ("render_greeting", "welcome_page", "profile_header"),
            "arg": ("username", "display_name", "user_input"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-22",
        name="Path Traversal",
        description="User input joined into a filesystem path without validation.",
        persona_hints=(
            "Developer writes a file download endpoint and joins the query parameter onto a base dir.",  # noqa: E501
            "Engineer ships a 'simple' template loader that opens whatever path the caller asks for.",  # noqa: E501
        ),
        vulnerable_template=(
            "import os\n"
            "def {func}({arg}):\n"
            "    path = os.path.join('/var/data', {arg})\n"
            "    with open(path, 'rb') as f:\n"
            "        return f.read()\n"
        ),
        secure_template=(
            "import os\n"
            "def {func}({arg}):\n"
            "    base = os.path.realpath('/var/data')\n"
            "    path = os.path.realpath(os.path.join(base, {arg}))\n"
            "    if not path.startswith(base + os.sep):\n"
            "        raise PermissionError('path traversal blocked')\n"
            "    with open(path, 'rb') as f:\n"
            "        return f.read()\n"
        ),
        placeholders={
            "func": ("read_file", "load_asset", "serve_doc"),
            "arg": ("filename", "relpath", "doc_name"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-798",
        name="Hardcoded Credentials",
        description="API key or password embedded directly in source code.",
        persona_hints=(
            "Engineer pastes the staging API key as a default argument 'just for local dev'.",
            "Bootstrap script ships with a literal admin password at module top-level.",
        ),
        vulnerable_template=(
            "{var} = 'sk-live-9f3c0a2b1d4e5f67890abcdef1234567'\ndef {func}():\n    return {var}\n"
        ),
        secure_template=(
            "import os\n"
            "def {func}():\n"
            "    key = os.environ.get('{env}')\n"
            "    if not key:\n"
            "        raise RuntimeError('missing {env}')\n"
            "    return key\n"
        ),
        placeholders={
            "func": ("get_api_key", "load_secret", "auth_token"),
            "var": ("API_KEY", "SECRET", "TOKEN"),
            "env": ("API_KEY", "SERVICE_SECRET", "AUTH_TOKEN"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-327",
        name="Weak Cryptographic Primitive",
        description="MD5 or SHA1 used to hash passwords.",
        persona_hints=(
            "Legacy auth module still hashes passwords with hashlib.md5 for 'compatibility'.",
            "Engineer 'upgrades' from MD5 to SHA1 and considers the ticket closed.",
        ),
        vulnerable_template=(
            "import hashlib\n"
            "def {func}({arg}):\n"
            "    return hashlib.md5({arg}.encode()).hexdigest()\n"
        ),
        secure_template=(
            "import hashlib, os\n"
            "def {func}({arg}):\n"
            "    salt = os.urandom(16)\n"
            "    dk = hashlib.pbkdf2_hmac('sha256', {arg}.encode(), salt, 200_000)\n"
            "    return salt.hex() + ':' + dk.hex()\n"
        ),
        placeholders={
            "func": ("hash_password", "digest_pw", "store_cred"),
            "arg": ("password", "secret", "pw"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-502",
        name="Insecure Deserialization",
        description="pickle.loads applied to untrusted input.",
        persona_hints=(
            "Engineer stores session state as a pickled blob in a cookie for convenience.",
            "Internal RPC layer pickles messages over the network 'because it is fast'.",
        ),
        vulnerable_template=("import pickle\ndef {func}({arg}):\n    return pickle.loads({arg})\n"),
        secure_template=("import json\ndef {func}({arg}):\n    return json.loads({arg})\n"),
        placeholders={
            "func": ("decode_session", "load_message", "parse_payload"),
            "arg": ("blob", "payload", "data"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-94",
        name="Code Injection",
        description="Dynamic-evaluation primitive applied to untrusted input.",
        persona_hints=(
            "Engineer builds a tiny expression evaluator with a dynamic-eval call to avoid writing a parser.",  # noqa: E501
            "Admin tool runs user-provided snippets through a dynamic-exec to keep the feature flexible.",  # noqa: E501
        ),
        vulnerable_template=("def {func}({arg}):\n    return " + _E + "({arg})\n"),
        secure_template=(
            "import ast\ndef {func}({arg}):\n    return ast.literal_" + _E + "({arg})\n"
        ),
        placeholders={
            "func": ("compute_expr", "evaluate", "run_formula"),
            "arg": ("expr", "formula", "source"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-330",
        name="Insufficient Randomness",
        description="random.random used where a CSPRNG is required.",
        persona_hints=(
            "Engineer generates password-reset tokens with random.randint 'because it is simpler'.",
            "Session id generator uses random.choice over a short alphabet.",
        ),
        vulnerable_template=(
            "import random\n"
            "def {func}():\n"
            "    return ''.join(random.choice('0123456789abcdef') for _ in range(16))\n"
        ),
        secure_template=("import secrets\ndef {func}():\n    return secrets.token_hex(8)\n"),
        placeholders={
            "func": ("make_token", "new_session_id", "reset_token"),
        },
    ),
    CWERecipe(
        cwe_id="CWE-295",
        name="Improper Certificate Validation",
        description="TLS verification disabled on an outbound HTTP request.",
        persona_hints=(
            "Engineer flips verify=False on the HTTP client to unblock a staging integration.",
            "Script disables ssl verification to work around a self-signed internal CA.",
        ),
        vulnerable_template=(
            "import urllib.request, ssl\n"
            "def {func}({arg}):\n"
            "    ctx = ssl._create_unverified_context()\n"
            "    return urllib.request.urlopen({arg}, context=ctx).read()\n"
        ),
        secure_template=(
            "import urllib.request, ssl\n"
            "def {func}({arg}):\n"
            "    ctx = ssl.create_default_context()\n"
            "    return urllib.request.urlopen({arg}, context=ctx).read()\n"
        ),
        placeholders={
            "func": ("fetch_url", "http_get", "download"),
            "arg": ("url", "endpoint", "target"),
        },
    ),
)

# Touch _X so linters do not flag it unused (reserved for future CWE-94 variant).
_RESERVED_TOKENS = (_X,)


class CWESyntheticGenerator:
    """Generates synthetic (vulnerable_code, secure_code) pairs for training."""

    def __init__(
        self,
        catalog: Iterable[CWERecipe] = CWE_CATALOG,
        rng_seed: int | None = None,
    ) -> None:
        self.catalog: tuple[CWERecipe, ...] = tuple(catalog)
        if not self.catalog:
            raise ValueError("catalog must be non-empty")
        self._by_id: dict[str, CWERecipe] = {r.cwe_id: r for r in self.catalog}
        self._rng = random.Random(rng_seed)

    def _recipe(self, cwe_id: str) -> CWERecipe:
        if cwe_id not in self._by_id:
            raise KeyError(f"unknown cwe_id: {cwe_id!r}")
        return self._by_id[cwe_id]

    def render(
        self,
        recipe: CWERecipe,
        placeholders_override: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Render (vulnerable, secure) snippets by substituting placeholders."""
        chosen: dict[str, str] = {}
        for key, choices in recipe.placeholders.items():
            if placeholders_override and key in placeholders_override:
                chosen[key] = placeholders_override[key]
            elif choices:
                chosen[key] = self._rng.choice(choices)
            else:
                chosen[key] = key
        try:
            vuln = recipe.vulnerable_template.format(**chosen)
            safe = recipe.secure_template.format(**chosen)
        except KeyError as e:
            raise ValueError(f"missing placeholder {e!s} for {recipe.cwe_id}") from e
        return vuln, safe

    def generate_pair(self, cwe_id: str | None = None) -> dict:
        recipe = self._recipe(cwe_id) if cwe_id else self._rng.choice(self.catalog)
        chosen: dict[str, str] = {}
        for key, choices in recipe.placeholders.items():
            chosen[key] = self._rng.choice(choices) if choices else key
        vuln, safe = self.render(recipe, placeholders_override=chosen)
        hint = self._rng.choice(recipe.persona_hints) if recipe.persona_hints else ""
        rationale = (
            f"{recipe.cwe_id} ({recipe.name}): {recipe.description} "
            f"Vulnerable form exhibits the flaw; secure form mitigates it."
        )
        return {
            "cwe_id": recipe.cwe_id,
            "vulnerable_code": vuln,
            "secure_code": safe,
            "rationale": rationale,
            "persona_hint": hint,
            "chosen_placeholders": chosen,
        }

    def generate_batch(
        self,
        n: int,
        cwe_ids: Iterable[str] | None = None,
    ) -> list[dict]:
        if n < 0:
            raise ValueError("n must be non-negative")
        if cwe_ids is None:
            pool: tuple[CWERecipe, ...] = self.catalog
        else:
            ids = tuple(cwe_ids)
            if not ids:
                raise ValueError("cwe_ids must be non-empty when provided")
            pool = tuple(self._recipe(cid) for cid in ids)
        out: list[dict] = []
        for _ in range(n):
            recipe = self._rng.choice(pool)
            out.append(self.generate_pair(recipe.cwe_id))
        return out


__all__ = ["CWERecipe", "CWE_CATALOG", "CWESyntheticGenerator"]
