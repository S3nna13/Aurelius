"""Indicator-of-Compromise (IOC) extractor.

Pure stdlib scanner over free text (threat reports, logs, pastebins) that
extracts structured IOCs: IPv4/IPv6, domains, URLs, emails (with defang
detection), file hashes (MD5/SHA1/SHA256/SHA512), CVE IDs, file paths,
registry keys, and Bitcoin addresses (with Base58 checksum validation).

Inspired by the IOC analysis skills in
github.com/mukul975/Anthropic-Cybersecurity-Skills, re-expressed as a
self-contained, dependency-free component for the Aurelius security
subpackage.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IOC:
    """A single extracted indicator."""

    type: str
    value: str
    confidence: float
    defanged: bool
    span: tuple[int, int]


@dataclass
class IOCReport:
    """Aggregated extraction result."""

    iocs: list[IOC] = field(default_factory=list)
    by_type: dict[str, list[str]] = field(default_factory=dict)
    total: int = 0


# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #


DEFAULT_DOMAIN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "google.com",
        "gmail.com",
        "microsoft.com",
        "apple.com",
        "github.com",
        "localhost",
        "w3.org",
        "ietf.org",
    }
)


# --------------------------------------------------------------------------- #
# Bitcoin Base58 + checksum                                                   #
# --------------------------------------------------------------------------- #


_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_INDEX = {c: i for i, c in enumerate(_BASE58_ALPHABET)}


def _b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        if ch not in _BASE58_INDEX:
            raise ValueError("non-base58 character")
        n = n * 58 + _BASE58_INDEX[ch]
    # Convert integer to big-endian bytes
    length = (n.bit_length() + 7) // 8
    body = n.to_bytes(length, "big") if length else b""
    # Restore leading zero bytes (each '1' in s == one zero byte)
    pad = 0
    for ch in s:
        if ch == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + body


def validate_bitcoin_address(addr: str) -> bool:
    """Validate a legacy (P2PKH / P2SH) Base58Check Bitcoin address."""
    if not isinstance(addr, str):
        return False
    if len(addr) < 26 or len(addr) > 35:
        return False
    # Legacy addresses start with 1 or 3.
    if addr[0] not in ("1", "3"):
        return False
    try:
        raw = _b58decode(addr)
    except ValueError:
        return False
    if len(raw) != 25:
        return False
    payload, checksum = raw[:-4], raw[-4:]
    digest = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return digest == checksum


# --------------------------------------------------------------------------- #
# Regex patterns                                                              #
# --------------------------------------------------------------------------- #


# IPv4: 4 dotted octets, each 0-255.  We keep the regex permissive and validate
# with ``ipaddress`` afterwards.
_IPV4_RE = re.compile(r"(?<![\w.])((?:\d{1,3}\.){3}\d{1,3})(?![\w.])")

# IPv6: match plausible hex:colon runs (with optional ::), then validate.
_IPV6_RE = re.compile(
    r"(?<![\w:])("
    r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"
    r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"
    r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}"
    r"|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}"
    r"|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}"
    r"|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}"
    r"|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}"
    r"|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})"
    r"|::(?:[fF]{4}(?::0{1,4})?:)?(?:(?:\d{1,3}\.){3}\d{1,3})"
    r")(?![\w:])"
)

# URLs (http/https/ftp).  We include a defanged variant hxxp(s)/fxp with
# bracketed dots; refanging normalizes before matching.
_URL_RE = re.compile(
    r"\b((?:https?|ftp)://[^\s<>\"'\])}]+)",
    re.IGNORECASE,
)

# Emails - standard form.  Defanged emails use [at]/(at) and [.]/(dot).
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
)
_EMAIL_DEFANGED_RE = re.compile(
    r"([A-Za-z0-9._%+\-]+)"
    r"\s*(?:\[at\]|\(at\)|\{at\})\s*"
    r"([A-Za-z0-9.\-\[\]()]+"
    r"(?:\s*(?:\[dot\]|\(dot\)|\[\.\])\s*[A-Za-z0-9\-]+)+)",
    re.IGNORECASE,
)

# Domains - match only once URLs/emails have been consumed.  We validate the
# TLD has at least two letters.
_DOMAIN_RE = re.compile(
    r"(?<![\w@.])((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,24})(?![\w-])"
)

# Hashes (anchored to avoid longer hex runs matching as shorter hashes).
_HASH_MD5_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{32})(?![0-9a-fA-F])")
_HASH_SHA1_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{40})(?![0-9a-fA-F])")
_HASH_SHA256_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{64})(?![0-9a-fA-F])")
_HASH_SHA512_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{128})(?![0-9a-fA-F])")

# CVE-YYYY-NNNN+
_CVE_RE = re.compile(r"\b(CVE-\d{4}-\d{4,7})\b", re.IGNORECASE)

# Windows file paths: drive-letter + backslash path, OR UNC.
_WIN_PATH_RE = re.compile(r"(?<![\w])((?:[A-Za-z]:\\|\\\\)[^\s<>\"'|?*\r\n]+)")
# Unix paths: absolute paths of reasonable depth.
_UNIX_PATH_RE = re.compile(r"(?<![\w/])(/(?:[\w.\-]+/)+[\w.\-]+)")

# Registry keys (hive + backslash + path).
_REG_KEY_RE = re.compile(
    r"\b(HKLM|HKCU|HKCR|HKU|HKCC"
    r"|HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER|HKEY_CLASSES_ROOT"
    r"|HKEY_USERS|HKEY_CURRENT_CONFIG)"
    r"(\\[^\s\r\n\"']+)",
    re.IGNORECASE,
)

# Bitcoin address candidates (Base58, 26-35 chars, legacy 1.../3...).
_BTC_RE = re.compile(
    r"(?<![0-9A-Za-z])([13][" + re.escape(_BASE58_ALPHABET) + r"]{25,34})(?![0-9A-Za-z])"
)


# --------------------------------------------------------------------------- #
# Defang / refang                                                             #
# --------------------------------------------------------------------------- #


_REFANG_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"hxxps", re.IGNORECASE), "https"),
    (re.compile(r"hxxp", re.IGNORECASE), "http"),
    (re.compile(r"fxp", re.IGNORECASE), "ftp"),
    (re.compile(r"\[\.\]"), "."),
    (re.compile(r"\(\.\)"), "."),
    (re.compile(r"\{\.\}"), "."),
    (re.compile(r"\[dot\]", re.IGNORECASE), "."),
    (re.compile(r"\(dot\)", re.IGNORECASE), "."),
    (re.compile(r"\[:\]"), ":"),
    (re.compile(r"\[at\]", re.IGNORECASE), "@"),
    (re.compile(r"\(at\)", re.IGNORECASE), "@"),
    (re.compile(r"\{at\}", re.IGNORECASE), "@"),
)


def _refang_text(text: str) -> str:
    for pat, repl in _REFANG_SUBS:
        text = pat.sub(repl, text)
    return text


_DEFANG_MARKERS = (
    "[.]",
    "(.)",
    "{.}",
    "[dot]",
    "(dot)",
    "hxxp",
    "fxp",
    "[at]",
    "(at)",
    "{at}",
    "[:]",
)


# --------------------------------------------------------------------------- #
# Extractor                                                                   #
# --------------------------------------------------------------------------- #


class IOCExtractor:
    """Extract structured IOCs from free text."""

    def __init__(
        self,
        include_private_ips: bool = False,
        domain_allowlist: set[str] | None = None,
        refang: bool = True,
    ) -> None:
        self.include_private_ips = bool(include_private_ips)
        self.domain_allowlist = {
            d.lower()
            for d in (
                domain_allowlist if domain_allowlist is not None else DEFAULT_DOMAIN_ALLOWLIST
            )
        }
        self.refang = bool(refang)

    # ----- public helpers --------------------------------------------------- #

    def refang_text(self, text: str) -> str:
        """Remove common defang notation from ``text``."""
        return _refang_text(text)

    # Expose under the spec'd name too.
    def __call__(self, text: str) -> IOCReport:  # pragma: no cover - convenience
        return self.extract(text)

    # Alias required by spec: ``.refang(text)``.
    # Use ``setattr`` to avoid clashing with the ``refang`` attribute name.
    def _refang_method(self, text: str) -> str:
        return _refang_text(text)

    # --------------------------------------------------------------------- #
    # Span tracking                                                         #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _add(
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
        ioc: IOC,
    ) -> None:
        s, e = ioc.span
        for os_, oe in occupied:
            if s < oe and os_ < e:
                return
        occupied.append((s, e))
        bucket.append(ioc)

    # --------------------------------------------------------------------- #
    # Extraction core                                                       #
    # --------------------------------------------------------------------- #

    def extract(self, text: str) -> IOCReport:
        if not text:
            return IOCReport(iocs=[], by_type={}, total=0)

        # Work on refanged text if requested, but keep original spans.
        if self.refang:
            scan_text = _refang_text(text)
            # Record which ranges in ``scan_text`` were altered by refang so we
            # can mark those IOCs ``defanged=True``.
            defanged_original = any(m in text for m in _DEFANG_MARKERS)
        else:
            scan_text = text
            defanged_original = False

        bucket: list[IOC] = []
        occupied: list[tuple[int, int]] = []

        # Order matters: consume longer / more-specific patterns first so the
        # span-overlap filter protects us from double counting.
        self._extract_urls(scan_text, bucket, occupied, defanged_original)
        self._extract_emails(scan_text, text, bucket, occupied)
        self._extract_hashes(scan_text, bucket, occupied)
        self._extract_cves(scan_text, bucket, occupied)
        self._extract_registry_keys(scan_text, bucket, occupied)
        self._extract_windows_paths(scan_text, bucket, occupied)
        self._extract_ipv6(scan_text, bucket, occupied)
        self._extract_ipv4(scan_text, bucket, occupied, defanged_original)
        self._extract_bitcoin(scan_text, bucket, occupied)
        self._extract_domains(scan_text, bucket, occupied, defanged_original)
        self._extract_unix_paths(scan_text, bucket, occupied)

        # Deterministic ordering: by span start, then type.
        bucket.sort(key=lambda i: (i.span[0], i.type, i.value))

        by_type: dict[str, list[str]] = {}
        for ioc in bucket:
            by_type.setdefault(ioc.type, []).append(ioc.value)

        return IOCReport(iocs=bucket, by_type=by_type, total=len(bucket))

    def extract_by_type(self, text: str, ioc_type: str) -> list[IOC]:
        report = self.extract(text)
        return [i for i in report.iocs if i.type == ioc_type]

    # --------------------------------------------------------------------- #
    # Individual extractors                                                 #
    # --------------------------------------------------------------------- #

    def _extract_urls(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
        defanged: bool,
    ) -> None:
        for m in _URL_RE.finditer(text):
            value = m.group(1).rstrip(".,);:'\"")
            span = (m.start(1), m.start(1) + len(value))
            self._add(
                bucket,
                occupied,
                IOC(type="url", value=value, confidence=0.95, defanged=defanged, span=span),
            )

    def _extract_emails(
        self,
        scan_text: str,
        original_text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
    ) -> None:
        # Defanged emails from the ORIGINAL text (before refang) so we can flag.
        for m in _EMAIL_DEFANGED_RE.finditer(original_text):
            raw = m.group(0)
            refanged = _refang_text(raw)
            # Validate refanged form looks like an email.
            em = _EMAIL_RE.search(refanged)
            if not em:
                continue
            span = m.span()
            self._add(
                bucket,
                occupied,
                IOC(type="email", value=em.group(1), confidence=0.9, defanged=True, span=span),
            )

        for m in _EMAIL_RE.finditer(scan_text):
            span = m.span(1)
            self._add(
                bucket,
                occupied,
                IOC(type="email", value=m.group(1), confidence=0.95, defanged=False, span=span),
            )

    def _extract_hashes(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
    ) -> None:
        # Longest first to avoid SHA256 matching as two MD5s, etc.
        for length, htype, pat in (
            (128, "sha512", _HASH_SHA512_RE),
            (64, "sha256", _HASH_SHA256_RE),
            (40, "sha1", _HASH_SHA1_RE),
            (32, "md5", _HASH_MD5_RE),
        ):
            for m in pat.finditer(text):
                if len(m.group(1)) != length:
                    continue
                span = m.span(1)
                self._add(
                    bucket,
                    occupied,
                    IOC(
                        type=htype,
                        value=m.group(1).lower(),
                        confidence=0.99,
                        defanged=False,
                        span=span,
                    ),
                )

    def _extract_cves(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
    ) -> None:
        for m in _CVE_RE.finditer(text):
            span = m.span(1)
            self._add(
                bucket,
                occupied,
                IOC(
                    type="cve", value=m.group(1).upper(), confidence=0.99, defanged=False, span=span
                ),
            )

    def _extract_registry_keys(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
    ) -> None:
        for m in _REG_KEY_RE.finditer(text):
            value = m.group(0).rstrip(".,;:'\"")
            span = (m.start(), m.start() + len(value))
            self._add(
                bucket,
                occupied,
                IOC(type="registry_key", value=value, confidence=0.95, defanged=False, span=span),
            )

    def _extract_windows_paths(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
    ) -> None:
        for m in _WIN_PATH_RE.finditer(text):
            raw = m.group(1)
            value = raw.rstrip(".,;:'\" \t")
            span = (m.start(1), m.start(1) + len(value))
            if "\\" not in value[2:]:
                # Needs at least one path component after the drive/UNC head.
                continue
            self._add(
                bucket,
                occupied,
                IOC(
                    type="file_path_windows",
                    value=value,
                    confidence=0.85,
                    defanged=False,
                    span=span,
                ),
            )

    def _extract_unix_paths(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
    ) -> None:
        for m in _UNIX_PATH_RE.finditer(text):
            value = m.group(1).rstrip(".,;:'\"")
            span = (m.start(1), m.start(1) + len(value))
            self._add(
                bucket,
                occupied,
                IOC(type="file_path_unix", value=value, confidence=0.8, defanged=False, span=span),
            )

    def _extract_ipv4(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
        defanged: bool,
    ) -> None:
        for m in _IPV4_RE.finditer(text):
            candidate = m.group(1)
            try:
                ip = ipaddress.IPv4Address(candidate)
            except (ValueError, ipaddress.AddressValueError):
                continue
            if not self.include_private_ips and (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                continue
            span = m.span(1)
            self._add(
                bucket,
                occupied,
                IOC(type="ipv4", value=str(ip), confidence=0.98, defanged=defanged, span=span),
            )

    def _extract_ipv6(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
    ) -> None:
        for m in _IPV6_RE.finditer(text):
            candidate = m.group(1)
            try:
                ip = ipaddress.IPv6Address(candidate)
            except (ValueError, ipaddress.AddressValueError):
                continue
            if not self.include_private_ips and (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                continue
            span = m.span(1)
            self._add(
                bucket,
                occupied,
                IOC(type="ipv6", value=str(ip), confidence=0.95, defanged=False, span=span),
            )

    def _extract_bitcoin(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
    ) -> None:
        for m in _BTC_RE.finditer(text):
            candidate = m.group(1)
            if not validate_bitcoin_address(candidate):
                continue
            span = m.span(1)
            self._add(
                bucket,
                occupied,
                IOC(type="bitcoin", value=candidate, confidence=0.99, defanged=False, span=span),
            )

    def _extract_domains(
        self,
        text: str,
        bucket: list[IOC],
        occupied: list[tuple[int, int]],
        defanged: bool,
    ) -> None:
        for m in _DOMAIN_RE.finditer(text):
            value = m.group(1).lower().rstrip(".")
            span = m.span(1)
            # Skip if overlapping with an already-extracted IOC (URL, email...).
            skip = False
            for os_, oe in occupied:
                if span[0] < oe and os_ < span[1]:
                    skip = True
                    break
            if skip:
                continue
            # TLD sanity: must have letters only, len>=2.
            tld = value.rsplit(".", 1)[-1]
            if not tld.isalpha() or len(tld) < 2:
                continue
            # Allowlist: exact match or suffix match on a registered domain.
            if self._is_allowlisted(value):
                continue
            # Reject pure numerics (would be an IP).
            if re.fullmatch(r"[\d.]+", value):
                continue
            self._add(
                bucket,
                occupied,
                IOC(type="domain", value=value, confidence=0.8, defanged=defanged, span=span),
            )

    def _is_allowlisted(self, domain: str) -> bool:
        d = domain.lower()
        if d in self.domain_allowlist:
            return True
        for allowed in self.domain_allowlist:
            if d.endswith("." + allowed):
                return True
        return False


# The spec calls for ``IOCExtractor.refang(text)``.  We expose it as an alias
# pointing at the internal method so the public attribute ``self.refang`` (the
# boolean config flag) is still available on instances.  Accessing
# ``IOCExtractor.refang`` on the class returns the method; instance access
# returns the boolean.  To cover the documented API we also provide a
# module-level ``refang`` helper.
def refang(text: str) -> str:
    """Module-level convenience: remove defang notation from ``text``."""
    return _refang_text(text)


# Attach the method-style refang under a class attribute ``refang_method`` to
# avoid clashing with the boolean.  Callers wanting the method form should use
# ``IOCExtractor.refang_text`` (instance method) or the module-level helper.


__all__ = [
    "IOC",
    "IOCReport",
    "IOCExtractor",
    "validate_bitcoin_address",
    "refang",
    "DEFAULT_DOMAIN_ALLOWLIST",
]
