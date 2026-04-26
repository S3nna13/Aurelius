from __future__ import annotations

import csv
import json
import mimetypes
from dataclasses import dataclass
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from xml.etree import ElementTree as ET  # nosec B405 — stdlib only; input is local files / agent text, not untrusted network XML


class DocumentConversionError(Exception):
    """Raised when a document cannot be converted."""


@dataclass
class ConversionResult:
    markdown: str
    source_path: str | Path
    source_format: str
    success: bool
    error_message: str | None = None


class _HTMLToMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._lines: list[str] = []
        self._current: list[str] = []
        self._in_pre = False
        self._in_code = False
        self._in_a = False
        self._a_href = ""
        self._a_buffer: list[str] = []
        self._list_stack: list[tuple[str, int]] = []
        self._pending_space = False

    def _flush(self) -> None:
        if self._current:
            self._lines.append("".join(self._current))
            self._current = []

    def _add_text(self, text: str) -> None:
        if not text:
            return
        if self._pending_space and self._current and not text.startswith(" "):
            self._current.append(" ")
        self._pending_space = False
        self._current.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}

        if tag in ("h1", "h2", "h3"):
            self._flush()
            self._current.append({"h1": "# ", "h2": "## ", "h3": "### "}[tag])
        elif tag == "p":
            self._flush()
        elif tag == "a":
            self._in_a = True
            self._a_href = attrs_dict.get("href", "")
            self._a_buffer = []
        elif tag == "ul":
            self._flush()
            self._list_stack.append(("ul", 0))
        elif tag == "ol":
            self._flush()
            self._list_stack.append(("ol", 1))
        elif tag == "li":
            self._flush()
            if self._list_stack:
                list_type, counter = self._list_stack[-1]
                indent = "  " * max(0, len(self._list_stack) - 1)
                if list_type == "ul":
                    self._current.append(f"{indent}- ")
                else:
                    self._current.append(f"{indent}{counter}. ")
                self._list_stack[-1] = (list_type, counter + 1)
        elif tag == "pre":
            self._flush()
            self._in_pre = True
            self._current.append("```")
        elif tag == "code":
            if self._in_pre:
                # <code> inside <pre> should not add extra fences
                pass
            else:
                self._in_code = True
                self._current.append("`")
        elif tag == "br":
            self._flush()
        elif tag in ("div", "section", "article", "header", "footer", "nav", "aside"):
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h1", "h2", "h3"):
            self._flush()
        elif tag == "p":
            self._flush()
            self._lines.append("")
        elif tag == "a":
            self._in_a = False
            text = "".join(self._a_buffer).strip()
            href = self._a_href
            self._current.append(f"[{text}]({href})")
            self._a_buffer = []
        elif tag in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
            self._flush()
            self._lines.append("")
        elif tag == "li":
            self._flush()
        elif tag == "pre":
            self._in_pre = False
            self._flush()
            self._current.append("```")
            self._flush()
            self._lines.append("")
        elif tag == "code":
            if not self._in_pre:
                self._in_code = False
                self._current.append("`")
        elif tag in ("div", "section", "article", "header", "footer", "nav", "aside"):
            self._flush()
            self._lines.append("")

    def handle_data(self, data: str) -> None:
        if self._in_a:
            self._a_buffer.append(data)
            return

        if self._in_pre:
            self._current.append(data)
            return

        # Normalize whitespace for regular text
        text = " ".join(data.split())
        if text:
            self._add_text(text)

    def get_markdown(self) -> str:
        self._flush()
        return "\n".join(self._lines).strip()


class DocumentConverter:
    _EXT_MAP: dict[str, str] = {
        ".html": "html",
        ".htm": "html",
        ".json": "json",
        ".xml": "xml",
        ".csv": "csv",
        ".txt": "txt",
        ".md": "md",
    }

    def supported_formats(self) -> list[str]:
        return sorted(self._EXT_MAP.keys())

    def _detect_format(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext in self._EXT_MAP:
            return self._EXT_MAP[ext]
        # Fallback: mimetypes guess
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            if mime in ("text/html", "application/xhtml+xml"):
                return "html"
            if mime == "application/json":
                return "json"
            if mime in ("text/xml", "application/xml"):
                return "xml"
            if mime == "text/csv":
                return "csv"
            if mime == "text/markdown":
                return "md"
            if mime == "text/plain":
                return "txt"
        if ext == "":
            return "txt"
        return "unknown"

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")

    def convert(self, path: str | Path) -> ConversionResult:
        p = Path(path)
        try:
            raw = self._read_text(p)
        except OSError as exc:
            return ConversionResult(
                markdown="",
                source_path=p,
                source_format="",
                success=False,
                error_message=str(exc),
            )

        fmt = self._detect_format(p)
        return self.convert_text(raw, fmt, source_path=p)

    def convert_text(
        self,
        text: str,
        format_hint: str,
        source_path: str | Path = "",
        jq_path: str | None = None,
    ) -> ConversionResult:
        fmt = format_hint.lower().strip()
        handler_name = f"_convert_{fmt}"
        handler = getattr(self, handler_name, self._convert_unknown)

        try:
            if fmt == "json":
                markdown = handler(text, jq_path=jq_path)
            else:
                markdown = handler(text)
        except Exception as exc:
            return ConversionResult(
                markdown="",
                source_path=source_path,
                source_format=fmt,
                success=False,
                error_message=str(exc),
            )

        return ConversionResult(
            markdown=markdown,
            source_path=source_path,
            source_format=fmt,
            success=True,
        )

    def _convert_html(self, text: str) -> str:
        parser = _HTMLToMarkdownParser()
        parser.feed(text)
        return parser.get_markdown()

    def _convert_json(self, text: str, jq_path: str | None = None) -> str:
        data = json.loads(text)
        if jq_path:
            for part in jq_path.split("."):
                if isinstance(data, list):
                    data = data[int(part)]
                else:
                    data = data[part]
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return f"```json\n{pretty}\n```"

    def _convert_xml(self, text: str) -> str:
        root = ET.fromstring(text)  # nosec B314 — stdlib only; input is local files / agent text
        chunks = [chunk.strip() for chunk in root.itertext() if chunk.strip()]
        return "\n\n".join(chunks)

    def _convert_csv(self, text: str) -> str:
        reader = csv.reader(StringIO(text))
        rows = list(reader)
        if not rows:
            return ""

        def _escape_cell(cell: str) -> str:
            return cell.replace("|", "\\|").strip()

        lines: list[str] = []
        lines.append("| " + " | ".join(_escape_cell(c) for c in rows[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(_escape_cell(c) for c in row) + " |")
        return "\n".join(lines)

    def _convert_txt(self, text: str) -> str:
        return text

    def _convert_md(self, text: str) -> str:
        return text

    def _convert_unknown(self, text: str) -> str:
        return f"<!-- Warning: Unknown format, returning as plain text -->\n{text}"


DEFAULT_DOCUMENT_CONVERTER = DocumentConverter()

DOCUMENT_CONVERTER_REGISTRY: dict[str, DocumentConverter] = {
    "default": DEFAULT_DOCUMENT_CONVERTER,
}
