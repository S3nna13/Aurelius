"""Simple CSV parser for data loading with type inference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from typing import Any


@dataclass
class CSVParser:
    """Parse CSV data with basic type inference."""

    delimiter: str = ","

    def parse(self, text: str) -> list[dict[str, Any]]:
        reader = csv.DictReader(StringIO(text), delimiter=self.delimiter)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = {}
            for key, val in row.items():
                parsed[key] = self._infer_type(val)
            rows.append(parsed)
        return rows

    def _infer_type(self, val: str) -> Any:
        if val is None or val.strip() == "":
            return None
        v = val.strip()
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        low = v.lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        return v


CSV_PARSER = CSVParser()
