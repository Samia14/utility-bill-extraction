"""Final bill JSON encoding and level1.csv writing.

Numbers stay exact decimal text. Binary floats are refused so a printed
decimal cannot change while it is written.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

BILL_FIELDS: tuple[str, ...] = (
    "provider",
    "tariff",
    "sanctioned_load_kw",
    "bill_month",
    "reading_date",
    "issue_date",
    "due_date",
    "previous_reading",
    "current_reading",
    "units_consumed",
    "charges",
    "total_charges",
    "taxes",
    "total_taxes",
    "current_bill",
    "arrears",
    "payable_within_due_date",
    "payable_after_due_date",
)


def format_number(value: Decimal) -> str:
    """Return the shortest plain decimal text for a finite Decimal."""
    if not value.is_finite():
        raise ValueError("Amount is not a finite number.")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-", "-0"}:
        return "0"
    return text


def dumps_bill(data: Any) -> str:
    """Encode a bill as one compact JSON line with exact decimals."""
    return _encode(data)


def write_level1_csv(path: Path | str, rows: list[tuple[str, dict[str, Any]]]) -> None:
    """Write bill_id,json rows. The csv module quotes the JSON cell."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["bill_id", "json"])
        for bill_id, bill in rows:
            writer.writerow([bill_id, dumps_bill(bill)])


def rows_with_data(results: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Keep a CSV row only when a bill object is present."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for result in results:
        bill = result.get("data")
        if bill is None:
            continue
        bill_id = result.get("bill_id")
        if not bill_id:
            source = str(result.get("source") or "bill")
            bill_id = Path(source).stem or "bill"
        rows.append((str(bill_id), bill))
    return rows


def _encode(value: Any) -> str:
    if isinstance(value, Decimal):
        return format_number(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        raise TypeError("Bill JSON cannot contain binary floats.")
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings.")
            parts.append(json.dumps(key, ensure_ascii=False) + ":" + _encode(item))
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    raise TypeError(f"Cannot encode {type(value).__name__} in bill JSON.")
