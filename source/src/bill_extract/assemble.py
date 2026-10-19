"""Turn positioned OCR words into one bill record.

A number belongs to the label it sits beside. History is rebuilt as rows.
A printed figure that does not match another printed figure is kept, and a
warning is recorded, so a faulty bill stays visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bill_extract.errors import AmountParseError, DateParseError, field_error
from bill_extract.normalize import (
    charge_type,
    map_provider,
    parse_amount,
    parse_bill_month,
    parse_date,
    tax_type,
)
from bill_extract.schema import format_number

_MONTH_TOKEN = re.compile(
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s./-]*\d{2,4}",
    re.IGNORECASE,
)
_NET_WORDS = ("net meter", "net-meter", "solar", "import", "export", "generation")

_LABELS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("due_date", "date", ("due date",)),
    ("bill_month", "month", ("bill month",)),
    ("reading_date", "date", ("reading date",)),
    ("issue_date", "date", ("issue date",)),
    ("previous_reading", "amount", ("previous reading", "prev reading")),
    ("current_reading", "amount", ("present reading", "current reading")),
    ("units_consumed", "amount", ("units",)),
    ("sanctioned_load_kw", "amount", ("sanc load", "s load", "san load", "sanctioned load")),
    ("payable_within_due_date", "amount", ("payable within due date", "amount due")),
    ("current_bill", "amount", ("current bill", "electricity charges for current month")),
    ("arrears", "amount", ("arrears", "previous dues")),
    ("tariff", "text", ("tariff",)),
    ("unit_rate", "amount", ("unit rate",)),
    ("fixed_charge", "amount", ("fixed charge", "fixed charges")),
    ("total_charges", "amount", ("total electricity charges", "electricity charges")),
    ("total_taxes", "amount", ("total taxes", "taxes and duties")),
)


@dataclass(frozen=True)
class Token:
    """One OCR word on the full page."""

    text: str
    x: int
    y: int
    width: int
    height: int

    @property
    def cx(self) -> float:
        return self.x + (self.width / 2)

    @property
    def cy(self) -> float:
        return self.y + (self.height / 2)


@dataclass(frozen=True)
class PlacedToken:
    """A word plus whether its center sits in one window's unshared interior."""

    token: Token
    unshared: bool


def norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def same_place(left: Token, right: Token) -> bool:
    """True when two boxes land on the same word."""
    tol_x = max(8, 0.6 * max(left.width, right.width))
    tol_y = max(8, 0.6 * max(left.height, right.height))
    return abs(left.cx - right.cx) <= tol_x and abs(left.cy - right.cy) <= tol_y


def _find_match(token: Token, others: list[Token], used: set[int]) -> int | None:
    for index, other in enumerate(others):
        if index in used:
            continue
        if same_place(token, other):
            return index
    return None


def agree_reads(first: list[Token], second: list[Token]) -> tuple[list[Token], list[Token]]:
    """Keep a word only when the normal read and the zoomed read match.

    A word that appears in only one read, or that the two reads spell
    differently, is returned in the second list and is not kept.
    """
    used: set[int] = set()
    agreed: list[Token] = []
    conflicts: list[Token] = []
    for token in first:
        match = _find_match(token, second, used)
        if match is None:
            conflicts.append(token)
            continue
        used.add(match)
        other = second[match]
        if norm_text(token.text) == norm_text(other.text) and norm_text(token.text):
            agreed.append(token)
        else:
            conflicts.append(token)
    for index, other in enumerate(second):
        if index not in used:
            conflicts.append(other)
    return agreed, conflicts


def merge_tokens(placed: list[PlacedToken]) -> tuple[list[Token], list[Token]]:
    """Collapse overlapping windows into one word list.

    Equal reads of the same place become one word, taken from the unshared
    interior when that read exists. Different reads of the same place are a
    disagreement and are not kept.
    """
    clusters: list[list[PlacedToken]] = []
    for item in placed:
        cluster = next(
            (group for group in clusters if any(same_place(item.token, other.token) for other in group)),
            None,
        )
        if cluster is None:
            clusters.append([item])
        else:
            cluster.append(item)
    kept: list[Token] = []
    conflicts: list[Token] = []
    for cluster in clusters:
        texts = {norm_text(item.token.text) for item in cluster}
        texts.discard("")
        if len(texts) == 1:
            unshared = [item for item in cluster if item.unshared]
            chosen = unshared[0] if unshared else cluster[0]
            kept.append(chosen.token)
        elif texts:
            conflicts.append(cluster[0].token)
    return kept, conflicts


def assemble(tokens: list[Token], page_height: int) -> dict[str, Any]:
    """Build the bill record from words. Missing values stay null."""
    record = _empty_record()
    header = [token for token in tokens if page_height <= 0 or token.cy <= page_height * 0.2]
    record["provider"] = _provider(header or tokens[:8])
    for field, kind, phrases in _LABELS:
        label = _find_label(tokens, phrases)
        if label is None:
            continue
        value = _value_beside(label, tokens, kind)
        if value is not None:
            record[field] = value
    record["charges"] = _lines(tokens, charge_type, ("energy", "fixed", "fpa", "fca", "surcharge", "adjustment"))
    record["taxes"] = _lines(tokens, tax_type, ("gst", "duty", "tax", "municipal"))
    record["fuel_adjustment"] = _fuel(tokens)
    record["late_payment_tiers"] = _late_tiers(tokens)
    record["history"] = _history(tokens)
    record["connection"] = _connection(tokens)
    if record["payable_after_due_date"] is None:
        amounts = [tier["amount"] for tier in record["late_payment_tiers"] if tier["amount"] is not None]
        if amounts:
            record["payable_after_due_date"] = max(amounts)
    _note_printed_faults(record)
    return record


def _empty_record() -> dict[str, Any]:
    return {
        "provider": None,
        "tariff": None,
        "sanctioned_load_kw": None,
        "bill_month": None,
        "reading_date": None,
        "issue_date": None,
        "due_date": None,
        "previous_reading": None,
        "current_reading": None,
        "units_consumed": None,
        "charges": [],
        "total_charges": None,
        "taxes": [],
        "total_taxes": None,
        "current_bill": None,
        "arrears": None,
        "payable_within_due_date": None,
        "payable_after_due_date": None,
        "late_payment_tiers": [],
        "fuel_adjustment": None,
        "history": [],
        "connection": {"kind": None, "evidence": None, "units": None},
        "unit_rate": None,
        "fixed_charge": None,
        "warnings": [],
    }


def _provider(tokens: list[Token]) -> str | None:
    for token in tokens:
        found = map_provider(token.text)
        if found:
            return found
    if tokens:
        return map_provider(" ".join(token.text for token in tokens))
    return None


def _find_label(tokens: list[Token], phrases: tuple[str, ...]) -> Token | None:
    """Find a label that may be split across words on one row.

    The anchor is the last word of the matched phrase, so the value is the
    next word to its right.
    """
    hits: list[Token] = []
    for row in _rows(tokens):
        ordered = sorted(row, key=lambda token: token.x)
        for start, _token in enumerate(ordered):
            words: list[str] = []
            for token in ordered[start:]:
                words.append(token.text)
                text = norm_text(" ".join(words))
                if any(phrase in text for phrase in phrases):
                    hits.append(token)
                    break
                if len(text) > 60:
                    break
    if not hits:
        return None
    hits.sort(key=lambda token: (token.y, token.x))
    return hits[0]


def _value_beside(label: Token, tokens: list[Token], kind: str) -> Any:
    row = [
        token
        for token in tokens
        if token is not label
        and token.cx > label.cx
        and abs(token.cy - label.cy) <= max(label.height, token.height, 12)
    ]
    row.sort(key=lambda token: token.cx)
    for token in row:
        parsed = _parse_kind(token.text, kind)
        if parsed is not None:
            return parsed
    return None


def _parse_kind(text: str, kind: str) -> Any:
    if kind == "amount":
        try:
            return parse_amount(text)
        except AmountParseError:
            return None
    if kind == "date":
        try:
            return parse_date(text)
        except DateParseError:
            return None
    if kind == "month":
        try:
            return parse_bill_month(text)
        except DateParseError:
            return None
    cleaned = " ".join(text.split())
    return cleaned or None


def _lines(
    tokens: list[Token],
    type_of,
    needles: tuple[str, ...],
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for token in tokens:
        text = norm_text(token.text)
        if not any(needle in text for needle in needles):
            continue
        amount = _value_beside(token, tokens, "amount")
        if amount is None:
            continue
        lines.append({"type": type_of(token.text), "label": token.text.strip(), "amount": amount})
    return lines


def _fuel(tokens: list[Token]) -> dict[str, Any] | None:
    label = _find_label(tokens, ("fpa", "fca", "fuel adjustment", "fuel price"))
    if label is None:
        return None
    amount = _value_beside(label, tokens, "amount")
    month = _month_on_row(label, tokens)
    if amount is None and month is None:
        return None
    return {"label": label.text.strip(), "amount": amount, "month": month}


def _late_tiers(tokens: list[Token]) -> list[dict[str, Any]]:
    tiers: list[dict[str, Any]] = []
    for token in tokens:
        text = norm_text(token.text)
        if not any(word in text for word in ("after due", "till", "lp surcharge", "late payment")):
            continue
        amount = _value_beside(token, tokens, "amount")
        date = _value_beside(token, tokens, "date")
        if amount is None and date is None:
            continue
        tiers.append({"label": token.text.strip(), "date": date, "amount": amount})
    return tiers


def _history(tokens: list[Token]) -> list[dict[str, Any]]:
    rows = _rows(tokens)
    history: list[dict[str, Any]] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.x)
        month_token = ordered[0]
        if _MONTH_TOKEN.fullmatch(month_token.text.strip()) is None:
            continue
        try:
            month = parse_bill_month(month_token.text)
        except DateParseError:
            continue
        numbers = []
        for token in sorted(row, key=lambda item: item.x):
            if token is month_token:
                continue
            try:
                numbers.append(parse_amount(token.text))
            except AmountParseError:
                continue
        history.append(
            {
                "month": month,
                "units": numbers[0] if numbers else None,
                "billed_amount": numbers[1] if len(numbers) > 1 else None,
                "payment_amount": numbers[2] if len(numbers) > 2 else None,
            }
        )
    return history


def _connection(tokens: list[Token]) -> dict[str, Any]:
    for token in tokens:
        text = norm_text(token.text)
        if any(word in text for word in _NET_WORDS):
            return {
                "kind": "net_metered",
                "evidence": token.text.strip(),
                "units": _value_beside(token, tokens, "amount"),
            }
    return {"kind": None, "evidence": None, "units": None}


def _month_on_row(label: Token, tokens: list[Token]) -> str | None:
    if _MONTH_TOKEN.search(label.text):
        try:
            return parse_bill_month(label.text)
        except DateParseError:
            pass
    for token in sorted(tokens, key=lambda item: item.x):
        if abs(token.cy - label.cy) > max(label.height, token.height, 12):
            continue
        if not _MONTH_TOKEN.search(token.text):
            continue
        try:
            return parse_bill_month(token.text)
        except DateParseError:
            continue
    return None


def _rows(tokens: list[Token]) -> list[list[Token]]:
    ordered = sorted(tokens, key=lambda token: (token.cy, token.cx))
    rows: list[list[Token]] = []
    for token in ordered:
        if rows and abs(token.cy - rows[-1][0].cy) <= max(token.height, rows[-1][0].height, 12):
            rows[-1].append(token)
        else:
            rows.append([token])
    return rows


def _note_printed_faults(record: dict[str, Any]) -> None:
    """Record a mismatch and leave both printed figures unchanged."""
    warnings: list[str] = []
    previous = record.get("previous_reading")
    current = record.get("current_reading")
    units = record.get("units_consumed")
    if (
        isinstance(previous, Decimal)
        and isinstance(current, Decimal)
        and isinstance(units, Decimal)
        and current - previous != units
    ):
        warnings.append(
            "The printed units do not match the difference of the printed readings. "
            "The printed figures were kept."
        )
    charges = record.get("charges") or []
    total = record.get("total_charges")
    if charges and isinstance(total, Decimal):
        line_sum = sum((line["amount"] for line in charges), Decimal("0"))
        if line_sum != total:
            warnings.append(
                "The printed charges subtotal does not equal the sum of the charge lines. "
                "The printed subtotal was kept."
            )
    taxes = record.get("taxes") or []
    tax_total = record.get("total_taxes")
    if taxes and isinstance(tax_total, Decimal):
        line_sum = sum((line["amount"] for line in taxes), Decimal("0"))
        if line_sum != tax_total:
            warnings.append(
                "The printed tax subtotal does not equal the sum of the tax lines. "
                "The printed subtotal was kept."
            )
    record["warnings"] = warnings


def refresh_printed_faults(record: dict[str, Any]) -> None:
    """Rebuild the printed-mismatch warnings from the figures currently stored."""
    _note_printed_faults(record)


def fault_errors(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Field errors for the report. The record values are already the printed ones."""
    errors: list[dict[str, Any]] = []
    for message in record.get("warnings") or []:
        field = "units_consumed" if message.startswith("The printed units") else "total_charges"
        if message.startswith("The printed tax"):
            field = "total_taxes"
        errors.append(field_error(field, "printed_mismatch", message, None))
    return errors


def amount_text(value: Decimal | None) -> str:
    if not isinstance(value, Decimal):
        return "not on the bill"
    return format_number(value)


_NUMBER_TOKEN = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_MONTH_FIND = re.compile(
    r"\b(?:sept(?:ember)?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s./-]*\d{2,4}\b",
    re.IGNORECASE,
)


def prepare_answer_record(record: dict[str, Any]) -> dict[str, Any]:
    """Turn text that was stored for a structured field into rows the answers can read."""
    prepared = dict(record)
    prepared["history"] = coerce_history(record.get("history"))
    prepared["late_payment_tiers"] = coerce_late_tiers(record.get("late_payment_tiers"))
    fuel = coerce_fuel(record.get("fuel_adjustment"))
    if fuel is not None or not isinstance(record.get("fuel_adjustment"), dict):
        prepared["fuel_adjustment"] = fuel
    connection = record.get("connection")
    if not isinstance(connection, dict):
        prepared["connection"] = {"kind": None, "evidence": None, "units": None}
    if not isinstance(record.get("warnings"), list):
        prepared["warnings"] = []
    return prepared


def coerce_history(value: Any) -> list[dict[str, Any]]:
    """Accept history rows, or a printed history block stored as one string."""
    if isinstance(value, str):
        return _history_from_text(value)
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        month = item.get("month")
        if isinstance(month, str):
            try:
                month = parse_bill_month(month)
            except DateParseError:
                month = None
        else:
            month = None
        rows.append(
            {
                "month": month,
                "units": _as_amount(item.get("units")),
                "billed_amount": _as_amount(item.get("billed_amount")),
                "payment_amount": _as_amount(item.get("payment_amount")),
            }
        )
    return rows


def coerce_late_tiers(value: Any) -> list[dict[str, Any]]:
    """Accept tier rows, or one printed surcharge line stored as text."""
    if isinstance(value, str):
        return _tiers_from_text(value)
    if not isinstance(value, list):
        return []
    tiers: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            tiers.extend(_tiers_from_text(item))
            continue
        if not isinstance(item, dict):
            continue
        tiers.append(
            {
                "label": item.get("label") or "Late payment",
                "date": item.get("date"),
                "amount": _as_amount(item.get("amount")),
            }
        )
    return tiers


def coerce_fuel(value: Any) -> dict[str, Any] | None:
    """Accept a fuel object, or a printed amount stored as text."""
    if isinstance(value, dict):
        amount = _as_amount(value.get("amount"))
        month = value.get("month")
        if isinstance(month, str):
            try:
                month = parse_bill_month(month)
            except DateParseError:
                month = None
        else:
            month = None
        if amount is None and month is None:
            return None
        return {"label": value.get("label") or "FPA", "amount": amount, "month": month}
    if isinstance(value, str):
        return _fuel_from_text(value)
    return None


def _history_from_text(text: str) -> list[dict[str, Any]]:
    matches = list(_MONTH_FIND.finditer(text))
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        try:
            month = parse_bill_month(match.group(0))
        except DateParseError:
            continue
        numbers = [_as_amount(raw) for raw in _NUMBER_TOKEN.findall(text[match.end() : end])]
        numbers = [number for number in numbers if number is not None]
        rows.append(
            {
                "month": month,
                "units": numbers[0] if numbers else None,
                "billed_amount": numbers[1] if len(numbers) > 1 else None,
                "payment_amount": numbers[2] if len(numbers) > 2 else None,
            }
        )
    return rows


def _tiers_from_text(text: str) -> list[dict[str, Any]]:
    tiers: list[dict[str, Any]] = []
    for raw in _NUMBER_TOKEN.findall(text):
        amount = _as_amount(raw)
        if amount is None:
            continue
        tiers.append({"label": "Late payment", "date": None, "amount": amount})
    return tiers


def _fuel_from_text(text: str) -> dict[str, Any] | None:
    amounts = [_as_amount(raw) for raw in _NUMBER_TOKEN.findall(text)]
    amounts = [amount for amount in amounts if amount is not None]
    if not amounts:
        return None
    month = None
    found = _MONTH_FIND.search(text)
    if found:
        try:
            month = parse_bill_month(found.group(0))
        except DateParseError:
            month = None
    return {"label": "FPA", "amount": amounts[-1], "month": month}


def _as_amount(value: Any) -> Decimal | None:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or value is None or isinstance(value, float):
        return None
    try:
        return parse_amount(value if isinstance(value, str) else str(value))
    except AmountParseError:
        return None
