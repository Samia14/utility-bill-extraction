"""Turn printed bill text into the final bill object.

Printed dates, amounts, and labels are normalized here. Units, taxes, totals,
and payables are never derived from other numbers. A subtotal is compared with
the sum of its lines only to record a warning. The printed subtotal is kept.
"""

from __future__ import annotations

import calendar
import re
from decimal import Decimal
from typing import Any

from bill_extract.errors import AmountParseError, DateParseError, field_error
from bill_extract.schema import BILL_FIELDS

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_MONTH_NAME = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)

_CURRENCY = re.compile(r"\b(?:rs|pkr|rupees)\.?", re.IGNORECASE)


def normalize(raw: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the bill object and any field errors. This does not raise for a bad field."""
    if not isinstance(raw, dict):
        errors = [
            field_error(
                "document",
                "missing_field",
                "The model result was not a JSON object, so the bill fields are empty.",
                raw,
            )
        ]
        return _empty_bill(), errors

    errors: list[dict[str, Any]] = []
    provider = _provider(raw.get("provider_name"), errors)
    tariff = _text_or_none(raw.get("tariff"))
    sanctioned_load_kw = _optional_amount(
        _strip_load_unit(raw.get("sanctioned_load")),
        "sanctioned_load_kw",
        errors,
        "The sanctioned load could not be read as a number.",
    )
    bill_month = _optional_bill_month(raw.get("bill_month"), errors)
    reading_date = _optional_date(raw.get("reading_date"), "reading_date", errors)
    issue_date = _optional_date(raw.get("issue_date"), "issue_date", errors)
    due_date = _optional_date(raw.get("due_date"), "due_date", errors)

    previous_reading, current_reading, units_consumed = _meter(raw, errors)
    charges, total_charges, current_from_charges = _charges(raw, errors)
    taxes, total_taxes = _taxes(raw, errors)
    current_bill = _optional_amount(
        raw.get("current_bill"),
        "current_bill",
        errors,
        "The current bill amount could not be read.",
    )
    if current_bill is None and current_from_charges:
        current_bill = current_from_charges[0]
    arrears = _arrears(raw.get("arrears"), errors)
    payable_within_due_date = _optional_amount(
        raw.get("payable_within_due_date"),
        "payable_within_due_date",
        errors,
        "The amount payable within the due date could not be read.",
    )
    payable_after_due_date = _highest_payable_after_due(
        raw.get("payable_after_due_date_amounts"),
        errors,
    )

    _warn_if_subtotal_differs("total_charges", charges, total_charges, errors)
    _warn_if_subtotal_differs("total_taxes", taxes, total_taxes, errors)

    bill = {
        "provider": provider,
        "tariff": tariff,
        "sanctioned_load_kw": sanctioned_load_kw,
        "bill_month": bill_month,
        "reading_date": reading_date,
        "issue_date": issue_date,
        "due_date": due_date,
        "previous_reading": previous_reading,
        "current_reading": current_reading,
        "units_consumed": units_consumed,
        "charges": charges,
        "total_charges": total_charges,
        "taxes": taxes,
        "total_taxes": total_taxes,
        "current_bill": current_bill,
        "arrears": arrears,
        "payable_within_due_date": payable_within_due_date,
        "payable_after_due_date": payable_after_due_date,
    }
    return {key: bill[key] for key in BILL_FIELDS}, errors


def parse_bill_month(value: str) -> str:
    """Parse a printed bill month into YYYY-MM. Two-digit years use 2000-2099."""
    year, month, _day = _parse_date_parts(value, require_day=False)
    return f"{year:04d}-{month:02d}"


def parse_date(value: str) -> str:
    """Parse a printed day into YYYY-MM-DD. Two-digit years use 2000-2099."""
    year, month, day = _parse_date_parts(value, require_day=True)
    if day is None:
        raise DateParseError("missing day")
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_amount(value: object) -> Decimal:
    """Parse one printed amount. A slash pair is rejected instead of guessed."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or value is None:
        raise AmountParseError("blank" if value is None else "invalid")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        raise AmountParseError("float")
    if not isinstance(value, str):
        raise AmountParseError("invalid")

    text = value.strip().replace("\u2212", "-").replace("\u2013", "-")
    if not text:
        raise AmountParseError("blank")
    if "/" in text:
        raise AmountParseError("slash")

    negative = False
    if "(" in text or ")" in text:
        if not re.search(r"\(\s*[\d,.]+\s*\)", text):
            raise AmountParseError("pattern")
        negative = True
        text = text.replace("(", " ").replace(")", " ")

    text = _CURRENCY.sub(" ", text)
    suffix = re.search(r"\b(cr|dr)\b", text, re.IGNORECASE)
    if suffix:
        if suffix.group(1).casefold() == "cr":
            negative = True
        text = text[: suffix.start()] + " " + text[suffix.end() :]

    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"([+-])?(\d+)(?:\.(\d+))?", text)
    if not match:
        raise AmountParseError("pattern")
    sign, whole, fraction = match.groups()
    if sign == "-":
        negative = True
    number = Decimal(f"{whole}.{fraction}" if fraction is not None else whole)
    if negative:
        number = -number
    return number


def expand_year(year: int) -> int:
    """Map a two-digit year onto 2000-2099. Every year uses the same rule."""
    if 0 <= year <= 99:
        return 2000 + year
    return year


def _provider(value: object, errors: list[dict[str, Any]]) -> str | None:
    if is_blank(value):
        errors.append(
            field_error(
                "provider",
                "missing_field",
                "The provider name is missing from the bill header.",
                value if isinstance(value, str) else None,
            )
        )
        return None
    if not isinstance(value, str):
        errors.append(
            field_error(
                "provider",
                "invalid_provider",
                "The bill header is not KE, LESCO, or IESCO.",
                value,
            )
        )
        return None
    mapped = map_provider(value)
    if mapped is None:
        errors.append(
            field_error(
                "provider",
                "invalid_provider",
                "The bill header is not KE, LESCO, or IESCO.",
                value,
            )
        )
        return None
    return mapped


def map_provider(value: str) -> str | None:
    """Map a printed header to KE, LESCO, or IESCO."""
    compact = re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()
    if "IESCO" in compact or "ISLAMABAD ELECTRIC" in compact:
        return "IESCO"
    if "LESCO" in compact or "LAHORE ELECTRIC" in compact:
        return "LESCO"
    if (
        compact in {"KE", "K E"}
        or "KESC" in compact
        or "K ELECTRIC" in compact
        or "KARACHI ELECTRIC" in compact
        or re.search(r"\bKE\b", compact)
    ):
        return "KE"
    return None


_ROW_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")


def _meter(
    raw: dict[str, Any],
    errors: list[dict[str, Any]],
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    previous_value = raw.get("previous_reading")
    current_value = raw.get("current_reading")
    units_value = raw.get("units_consumed")
    latest = _latest_meter_row(raw.get("meter_rows"))
    if latest is not None:
        if not is_blank(latest.get("previous_reading")):
            previous_value = latest.get("previous_reading")
        if not is_blank(latest.get("current_reading")):
            current_value = latest.get("current_reading")
        if is_blank(units_value) and not is_blank(latest.get("units")):
            units_value = latest.get("units")
    units = _optional_amount(
        units_value,
        "units_consumed",
        errors,
        "The units figure could not be read.",
    )
    previous = _optional_amount(
        previous_value,
        "previous_reading",
        errors,
        "The previous reading could not be read.",
    )
    current = _optional_amount(
        current_value,
        "current_reading",
        errors,
        "The current reading could not be read.",
    )
    return previous, current, units


def _latest_meter_row(value: object) -> dict[str, Any] | None:
    """Pick the meter row with the latest printed date and time. Do not add rows."""
    if not isinstance(value, list) or len(value) < 2:
        return None
    dated: list[tuple[tuple[int, int, int, int, int, int], dict[str, Any]]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        stamp = _meter_stamp(item.get("reading_at"))
        if stamp is not None:
            dated.append((stamp, item))
    if not dated:
        return None
    latest_stamp = max(stamp for stamp, _item in dated)
    chosen = [item for stamp, item in dated if stamp == latest_stamp]
    return chosen[-1]


def _meter_stamp(value: object) -> tuple[int, int, int, int, int, int] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    hour, minute, second = 0, 0, 0
    time_match = _ROW_TIME.search(value)
    date_text = value
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        second = int(time_match.group(3) or 0)
        if hour > 23 or minute > 59 or second > 59:
            return None
        date_text = _ROW_TIME.sub(" ", value)
    date_text = re.sub(r"\s+", " ", date_text).strip(" ,;-")
    try:
        parsed = parse_date(date_text)
    except DateParseError:
        return None
    year, month, day = (int(part) for part in parsed.split("-"))
    return year, month, day, hour, minute, second


def _charges(
    raw: dict[str, Any],
    errors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Decimal | None, list[Decimal]]:
    lines = _line_list(raw.get("charge_lines"), "charges", errors)
    entries: list[dict[str, Any]] = []
    primary: list[Decimal] = []
    net: list[Decimal] = []
    energy_summary: list[Decimal] = []
    current_bill_lines: list[Decimal] = []

    for line in lines:
        label, amount, skipped = _read_line(line, "charges", errors, drop_zero=False)
        if skipped or amount is None:
            continue
        kind = subtotal_kind(label)
        if kind == "primary":
            primary.append(amount)
        elif kind == "net":
            net.append(amount)
        elif kind == "energy_summary":
            energy_summary.append(amount)
        elif kind == "current_bill":
            current_bill_lines.append(amount)
        elif kind == "ignore":
            continue
        elif amount == 0:
            continue
        else:
            entries.append({"type": charge_type(label), "amount": amount})

    explicit_label = raw.get("charges_subtotal_label")
    explicit_amount = _optional_amount(
        raw.get("charges_subtotal_amount"),
        "total_charges",
        errors,
        "The charges subtotal could not be read.",
    )
    if explicit_amount is not None:
        kind = subtotal_kind(explicit_label if isinstance(explicit_label, str) else "")
        if kind == "net":
            net.insert(0, explicit_amount)
        elif kind == "energy_summary":
            energy_summary.insert(0, explicit_amount)
        elif kind == "current_bill":
            current_bill_lines.insert(0, explicit_amount)
        elif kind == "ignore":
            pass
        else:
            primary.insert(0, explicit_amount)

    if primary:
        total_charges = primary[0]
    elif energy_summary:
        total_charges = energy_summary[0]
    elif net:
        total_charges = net[0]
    else:
        total_charges = None
    return entries, total_charges, current_bill_lines


def _taxes(
    raw: dict[str, Any],
    errors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Decimal | None]:
    lines = _line_list(raw.get("tax_lines"), "taxes", errors)
    itemized: list[dict[str, Any]] = []
    generic: list[Decimal] = []

    for line in lines:
        label, amount, skipped = _read_line(line, "taxes", errors, drop_zero=False)
        if skipped or amount is None or amount == 0:
            continue
        if is_generic_tax_label(label):
            generic.append(amount)
        else:
            itemized.append({"type": tax_type(label), "amount": amount})

    subtotal = _optional_amount(
        raw.get("taxes_subtotal_amount"),
        "total_taxes",
        errors,
        "The tax subtotal could not be read.",
    )
    combined_flag = _as_bool(raw.get("taxes_are_combined"))
    one_combined_line = combined_flag and len(itemized) == 1 and not generic
    if itemized and not one_combined_line:
        total_taxes = subtotal if subtotal is not None else (generic[0] if generic else None)
        return itemized, total_taxes

    if subtotal is not None:
        return [], subtotal
    if itemized:
        return [], itemized[0]["amount"]
    if generic:
        return [], generic[0]
    return [], None


def _arrears_amount_text(value: object) -> object:
    """Keep the amount before a slash. The part after the slash is the month count."""
    if not isinstance(value, str):
        return value
    if not re.search(r"[/\\]", value):
        return value
    left, _months = re.split(r"\s*[/\\]\s*", value.strip(), maxsplit=1)
    return left.strip()


def _arrears(value: object, errors: list[dict[str, Any]]) -> Decimal | None:
    if is_blank(value):
        return None
    try:
        return parse_amount(_arrears_amount_text(value))
    except AmountParseError:
        errors.append(
            field_error("arrears", "unparseable_amount", "The arrears amount could not be read.", value)
        )
        return None


def _highest_payable_after_due(value: object, errors: list[dict[str, Any]]) -> Decimal | None:
    """Choose the highest printed late-payment payable total. Do not add a surcharge."""
    if is_blank(value):
        return None
    if not isinstance(value, list):
        errors.append(
            field_error(
                "payable_after_due_date",
                "unparseable_amount",
                "The late-payment payable amounts were not a list.",
                value,
            )
        )
        return None

    parsed: list[Decimal] = []
    for item in value:
        if is_blank(item):
            continue
        try:
            parsed.append(parse_amount(item))
        except AmountParseError:
            errors.append(
                field_error(
                    "payable_after_due_date",
                    "unparseable_amount",
                    "A late-payment payable amount could not be read.",
                    item,
                )
            )
    if not parsed:
        return None
    return max(parsed)


def _warn_if_subtotal_differs(
    field: str,
    lines: list[dict[str, Any]],
    subtotal: Decimal | None,
    errors: list[dict[str, Any]],
) -> None:
    if not lines or subtotal is None:
        return
    line_sum = sum((line["amount"] for line in lines), Decimal("0"))
    if line_sum == subtotal:
        return
    label = "charges" if field == "total_charges" else "taxes"
    errors.append(
        field_error(
            field,
            "subtotal_mismatch",
            f"The printed {label} subtotal does not equal the sum of the {label} lines. "
            "The printed subtotal was kept.",
            format(subtotal, "f"),
        )
    )


def charge_type(label: str) -> str:
    """Map a charge-section label to a stable type. Unknown labels are kept."""
    text = norm_label(label)
    if ("quarterly" in text or re.search(r"\bqtr\b", text)) and "adjust" in text:
        return "quarterly_adjustment"
    if (
        re.search(r"\bfca\b", text)
        or re.search(r"\bfpa\b", text)
        or "fuel price" in text
        or "fuel charge adjustment" in text
        or "fuel adjustment" in text
    ):
        return "fpa"
    if "late payment" in text or re.search(r"\blp surcharge\b", text):
        return "late_payment_surcharge"
    if "surcharge" in text or re.search(r"\bphl\b", text):
        return "surcharge"
    if "fixed" in text and "charge" in text:
        return "fixed"
    if "variable" in text or "unit charge" in text or ("energy" in text and "charge" in text):
        return "energy"
    return snake_case(label)


def tax_type(label: str) -> str:
    """Map a tax-section label to a stable type."""
    text = norm_label(label)
    if "electricity duty" in text or "elec duty" in text or re.search(r"\bed\b", text):
        return "electricity_duty"
    if "sales tax" in text or "general sales tax" in text or re.search(r"\bgst\b", text):
        return "gst"
    if "muct" in text or "municipal" in text:
        return "municipal_tax"
    return snake_case(label)


def subtotal_kind(label: str) -> str | None:
    """Classify a printed total so it is not stored as a charge line."""
    text = norm_label(label)
    if not text:
        return None
    if "for current month" in text or text == "current bill" or text.startswith("current bill "):
        return "current_bill"
    if text.startswith("grand total") or text == "amount due" or text.startswith("amount payable"):
        return "ignore"
    if text.startswith("net electricity"):
        return "net"
    if text in {"electricity charges", "electricity charge"} or text.startswith("total electricity"):
        return "primary"
    if text in {"energy charges", "total charges", "total energy charges"}:
        return "energy_summary"
    return None


def is_generic_tax_label(label: str) -> bool:
    """True when the line is a combined or total tax figure, not one tax type."""
    text = norm_label(label)
    if not text:
        return False
    itemized_markers = (
        "electricity duty",
        "elec duty",
        "sales tax",
        "gst",
        "municipal",
        "muct",
        "income tax",
        "further tax",
        "tv fee",
        "ptv",
    )
    if any(marker in text for marker in itemized_markers):
        return False
    if text in {
        "tax",
        "taxes",
        "duty",
        "duties",
        "taxes and duties",
        "duties and taxes",
        "total taxes",
        "total tax",
        "govt charges",
        "government charges",
        "government dues",
    }:
        return True
    return text.startswith("taxes ") or text.startswith("tax ") or text.startswith("total taxes")


def norm_label(label: str) -> str:
    text = label.casefold().replace("&", " and ").replace("%", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def snake_case(label: str) -> str:
    text = label.casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or "unknown_charge"


def is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _optional_amount(
    value: object,
    field: str,
    errors: list[dict[str, Any]],
    message: str,
) -> Decimal | None:
    if is_blank(value):
        return None
    try:
        return parse_amount(value)
    except AmountParseError as exc:
        if exc.reason == "slash":
            message = "The value has a slash between two numbers, so it was not read as an amount."
        errors.append(field_error(field, "unparseable_amount", message, value))
        return None


def _strip_load_unit(value: object) -> object:
    if not isinstance(value, str):
        return value
    return re.sub(r"\b(?:kw|kva|kilowatts?)\b\.?", " ", value, flags=re.IGNORECASE).strip()


def _optional_date(value: object, field: str, errors: list[dict[str, Any]]) -> str | None:
    if is_blank(value):
        return None
    if not isinstance(value, str):
        errors.append(
            field_error(field, "unparseable_date", "The date could not be read.", value)
        )
        return None
    try:
        return parse_date(value)
    except DateParseError:
        errors.append(
            field_error(field, "unparseable_date", "The date could not be read.", value)
        )
        return None


def _optional_bill_month(value: object, errors: list[dict[str, Any]]) -> str | None:
    if is_blank(value):
        return None
    if not isinstance(value, str):
        errors.append(
            field_error("bill_month", "unparseable_date", "The bill month could not be read.", value)
        )
        return None
    try:
        return parse_bill_month(value)
    except DateParseError:
        errors.append(
            field_error("bill_month", "unparseable_date", "The bill month could not be read.", value)
        )
        return None


def _read_line(
    line: object,
    field: str,
    errors: list[dict[str, Any]],
    drop_zero: bool,
) -> tuple[str, Decimal | None, bool]:
    """Return label, amount, and whether the line should be skipped."""
    if not isinstance(line, dict):
        errors.append(
            field_error(field, "unparseable_amount", "A bill line was not a label and an amount.", line)
        )
        return "", None, True
    label = line.get("label")
    amount_raw = line.get("amount")
    label_text = label.strip() if isinstance(label, str) else ""
    if is_blank(label_text) and is_blank(amount_raw):
        return "", None, True
    if is_blank(amount_raw):
        return label_text, None, True
    try:
        amount = parse_amount(amount_raw)
    except AmountParseError:
        errors.append(
            field_error(
                field,
                "unparseable_amount",
                f"The amount for '{label_text or 'a line'}' could not be read.",
                amount_raw,
            )
        )
        return label_text, None, True
    if drop_zero and amount == 0:
        return label_text, None, True
    return label_text, amount, False


def _line_list(value: object, field: str, errors: list[dict[str, Any]]) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    errors.append(
        field_error(field, "unparseable_amount", "The lines for this section were not a list.", value)
    )
    return []


def _text_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "1"}
    return False


def _empty_bill() -> dict[str, Any]:
    bill: dict[str, Any] = {key: None for key in BILL_FIELDS}
    bill["charges"] = []
    bill["taxes"] = []
    return bill


def _parse_date_parts(value: str, require_day: bool) -> tuple[int, int, int | None]:
    text = value.strip()
    if not text:
        raise DateParseError("blank")
    iso_day = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso_day:
        year, month, day = (int(part) for part in iso_day.groups())
        _validate_date(year, month, day)
        return year, month, day
    iso_month = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if iso_month:
        if require_day:
            raise DateParseError("missing day")
        year, month = (int(part) for part in iso_month.groups())
        if not 1 <= month <= 12:
            raise DateParseError("month")
        return year, month, None

    cleaned = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", " ")
    cleaned = re.sub(r"[-/.]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    named_day = re.fullmatch(rf"(\d{{1,2}}) ({_MONTH_NAME}) (\d{{2,4}})", cleaned, re.IGNORECASE)
    if named_day:
        day_text, month_text, year_text = named_day.groups()
        return _complete_date(int(year_text), _month_number(month_text), int(day_text))

    named_month_first = re.fullmatch(
        rf"({_MONTH_NAME}) (\d{{1,2}}) (\d{{2,4}})",
        cleaned,
        re.IGNORECASE,
    )
    if named_month_first:
        month_text, day_text, year_text = named_month_first.groups()
        return _complete_date(int(year_text), _month_number(month_text), int(day_text))

    named_month = re.fullmatch(rf"({_MONTH_NAME}) (\d{{2,4}})", cleaned, re.IGNORECASE)
    if named_month:
        if require_day:
            raise DateParseError("missing day")
        month_text, year_text = named_month.groups()
        year = expand_year(int(year_text))
        month = _month_number(month_text)
        return year, month, None

    numeric = re.fullmatch(r"(\d{1,4}) (\d{1,2}) (\d{2,4})", cleaned)
    if numeric:
        first, second, third = (int(part) for part in numeric.groups())
        if first > 31:
            year, month, day = expand_year(first), second, third
        else:
            day, month, year = first, second, expand_year(third)
        _validate_date(year, month, day)
        return year, month, day

    numeric_month = re.fullmatch(r"(\d{1,4}) (\d{2,4})", cleaned)
    if numeric_month and not require_day:
        first, second = (int(part) for part in numeric_month.groups())
        if first > 12:
            year, month = expand_year(first), second
        else:
            month, year = first, expand_year(second)
        if not 1 <= month <= 12:
            raise DateParseError("month")
        return year, month, None

    raise DateParseError("pattern")


def _complete_date(year: int, month: int, day: int) -> tuple[int, int, int]:
    year = expand_year(year)
    _validate_date(year, month, day)
    return year, month, day


def _month_number(name: str) -> int:
    try:
        return _MONTHS[name.casefold()]
    except KeyError as exc:
        raise DateParseError("month") from exc


def _validate_date(year: int, month: int, day: int) -> None:
    try:
        last_day = calendar.monthrange(year, month)[1]
    except calendar.IllegalMonthError as exc:
        raise DateParseError("month") from exc
    if day < 1 or day > last_day:
        raise DateParseError("day")
