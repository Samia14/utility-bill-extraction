"""The fixed questions that can be answered from a saved bill record."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from bill_extract.assemble import amount_text
from bill_extract.errors import DateParseError
from bill_extract.normalize import parse_bill_month
from bill_extract.question_store import normalize_question

Answerer = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class FixedQuestion:
    question_id: str
    wording: str
    answer: Answerer


def match_question(question: str) -> FixedQuestion | None:
    """Return the catalog entry whose wording matches. Nothing else matches."""
    key = normalize_question(question)
    if not key:
        return None
    for item in CATALOG:
        if normalize_question(item.wording) == key:
            return item
    return None


def _series(record: dict[str, Any]) -> list[tuple[str, Decimal]]:
    rows: list[tuple[str, Decimal]] = []
    seen: set[str] = set()
    for row in _dict_rows(record.get("history")):
        month = row.get("month")
        units = row.get("units")
        if isinstance(month, str) and isinstance(units, Decimal):
            rows.append((month, units))
            seen.add(month)
    current = record.get("bill_month")
    units = record.get("units_consumed")
    if isinstance(current, str) and isinstance(units, Decimal) and current not in seen:
        rows.append((current, units))
    return rows


def _month_number(month: str) -> int | None:
    try:
        parsed = parse_bill_month(month)
    except DateParseError:
        return None
    return int(parsed[5:7])


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _answer_connection(record: dict[str, Any]) -> str:
    connection = record.get("connection")
    if not isinstance(connection, dict):
        connection = {}
    evidence = connection.get("evidence")
    if connection.get("kind") == "net_metered" and evidence:
        units = connection.get("units")
        extra = ""
        if isinstance(units, Decimal):
            extra = f" The units printed with it are {amount_text(units)}."
        return (
            f"This is a net-metered connection. The bill prints '{evidence}'.{extra}"
        )
    return (
        "The bill does not print a net-metering, solar, import, or export label. "
        "On that basis it reads as a standard connection."
    )


def _answer_highest(record: dict[str, Any]) -> str:
    series = _series(record)
    if not series:
        return "The units for the months on this bill are not on the bill."
    highest = max(units for _month, units in series)
    months = [month for month, units in series if units == highest]
    names = ", ".join(months)
    if len(months) == 1:
        return f"The highest units were {amount_text(highest)} in {names}."
    return f"The highest units were {amount_text(highest)}, tied in {names}."


def _answer_fuel(record: dict[str, Any]) -> str:
    fuel = record.get("fuel_adjustment")
    if not isinstance(fuel, dict) or fuel.get("amount") is None:
        return "The fuel adjustment amount is not on the bill."
    month = fuel.get("month")
    amount = amount_text(fuel.get("amount"))
    if not month:
        return f"The fuel adjustment is {amount}. The month for that charge is not on the bill."
    return f"The fuel adjustment is {amount} for {month}."


def _answer_late(record: dict[str, Any]) -> str:
    tiers = _dict_rows(record.get("late_payment_tiers"))
    printed = [tier for tier in tiers if tier.get("amount") is not None or tier.get("date")]
    if not printed:
        return "The late-payment amounts are not on the bill."
    parts: list[str] = []
    for tier in printed:
        label = str(tier.get("label") or "Late payment").strip()
        amount = amount_text(tier.get("amount")) if tier.get("amount") is not None else "an amount that is not printed"
        when = f" on {tier['date']}" if tier.get("date") else ""
        parts.append(f"{label} is {amount}{when}")
    return "The bill prints these late-payment figures: " + "; ".join(parts) + "."


def _answer_last_year(record: dict[str, Any]) -> str:
    current_month = record.get("bill_month")
    current_units = record.get("units_consumed")
    if not isinstance(current_month, str) or not isinstance(current_units, Decimal):
        return "This month's units are not on the bill."
    try:
        year = int(current_month[:4])
        month = int(current_month[5:7])
    except (ValueError, IndexError):
        return "This month's bill month is not on the bill."
    target = f"{year - 1:04d}-{month:02d}"
    previous = next((units for name, units in _series(record) if name == target), None)
    if previous is None:
        return f"The units for {target} are not on the bill."
    if current_units > previous:
        return (
            f"Yes. This month is {amount_text(current_units)} units and "
            f"{target} was {amount_text(previous)} units."
        )
    if current_units < previous:
        return (
            f"No. This month is {amount_text(current_units)} units and "
            f"{target} was {amount_text(previous)} units."
        )
    return (
        f"This month uses the same {amount_text(current_units)} units as {target}."
    )


def _answer_average(record: dict[str, Any]) -> str:
    series = _series(record)
    if not series:
        return "The monthly units are not on the bill."
    total = sum((units for _month, units in series), Decimal("0"))
    average = (total / Decimal(len(series))).quantize(Decimal("0.01"))
    current = record.get("units_consumed")
    sentence = (
        f"The average of the {len(series)} months printed on the bill is "
        f"{amount_text(average)} units."
    )
    if isinstance(current, Decimal):
        if current > average:
            sentence += f" This month's {amount_text(current)} units are above that average."
        elif current < average:
            sentence += f" This month's {amount_text(current)} units are below that average."
        else:
            sentence += f" This month's {amount_text(current)} units match that average."
    return sentence


def _answer_seasons(record: dict[str, Any]) -> str:
    summer_months = {6, 7, 8}
    winter_months = {12, 1, 2}
    summer: list[Decimal] = []
    winter: list[Decimal] = []
    summer_found: set[int] = set()
    winter_found: set[int] = set()
    for month, units in _series(record):
        number = _month_number(month)
        if number in summer_months:
            summer.append(units)
            summer_found.add(number)
        elif number in winter_months:
            winter.append(units)
            winter_found.add(number)
    if not summer or not winter:
        return "The bill does not show both a summer month and a winter month."
    difference = sum(summer, Decimal("0")) - sum(winter, Decimal("0"))
    missing = []
    for number, name in (
        (6, "June"),
        (7, "July"),
        (8, "August"),
        (12, "December"),
        (1, "January"),
        (2, "February"),
    ):
        found = summer_found if number in summer_months else winter_found
        if number not in found:
            missing.append(name)
    sentence = (
        f"Summer units minus winter units, using the months printed on the bill, "
        f"is {amount_text(difference)}."
    )
    if missing:
        sentence += " These months are not on the bill: " + ", ".join(missing) + "."
    return sentence


def _answer_payments(record: dict[str, Any]) -> str:
    history = _dict_rows(record.get("history"))
    if not history:
        return "The payment history is not on the bill."
    unpaid: list[str] = []
    missing: list[str] = []
    for row in history:
        month = str(row.get("month") or "a month")
        payment = row.get("payment_amount")
        if payment is None:
            missing.append(month)
        elif isinstance(payment, Decimal) and payment == 0:
            unpaid.append(month)
    if unpaid:
        return "No. These months show a payment of 0: " + ", ".join(unpaid) + "."
    if missing:
        return "A payment amount is not printed for " + ", ".join(missing) + "."
    return "Yes. Every month shown on the bill has a payment amount above 0."


def _answer_trend(record: dict[str, Any]) -> str:
    series = _series(record)
    if len(series) < 3:
        return "The bill does not show three months of units."
    last = series[-3:]
    first, middle, end = last
    if first[1] < middle[1] < end[1]:
        direction = "up"
    elif first[1] > middle[1] > end[1]:
        direction = "down"
    else:
        direction = "mixed"
    detail = ", ".join(f"{month} {amount_text(units)}" for month, units in last)
    return f"Over the last 3 printed months, units have gone {direction}: {detail}."


def _answer_next(record: dict[str, Any]) -> str:
    series = _series(record)
    if len(series) < 1:
        return "The monthly units are not on the bill, so the next month cannot be estimated."
    last = series[-3:] if len(series) >= 3 else series
    total = sum((units for _month, units in last), Decimal("0"))
    average = (total / Decimal(len(last))).quantize(Decimal("0.01"))
    return (
        f"Estimate: about {amount_text(average)} units. "
        f"This uses the average of the last {len(last)} printed months."
    )


def _answer_set_aside(record: dict[str, Any]) -> str:
    amounts = [
        row["billed_amount"]
        for row in _dict_rows(record.get("history"))
        if isinstance(row.get("billed_amount"), Decimal)
    ]
    if not amounts:
        return "The billed amounts for the months on this bill are not on the bill."
    average = (sum(amounts, Decimal("0")) / Decimal(len(amounts))).quantize(Decimal("0.01"))
    return (
        f"Estimate: set aside {amount_text(average)} each month. "
        f"This is the average of the {len(amounts)} billed amounts printed on the bill."
    )


def _answer_same_units(record: dict[str, Any]) -> str:
    history = [
        row for row in _dict_rows(record.get("history")) if isinstance(row.get("units"), Decimal)
    ]
    if not history:
        return "Last month's units are not on the bill."
    last_units = history[-1]["units"]
    rate = record.get("unit_rate")
    if not isinstance(rate, Decimal):
        return "The bill does not show a unit rate, so a bill for those units cannot be estimated."
    energy = (rate * last_units).quantize(Decimal("0.01"))
    fixed = record.get("fixed_charge")
    if isinstance(fixed, Decimal):
        total = energy + fixed
        return (
            f"Estimate: about {amount_text(total)}. "
            f"This is last month's {amount_text(last_units)} units times the printed rate "
            f"{amount_text(rate)}, plus the printed fixed charge {amount_text(fixed)}."
        )
    return (
        f"Estimate: about {amount_text(energy)} for the energy. "
        f"This is last month's {amount_text(last_units)} units times the printed rate "
        f"{amount_text(rate)}. The bill does not show a fixed charge to add."
    )


CATALOG: tuple[FixedQuestion, ...] = (
    FixedQuestion(
        "Q1",
        "Is this a net-metered (solar) connection or a standard connection? How did you tell?",
        _answer_connection,
    ),
    FixedQuestion(
        "Q2",
        "Which month in the last 12 months had the highest units consumed?",
        _answer_highest,
    ),
    FixedQuestion(
        "Q3",
        "How much fuel adjustment am I being charged (shown as FPA or FCA on the bill), and for which month?",
        _answer_fuel,
    ),
    FixedQuestion(
        "Q4",
        "If I don't pay on time, what late payment surcharges will be added?",
        _answer_late,
    ),
    FixedQuestion(
        "Q5",
        "Am I consuming more units than in the same month last year?",
        _answer_last_year,
    ),
    FixedQuestion(
        "Q6",
        "What are my average monthly units consumed, and how does this month compare?",
        _answer_average,
    ),
    FixedQuestion(
        "Q7",
        "How many more units do I consume in summer (June-August) than in winter (December-February)?",
        _answer_seasons,
    ),
    FixedQuestion(
        "Q8",
        "Looking at the payments for the 12 months shown on the bill, have I been paying every bill?",
        _answer_payments,
    ),
    FixedQuestion(
        "Q9",
        "Have my units consumed gone up or down over the last 3 months?",
        _answer_trend,
    ),
    FixedQuestion(
        "Q10",
        "Based on the 12 months trend, how many units should I expect to consume next month?",
        _answer_next,
    ),
    FixedQuestion(
        "Q11",
        "If I wanted to spread my electricity cost evenly across the year, how much should I set aside each month?",
        _answer_set_aside,
    ),
    FixedQuestion(
        "Q12",
        "If I consume the same units as last month, what would my bill be roughly?",
        _answer_same_units,
    ),
)
