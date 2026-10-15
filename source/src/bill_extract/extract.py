"""Ask a vision model for the printed text on a bill. Amounts are not calculated here."""

from __future__ import annotations

import base64
import json
import re
import time
from decimal import Decimal
from typing import Any

from bill_extract.config import Config
from bill_extract.errors import DependencyError, ModelError, ModelResponseError

_SYSTEM_PROMPT = """You read Pakistani electricity bills from K-Electric (KE), LESCO, and IESCO.
Read only text that is printed on the page. Copy labels, dates, and amounts as printed.
Do not calculate. Do not subtract readings. Do not multiply units by a rate.
Do not add a within-due amount to a surcharge. Do not invent a missing figure.
If a value is not printed, use null. Keep every amount as a string, including commas, decimals, minus signs, and a CR suffix.

Return one JSON object with these fields:
- provider_name: the utility name in the bill header. Do not take it from a file name.
- tariff: the printed tariff code, copied exactly.
- sanctioned_load: the number printed as Sanc Load, S.Load, or San Load. If the bill only shows Conn Load or MDI, use null.
- bill_month, reading_date, issue_date: the printed text.
- due_date: the due date, which is the last date to pay with no late-payment surcharge. Do not return a Till date or an After date.
- meter_ambiguous: always false. Extra meter rows are not an error.
- meter_rows: one object per printed meter row, in printed order. Each object has previous_reading, current_reading, units, and reading_at. reading_at is the date and time printed on that row, copied as printed, or null when that row has no date. Do not copy the bill month or the single bill reading date onto a row that does not print its own date. Do not add rows together.
- previous_reading and current_reading: when several rows print a date or time, copy the previous and current readings from the latest row. Otherwise copy the printed billed row. Do not leave them null because other rows exist, and do not add rows together.
- units_consumed: the printed units for the month, or the single UNITS cell for the billed register. Do not add off-peak, peak, and import units. Do not subtract the readings.
- meter_note: null. A short note is not required.
- charge_lines: every line in the charges section, in printed order, including zero amounts. Each item is {"label", "amount"}. Do not put tax lines, arrears, subtotals, current-bill totals, or payable totals in this list.
- charges_subtotal_label and charges_subtotal_amount: the printed Electricity Charges or Total Electricity Charges line. Do not use Net Electricity Charges when a total line is also printed. Do not use Electricity Charges for current month, Grand Total, or Amount Due. Use null when that subtotal is not printed. Do not add the lines yourself.
- tax_lines: every line in the tax or government section, including zero amounts. If the bill prints only one combined tax amount, put that one line here.
- taxes_are_combined: true only when the bill prints one combined tax amount and does not list separate tax lines. Otherwise false.
- taxes_subtotal_label and taxes_subtotal_amount: the printed Taxes and Duties total, or the combined Taxes line. Use null when no tax total is printed. Do not add the tax lines yourself.
- current_bill: Electricity Charges for current month, or Current Bill. Not Amount Due and not Grand Total.
- arrears: previous dues carried forward, with the printed sign or CR suffix. Use null when the space is blank. If the print is an amount and a month count separated by / or \\, copy that full text unchanged.
- payable_within_due_date: the amount payable by the due date.
- payable_after_due_date_amounts: every printed payable total that applies after the due date, including each Till and After tier. Use the payable totals only. Do not include surcharge-only amounts such as LP surcharge. Do not add the within-due amount to a surcharge.
- unreadable_fields: field names you could not read. Use an empty array when every located field was readable.
- field_boxes: one object for each field you can locate, with field, x, y, width, and height. x and y are the top-left corner as fractions of the image width and height, from 0 to 1. width and height are fractions of the image too. Use an empty array when you cannot locate a field.
- charges_section_present: true when a charges or bill-calculation section is printed, even if its lines could not be read.
- taxes_section_present: true when a tax or government section is printed, even if its lines could not be read.
"""


def _nullable(kind: str) -> dict[str, Any]:
    return {"anyOf": [{"type": kind}, {"type": "null"}]}


def _box_array() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["field", "x", "y", "width", "height"],
            "properties": {
                "field": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number"},
                "height": {"type": "number"},
            },
        },
    }


def _meter_row_array() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["previous_reading", "current_reading", "units", "reading_at"],
            "properties": {
                "previous_reading": _nullable("string"),
                "current_reading": _nullable("string"),
                "units": _nullable("string"),
                "reading_at": _nullable("string"),
            },
        },
    }


def _line_array() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "amount"],
            "properties": {
                "label": {"type": "string"},
                "amount": _nullable("string"),
            },
        },
    }


_RAW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "provider_name",
        "tariff",
        "sanctioned_load",
        "bill_month",
        "reading_date",
        "issue_date",
        "due_date",
        "meter_ambiguous",
        "meter_rows",
        "previous_reading",
        "current_reading",
        "units_consumed",
        "meter_note",
        "charge_lines",
        "charges_subtotal_label",
        "charges_subtotal_amount",
        "tax_lines",
        "taxes_are_combined",
        "taxes_subtotal_label",
        "taxes_subtotal_amount",
        "current_bill",
        "arrears",
        "payable_within_due_date",
        "payable_after_due_date_amounts",
        "unreadable_fields",
        "field_boxes",
        "charges_section_present",
        "taxes_section_present",
    ],
    "properties": {
        "provider_name": _nullable("string"),
        "tariff": _nullable("string"),
        "sanctioned_load": _nullable("string"),
        "bill_month": _nullable("string"),
        "reading_date": _nullable("string"),
        "issue_date": _nullable("string"),
        "due_date": _nullable("string"),
        "meter_ambiguous": {"type": "boolean"},
        "meter_rows": _meter_row_array(),
        "previous_reading": _nullable("string"),
        "current_reading": _nullable("string"),
        "units_consumed": _nullable("string"),
        "meter_note": _nullable("string"),
        "charge_lines": _line_array(),
        "charges_subtotal_label": _nullable("string"),
        "charges_subtotal_amount": _nullable("string"),
        "tax_lines": _line_array(),
        "taxes_are_combined": {"type": "boolean"},
        "taxes_subtotal_label": _nullable("string"),
        "taxes_subtotal_amount": _nullable("string"),
        "current_bill": _nullable("string"),
        "arrears": _nullable("string"),
        "payable_within_due_date": _nullable("string"),
        "payable_after_due_date_amounts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unreadable_fields": {
            "type": "array",
            "items": {"type": "string"},
        },
        "field_boxes": _box_array(),
        "charges_section_present": {"type": "boolean"},
        "taxes_section_present": {"type": "boolean"},
    },
}


def build_client(config: Config):
    """Create the OpenAI client. The key is not written to logs."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise DependencyError(
            "The openai package is not installed. Install the packages in requirements.txt and try again."
        ) from exc
    return OpenAI(api_key=config.api_key, timeout=120.0, max_retries=0)


def extract_raw(png_bytes: bytes, config: Config, client: Any) -> dict[str, Any]:
    """Read one page and return the raw printed fields. One repair call is allowed."""
    encoded = base64.standard_b64encode(png_bytes).decode("ascii")
    return _read_with_repair(client, config, _vision_payload(config, encoded))


def extract_fields(
    png_bytes: bytes,
    config: Config,
    client: Any,
    fields: list[str],
) -> dict[str, Any]:
    """Read only the requested fields from a crop. One repair call is allowed."""
    encoded = base64.standard_b64encode(png_bytes).decode("ascii")
    return _read_with_repair(client, config, _crop_payload(config, encoded, fields))


def _read_with_repair(client: Any, config: Config, payload: dict[str, Any]) -> dict[str, Any]:
    text = _complete(client, payload)
    try:
        return parse_model_json(text)
    except ModelResponseError as first_error:
        try:
            repaired = _complete(client, _repair_payload(config, text, str(first_error)))
            return parse_model_json(repaired)
        except ModelResponseError as exc:
            raise ModelResponseError(
                "The model returned invalid JSON, and the repair attempt also failed."
            ) from exc


def parse_model_json(text: str) -> dict[str, Any]:
    """Parse model JSON. Numbers are Decimals so later steps do not use binary floats."""
    if not isinstance(text, str) or not text.strip():
        raise ModelResponseError("The model returned an empty response.")
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate, parse_float=Decimal, parse_int=Decimal)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ModelResponseError("The model response was not valid JSON.")
        try:
            value = json.loads(candidate[start : end + 1], parse_float=Decimal, parse_int=Decimal)
        except json.JSONDecodeError as exc:
            raise ModelResponseError("The model response was not valid JSON.") from exc
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict):
        raise ModelResponseError("The model JSON was not an object.")
    return value


def _response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "printed_bill",
            "strict": True,
            "schema": _RAW_SCHEMA,
        },
    }


def _vision_payload(config: Config, encoded_png: str) -> dict[str, Any]:
    return {
        "model": config.model_name,
        "temperature": 0,
        "max_completion_tokens": 16000,
        "response_format": _response_format(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this electricity bill and return the JSON object."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_png}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }


def _crop_payload(config: Config, encoded_png: str, fields: list[str]) -> dict[str, Any]:
    names = ", ".join(fields) if fields else "the unread fields"
    return {
        "model": config.model_name,
        "temperature": 0,
        "max_completion_tokens": 16000,
        "response_format": _response_format(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This image is a zoomed copy of an electricity bill. "
                            f"Read only these fields: {names}. "
                            "Copy the printed text. Do not calculate. "
                            "Leave every other field null or empty. "
                            "If a requested field is still unreadable, leave it null and "
                            "include its name in unreadable_fields."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_png}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }


def _repair_payload(config: Config, bad_text: str, problem: str) -> dict[str, Any]:
    clipped = (bad_text or "")[:12000]
    return {
        "model": config.model_name,
        "temperature": 0,
        "max_completion_tokens": 16000,
        "response_format": _response_format(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "The previous response was not valid JSON. "
                    f"Problem: {problem}\n"
                    "Return only the corrected JSON object. "
                    "Keep every amount and date exactly as it appeared. "
                    "Do not calculate any new amount. If a value was not present, use null.\n\n"
                    f"Previous response:\n{clipped}"
                ),
            },
        ],
    }


def _complete(client: Any, payload: dict[str, Any]) -> str:
    """Call the model. Retry timeout, 429, and 5xx with backoff. Never log the API key."""
    delay_seconds = 1.0
    last_message = "The model request failed."
    for attempt in range(3):
        try:
            response = _create(client, payload)
            return _message_text(response)
        except (ModelError, ModelResponseError):
            raise
        except Exception as exc:
            if not _is_retryable(exc):
                raise ModelError(_public_model_message(exc)) from exc
            last_message = _public_model_message(exc)
            if attempt == 2:
                break
            time.sleep(delay_seconds)
            delay_seconds *= 2
    raise ModelError(last_message)


def _create(
    client: Any,
    payload: dict[str, Any],
    *,
    allow_temperature: bool = True,
    allow_schema: bool = True,
):
    body = dict(payload)
    if not allow_temperature:
        body.pop("temperature", None)
    if not allow_schema:
        body["response_format"] = {"type": "json_object"}
    try:
        return client.chat.completions.create(**body)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        detail = _safe_detail(exc)
        if allow_temperature and status == 400 and "temperature" in detail:
            return _create(client, payload, allow_temperature=False, allow_schema=allow_schema)
        token_rejected = "max_completion_tokens" in detail
        schema_rejected = "response_format" in detail or "json_schema" in detail
        if allow_schema and status == 400 and token_rejected and not schema_rejected:
            alternate = dict(payload)
            alternate.pop("max_completion_tokens", None)
            alternate["max_tokens"] = payload.get("max_completion_tokens", 16000)
            return _create(client, alternate, allow_temperature=allow_temperature, allow_schema=True)
        if allow_schema and status == 400 and schema_rejected:
            return _create(client, payload, allow_temperature=allow_temperature, allow_schema=False)
        raise


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"APITimeoutError", "APIConnectionError", "RateLimitError"}:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def _public_model_message(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status == 429 or name == "RateLimitError":
        return "The model request was rate limited."
    if "Timeout" in name:
        return "The model request timed out."
    if "Connection" in name:
        return "The model request could not connect."
    if isinstance(status, int):
        return f"The model request failed with status {status}."
    return "The model request failed."


def _safe_detail(exc: Exception) -> str:
    text = _redact(str(exc)).casefold()
    if len(text) > 2000:
        return text[:2000]
    return text


def _redact(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_\-]+", "***", text)


def _message_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ModelResponseError("The model returned no choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ModelResponseError("The model returned an empty response.")
    content = getattr(message, "content", None)
    text = _content_text(content)
    if text.strip():
        return text
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ModelResponseError("The model declined to read the bill.")
    raise ModelResponseError("The model returned an empty response.")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(str(part_text))
        return "".join(parts)
    return ""
