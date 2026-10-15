"""Exception types and field-error records for bill extraction."""

from __future__ import annotations

import json
from typing import Any


class BillExtractError(Exception):
    """Base error for bill extraction."""


class ConfigurationError(BillExtractError):
    """The API key or model name is missing or blank."""


class DependencyError(BillExtractError):
    """A required package is not installed."""


class DocumentLoadError(BillExtractError):
    """A bill file is missing, unsupported, or unreadable."""


class ModelError(BillExtractError):
    """The model request failed after the allowed retries."""


class ModelResponseError(BillExtractError):
    """The model response was not valid JSON after one repair attempt."""


class AmountParseError(ValueError):
    """A printed amount could not be read."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class DateParseError(ValueError):
    """A printed date could not be read."""


def field_error(field: str, code: str, message: str, raw: Any = None) -> dict[str, Any]:
    """Build one field error. The raw printed text is kept when we have it."""
    return {
        "field": field,
        "code": code,
        "message": message,
        "raw": _raw_text(raw),
    }


def _raw_text(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, ensure_ascii=False, default=str)
    return str(raw)
