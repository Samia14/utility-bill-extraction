"""Second read for fields the full-page model left unread.

A temporary copy of the image is zoomed with OpenCV. The zoomed copy is read
again, and OCR is used only for a value that is still missing. The original
file is never modified. The copy is deleted even when the later reads fail.
Already read values are not replaced, and missing lines are not invented.
"""

from __future__ import annotations

import copy
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from bill_extract.errors import field_error

_CORE_FIELDS = (
    "provider_name",
    "tariff",
    "bill_month",
    "reading_date",
    "issue_date",
    "due_date",
    "units_consumed",
    "current_bill",
    "payable_within_due_date",
)

_OPTIONAL_FIELDS = (
    "sanctioned_load",
    "arrears",
    "charges_subtotal_amount",
    "taxes_subtotal_amount",
    "payable_after_due_date_amounts",
)

_UNREADABLE_TOKENS = {"unreadable", "illegible", "not readable", "cannot read"}

Reader = Callable[[bytes, list[str]], dict[str, Any]]
Zoomer = Callable[[Path], bytes]
TextReader = Callable[[bytes], str]

_MONTH_NAME = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)
_DATE_TEXT = re.compile(
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAME})[\s,./-]+\d{{2,4}}"
    rf"|(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?[\s,./-]+\d{{2,4}}"
    rf"|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}",
    re.IGNORECASE,
)
_MONTH_TEXT = re.compile(rf"(?:{_MONTH_NAME})[\s,./-]+\d{{2,4}}", re.IGNORECASE)
_AMOUNT_TEXT = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*[/\\]\s*\d+)?(?:\s*cr)?",
    re.IGNORECASE,
)
_TARIFF_TEXT = re.compile(r"\b[A-Z]-?\d[A-Z0-9()/-]*", re.IGNORECASE)
_DATE_FIELDS = {"reading_date", "issue_date", "due_date"}
_AMOUNT_FIELDS = {
    "sanctioned_load",
    "previous_reading",
    "current_reading",
    "units_consumed",
    "current_bill",
    "arrears",
    "payable_within_due_date",
    "charges_subtotal_amount",
    "taxes_subtotal_amount",
}


def find_missing_fields(raw: dict[str, Any]) -> list[str]:
    """Return field names that are blank or marked unreadable."""
    if not isinstance(raw, dict):
        return []
    unreadable = _unreadable_names(raw)
    missing: list[str] = []

    for field in _CORE_FIELDS:
        if field in unreadable or not _filled(raw.get(field)):
            missing.append(field)

    for field in ("previous_reading", "current_reading"):
        if field in unreadable or not _filled(raw.get(field)):
            missing.append(field)

    for field in _OPTIONAL_FIELDS:
        if field in unreadable or _is_unreadable_token(raw.get(field)):
            missing.append(field)

    if _charges_section_present(raw, unreadable) and not _has_items(raw.get("charge_lines")):
        missing.append("charge_lines")
    if _taxes_need_lines(raw, unreadable):
        missing.append("tax_lines")

    return list(dict.fromkeys(missing))


def merge_unread(base: dict[str, Any], crop: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Fill only fields that are still unread. Keep every value the full page already returned."""
    merged = copy.deepcopy(base)
    if not isinstance(crop, dict):
        return merged
    for field in fields:
        incoming = crop.get(field)
        current = merged.get(field)
        if field in {"charge_lines", "tax_lines", "payable_after_due_date_amounts"}:
            if not _has_items(current) and _has_items(incoming):
                merged[field] = copy.deepcopy(incoming)
            continue
        if not _filled(current) and _filled(incoming):
            merged[field] = copy.deepcopy(incoming) if isinstance(incoming, (dict, list)) else incoming
    return merged


def reread_missing(
    source_path: Path | None,
    image_bytes: bytes,
    raw: dict[str, Any],
    reader: Reader,
    zoomer: Zoomer | None = None,
    ocr_reader: TextReader | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Zoom a temporary copy with OpenCV and fill unread fields from that image."""
    if not isinstance(raw, dict):
        return raw, []
    missing = find_missing_fields(raw)
    if not missing:
        return raw, []

    enlarge = zoomer or zoom_image_file
    read_text = ocr_reader or ocr_zoomed_text
    merged = copy.deepcopy(raw)
    errors: list[dict[str, Any]] = []
    temp_path: Path | None = None
    try:
        try:
            temp_path = write_temp_copy(source_path, image_bytes)
        except OSError:
            return raw, [_missing_error(field, "A temporary copy of the image could not be made.") for field in missing]

        try:
            zoomed = enlarge(temp_path)
        except Exception:
            return merged, [
                _missing_error(
                    field,
                    "The copy could not be zoomed for this field. The other fields were kept.",
                )
                for field in missing
            ]

        ocr_text: str | None = None
        for field in missing:
            try:
                extra = reader(zoomed, [field])
                if isinstance(extra, dict):
                    merged = merge_unread(merged, extra, [field])
            except Exception:
                pass
            if not _field_was_filled(raw, merged, field):
                try:
                    if ocr_text is None:
                        ocr_text = read_text(zoomed)
                    merged = merge_unread(merged, reading_from_ocr(field, ocr_text), [field])
                except Exception:
                    ocr_text = ""
            if not _field_was_filled(raw, merged, field):
                errors.append(
                    _missing_error(
                        field,
                        "The zoomed copy could not supply this field, so it was left empty.",
                    )
                )
        return merged, errors
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def write_temp_copy(source_path: Path | None, image_bytes: bytes) -> Path:
    """Copy the original image to a temp file. The original is only read."""
    suffix = ".png"
    if source_path is not None and source_path.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
        suffix = source_path.suffix
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.close()
    destination = Path(handle.name)
    try:
        if source_path is not None and source_path.is_file():
            shutil.copyfile(source_path, destination)
        else:
            destination.write_bytes(image_bytes)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def zoom_image_file(path: Path) -> bytes:
    """Enlarge the temporary copy with OpenCV. The original file is not opened for writing."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed, so the copy could not be zoomed.") from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("The temporary image could not be opened.")
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side < 1200:
        factor = 3.0
    elif long_side < 2200:
        factor = 2.0
    else:
        factor = 1.5
    if long_side * factor > 3600:
        factor = 3600 / long_side
    zoomed = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".png", zoomed)
    if not ok:
        raise RuntimeError("The zoomed copy could not be encoded.")
    return encoded.tobytes()


def ocr_zoomed_text(image_png: bytes) -> str:
    """Read the zoomed copy with Tesseract. Empty text means the value stayed unread."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("OCR is not installed, so the zoomed copy could not be read.") from exc

    import io

    _use_installed_tesseract(pytesseract)
    image = Image.open(io.BytesIO(image_png))
    try:
        text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
    except pytesseract.TesseractError:
        text = pytesseract.image_to_string(image, config="--psm 6")
    return text or ""


def _use_installed_tesseract(pytesseract: Any) -> None:
    import shutil

    if shutil.which("tesseract"):
        return
    installed = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if installed.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(installed)


def reading_from_ocr(field: str, text: str) -> dict[str, Any]:
    """Turn OCR text for one unread area into that field. Do not calculate a new amount."""
    if not isinstance(text, str) or not text.strip():
        return {}
    if field in _DATE_FIELDS:
        found = _search(_DATE_TEXT, text)
        return {field: found} if found else {}
    if field == "bill_month":
        found = _search(_MONTH_TEXT, text) or _search(_DATE_TEXT, text)
        return {field: found} if found else {}
    if field == "arrears":
        pair = re.search(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*[/\\]\s*\d+", text)
        if pair:
            return {field: pair.group(0)}
        found = _search(_AMOUNT_TEXT, text)
        return {field: found} if found else {}
    if field in _AMOUNT_FIELDS:
        found = _last_amount(text)
        return {field: found} if found else {}
    if field == "payable_after_due_date_amounts":
        amounts = _AMOUNT_TEXT.findall(text)
        return {field: amounts} if amounts else {}
    if field in {"charge_lines", "tax_lines"}:
        lines = _ocr_lines(text)
        return {field: lines} if lines else {}
    if field == "provider_name":
        found = _provider_text(text)
        return {field: found} if found else {}
    if field == "tariff":
        found = _search(_TARIFF_TEXT, text)
        return {field: found} if found else {}
    stripped = text.strip()
    return {field: stripped} if stripped else {}


def _search(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(0).strip()


def _last_amount(text: str) -> str | None:
    amounts = [item.strip() for item in _AMOUNT_TEXT.findall(text) if item.strip()]
    if not amounts:
        return None
    return amounts[-1]


def _ocr_lines(text: str) -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"(.+?)\s+([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*$", line)
        if match is None:
            continue
        label = match.group(1).strip(" :-")
        amount = match.group(2).strip()
        if label and amount and amount not in {"0", "0.00"}:
            lines.append({"label": label, "amount": amount})
    return lines


def _provider_text(text: str) -> str | None:
    compact = re.sub(r"[^A-Z0-9]+", " ", text.upper())
    if "IESCO" in compact or "ISLAMABAD ELECTRIC" in compact:
        return "IESCO"
    if "LESCO" in compact or "LAHORE ELECTRIC" in compact:
        return "LESCO"
    if "KESC" in compact or "K ELECTRIC" in compact or "KARACHI ELECTRIC" in compact or re.search(r"\bKE\b", compact):
        return "KE"
    return None


def _missing_error(field: str, message: str) -> dict[str, Any]:
    return field_error(field, "missing_field", message, None)


def _field_was_filled(before: dict[str, Any], after: dict[str, Any], field: str) -> bool:
    if field in {"charge_lines", "tax_lines", "payable_after_due_date_amounts"}:
        return _has_items(after.get(field)) and not _has_items(before.get(field)) or _has_items(before.get(field))
    return _filled(after.get(field))


def _charges_section_present(raw: dict[str, Any], unreadable: set[str]) -> bool:
    if raw.get("charges_section_present") is True or "charge_lines" in unreadable:
        return True
    return _filled(raw.get("charges_subtotal_amount")) and not _has_items(raw.get("charge_lines"))


def _taxes_need_lines(raw: dict[str, Any], unreadable: set[str]) -> bool:
    if _has_items(raw.get("tax_lines")):
        return False
    if raw.get("taxes_are_combined") is True and _filled(raw.get("taxes_subtotal_amount")):
        return False
    if raw.get("taxes_section_present") is True or "tax_lines" in unreadable:
        return True
    return _filled(raw.get("taxes_subtotal_amount"))


def _unreadable_names(raw: dict[str, Any]) -> set[str]:
    values = raw.get("unreadable_fields")
    if not isinstance(values, list):
        return set()
    return {item.strip() for item in values if isinstance(item, str) and item.strip()}


def _filled(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and text.casefold() not in _UNREADABLE_TOKENS
    if isinstance(value, (list, dict)):
        return _has_items(value) if isinstance(value, list) else bool(value)
    return True


def _is_unreadable_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() in _UNREADABLE_TOKENS


def _has_items(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, dict):
            if _filled(item.get("label")) or _filled(item.get("amount")):
                return True
        elif _filled(item):
            return True
    return False

