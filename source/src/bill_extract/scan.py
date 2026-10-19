"""Read a bill by sliding overlapping windows, then zoom any window that disagrees.

OpenCV prepares a copy of the page. OCR reads each window at normal size and
again zoomed. A word is kept only when both reads match. The model is asked
only for a window whose reads disagree or whose ink produced no words. A
printed mismatch is not sent to the model.
"""

from __future__ import annotations

from typing import Any, Callable

from bill_extract.assemble import (
    PlacedToken,
    Token,
    agree_reads,
    assemble,
    coerce_fuel,
    coerce_history,
    coerce_late_tiers,
    fault_errors,
    merge_tokens,
    refresh_printed_faults,
)
from bill_extract.errors import AmountParseError, DateParseError, DependencyError, field_error
from bill_extract.normalize import parse_amount, parse_bill_month, parse_date
from bill_extract.windows import Window, quarter_windows

ReadCrop = Callable[[Any], list[Token]]
FillGap = Callable[[Any, list[str]], dict[str, Any]]

_GAP_FIELDS: tuple[str, ...] = (
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
    "current_bill",
    "arrears",
    "payable_within_due_date",
    "payable_after_due_date",
    "unit_rate",
    "fixed_charge",
    "total_charges",
    "total_taxes",
    "history",
    "fuel_adjustment",
    "late_payment_tiers",
)


def scan_image(
    image: Any,
    read_crop: ReadCrop,
    fill_gap: FillGap | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one page array. The array itself is not modified."""
    page = _copy_page(image)
    try:
        height, width = int(page.shape[0]), int(page.shape[1])
        placed: list[PlacedToken] = []
        gap_windows: list[Any] = []
        errors: list[dict[str, Any]] = []
        for window in quarter_windows(width, height):
            crop = page[window.y : window.y + window.height, window.x : window.x + window.width]
            try:
                agreed, disagreed = _read_window(crop, read_crop)
            except DependencyError:
                raise
            except Exception as exc:
                errors.append(
                    field_error(
                        "document",
                        "ocr_error",
                        "One window could not be read. The rest of the bill was kept.",
                        str(exc),
                    )
                )
                continue
            if disagreed or (_has_ink(crop) and not agreed):
                gap_windows.append(crop)
            for token in agreed:
                page_token = _offset(token, window)
                placed.append(PlacedToken(page_token, _in_interior(page_token, window)))
        kept, border_conflicts = merge_tokens(placed)
        if border_conflicts and gap_windows:
            pass
        elif border_conflicts:
            gap_windows.append(page)
        record = assemble(kept, height)
        if fill_gap is not None and gap_windows:
            _fill_empty(record, gap_windows, fill_gap, errors)
            refresh_printed_faults(record)
        errors.extend(fault_errors(record))
        return record, errors
    finally:
        del page


def tesseract_tokens(image: Any) -> list[Token]:
    """Read one crop with Tesseract. Words come back in crop coordinates."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise DependencyError(
            "OCR is not installed. Install pytesseract and the Tesseract program, then try again."
        ) from exc

    from bill_extract.fallback import _use_installed_tesseract

    _use_installed_tesseract(pytesseract)
    pil = Image.fromarray(_to_rgb(image))
    try:
        data = pytesseract.image_to_data(
            pil, lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT
        )
    except Exception:
        try:
            data = pytesseract.image_to_data(pil, config="--psm 6", output_type=pytesseract.Output.DICT)
        except Exception as exc:
            raise DependencyError(
                "Tesseract could not read this window. Install Tesseract and make sure it is on the PATH."
            ) from exc
    tokens: list[Token] = []
    count = len(data.get("text") or [])
    for index in range(count):
        text = str(data["text"][index]).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1
        if confidence < 0:
            continue
        tokens.append(
            Token(
                text,
                int(data["left"][index]),
                int(data["top"][index]),
                max(1, int(data["width"][index])),
                max(1, int(data["height"][index])),
            )
        )
    return tokens


def _read_window(crop: Any, read_crop: ReadCrop) -> tuple[list[Token], list[Token]]:
    normal = read_crop(crop)
    zoomed = _enlarge(crop)
    zoom_tokens = _scale_back(read_crop(zoomed), _zoom_factor(crop, zoomed))
    return agree_reads(normal, zoom_tokens)


def _fill_empty(
    record: dict[str, Any],
    crops: list[Any],
    fill_gap: FillGap,
    errors: list[dict[str, Any]],
) -> None:
    """Ask the model only for fields that are still empty. Agreed values stay."""
    pending = [field for field in _GAP_FIELDS if _is_empty(record.get(field))]
    if not pending:
        return
    for crop in crops:
        if not pending:
            return
        try:
            filled = fill_gap(crop, list(pending))
        except Exception as exc:
            errors.append(
                field_error(
                    "document",
                    "gap_fill_error",
                    "A gap could not be filled. The printed values already read were kept.",
                    str(exc),
                )
            )
            return
        if not isinstance(filled, dict):
            continue
        for field in list(pending):
            if field not in filled or not _is_empty(record.get(field)):
                continue
            value = _coerce_gap(field, filled[field])
            if value is None or value == []:
                continue
            record[field] = value
            pending.remove(field)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if value == []:
        return True
    if isinstance(value, dict) and value.get("kind") is None and value.get("evidence") is None:
        return True
    return False


def _copy_page(image: Any):
    try:
        return image.copy()
    except Exception as exc:
        raise DependencyError("The bill image could not be copied for reading.") from exc


def _enlarge(image: Any):
    try:
        import cv2
    except ImportError:
        import numpy as np

        factor = 2
        if getattr(image, "ndim", 0) == 2:
            return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
    return cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def _zoom_factor(crop: Any, zoomed: Any) -> float:
    crop_height = max(1, int(crop.shape[0]))
    return max(1.0, float(zoomed.shape[0]) / float(crop_height))


def _scale_back(tokens: list[Token], factor: float) -> list[Token]:
    if factor == 1:
        return tokens
    scaled: list[Token] = []
    for token in tokens:
        scaled.append(
            Token(
                token.text,
                int(token.x / factor),
                int(token.y / factor),
                max(1, int(token.width / factor)),
                max(1, int(token.height / factor)),
            )
        )
    return scaled


def _offset(token: Token, window: Window) -> Token:
    return Token(
        token.text,
        token.x + window.x,
        token.y + window.y,
        token.width,
        token.height,
    )


def _in_interior(token: Token, window: Window) -> bool:
    left, top, width, height = window.interior()
    return left <= token.cx < left + width and top <= token.cy < top + height


def _has_ink(image: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return True
    gray = image.mean(axis=2) if getattr(image, "ndim", 0) == 3 else image
    if gray.size == 0:
        return False
    return bool((np.asarray(gray) < 200).mean() > 0.01)


def _coerce_gap(field: str, value: Any) -> Any:
    """Turn model text into the record type. A value that does not parse is dropped."""
    if value is None or value == "":
        return None
    amount_fields = {
        "sanctioned_load_kw",
        "previous_reading",
        "current_reading",
        "units_consumed",
        "current_bill",
        "arrears",
        "payable_within_due_date",
        "payable_after_due_date",
        "unit_rate",
        "fixed_charge",
        "total_charges",
        "total_taxes",
    }
    date_fields = {"reading_date", "issue_date", "due_date"}
    try:
        if field in amount_fields:
            return parse_amount(value if isinstance(value, str) else str(value))
        if field in date_fields:
            return parse_date(str(value))
        if field == "bill_month":
            return parse_bill_month(str(value))
        if field == "history":
            rows = coerce_history(value)
            return rows or None
        if field == "fuel_adjustment":
            return coerce_fuel(value)
        if field == "connection" and isinstance(value, dict):
            return value
        if field == "late_payment_tiers":
            tiers = coerce_late_tiers(value)
            return tiers or None
        if isinstance(value, str):
            return value.strip() or None
    except (AmountParseError, DateParseError):
        return None
    return None


def _to_rgb(image: Any):
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.shape[2] == 3:
        return array[:, :, ::-1]
    return array[:, :, :3]
