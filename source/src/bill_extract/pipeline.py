"""Run extraction over one file or a folder and write level1.csv."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from bill_extract.config import Config
from bill_extract.errors import (
    DependencyError,
    DocumentLoadError,
    ModelError,
    ModelResponseError,
)
from bill_extract.extract import build_client, extract_gap_values
from bill_extract.load import ensure_pillow, is_supported, load_pages
from bill_extract.scan import scan_image, tesseract_tokens
from bill_extract.schema import rows_with_data, write_level1_csv


def collect_inputs(path: Path) -> list[Path]:
    """List supported bill files. A folder skips unrelated files."""
    if path.is_dir():
        files = [
            item
            for item in path.iterdir()
            if item.is_file() and not item.name.startswith(".") and is_supported(item)
        ]
        return sorted(files, key=lambda item: item.name.casefold())
    return [path]


def run_batch(
    input_path: Path,
    output_path: Path,
    config: Config | None,
    client: Any = None,
) -> int:
    """Extract every bill. Return 0 when all files succeed, otherwise 1.

    Configuration is checked by the caller before this function reads files.
    One bad file does not stop the rest.
    """
    try:
        ensure_pillow()
        if client is None and config is not None:
            client = build_client(config)
    except DependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not input_path.exists():
        print(f"The input path was not found: {input_path}", file=sys.stderr)
        return 1

    files = collect_inputs(input_path)
    results: list[dict[str, Any]] = []
    if not files:
        print("No JPG or PNG bill images were found in the input folder.", file=sys.stderr)

    for file_path in files:
        results.extend(_process_file(file_path, config, client))

    try:
        write_level1_csv(output_path, rows_with_data(results))
    except OSError:
        print(f"The CSV file could not be written: {output_path}", file=sys.stderr)
        return 1

    _report(results, output_path)
    if any(not result["ok"] for result in results):
        return 1
    return 0


def _process_file(path: Path, config: Config, client: Any) -> list[dict[str, Any]]:
    try:
        pages = load_pages(path)
    except DocumentLoadError as exc:
        return [_failure(str(path), path.stem, "document_load", str(exc))]
    except DependencyError as exc:
        return [_failure(str(path), path.stem, "document_load", str(exc))]
    except OSError:
        return [
            _failure(
                str(path),
                path.stem,
                "document_load",
                f"The file '{path.name}' could not be read.",
            )
        ]
    except Exception as exc:
        return [_failure(str(path), path.stem, "unexpected_error", _unexpected_message(exc))]

    results: list[dict[str, Any]] = []
    for page in pages:
        results.append(_process_page(page, config, client))
    return results


def _process_page(page: Any, config: Config | None, client: Any) -> dict[str, Any]:
    try:
        image = _page_image(page.png_bytes)
        data, errors = scan_image(
            image,
            tesseract_tokens,
            fill_gap=_gap_filler(config, client),
        )
    except DependencyError as exc:
        return _failure(page.source, page.bill_id, "document_load", str(exc))
    except DocumentLoadError as exc:
        return _failure(page.source, page.bill_id, "document_load", str(exc))
    except ModelError as exc:
        return _failure(page.source, page.bill_id, "model_error", str(exc))
    except ModelResponseError as exc:
        return _failure(page.source, page.bill_id, "invalid_model_json", str(exc))
    except Exception as exc:
        return _failure(page.source, page.bill_id, "unexpected_error", _unexpected_message(exc))
    return {
        "source": page.source,
        "ok": True,
        "data": data,
        "errors": errors,
        "bill_id": page.bill_id,
    }


def _page_image(png_bytes: bytes):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise DependencyError(
            "OpenCV is not installed, so the bill image could not be read."
        ) from exc
    array = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise DocumentLoadError("The bill image could not be opened.")
    return image


def _gap_filler(config: Config | None, client: Any):
    """Return a model filler, or None when no model is configured."""
    if config is None or client is None:
        return None

    def fill(crop: Any, fields: list[str]) -> dict[str, Any]:
        import cv2

        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            return {}
        return extract_gap_values(encoded.tobytes(), config, client, fields)

    return fill


def _failure(source: str, bill_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "source": source,
        "ok": False,
        "data": None,
        "errors": [
            {
                "field": "document",
                "code": code,
                "message": message,
                "raw": None,
            }
        ],
        "bill_id": bill_id,
    }


def _unexpected_message(exc: Exception) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_\-]+", "***", str(exc))
    text = " ".join(text.split())
    if len(text) > 300:
        text = text[:300] + "..."
    if not text:
        text = type(exc).__name__
    return f"This file was skipped because an unexpected error occurred: {text}"


def _report(results: list[dict[str, Any]], output_path: Path) -> None:
    interesting = [result for result in results if (not result["ok"]) or result.get("errors")]
    for result in results:
        label = "Failed" if not result["ok"] else "Warning"
        for error in result.get("errors") or []:
            print(f"{label}: {result['source']}: {error['message']}", file=sys.stderr)
    if not interesting:
        return
    report_path = output_path.with_suffix(".errors.json")
    payload = [
        {
            "source": result["source"],
            "bill_id": result.get("bill_id"),
            "ok": result["ok"],
            "errors": result.get("errors") or [],
        }
        for result in interesting
    ]
    try:
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        print(f"The error report could not be written: {report_path}", file=sys.stderr)
        return
    print(f"Error report written to {report_path}", file=sys.stderr)
