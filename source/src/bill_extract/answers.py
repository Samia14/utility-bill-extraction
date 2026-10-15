"""Answer each question row and write level2.csv.

Each row names a bill_id. The bill JSON comes from level1.csv, which is the
file written when that bill image was extracted. The question is in the
question column. The answer is written on the same row.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bill_extract.ask import answer_about_bill, session_file_id
from bill_extract.config import Config, load_config
from bill_extract.errors import (
    ConfigurationError,
    DependencyError,
    DocumentLoadError,
    ModelError,
    ModelResponseError,
)
from bill_extract.guardrail import EMPTY_QUESTION_MESSAGE, ScopeCheck

LEVEL2_FIELDS: tuple[str, ...] = ("bill_id", "question_id", "question", "answer")

MISSING_BILL_MESSAGE = (
    "This bill was not found in the extracted file, so the question cannot be answered."
)
EMPTY_BILL_MESSAGE = (
    "This bill has no extracted data, so the question cannot be answered."
)
INVALID_BILL_MESSAGE = (
    "The extracted data for this bill could not be read, so the question cannot be answered."
)


@dataclass(frozen=True)
class AnswerBatch:
    """How many question rows were written, and how many of them failed."""

    written: int
    failed: int
    output_path: Path


def answer_questions(
    bills_path: Path,
    questions_path: Path,
    output_path: Path,
    *,
    config: Config | None = None,
    storage_dir: Path | None = None,
    classify: Callable[[str], ScopeCheck] | None = None,
    answer: Callable[[str], str] | None = None,
) -> AnswerBatch:
    """Fill one answer per question. One bad row does not stop the rest."""
    bills, empty_ids = _load_bills(bills_path)
    questions = _load_questions(questions_path)
    rows: list[dict[str, str]] = []
    failed = 0
    for index, question_row in enumerate(questions, start=1):
        answer_text, ok = _answer_row(
            question_row,
            bills,
            empty_ids,
            row_number=index,
            config=config,
            storage_dir=storage_dir,
            classify=classify,
            answer=answer,
        )
        if not ok:
            failed += 1
        filled = dict(question_row)
        filled["answer"] = answer_text
        rows.append(filled)
    _write_level2(output_path, rows)
    return AnswerBatch(written=len(rows), failed=failed, output_path=output_path)


def answers_main(argv: list[str] | None = None) -> int:
    """Write level2.csv. Return 0 when every row is answered, 1 on a row or file error, 2 on bad settings."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            return 0
        if isinstance(code, int):
            return code
        return 2

    bills_path = Path(args.bills)
    questions_path = Path(args.questions)
    output_path = Path(args.output) if args.output else bills_path.parent / "level2.csv"
    try:
        config = load_config()
        batch = answer_questions(
            bills_path,
            questions_path,
            output_path,
            config=config,
            storage_dir=Path(args.storage) if args.storage else None,
        )
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (DocumentLoadError, ModelError, ModelResponseError, DependencyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError:
        print(f"The answers file could not be written: {output_path}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(_unexpected_message(exc), file=sys.stderr)
        return 1

    print(f"Wrote {batch.written} answers to {batch.output_path}", file=sys.stderr)
    if batch.failed:
        print(f"{batch.failed} questions could not be answered.", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bill_extract answers",
        description=(
            "Read each question, look up that bill in level1.csv, "
            "and write the answer in level2.csv."
        ),
    )
    parser.add_argument(
        "--bills",
        required=True,
        help="The level1.csv file written when the bill images were extracted.",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="A CSV file with bill_id and question columns. question_id is kept when it is present.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write level2.csv. When omitted, it is written next to the bills file.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Folder for the JSON conversation files. The default is agent_sessions next to the project.",
    )
    return parser


def _load_bills(path: Path) -> tuple[dict[str, str], set[str]]:
    """Return bill JSON by id, and the ids whose JSON cell is empty."""
    if not path.is_file():
        raise DocumentLoadError(f"The extracted bills file was not found: {path.name}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:
        raise DocumentLoadError(f"The extracted bills file could not be read: {path.name}") from exc
    except csv.Error as exc:
        raise DocumentLoadError(f"The extracted bills file is not a valid CSV file: {path.name}") from exc
    if "bill_id" not in fieldnames or "json" not in fieldnames:
        raise DocumentLoadError("The extracted bills file must have bill_id and json columns.")

    bills: dict[str, str] = {}
    empty_ids: set[str] = set()
    for row in rows:
        bill_id = str(row.get("bill_id") or "").strip()
        if not bill_id or bill_id in bills or bill_id in empty_ids:
            continue
        payload = str(row.get("json") or "").strip()
        if not payload:
            empty_ids.add(bill_id)
            continue
        bills[bill_id] = payload
    return bills, empty_ids


def _load_questions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DocumentLoadError(f"The questions file was not found: {path.name}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            raw_rows = list(reader)
    except OSError as exc:
        raise DocumentLoadError(f"The questions file could not be read: {path.name}") from exc
    except csv.Error as exc:
        raise DocumentLoadError(f"The questions file is not a valid CSV file: {path.name}") from exc
    if "bill_id" not in fieldnames or "question" not in fieldnames:
        raise DocumentLoadError("The questions file must have bill_id and question columns.")

    questions: list[dict[str, str]] = []
    for raw in raw_rows:
        bill_id = str(raw.get("bill_id") or "").strip()
        question = str(raw.get("question") or "").strip()
        if not bill_id and not question:
            continue
        row = {key: str(raw.get(key) or "") for key in fieldnames}
        row["bill_id"] = bill_id
        row["question"] = str(raw.get("question") or "")
        if "question_id" not in row:
            row["question_id"] = ""
        questions.append(row)
    if not questions:
        raise DocumentLoadError("The questions file has no questions.")
    return questions


def _answer_row(
    question_row: dict[str, str],
    bills: dict[str, str],
    empty_ids: set[str],
    *,
    row_number: int,
    config: Config | None,
    storage_dir: Path | None,
    classify: Callable[[str], ScopeCheck] | None,
    answer: Callable[[str], str] | None,
) -> tuple[str, bool]:
    bill_id = question_row.get("bill_id", "").strip()
    question_id = question_row.get("question_id", "").strip()
    label = question_id or f"row {row_number}"
    question = question_row.get("question", "")
    if not bill_id:
        message = MISSING_BILL_MESSAGE
        print(f"{label}: {message}", file=sys.stderr)
        return message, False
    if bill_id in empty_ids:
        print(f"{bill_id} {label}: {EMPTY_BILL_MESSAGE}", file=sys.stderr)
        return EMPTY_BILL_MESSAGE, False
    bill_json = bills.get(bill_id)
    if bill_json is None:
        print(f"{bill_id} {label}: {MISSING_BILL_MESSAGE}", file=sys.stderr)
        return MISSING_BILL_MESSAGE, False
    if not question.strip():
        print(f"{bill_id} {label}: {EMPTY_QUESTION_MESSAGE}", file=sys.stderr)
        return EMPTY_QUESTION_MESSAGE, False
    try:
        session_id = session_file_id(bill_id, f"{bill_id}-{question_id or row_number}")
        result = answer_about_bill(
            question,
            bill_id,
            bill_json,
            session_id=session_id,
            config=config,
            storage_dir=storage_dir,
            classify=classify,
            answer=answer,
        )
    except (DocumentLoadError, ModelError, ModelResponseError, DependencyError, ConfigurationError) as exc:
        message = str(exc)
        print(f"{bill_id} {label}: {message}", file=sys.stderr)
        return message, False
    except Exception as exc:
        message = _unexpected_message(exc)
        print(f"{bill_id} {label}: {message}", file=sys.stderr)
        return message, False
    print(f"{bill_id} {label}: answered", file=sys.stderr)
    return result.text, True


def _write_level2(path: Path, rows: list[dict[str, str]]) -> None:
    """Write level2.csv. A same-path input is replaced only after the new file is complete."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extra = [key for key in rows[0] if key not in LEVEL2_FIELDS] if rows else []
    fieldnames = list(LEVEL2_FIELDS) + extra
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
                handle.flush()
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise


def _unexpected_message(exc: Exception) -> str:
    text = " ".join(re.sub(r"sk-[A-Za-z0-9_\-]+", "***", str(exc)).split())
    if len(text) > 300:
        text = text[:300] + "..."
    if not text:
        text = type(exc).__name__
    return f"This question was not answered because an unexpected error occurred: {text}"


if __name__ == "__main__":
    raise SystemExit(answers_main())
