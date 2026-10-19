"""Answer a fixed question from the saved bill record.

A question that is not in the catalog is refused and stored on the guardrail.
The image is not opened, and no model is called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bill_extract.assemble import prepare_answer_record
from bill_extract.catalog import match_question
from bill_extract.guardrail import REFUSAL_MESSAGE
from bill_extract.question_store import QuestionStore

OUT_OF_CONTEXT_REASON = "The question is not one of the fixed bill questions."


def answer_from_record(
    question: str,
    record: dict[str, Any],
    store_path: Path,
) -> tuple[str, bool]:
    """Return the answer text and whether the question was in the catalog."""
    store = QuestionStore(store_path)
    matched = match_question(question)
    if matched is None:
        if not store.contains(question):
            store.add(question, OUT_OF_CONTEXT_REASON)
        return REFUSAL_MESSAGE, False
    text = matched.answer(prepare_answer_record(record)).strip()
    warnings = [str(item) for item in (record.get("warnings") or []) if str(item).strip()]
    if warnings:
        text = text + " " + " ".join(warnings)
    return text, True
