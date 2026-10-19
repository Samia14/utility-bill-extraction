"""Remember questions that are outside the fixed catalog.

A catalog question is never stored. The same normalized question is refused
from this list on the next ask.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from bill_extract.errors import DocumentLoadError

_WORD = re.compile(r"[^a-z0-9]+")


def normalize_question(question: str) -> str:
    """Lowercase the question and drop punctuation so spacing does not matter."""
    return _WORD.sub(" ", question.casefold()).strip()


class QuestionStore:
    """A JSON list of questions that must be refused."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def contains(self, question: str) -> bool:
        key = normalize_question(question)
        if not key:
            return False
        return any(item.get("normalized") == key for item in self._read())

    def add(self, question: str, reason: str) -> None:
        """Append one out-of-context question. A catalog wording must not be passed in."""
        key = normalize_question(question)
        if not key or self.contains(question):
            return
        items = self._read()
        items.append(
            {
                "question": " ".join(question.split()),
                "normalized": key,
                "reason": reason,
            }
        )
        self._write(items)

    def _read(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DocumentLoadError(
                "The guardrail file could not be read."
            ) from exc
        if not text.strip():
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DocumentLoadError(
                "The guardrail file is not valid JSON."
            ) from exc
        if not isinstance(value, list):
            raise DocumentLoadError("The guardrail file must contain a list.")
        items: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            normalized = str(item.get("normalized") or "")
            if not normalized:
                continue
            items.append(
                {
                    "question": str(item.get("question") or ""),
                    "normalized": normalized,
                    "reason": str(item.get("reason") or ""),
                }
            )
        return items

    def _write(self, items: list[dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".partial")
        try:
            temporary.write_text(
                json.dumps(items, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError as exc:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            raise DocumentLoadError(
                "The guardrail file could not be written."
            ) from exc
