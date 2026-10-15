"""Tests for level2.csv answers. These tests do not call the model."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from bill_extract.answers import (  # noqa: E402
    EMPTY_BILL_MESSAGE,
    MISSING_BILL_MESSAGE,
    answer_questions,
)
from bill_extract.guardrail import ScopeCheck  # noqa: E402


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class AnswerCsvTest(unittest.TestCase):
    def test_each_question_is_answered_from_its_bill(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bills = root / "level1.csv"
            questions = root / "questions.csv"
            output = root / "level2.csv"
            _write_csv(
                bills,
                ["bill_id", "json"],
                [
                    ["KESC_0008", '{"provider":"KE","payable_within_due_date":100}'],
                    ["LESCO_0001", '{"provider":"LESCO","payable_within_due_date":50}'],
                ],
            )
            _write_csv(
                questions,
                ["bill_id", "question_id", "question", "answer"],
                [
                    ["KESC_0008", "Q1", "What is the amount due?", ""],
                    ["LESCO_0001", "Q1", "What do I owe?", ""],
                ],
            )
            seen: list[tuple[str, str]] = []

            def answer(question: str) -> str:
                seen.append((question, "called"))
                if "amount due" in question:
                    return "The amount due is 100."
                return "The amount due is 50."

            batch = answer_questions(
                bills,
                questions,
                output,
                classify=lambda _question: ScopeCheck(in_scope=True, reason="bill"),
                answer=answer,
            )
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(batch.written, 2)
        self.assertEqual(batch.failed, 0)
        self.assertEqual(rows[0]["bill_id"], "KESC_0008")
        self.assertEqual(rows[0]["question_id"], "Q1")
        self.assertEqual(rows[0]["question"], "What is the amount due?")
        self.assertEqual(rows[0]["answer"], "The amount due is 100.")
        self.assertEqual(rows[1]["answer"], "The amount due is 50.")
        self.assertEqual(len(seen), 2)

    def test_a_missing_bill_does_not_stop_the_next_question(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bills = root / "level1.csv"
            questions = root / "questions.csv"
            output = root / "level2.csv"
            _write_csv(
                bills,
                ["bill_id", "json"],
                [
                    ["KESC_0008", ""],
                    ["LESCO_0001", '{"provider":"LESCO"}'],
                ],
            )
            _write_csv(
                questions,
                ["bill_id", "question_id", "question", "answer"],
                [
                    ["KESC_0008", "Q1", "What is due?", ""],
                    ["MISSING", "Q2", "What is due?", ""],
                    ["LESCO_0001", "Q3", "What is due, exactly?", ""],
                ],
            )

            def answer(question: str) -> str:
                return f"Answer: {question}"

            batch = answer_questions(
                bills,
                questions,
                output,
                classify=lambda _question: ScopeCheck(in_scope=True, reason="bill"),
                answer=answer,
            )
            with output.open(encoding="utf-8", newline="") as handle:
                text = handle.read()
                handle.seek(0)
                rows = list(csv.DictReader(handle))

        self.assertEqual(batch.failed, 2)
        self.assertEqual(rows[0]["answer"], EMPTY_BILL_MESSAGE)
        self.assertEqual(rows[1]["answer"], MISSING_BILL_MESSAGE)
        self.assertEqual(rows[2]["answer"], "Answer: What is due, exactly?")
        self.assertIn('"Answer: What is due, exactly?"', text)

    def test_answers_command_requires_the_files(self):
        from bill_extract.cli import main

        self.assertEqual(main(["answers"]), 2)


if __name__ == "__main__":
    unittest.main()
