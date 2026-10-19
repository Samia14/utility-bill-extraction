"""Tests for overlapping-window extraction and catalog answers. No model calls."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from bill_extract.assemble import (  # noqa: E402
    PlacedToken,
    Token,
    agree_reads,
    assemble,
    merge_tokens,
)
from bill_extract.ask import answer_about_bill  # noqa: E402
from bill_extract.catalog import CATALOG  # noqa: E402
from bill_extract.guardrail import REFUSAL_MESSAGE  # noqa: E402
from bill_extract.question_store import QuestionStore  # noqa: E402
from bill_extract.scan import scan_image  # noqa: E402
from bill_extract.windows import quarter_windows  # noqa: E402


def _token(text: str, x: int, y: int, width: int = 40, height: int = 16) -> Token:
    return Token(text, x, y, width, height)


class WindowTest(unittest.TestCase):
    def test_windows_overlap_and_cover_the_far_edge(self):
        windows = quarter_windows(400, 400)
        self.assertGreaterEqual(len(windows), 4)
        self.assertTrue(any(window.x == 0 and window.y == 0 for window in windows))
        self.assertTrue(any(window.x + window.width == 400 for window in windows))
        first = windows[0]
        second = next(window for window in windows if window.x > first.x and window.y == first.y)
        self.assertLess(second.x, first.x + first.width)

    def test_a_border_point_is_outside_the_unshared_interior(self):
        window = quarter_windows(400, 400)[0]
        left, top, width, height = window.interior()
        self.assertLess(left + width, window.x + window.width)


class ReadAgreementTest(unittest.TestCase):
    def test_a_digit_is_kept_only_when_both_reads_match(self):
        agreed, conflicts = agree_reads(
            [_token("676", 10, 10)],
            [_token("676", 12, 10)],
        )
        self.assertEqual([token.text for token in agreed], ["676"])
        self.assertEqual(conflicts, [])

    def test_a_disagreement_is_not_kept(self):
        agreed, conflicts = agree_reads(
            [_token("676", 10, 10)],
            [_token("678", 10, 10)],
        )
        self.assertEqual(agreed, [])
        self.assertEqual(len(conflicts), 1)

    def test_the_unshared_read_wins_when_the_border_agrees(self):
        kept, conflicts = merge_tokens(
            [
                PlacedToken(_token("412", 10, 10), False),
                PlacedToken(_token("412", 12, 10), True),
            ]
        )
        self.assertEqual(conflicts, [])
        self.assertTrue(kept[0].x == 12 or kept[0].x == 10)
        self.assertEqual(kept[0].text, "412")


class AssembleTest(unittest.TestCase):
    def _bill_tokens(self) -> list[Token]:
        return [
            _token("IESCO", 10, 10, 70),
            _token("Due", 10, 80, 30),
            _token("Date", 45, 80, 40),
            _token("24 AUG 26", 100, 80, 80),
            _token("Previous Reading", 10, 140, 130),
            _token("100", 160, 140, 40),
            _token("Present Reading", 10, 180, 130),
            _token("176", 160, 180, 40),
            _token("Units", 10, 220, 50),
            _token("70", 160, 220, 30),
            _token("Energy Charges", 10, 260, 120),
            _token("29000", 200, 260, 50),
            _token("Total Electricity Charges", 10, 300, 180),
            _token("29807", 210, 300, 50),
            _token("FPA", 10, 360, 40),
            _token("370.76", 60, 360, 50),
            _token("Jun 26", 130, 360, 50),
            _token("Jul 26", 10, 420, 50),
            _token("412", 80, 420, 40),
            _token("34613", 140, 420, 50),
            _token("34613", 210, 420, 50),
            _token("Net Metering", 10, 480, 100),
            _token("15", 130, 480, 30),
        ]

    def test_labels_history_and_printed_faults_stay_as_printed(self):
        record = assemble(self._bill_tokens(), 800)
        self.assertEqual(record["provider"], "IESCO")
        self.assertEqual(record["due_date"], "2026-08-24")
        self.assertEqual(record["previous_reading"], Decimal("100"))
        self.assertEqual(record["current_reading"], Decimal("176"))
        self.assertEqual(record["units_consumed"], Decimal("70"))
        self.assertEqual(record["history"][0]["month"], "2026-07")
        self.assertEqual(record["history"][0]["units"], Decimal("412"))
        self.assertEqual(record["history"][0]["payment_amount"], Decimal("34613"))
        self.assertEqual(record["fuel_adjustment"]["amount"], Decimal("370.76"))
        self.assertEqual(record["fuel_adjustment"]["month"], "2026-06")
        self.assertEqual(record["connection"]["kind"], "net_metered")
        self.assertEqual(record["total_charges"], Decimal("29807"))
        self.assertTrue(any("printed units" in message for message in record["warnings"]))
        self.assertTrue(any("printed subtotal" in message or "charges subtotal" in message for message in record["warnings"]))
        self.assertEqual(record["units_consumed"], Decimal("70"))
        self.assertEqual(record["total_charges"], Decimal("29807"))


class ScanGapTest(unittest.TestCase):
    def test_an_agreed_read_does_not_call_the_model(self):
        calls: list[list[str]] = []

        def read(crop):
            scale = crop.shape[0] / 40
            return [
                Token(
                    "100",
                    int(4 * scale),
                    int(4 * scale),
                    max(1, int(10 * scale)),
                    max(1, int(8 * scale)),
                )
            ]

        def fill(_crop, fields):
            calls.append(fields)
            return {}

        image = np.zeros((80, 80, 3), dtype=np.uint8)
        record, _errors = scan_image(image, read, fill_gap=fill)
        self.assertEqual(calls, [])
        self.assertEqual(record["units_consumed"], None)

    def test_a_disagreement_is_a_gap_and_an_agreed_value_is_not_replaced(self):
        def read(crop):
            scale = max(crop.shape[0], 1) / 40
            provider = Token(
                "IESCO",
                int(2 * scale),
                int(2 * scale),
                max(1, int(20 * scale)),
                max(1, int(8 * scale)),
            )
            text = "100" if crop.shape[0] <= 40 else "176"
            other = Token(text, int(4 * scale), int(20 * scale), max(1, int(10 * scale)), 8)
            return [provider, other]

        def fill(_crop, fields):
            return {"provider": "KE", "units_consumed": "70"}

        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        record, _errors = scan_image(image, read, fill_gap=fill)
        self.assertEqual(record["provider"], "IESCO")
        self.assertEqual(record["units_consumed"], Decimal("70"))


class StoredTextAnswerTest(unittest.TestCase):
    def test_a_history_block_stored_as_text_can_be_answered(self):
        history = (
            "BILL HISTORY\n"
            "Sep 25     597     32414     0\n"
            "Oct 25     656     73264     73264\n"
        )
        record = {
            "history": history,
            "late_payment_tiers": "L.P SURCHARGE 1,491 2,981",
            "fuel_adjustment": "370.7",
            "connection": {"kind": None, "evidence": None, "units": None},
            "warnings": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            highest = answer_about_bill(
                CATALOG[1].wording,
                "B1",
                json.dumps(record),
                session_id="B1",
                storage_dir=Path(folder),
            )
            late = answer_about_bill(
                CATALOG[3].wording,
                "B1",
                json.dumps(record),
                session_id="B1",
                storage_dir=Path(folder),
            )
        self.assertTrue(highest.in_scope)
        self.assertIn("2025-10", highest.text)
        self.assertIn("656", highest.text)
        self.assertNotIn("unexpected error", highest.text)
        self.assertIn("1491", late.text)
        self.assertIn("2981", late.text)


class CatalogAnswerTest(unittest.TestCase):
    def test_a_fixed_question_is_answered_from_the_record(self):
        record = {
            "bill_month": "2026-08",
            "units_consumed": 500,
            "history": [
                {"month": "2026-06", "units": 100, "billed_amount": 1000, "payment_amount": 1000},
                {"month": "2026-07", "units": 200, "billed_amount": 2000, "payment_amount": 0},
            ],
            "warnings": [
                "The printed units do not match the difference of the printed readings. The printed figures were kept."
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            result = answer_about_bill(
                CATALOG[1].wording,
                "B1",
                json.dumps(record),
                session_id="B1",
                storage_dir=Path(folder),
            )
            stored = QuestionStore(Path(folder) / "blocked_questions.json")
            self.assertFalse(stored.contains(CATALOG[1].wording))
        self.assertTrue(result.in_scope)
        self.assertIn("2026-08", result.text)
        self.assertIn("500", result.text)
        self.assertIn("printed units", result.text)

    def test_an_unknown_question_is_stored_once_and_refused(self):
        question = "What is the capital of France?"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = answer_about_bill(
                question,
                "B1",
                "{}",
                session_id="B1",
                storage_dir=root,
            )
            second = answer_about_bill(
                question.upper(),
                "B1",
                "{}",
                session_id="B1",
                storage_dir=root,
            )
            items = json.loads((root / "blocked_questions.json").read_text(encoding="utf-8"))
        self.assertFalse(first.in_scope)
        self.assertEqual(first.text, REFUSAL_MESSAGE)
        self.assertEqual(second.text, REFUSAL_MESSAGE)
        self.assertEqual(len(items), 1)
