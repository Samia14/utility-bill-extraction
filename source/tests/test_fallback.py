"""Unit tests for the zoom fallback. These tests do not call the network."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from bill_extract.fallback import (  # noqa: E402
    find_missing_fields,
    merge_unread,
    reading_from_ocr,
    reread_missing,
)


def _complete_raw(**overrides):
    raw = {
        "provider_name": "KE",
        "tariff": "A1-R",
        "sanctioned_load": "3",
        "bill_month": "Apr-26",
        "reading_date": "03-Apr-26",
        "issue_date": "07-Apr-26",
        "due_date": "21 Apr 2026",
        "meter_ambiguous": False,
        "previous_reading": "10",
        "current_reading": "20",
        "units_consumed": "10",
        "charge_lines": [{"label": "Fixed Charges", "amount": "10"}],
        "charges_subtotal_amount": "10",
        "charges_section_present": True,
        "tax_lines": [{"label": "Sales Tax", "amount": "2"}],
        "taxes_section_present": True,
        "taxes_are_combined": False,
        "current_bill": "12",
        "arrears": "0",
        "payable_within_due_date": "12",
        "payable_after_due_date_amounts": ["13"],
        "unreadable_fields": [],
        "field_boxes": [],
    }
    raw.update(overrides)
    return raw


class MergeUnreadTest(unittest.TestCase):
    def test_merge_fills_only_the_missing_field(self):
        base = _complete_raw(due_date=None, current_bill="100")
        crop = {
            "provider_name": "LESCO",
            "tariff": "Z-9",
            "due_date": "21 Apr 2026",
            "current_bill": "999",
            "charge_lines": [{"label": "Other", "amount": "5"}],
        }
        merged = merge_unread(base, crop, ["due_date"])
        self.assertEqual(merged["provider_name"], "KE")
        self.assertEqual(merged["tariff"], "A1-R")
        self.assertEqual(merged["current_bill"], "100")
        self.assertEqual(merged["due_date"], "21 Apr 2026")
        self.assertEqual(merged["charge_lines"], base["charge_lines"])

    def test_failed_zoom_does_not_discard_good_fields(self):
        base = _complete_raw(due_date=None, current_bill="100")
        seen = {}

        def zoomer(path):
            seen["existed"] = path.exists()
            seen["path"] = path
            raise RuntimeError("zoom failed")

        def reader(image_bytes, fields):
            raise AssertionError("the model should not be called when the zoom fails")

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "bill.png"
            source.write_bytes(b"original-bytes")
            merged, errors = reread_missing(source, b"original-bytes", base, reader, zoomer=zoomer)
            self.assertEqual(source.read_bytes(), b"original-bytes")

        self.assertEqual(merged["provider_name"], "KE")
        self.assertEqual(merged["current_bill"], "100")
        self.assertEqual(merged["tariff"], "A1-R")
        self.assertIsNone(merged["due_date"])
        self.assertTrue(seen["existed"])
        self.assertFalse(seen["path"].exists())
        self.assertEqual(errors[0]["field"], "due_date")
        self.assertEqual(errors[0]["code"], "missing_field")

    def test_complete_page_does_not_request_a_crop(self):
        self.assertEqual(find_missing_fields(_complete_raw()), [])

        def reader(crop_bytes, fields):
            raise AssertionError("a complete page should not be cropped")

        merged, errors = reread_missing(None, b"bytes", _complete_raw(), reader)
        self.assertEqual(merged["provider_name"], "KE")
        self.assertEqual(errors, [])

    def test_unread_value_is_read_from_the_zoomed_copy(self):
        base = _complete_raw(due_date=None, current_bill="100")
        seen = {}

        def zoomer(path):
            seen["zoomed_path"] = path
            return b"zoomed-bytes"

        def reader(image_bytes, fields):
            seen["model_bytes"] = image_bytes
            seen["fields"] = fields
            return {"due_date": None, "current_bill": "999"}

        def ocr_reader(zoomed):
            seen["ocr_bytes"] = zoomed
            return "Due Date 21 Apr 2026"

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "bill.png"
            source.write_bytes(b"original-bytes")
            merged, errors = reread_missing(
                source,
                b"original-bytes",
                base,
                reader,
                zoomer=zoomer,
                ocr_reader=ocr_reader,
            )
            self.assertEqual(source.read_bytes(), b"original-bytes")

        self.assertEqual(errors, [])
        self.assertEqual(merged["due_date"], "21 Apr 2026")
        self.assertEqual(merged["current_bill"], "100")
        self.assertEqual(merged["provider_name"], "KE")
        self.assertEqual(seen["model_bytes"], b"zoomed-bytes")
        self.assertEqual(seen["ocr_bytes"], b"zoomed-bytes")
        self.assertFalse(seen["zoomed_path"].exists())

    def test_ocr_text_keeps_the_arrears_amount_before_the_slash(self):
        self.assertEqual(reading_from_ocr("arrears", "Arrears 39411 / 2"), {"arrears": "39411 / 2"})
        self.assertEqual(reading_from_ocr("due_date", "no date here"), {})

    def test_empty_charges_are_missing_only_when_the_section_is_present(self):
        quiet = _complete_raw(charge_lines=[], charges_section_present=False, charges_subtotal_amount=None)
        self.assertNotIn("charge_lines", find_missing_fields(quiet))
        present = _complete_raw(charge_lines=[], charges_section_present=True)
        self.assertIn("charge_lines", find_missing_fields(present))


if __name__ == "__main__":
    unittest.main()
