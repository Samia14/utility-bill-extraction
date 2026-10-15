"""Unit tests for level1.csv writing. These tests do not call the network."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from bill_extract.normalize import normalize  # noqa: E402
from bill_extract.schema import dumps_bill, rows_with_data, write_level1_csv  # noqa: E402
from test_normalize import CONTRACT_JSON, contract_raw  # noqa: E402


class CsvWriteTest(unittest.TestCase):
    def test_contract_row_quotes_json_on_one_line(self):
        bill, errors = normalize(contract_raw())
        self.assertEqual(errors, [])
        self.assertEqual(dumps_bill(bill), CONTRACT_JSON)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "level1.csv"
            write_level1_csv(path, [("KESC_0008", bill)])
            text = path.read_text(encoding="utf-8")

        lines = text.splitlines()
        self.assertEqual(lines[0], "bill_id,json")
        self.assertEqual(len(lines), 2)
        self.assertIn('"{""provider"":""KE""', lines[1])
        self.assertIn('""amount"":663.51', lines[1])
        self.assertNotIn("663.510000", lines[1])
        self.assertNotIn("\n", lines[1])

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "level1.csv"
            write_level1_csv(path, [("KESC_0008", bill)])
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["bill_id", "json"])
        self.assertEqual(rows[1][0], "KESC_0008")
        self.assertEqual(rows[1][1], CONTRACT_JSON)

    def test_failed_files_are_left_out_of_the_csv(self):
        bill, _errors = normalize(contract_raw())
        results = [
            {"source": "missing.png", "ok": False, "data": None, "errors": [], "bill_id": "missing"},
            {"source": "ok.png", "ok": True, "data": bill, "errors": [], "bill_id": "ok-bill"},
        ]
        self.assertEqual(rows_with_data(results), [("ok-bill", bill)])

    def test_encoder_refuses_binary_floats(self):
        with self.assertRaises(TypeError):
            dumps_bill({"amount": 0.1})
        self.assertEqual(dumps_bill({"amount": Decimal("663.50")}), '{"amount":663.5}')
        self.assertEqual(dumps_bill({"amount": Decimal("20.00")}), '{"amount":20}')


if __name__ == "__main__":
    unittest.main()
