"""Unit tests for printed-bill normalization. These tests do not call the network."""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from bill_extract.errors import AmountParseError  # noqa: E402
from bill_extract.normalize import (  # noqa: E402
    normalize,
    parse_amount,
    parse_bill_month,
    parse_date,
)
from bill_extract.schema import dumps_bill  # noqa: E402

# Printed strings from the KE April bill contract. These belong in the test,
# not in production code, and the assertion calls normalize.
CONTRACT_JSON = (
    '{"provider":"KE","tariff":"A1-R","sanctioned_load_kw":3,'
    '"bill_month":"2026-04","reading_date":"2026-04-03","issue_date":"2026-04-07",'
    '"due_date":"2026-04-21","previous_reading":8819,"current_reading":8970,'
    '"units_consumed":151,"charges":['
    '{"type":"fixed","amount":900},'
    '{"type":"energy","amount":1054},'
    '{"type":"energy","amount":663.51},'
    '{"type":"quarterly_adjustment","amount":52.91},'
    '{"type":"fpa","amount":125.27},'
    '{"type":"surcharge","amount":64.93}'
    '],"total_charges":2860.62,"taxes":['
    '{"type":"electricity_duty","amount":29.41},'
    '{"type":"gst","amount":520.21},'
    '{"type":"municipal_tax","amount":20}'
    '],"total_taxes":569.62,"current_bill":3430.24,"arrears":-0.53,'
    '"payable_within_due_date":3430,"payable_after_due_date":3716}'
)


def blank_raw(**overrides):
    raw = {
        "provider_name": "KE",
        "tariff": "A1-R",
        "sanctioned_load": "1",
        "bill_month": "Jan-24",
        "reading_date": "02-Jan-24",
        "issue_date": "05-Jan-2024",
        "due_date": "20 Jan 2024",
        "meter_ambiguous": False,
        "previous_reading": "10",
        "current_reading": "25",
        "units_consumed": "4",
        "charge_lines": [],
        "charges_subtotal_label": None,
        "charges_subtotal_amount": None,
        "tax_lines": [],
        "taxes_are_combined": False,
        "taxes_subtotal_label": None,
        "taxes_subtotal_amount": None,
        "current_bill": None,
        "arrears": None,
        "payable_within_due_date": None,
        "payable_after_due_date_amounts": [],
    }
    raw.update(overrides)
    return raw


def contract_raw():
    """Raw printed strings that should normalize to the KE contract object."""
    return {
        "provider_name": "K-Electric",
        "tariff": "A1-R",
        "sanctioned_load": "3",
        "bill_month": "Apr-26",
        "reading_date": "03-Apr-26",
        "issue_date": "07-Apr-2026",
        "due_date": "21st Apr 2026",
        "meter_ambiguous": False,
        "previous_reading": "8819",
        "current_reading": "8970",
        "units_consumed": "151",
        "charge_lines": [
            {"label": "", "amount": ""},
            {"label": "Fixed Charges", "amount": "900.00"},
            {"label": "Variable (Upto 100 Units) Protected", "amount": "1,054.00"},
            {"label": "Variable (Upto 200 Units) Protected", "amount": "663.51"},
            {"label": "Uniform Quarterly Adjustment", "amount": "52.91"},
            {"label": "FCA Jan-26", "amount": "125.27"},
            {"label": "Additional Surcharge (PHL)", "amount": "64.93"},
            {"label": "Meter Rent", "amount": "0.00"},
            {"label": "Net Electricity Charges", "amount": "999.99"},
            {"label": "Electricity Charges", "amount": "2,860.62"},
        ],
        "charges_subtotal_label": "Electricity Charges",
        "charges_subtotal_amount": "2,860.62",
        "tax_lines": [
            {"label": "Electricity Duty", "amount": "29.41"},
            {"label": "Sales Tax (18%)", "amount": "520.21"},
            {"label": "MUCT (KMC)", "amount": "20.00"},
            {"label": "TV Fee", "amount": "0.00"},
        ],
        "taxes_are_combined": False,
        "taxes_subtotal_label": "Taxes and Duties",
        "taxes_subtotal_amount": "569.62",
        "current_bill": "3,430.24",
        "arrears": "Rs. -0.53",
        "payable_within_due_date": "Rs. 3,430",
        "payable_after_due_date_amounts": ["3,574", "3,716"],
    }


class NormalizeContractTest(unittest.TestCase):
    def test_ke_printed_strings_match_contract(self):
        bill, errors = normalize(contract_raw())
        self.assertEqual(errors, [])
        self.assertEqual(dumps_bill(bill), CONTRACT_JSON)
        self.assertEqual([line["type"] for line in bill["charges"]].count("energy"), 2)
        self.assertEqual(bill["arrears"], Decimal("-0.53"))
        self.assertEqual(bill["payable_after_due_date"], Decimal("3716"))
        self.assertEqual(bill["current_bill"], Decimal("3430.24"))
        self.assertNotEqual(bill["payable_within_due_date"], bill["current_bill"])


class NormalizeRulesTest(unittest.TestCase):
    def test_provider_comes_from_header_text(self):
        cases = {
            "K-Electric": "KE",
            "KESC": "KE",
            "KE": "KE",
            "Lahore Electric Supply Company": "LESCO",
            "LESCO": "LESCO",
            "Islamabad Electric Supply Company": "IESCO",
            "IESCO": "IESCO",
        }
        for header, expected in cases.items():
            bill, errors = normalize(blank_raw(provider_name=header))
            self.assertEqual(bill["provider"], expected, header)
            self.assertEqual(errors, [], header)

    def test_invalid_provider_keeps_other_fields(self):
        bill, errors = normalize(blank_raw(provider_name="MEPCO", tariff="A-1a(01)"))
        self.assertIsNone(bill["provider"])
        self.assertEqual(bill["tariff"], "A-1a(01)")
        self.assertEqual(errors[0]["code"], "invalid_provider")
        self.assertEqual(errors[0]["raw"], "MEPCO")

    def test_missing_provider(self):
        bill, errors = normalize(blank_raw(provider_name=None))
        self.assertIsNone(bill["provider"])
        self.assertEqual(errors[0]["code"], "missing_field")
        self.assertEqual(bill["tariff"], "A1-R")

    def test_tariff_is_copied_exactly(self):
        bill, _errors = normalize(blank_raw(tariff="  A-1a(01)  "))
        self.assertEqual(bill["tariff"], "A-1a(01)")

    def test_sanctioned_load_strips_unit_word(self):
        bill, errors = normalize(blank_raw(sanctioned_load="3 KW"))
        self.assertEqual(errors, [])
        self.assertEqual(bill["sanctioned_load_kw"], Decimal("3"))

    def test_dates_and_two_digit_years(self):
        self.assertEqual(parse_bill_month("Apr-26"), "2026-04")
        self.assertEqual(parse_bill_month("JUL 2026"), "2026-07")
        self.assertEqual(parse_bill_month("SEP 26"), "2026-09")
        self.assertEqual(parse_date("03-Apr-26"), "2026-04-03")
        self.assertEqual(parse_date("07-Apr-2026"), "2026-04-07")
        self.assertEqual(parse_date("21st Apr 2026"), "2026-04-21")
        self.assertEqual(parse_date("13th Jul 2026"), "2026-07-13")
        self.assertEqual(parse_date("08 OCT 26"), "2026-10-08")
        self.assertEqual(parse_date("17 JUL 26"), "2026-07-17")
        self.assertEqual(parse_date("25-Jun-26"), "2026-06-25")
        self.assertEqual(parse_date("15-Jan-99"), "2099-01-15")
        self.assertEqual(parse_date("01-Feb-00"), "2000-02-01")
        self.assertEqual(parse_date("29-Feb-00"), "2000-02-29")
        bill, errors = normalize(blank_raw(due_date="32-Apr-26"))
        self.assertIsNone(bill["due_date"])
        self.assertEqual(errors[0]["code"], "unparseable_date")
        self.assertEqual(errors[0]["raw"], "32-Apr-26")
        self.assertEqual(bill["bill_month"], "2024-01")

    def test_units_are_not_calculated_from_readings(self):
        bill, errors = normalize(
            blank_raw(previous_reading="100", current_reading="180", units_consumed=None)
        )
        self.assertEqual(errors, [])
        self.assertEqual(bill["previous_reading"], Decimal("100"))
        self.assertEqual(bill["current_reading"], Decimal("180"))
        self.assertIsNone(bill["units_consumed"])

    def test_printed_units_win_over_the_reading_gap(self):
        bill, errors = normalize(
            blank_raw(previous_reading="100", current_reading="250", units_consumed="40")
        )
        self.assertEqual(errors, [])
        self.assertEqual(bill["units_consumed"], Decimal("40"))

    def test_latest_dated_meter_row_supplies_the_readings(self):
        bill, errors = normalize(
            blank_raw(
                previous_reading="1",
                current_reading="2",
                units_consumed="9",
                meter_rows=[
                    {
                        "previous_reading": "100",
                        "current_reading": "110",
                        "units": "10",
                        "reading_at": "01-Jan-24",
                    },
                    {
                        "previous_reading": "300",
                        "current_reading": "340",
                        "units": "40",
                        "reading_at": "15-Mar-24 18:30:00",
                    },
                    {
                        "previous_reading": "200",
                        "current_reading": "220",
                        "units": "20",
                        "reading_at": "15-Mar-24 09:00",
                    },
                ],
            )
        )
        self.assertEqual(errors, [])
        self.assertEqual(bill["previous_reading"], Decimal("300"))
        self.assertEqual(bill["current_reading"], Decimal("340"))
        self.assertEqual(bill["units_consumed"], Decimal("9"))

    def test_meter_rows_without_dates_keep_the_printed_pair(self):
        bill, errors = normalize(
            blank_raw(
                previous_reading="100",
                current_reading="180",
                units_consumed="40",
                meter_rows=[
                    {
                        "previous_reading": "5",
                        "current_reading": "6",
                        "units": "1",
                        "reading_at": None,
                    },
                    {
                        "previous_reading": "7",
                        "current_reading": "8",
                        "units": "1",
                        "reading_at": None,
                    },
                ],
            )
        )
        self.assertEqual(errors, [])
        self.assertEqual(bill["previous_reading"], Decimal("100"))
        self.assertEqual(bill["current_reading"], Decimal("180"))
        self.assertEqual(bill["units_consumed"], Decimal("40"))

    def test_extra_meter_rows_keep_the_printed_readings(self):
        bill, errors = normalize(
            blank_raw(
                meter_ambiguous=True,
                previous_reading="100",
                current_reading="180",
                units_consumed="40",
                meter_note="off peak and peak",
            )
        )
        self.assertEqual(errors, [])
        self.assertEqual(bill["previous_reading"], Decimal("100"))
        self.assertEqual(bill["current_reading"], Decimal("180"))
        self.assertEqual(bill["units_consumed"], Decimal("40"))

    def test_zero_lines_drop_and_unknown_charge_is_kept(self):
        bill, errors = normalize(
            blank_raw(
                charge_lines=[
                    {"label": "Service Rent", "amount": "15"},
                    {"label": "Fixed Charges", "amount": "0.00"},
                    {"label": "", "amount": ""},
                    {"label": "Fuel Price Adjustment", "amount": "1.25"},
                ]
            )
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            [(line["type"], line["amount"]) for line in bill["charges"]],
            [("service_rent", Decimal("15")), ("fpa", Decimal("1.25"))],
        )
        self.assertIsNone(bill["total_charges"])

    def test_repeated_charge_types_stay_separate(self):
        bill, _errors = normalize(
            blank_raw(
                charge_lines=[
                    {"label": "Uniform Quarterly Adjustment", "amount": "10.00"},
                    {"label": "Uniform Quarterly Adjustment", "amount": "-2.50"},
                    {"label": "Variable Off Peak", "amount": "8"},
                    {"label": "Variable Peak", "amount": "3"},
                ]
            )
        )
        self.assertEqual(
            [(line["type"], line["amount"]) for line in bill["charges"]],
            [
                ("quarterly_adjustment", Decimal("10")),
                ("quarterly_adjustment", Decimal("-2.50")),
                ("energy", Decimal("8")),
                ("energy", Decimal("3")),
            ],
        )

    def test_net_is_not_used_when_a_total_line_is_printed(self):
        bill, errors = normalize(
            blank_raw(
                charge_lines=[{"label": "Fixed Charges", "amount": "100"}],
                charges_subtotal_label="Total Electricity Charges",
                charges_subtotal_amount="100",
            )
        )
        self.assertEqual(errors, [])
        self.assertEqual(bill["total_charges"], Decimal("100"))

        with_net, net_errors = normalize(
            blank_raw(
                charge_lines=[
                    {"label": "Fixed Charges", "amount": "80"},
                    {"label": "Net Electricity Charges", "amount": "50"},
                ],
                charges_subtotal_label="Electricity Charges",
                charges_subtotal_amount="80",
            )
        )
        self.assertEqual(net_errors, [])
        self.assertEqual(with_net["total_charges"], Decimal("80"))
        self.assertEqual(len(with_net["charges"]), 1)

    def test_printed_subtotal_is_kept_when_lines_differ(self):
        bill, errors = normalize(
            blank_raw(
                charge_lines=[
                    {"label": "Fixed Charges", "amount": "100"},
                    {"label": "Variable Charges", "amount": "50"},
                ],
                charges_subtotal_label="Electricity Charges",
                charges_subtotal_amount="200",
            )
        )
        self.assertEqual(bill["total_charges"], Decimal("200"))
        self.assertEqual(errors[0]["code"], "subtotal_mismatch")
        self.assertEqual(errors[0]["field"], "total_charges")

    def test_missing_subtotal_is_not_replaced_by_a_sum(self):
        bill, errors = normalize(
            blank_raw(
                charge_lines=[{"label": "Fixed Charges", "amount": "40"}],
                tax_lines=[
                    {"label": "Electricity Duty", "amount": "2"},
                    {"label": "Sales Tax", "amount": "3"},
                ],
            )
        )
        self.assertEqual(errors, [])
        self.assertIsNone(bill["total_charges"])
        self.assertIsNone(bill["total_taxes"])
        self.assertEqual(len(bill["taxes"]), 2)

    def test_combined_tax_collapses_to_total(self):
        bill, errors = normalize(
            blank_raw(
                taxes_are_combined=True,
                tax_lines=[{"label": "Taxes 15.24%", "amount": "120.50"}],
            )
        )
        self.assertEqual(errors, [])
        self.assertEqual(bill["taxes"], [])
        self.assertEqual(bill["total_taxes"], Decimal("120.50"))

    def test_arrears_sign_zero_blank_and_slash(self):
        credit, credit_errors = normalize(blank_raw(arrears="0.53 CR"))
        self.assertEqual(credit_errors, [])
        self.assertEqual(credit["arrears"], Decimal("-0.53"))

        zero, zero_errors = normalize(blank_raw(arrears="Rs. 0"))
        self.assertEqual(zero_errors, [])
        self.assertEqual(zero["arrears"], Decimal("0"))

        blank, blank_errors = normalize(blank_raw(arrears="  "))
        self.assertEqual(blank_errors, [])
        self.assertIsNone(blank["arrears"])

        slash, slash_errors = normalize(blank_raw(arrears="39411 / 2"))
        self.assertEqual(slash_errors, [])
        self.assertEqual(slash["arrears"], Decimal("39411"))
        self.assertEqual(slash["tariff"], "A1-R")

        backslash, backslash_errors = normalize(blank_raw(arrears="39,411 \\ 2"))
        self.assertEqual(backslash_errors, [])
        self.assertEqual(backslash["arrears"], Decimal("39411"))

    def test_highest_after_due_payable_is_selected_not_added(self):
        bill, errors = normalize(
            blank_raw(
                payable_within_due_date="1000",
                payable_after_due_date_amounts=["1100", "1300", "1200.50"],
            )
        )
        self.assertEqual(errors, [])
        self.assertEqual(bill["payable_within_due_date"], Decimal("1000"))
        self.assertEqual(bill["payable_after_due_date"], Decimal("1300"))

    def test_bad_after_due_value_does_not_drop_the_valid_ones(self):
        bill, errors = normalize(
            blank_raw(payable_after_due_date_amounts=["100", "39411 / 2", "250"])
        )
        self.assertEqual(bill["payable_after_due_date"], Decimal("250"))
        self.assertEqual(errors[0]["code"], "unparseable_amount")
        self.assertEqual(errors[0]["raw"], "39411 / 2")

    def test_binary_float_amounts_are_refused(self):
        with self.assertRaises(AmountParseError):
            parse_amount(0.1)

    def test_late_payment_surcharge_is_not_typed_as_phl(self):
        bill, _errors = normalize(
            blank_raw(
                charge_lines=[{"label": "Late Payment Surcharge", "amount": "5"}],
            )
        )
        self.assertEqual(bill["charges"][0]["type"], "late_payment_surcharge")


class ProductionSourceTest(unittest.TestCase):
    def test_src_does_not_embed_training_fixtures(self):
        text = "\n".join(path.read_text(encoding="utf-8") for path in SRC.rglob("*.py"))
        banned = [
            "663.51",
            "2860.62",
            "3430.24",
            "KESC_0008",
            "KESC_0004",
            "LESCO_0005",
            "LESCO_0006",
            "IESCO_0004",
            "previous_reading -",
            "current_reading -",
            "payable_within_due_date +",
        ]
        for token in banned:
            self.assertNotIn(token, text)
        self.assertIn("2000 + year", text)
        self.assertNotIn("2026", text)


if __name__ == "__main__":
    unittest.main()
