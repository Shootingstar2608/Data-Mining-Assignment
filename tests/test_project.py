"""Regression tests for preprocessing, score ties and input provenance."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from etl_rfm import clean_transactions, build_rfm
from features import refine_transactions, load_cancellations, build_customer_features
from gmm_modeling import load_rfm, select_candidate, preprocess_rfm
from modeling import quintile_score, rfm_segments
from reproducibility import file_fingerprint, matches_input


def fixture():
    rows = [
        ("1", "12345", 2, "2011-01-01", 10., 1., "UK"),
        ("1", "12346", 1, "2011-01-01", 5., 1., "UK"),
        ("1", "12346", 1, "2011-01-01", 5., 1., "UK"),  # duplicate
        ("2", "12345", 1, "2011-01-03", 10., 1., "UK"),
        ("2", "POST", 1, "2011-01-03", 99., 1., "UK"),
        ("C1", "12345", -1, "2011-01-04", 10., 1., "UK"),
        ("C1", "12345", -1, "2011-01-04", 10., 1., "UK"),  # duplicate return
        ("3", "12345", 1, "2011-01-02", 10., 2., "UK"),
        ("C3", "12345", -1, "2011-01-04", 10., 2., "UK"),  # fully returned
        ("4", "12345", 1, "2011-01-02", 0., 3., "UK"),
        ("5", "12345", 1, "2011-01-02", 10., np.nan, "UK"),
        ("C4", "12345", 1, "2011-01-04", 10., 1., "UK"),  # invalid return sign
        ("C5", "12345", -1, "2011-01-04", -10., 1., "UK"),
    ]
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate",
                                       "UnitPrice", "CustomerID", "Country"])


class TransactionTests(unittest.TestCase):
    def test_clean_excludes_invalid_purchases(self):
        clean = clean_transactions(fixture())
        self.assertTrue((clean.Quantity > 0).all())
        self.assertTrue((clean.UnitPrice > 0).all())
        self.assertFalse(clean.InvoiceNo.str.startswith("C").any())
        self.assertFalse(clean.CustomerID.isna().any())

    def test_unique_invoice_frequency_and_net_returns(self):
        raw = fixture()
        purchases = refine_transactions(clean_transactions(raw))
        cancellations = load_cancellations(raw=raw)
        self.assertEqual(len(cancellations), 2)
        features = build_customer_features(purchases, cancellations)
        self.assertEqual(features.CustomerID.tolist(), [1.])
        row = features.iloc[0]
        self.assertEqual(row.Frequency, 2)
        self.assertEqual(row.Monetary, 25.)
        self.assertEqual(row.AvgOrderValue, 12.5)
        self.assertEqual(row.Recency, 1)
        self.assertAlmostEqual(row.ReturnRate, 10 / 35)

    def test_gross_rfm_uses_unique_invoices(self):
        table = build_rfm(clean_transactions(fixture())).set_index("CustomerID")
        self.assertEqual(table.loc[1., "Frequency"], 2)

    def test_missing_column_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            clean_transactions(fixture().drop(columns="InvoiceDate"))


class ScoringTests(unittest.TestCase):
    def test_ties_and_row_order(self):
        frame = pd.DataFrame({"CustomerID": range(100), "Frequency": [1]*45 + [2]*25 + list(range(3, 33))})
        frame["score"] = quintile_score(frame.Frequency)
        self.assertEqual(frame.loc[frame.Frequency.eq(1), "score"].nunique(), 1)
        shuffled = frame.sample(frac=1, random_state=42)
        pd.testing.assert_series_equal(quintile_score(shuffled.Frequency).sort_index(), frame.score.rename(None))

    def test_direction_and_constant_values(self):
        values = pd.Series([1., 2., 3., 4., 5.])
        self.assertLess(quintile_score(values).iloc[0], quintile_score(values).iloc[-1])
        self.assertGreater(quintile_score(values, False).iloc[0], quintile_score(values, False).iloc[-1])
        self.assertEqual(quintile_score(pd.Series([1]*30)).nunique(), 1)

    def test_all_rule_codes_are_mapped(self):
        # The rule map covers the 25 possible R/F score combinations.
        values = pd.DataFrame([(r, f) for r in range(1, 6) for f in range(1, 6)], columns=["Recency", "Frequency"])
        values["Monetary"] = np.arange(1, 26)
        self.assertFalse(rfm_segments(values).RFM_Segment.str.fullmatch(r"\d\d").any())


class ProvenanceTests(unittest.TestCase):
    def test_line_endings_and_content_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/"a.csv", Path(directory)/"b.csv"
            a.write_bytes(b"CustomerID,Monetary\n1,10\n")
            b.write_bytes(a.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(file_fingerprint(a), file_fingerprint(b))
            summary = {"input_sha256": file_fingerprint(a), "fingerprint_format": "sha256-lf-v1"}
            self.assertTrue(matches_input(b, summary))
            legacy = {"input_sha256": hashlib.sha256(b.read_bytes()).hexdigest()}
            self.assertTrue(matches_input(a, legacy))
            b.write_bytes(b"CustomerID,Monetary\n1,11\n")
            self.assertFalse(matches_input(b, summary))
            self.assertFalse(matches_input(b, legacy))

    def test_invalid_rfm_rejected(self):
        valid = pd.DataFrame({"CustomerID": [1, 2], "Recency": [1, 10], "Frequency": [2, 3], "Monetary": [10, 30]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"rfm.csv"
            for column, bad in [("CustomerID", [1, 1]), ("Recency", [0, 1]),
                                ("Monetary", [np.inf, 1]), ("Frequency", [np.nan, 1])]:
                with self.subTest(column=column):
                    invalid = valid.copy(); invalid[column] = bad; invalid.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_rfm(path)

    def test_no_cap_matches_shared_matrix(self):
        from sklearn.preprocessing import StandardScaler
        frame = pd.DataFrame({"Recency": [1, 2, 300], "Frequency": [1, 3, 100], "Monetary": [1, 300, 100000]})
        got, _, _, _ = preprocess_rfm(frame, 1.)
        np.testing.assert_allclose(got, StandardScaler().fit_transform(np.log1p(frame)))

    def test_selection_requires_eligible_candidate(self):
        scores = pd.DataFrame({"bic": [1., 2., 3.], "stable_candidate": [False, True, True]})
        self.assertEqual(select_candidate(scores).bic, 2.)
        with self.assertRaises(RuntimeError):
            select_candidate(scores.assign(stable_candidate=False))


if __name__ == "__main__":
    unittest.main()
