"""Check demo data, inference and the interactive presentation paths."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from demo_data import COLORS, SEGMENTS, filter_customers, load_dashboard, load_models, predict_scenario


class DemoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dashboard(ROOT)
        cls.bundles = load_models(ROOT)

    def test_filter_matches_country_and_segment(self):
        c = self.data["customers"]
        actual = filter_customers(c, ["VIP"], "United Kingdom")
        expected = c[c.Segment.eq("VIP") & c.Country.eq("United Kingdom")]
        pd.testing.assert_frame_equal(actual, expected)
        self.assertTrue(filter_customers(c, []).empty)

    def test_saved_model_matches_customers_in_every_segment(self):
        rows = self.data["customers"].groupby("Segment").head(2)
        for _, row in rows.iterrows():
            result = predict_scenario([row.Recency, row.Frequency, row.Monetary], self.bundles)
            self.assertEqual(result["segment"], SEGMENTS[row.Segment])
            self.assertAlmostEqual(result["probabilities"]["Xác suất"].sum(), 1.)
            self.assertTrue(result["probabilities"]["Xác suất"].between(0, 1).all())

    def test_scenario_rejects_invalid_rfm(self):
        for values in [[0, 1, 10], [1, -1, 10], [1, 1, np.nan], [1, 10]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                predict_scenario(values, self.bundles)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Install requirements-demo.txt for UI tests")
class DemoInteractionTests(unittest.TestCase):
    def make_app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def test_all_four_pages_and_chart_controls(self):
        app = self.make_app()
        self.assertEqual(app.metric[0].value, "4,320")
        for page in ["EDA", "Modeling", "Visualize"]:
            app.radio(key="demo_page").set_value(page).run()
            self.assertFalse(app.exception, page)
        app.radio(key="demo_page").set_value("EDA").run()
        app.selectbox(key="eda_variable").set_value("Frequency").run()
        app.toggle(key="eda_logged").set_value(False).run()
        self.assertFalse(app.exception)
        app.radio(key="demo_page").set_value("Modeling").run()
        app.selectbox(key="model_metric").set_value("DaviesBouldin").run()
        app.radio(key="k_metric").set_value("Inertia").run()
        self.assertFalse(app.exception)

    def test_empty_filter_and_customer_search(self):
        app = self.make_app()
        app.radio(key="demo_page").set_value("Visualize").run()
        app.multiselect(key="segment_filter").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Không có khách" in e.value for e in app.info))
        app.multiselect(key="segment_filter").set_value(["VIP"]).run()
        app.selectbox(key="country_filter").set_value("United Kingdom").run()
        app.text_input(key="customer_search").set_value("không-tồn-tại").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 0)

    def test_scenario_submit_and_extrapolation(self):
        app = self.make_app()
        app.radio(key="demo_page").set_value("Visualize").run()
        app.button(key="FormSubmitter:scenario_form-Xem phân khúc").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("VIP" in item.value for item in app.success))
        app.number_input(key="scenario_r").set_value(730)
        app.number_input(key="scenario_f").set_value(1)
        app.number_input(key="scenario_m").set_value(10.)
        app.button(key="FormSubmitter:scenario_form-Xem phân khúc").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("ngoài khoảng" in item.value for item in app.warning))


if __name__ == "__main__":
    unittest.main()
