"""Contract tests for the feature list, the split, the TFT cell, and leakage."""

from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from housing_tft.calendar import national_macro, project_known_future  # noqa: E402
from housing_tft.data import (  # noqa: E402
    assign_splits,
    build_windows,
    category_index,
    fit_scaler,
    group_rows,
)
from housing_tft.baselines import aligned_proba  # noqa: E402
from housing_tft.model import TemporalFusionTransformer, manual_lstm  # noqa: E402
from housing_tft.simulate import simulate, tier_from_state  # noqa: E402
from housing_tft.spec import (  # noqa: E402
    EXPECTED_NAMES,
    FEATURES,
    KNOWN_CAT,
    KNOWN_NUM,
    VOCABS,
)


class SpecTests(unittest.TestCase):
    def test_fifty_names_in_order(self):
        expected = [
            "tenure_start", "lease_months_remaining", "rent_arrears", "persons_per_bedroom", "moves_12m",
            "emergency_housing_stay", "deprivation_category", "mortgage_payment", "housing_cost_burden",
            "disaster_displaced", "month_of_year", "quarter", "school_holiday", "public_holidays", "age_band",
            "residency_status", "ethnicity", "local_board", "nzdep_decile", "local_median_rent",
            "local_affordability_index", "local_price_to_income", "local_homelessness_rate", "local_eh_grants",
            "local_service_spend", "whanau_type", "sole_parent", "dependants_count", "household_size_start",
            "household_size_change", "weekly_income", "main_benefit", "hourly_wage", "essential_spend_share",
            "debt_balance", "benefit_indexation", "planned_support_places", "emergency_accommodation_type",
            "disability", "substance_service_contact", "prior_justice_history", "justice_event",
            "gambling_spend_share", "ocr_rate", "cpi_inflation_annual", "unemployment_rate", "petrol_price",
            "pandemic_restriction_level", "natural_disaster_severity", "major_event_context",
        ]
        self.assertEqual(list(EXPECTED_NAMES), expected)
        self.assertEqual(len(FEATURES), 50)
        self.assertEqual(len({f.name for f in FEATURES}), 50)

    def test_every_category_has_a_vocab(self):
        for feature in FEATURES:
            if feature.kind == "categorical":
                self.assertIn(feature.name, VOCABS)
                self.assertGreaterEqual(len(VOCABS[feature.name]), 2)

    def test_tier_legend_cases(self):
        self.assertEqual(tier_from_state("emergency", 0.2), 1)
        self.assertEqual(tier_from_state("transitional", 0.2), 2)
        self.assertEqual(tier_from_state("market_rent", 0.7), 3)
        self.assertEqual(tier_from_state("social", 0.2), 4)
        self.assertEqual(tier_from_state("market_own", 0.2), 5)


class SplitTests(unittest.TestCase):
    def test_seeded_300_200_is_stable_and_disjoint(self):
        ids = [f"W{i:04d}" for i in range(1, 501)]
        first = assign_splits(ids, seed=42)
        second = assign_splits(list(reversed(ids)), seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first.fit), 240)
        self.assertEqual(len(first.val), 60)
        self.assertEqual(len(first.test), 200)
        self.assertEqual(len(set(first.train)), 300)
        self.assertEqual(set(first.fit) & set(first.val), set())
        self.assertEqual(set(first.train) & set(first.test), set())
        self.assertEqual(set(first.fit) | set(first.val) | set(first.test), set(ids))


class ModelTests(unittest.TestCase):
    def test_manual_lstm_matches_pytorch(self):
        torch.manual_seed(0)
        cell = torch.nn.LSTM(5, 7, batch_first=True)
        x = torch.randn(4, 6, 5)
        h0 = torch.randn(1, 4, 7)
        c0 = torch.randn(1, 4, 7)
        y, (hn, cn) = cell(x, (h0, c0))
        y2, h2, c2 = manual_lstm(x, h0[0], c0[0], cell.weight_ih_l0, cell.weight_hh_l0, cell.bias_ih_l0, cell.bias_hh_l0)
        self.assertTrue(torch.allclose(y, y2, atol=1e-5))
        self.assertTrue(torch.allclose(hn[0], h2, atol=1e-5))
        self.assertTrue(torch.allclose(cn[0], c2, atol=1e-5))

    def test_forward_shapes_and_selection_weights(self):
        torch.manual_seed(1)
        model = TemporalFusionTransformer(d_model=16, d_hidden=24, n_heads=4, dropout=0.0)
        model.eval()
        batch = 3
        static_cat = torch.randint(0, 2, (batch, len(model.static_cat.embs)))
        static_num = torch.randn(batch, model.static_num.w.shape[0])
        hist_cat = torch.randint(0, 2, (batch, 12, len(model.obs_cat.embs)))
        hist_num = torch.randn(batch, 12, model.obs_num.w.shape[0])
        known_cat = torch.randint(0, 2, (batch, 12, len(model.known_cat.embs)))
        known_num = torch.randn(batch, 12, model.known_num.w.shape[0])
        known_cat_f = torch.randint(0, 2, (batch, 12, len(model.known_cat.embs)))
        known_num_f = torch.randn(batch, 12, model.known_num.w.shape[0])
        now, future, attn, w_static, w_hist, w_future = model(
            static_cat, static_num, hist_cat, hist_num, known_cat, known_num, known_cat_f, known_num_f
        )
        self.assertEqual(tuple(now.shape), (3, 5))
        self.assertEqual(tuple(future.shape), (3, 12, 5))
        self.assertEqual(tuple(attn.shape), (3, 12, 24))
        self.assertTrue(torch.allclose(w_static.sum(-1), torch.ones(3), atol=1e-4))
        self.assertTrue(torch.allclose(w_hist.sum(-1), torch.ones(3, 12), atol=1e-4))
        self.assertTrue(torch.allclose(w_future.sum(-1), torch.ones(3, 12), atol=1e-4))
        self.assertTrue(torch.allclose(attn.sum(-1), torch.ones(3, 12), atol=1e-4))


class PanelTests(unittest.TestCase):
    def test_small_panel_and_lease_countdown(self):
        rows = simulate(n=12, months=60, seed=7)
        self.assertEqual(len(rows), 12 * 60)
        grouped = group_rows(rows)
        for series in grouped.values():
            opening = series[0]["household_size_start"]
            self.assertTrue(all(row["household_size_start"] == opening for row in series))
            self.assertTrue(all(row["stability_tier"] in {1, 2, 3, 4, 5} for row in series))
        from housing_tft.labels import tier_id
        self.assertEqual(tier_id({"deprivation_category": "Without shelter"}), 1)
        self.assertEqual(tier_id({"deprivation_category": "Temporary accommodation", "emergency_housing_stay": 1}), 2)
        self.assertEqual(tier_id({"deprivation_category": "Housed", "tenure_start": "Social housing", "housing_cost_burden": 0.2}), 4)
        self.assertEqual(tier_id({"deprivation_category": "Housed", "tenure_start": "Private rental", "housing_cost_burden": 0.1, "rent_arrears": 0, "persons_per_bedroom": 1}), 5)
        rows = []
        for wid in ("A", "B"):
            for t in range(1, 61):
                rows.append({
                    "whanau_id": wid, "time_idx": t, "month_of_year": ((t - 1) % 12) + 1, "quarter": ((t - 1) % 12) // 3 + 1,
                    "school_holiday": 0, "public_holidays": 0, "benefit_indexation": 0, "planned_support_places": 4,
                    "lease_months_remaining": 99 if wid == "A" else 6, "tenure_start": "Private rental", "age_band": "35-44",
                    "residency_status": "Citizen", "local_board": "Howick", "whanau_type": "Couple only",
                    "nzdep_decile": 5, "sole_parent": 0, "dependants_count": 0, "household_size_start": 2, "disability": 0,
                    "prior_justice_history": 0, "deprivation_category": "Housed", "main_benefit": "None",
                    "emergency_accommodation_type": "None", "ethnicity": "European",
                    **{name: 0 for name in ("rent_arrears", "persons_per_bedroom", "moves_12m", "emergency_housing_stay", "mortgage_payment", "housing_cost_burden", "disaster_displaced", "local_median_rent", "local_affordability_index", "local_price_to_income", "local_homelessness_rate", "local_eh_grants", "local_service_spend", "household_size_change", "weekly_income", "hourly_wage", "essential_spend_share", "debt_balance", "substance_service_contact", "justice_event", "gambling_spend_share", "ocr_rate", "cpi_inflation_annual", "unemployment_rate", "petrol_price", "pandemic_restriction_level", "natural_disaster_severity")},
                })
        grouped2 = group_rows(rows)
        scaler = fit_scaler(grouped2, ["A"])
        arrays, _meta = build_windows(grouped2, ["B"], scaler, origins=[48])
        lease_at = [f.name for f in KNOWN_NUM].index("lease_months_remaining")
        mean, std = scaler.mean["lease_months_remaining"], scaler.std["lease_months_remaining"]
        future = arrays["known_num_fut"][0, :, lease_at] * std + mean
        self.assertTrue(np.allclose(future, np.maximum(0, 6 - np.arange(1, 13)), atol=1e-3))


class BaselineTests(unittest.TestCase):
    def test_aligned_proba_pads_missing_classes(self):
        sklearn = mock.MagicMock()
        sklearn.classes_ = np.array([1, 3, 4])
        sklearn.predict_proba.return_value = np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
        out = aligned_proba(sklearn, np.zeros((2, 3)))
        self.assertAlmostEqual(out[0, 1], 0.2)
        self.assertAlmostEqual(out[0, 3], 0.5)
        self.assertAlmostEqual(out.sum(axis=1).tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
