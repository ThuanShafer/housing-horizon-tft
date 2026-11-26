"""Add Random Forest metrics to an existing browser bundle without retraining the TFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from housing_tft.baselines import random_forest_evaluation
from housing_tft.data import HEADLINE_ORIGIN, assign_splits, build_windows, fit_scaler, group_rows, load_panel
from housing_tft.train import design_matrix

ROOT = Path(__file__).resolve().parents[2]


def patch_bundle(bundle: Path, seed: int | None = None) -> dict:
    metrics_path = bundle / "metrics.json"
    doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    seed = seed if seed is not None else int(doc["seed"])
    rows = load_panel(ROOT / "data" / "panel.xlsx")
    grouped = group_rows(rows)
    split = assign_splits(list(grouped), seed=seed)
    scaler = fit_scaler(grouped, split.fit)
    fit_w, _ = build_windows(grouped, split.fit, scaler)
    test_one, _ = build_windows(grouped, split.test, scaler, origins=[HEADLINE_ORIGIN])
    doc["random_forest"] = random_forest_evaluation(fit_w, test_one, design_matrix, seed)
    metrics_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc["random_forest"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Random Forest and patch metrics.json")
    parser.add_argument("--bundle", type=Path, default=ROOT / "web" / "model")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    block = patch_bundle(args.bundle.resolve(), args.seed)
    print(f"random forest h6 macro-F1 {block['test_origin']['h6']['macro_f1']:.3f}")
    print(f"wrote {args.bundle / 'metrics.json'}")


if __name__ == "__main__":
    main()