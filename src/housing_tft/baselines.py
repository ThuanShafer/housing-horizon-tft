"""Snapshot classifiers on the same splits as the TFT."""

from __future__ import annotations

import numpy as np

from housing_tft.metrics import performance, scores


def aligned_proba(classifier, x: np.ndarray, n: int = 5) -> np.ndarray:
    raw = classifier.predict_proba(x)
    out = np.zeros((len(x), n), dtype=np.float64)
    for col, cls in enumerate(classifier.classes_):
        out[:, int(cls)] = raw[:, col]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum


def fit_random_forest_classifier(x_fit: np.ndarray, y_fit: np.ndarray, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_fit, y_fit)
    return model


def random_forest_evaluation(fit_w, test_one, design_matrix, seed: int) -> dict:
    """Train one forest per horizon on the fit split; score the 200 origin windows."""
    x_fit = design_matrix(fit_w, True)
    x_test = design_matrix(test_one, True)
    report_horizons = {}
    models = {}
    for name, index in (("now", None), ("h3", 2), ("h6", 5), ("h12", 11)):
        y_fit = fit_w["y_now"] if index is None else fit_w["y_fut"][:, index]
        y_test = test_one["y_now"] if index is None else test_one["y_fut"][:, index]
        clf = fit_random_forest_classifier(x_fit, y_fit, seed + (index if index is not None else 0))
        guessed = clf.predict(x_test).astype(np.int64)
        report_horizons[name] = scores(y_test, guessed)
        models[name] = clf
    h6_prob = aligned_proba(models["h6"], x_test)
    guessed_h6 = models["h6"].predict(x_test).astype(np.int64)
    y_h6 = test_one["y_fut"][:, 5]
    return {
        "task": "Five-tier housing stability, Random Forest",
        "note": (
            "400 trees per horizon, balanced subsample weights, sqrt features. "
            "Last-month numeric inputs, 3- and 6-month deltas, and one-hot categories at the origin month. "
            "Same 240 fit families and 200 test whānau as the transformer."
        ),
        "test_origin": report_horizons,
        "performance_h6": performance(y_h6, guessed_h6, h6_prob),
        "val_macro_f1_h6": float(report_horizons["h6"]["macro_f1"]),
    }