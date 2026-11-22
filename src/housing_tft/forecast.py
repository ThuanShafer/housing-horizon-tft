"""Turn a fitted TFT and a JSON payload into a five-tier forecast."""

from __future__ import annotations

import numpy as np
import torch

from housing_tft.data import INPUT_KEYS, encode_case, parse_payload
from housing_tft.spec import FEATURES, TIERS, VOCABS, feature_map


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@torch.no_grad()
def predict_batch(model, arrays: dict) -> dict[str, np.ndarray]:
    model.eval()
    batch = {key: torch.from_numpy(arrays[key]) for key in INPUT_KEYS}
    now, future, attn, w_static, w_hist, w_future = model(
        batch["static_cat"], batch["static_num"], batch["hist_cat"], batch["hist_num"],
        batch["known_cat_hist"], batch["known_num_hist"], batch["known_cat_fut"], batch["known_num_fut"],
    )
    return {
        "now": now.numpy(),
        "future": future.numpy(),
        "attention": attn.numpy(),
        "w_static": w_static.numpy(),
        "w_hist": w_hist.numpy(),
        "w_future": w_future.numpy(),
    }


def _pack(probs: np.ndarray) -> dict:
    cls = int(np.argmax(probs))
    tier = TIERS[cls]
    order = np.argsort(-probs)
    margin = float(probs[order[0]] - probs[order[1]])
    return {
        "class_index": cls,
        "tier_id": tier["id"],
        "name": tier["name"],
        "short": tier["short"],
        "probability": float(probs[cls]),
        "margin": margin,
        "low_confidence": bool(float(probs[cls]) < 0.4),
        "probabilities": [
            {"tier_id": TIERS[i]["id"], "name": TIERS[i]["name"], "short": TIERS[i]["short"], "probability": float(probs[i])}
            for i in range(5)
        ],
    }


def _top(names: list[str], weights: np.ndarray, limit: int) -> list[dict]:
    spec = feature_map()
    order = np.argsort(-weights)[:limit]
    rows = []
    for index in order:
        feature = spec[names[int(index)]]
        rows.append({
            "name": feature.name,
            "domain": feature.domain,
            "role": feature.role,
            "sensitive": feature.sensitive,
            "description": feature.description,
            "weight": float(weights[int(index)]),
        })
    return rows


def describe_case(model, pred: dict, info: dict) -> dict:
    now = _pack(softmax(pred["now"][0]))
    future_probs = softmax(pred["future"][0])
    path = []
    for step in range(future_probs.shape[0]):
        packed = _pack(future_probs[step])
        packed["step"] = step + 1
        packed["label"] = info["future_labels"][step]
        path.append(packed)
    primary = path[5]
    hist_mean = pred["w_hist"][0].mean(axis=0)
    attention = pred["attention"][0, 5]
    notes = list(info["notes"])
    if primary["low_confidence"]:
        notes.append("the 6-month probability is under 0.40, so this is a weak call rather than a firm classification")
    elif primary["margin"] < 0.08:
        notes.append("the top two tiers are close at the 6-month horizon")
    sensitive_top = [row["name"] for row in _top(model.hist_names, hist_mean, 8) if row["sensitive"]]
    if sensitive_top:
        notes.append(
            "sensitive inputs (" + ", ".join(sensitive_top) + ") are among the stronger variable-selection weights for this history. "
            "That is an association in the synthetic panel, not a reason to treat those inputs as causes."
        )
    return {
        "primary_horizon_months": 6,
        "classification": primary,
        "nowcast": now,
        "forecasts": {"3": path[2], "6": path[5], "12": path[11]},
        "path": path,
        "attention": {
            "horizon_months": 6,
            "labels": info["history_labels"] + info["future_labels"],
            "weights": [float(v) for v in attention],
        },
        "variables": {
            "history": _top(model.hist_names, hist_mean, 8),
            "static": _top(model.static_names, pred["w_static"][0], 5),
            "future": _top(model.future_names, pred["w_future"][0, 5], 5),
        },
        "history_months_provided": info["months_provided"],
        "padded": info["padded"],
        "origin": info["origin"],
        "origin_label": info["origin_label"],
        "whanau_id": info["last_row"].get("whanau_id"),
        "notes": notes,
        "disclaimer": "Synthetic-panel research prototype. This is not a determination of homelessness, eligibility, or who should receive support.",
    }


def forecast_payload(model, scaler, payload, calendar=None) -> dict:
    history, future = parse_payload(payload)
    arrays, info = encode_case(history, scaler, future, calendar)
    pred = predict_batch(model, arrays)
    return describe_case(model, pred, info)


def feature_catalog() -> list[dict]:
    return [
        {
            "name": feature.name,
            "role": feature.role,
            "kind": feature.kind,
            "domain": feature.domain,
            "unit": feature.unit,
            "sensitive": feature.sensitive,
            "description": feature.description,
            "values": list(VOCABS.get(feature.name, ())),
        }
        for feature in FEATURES
    ]
