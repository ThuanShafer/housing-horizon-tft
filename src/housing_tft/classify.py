"""Classify one JSON document with the fitted TFT.

    PYTHONPATH=src python -m housing_tft.classify path/to/case.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from housing_tft.data import Scaler
from housing_tft.forecast import forecast_payload
from housing_tft.model import TemporalFusionTransformer

ROOT = Path(__file__).resolve().parents[2]


def load_model(path: Path | None = None):
    checkpoint = torch.load(path or ROOT / "artifacts" / "tft.pt", map_location="cpu", weights_only=False)
    model = TemporalFusionTransformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, Scaler.from_json(checkpoint["scaler"]), checkpoint.get("calendar")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m housing_tft.classify case.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    model, scaler, calendar = load_model()
    print(json.dumps(forecast_payload(model, scaler, payload, calendar), indent=2))


if __name__ == "__main__":
    main()
