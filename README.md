# Housing Horizon

A Temporal Fusion Transformer that forecasts one of five housing-stability tiers for a whānau. The portal loads the fitted weights in the browser and classifies an uploaded JSON file.

This is a research prototype on a **synthetic** Auckland panel of 500 whānau × 60 months (January 2021–December 2025). It is not a determination of homelessness, eligibility, or who should receive support.

The linked spreadsheet returned HTTP 401, so the panel was generated to the same 50-feature schema (`PYTHONPATH=src python -m housing_tft.simulate`, seed 42). To train on another extract, replace `data/panel.csv` with columns `whanau_id`, `time_idx`, `stability_tier` (1–5) and the 50 feature names, then rerun training.

## Tiers

1. Severely Deprived (Acute Homelessness)
2. Unstable / Transitional
3. At Risk (Housing Insecure)
4. Subsidised / Assisted Stability
5. Stable / Market Secure

`stability_tier` is the target. It is not one of the 50 inputs.

## Split

Seed **42** assigns 300 whānau to the training pool and 200 to the test. Seed **43** holds 60 of the 300 for early stopping, so weights are fit on 240. No whānau appears in more than one group. The seed slider in the dashboard refits that same cut for any integer from 0 to 100.

The headline window is history January–December 2024 (`time_idx` 36–47) and a forecast through December 2025. The loss emphasises the 3-, 6- and 12-month steps.

Known-future lease months count down from the origin. A later renewal is not shown to the model, because that would leak the housing outcome.

## Run

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m unittest tests.test_core
PYTHONPATH=src python -m housing_tft.train
PYTHONPATH=src python -m housing_tft.classify web/examples/sample-history.json
cd web && python3 serve.py
```

Open `http://127.0.0.1:8765`. Node can check that the browser forward pass matches PyTorch:

```bash
node tests/parity.mjs
```

## JSON

One month, a list of months, or `{"history":[...]}`. Each month needs the 50 feature names. Include `time_idx` (0 = January 2021) when you have it, so the published calendar and support-place path can be rebuilt. Twelve monthly rows are the encoder length the model was trained on. A shorter history is repeated to fill that length, and the page says so.

## Model

Gated residual networks, variable selection, an LSTM encoder–decoder, static enrichment, and interpretable multi-head attention, following Lim, Arık, Loeff and Pfister (2021). The head is five-class rather than a quantile head, because the requested output is a tier. The scorecard on the page compares the six-month forecast with a no-change rule and a last-month linear model. On this panel most whānau do not change tier in six months, and the page reports that comparison directly.
