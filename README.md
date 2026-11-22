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

The network is the Lim, Arık, Loeff and Pfister (2021) Temporal Fusion Transformer, written as ordinary PyTorch modules in `src/housing_tft/model.py` and trained from a random start. Nothing is loaded from a library checkpoint or a pretrained transformer. The same math is copied in `web/tft.js`, so the portal can run the fitted weights without PyTorch.

Each example is one whānau, cut into a 12-month history and a 12-month forecast. Of the 50 schema columns, 48 are inputs. Ethnicity and `major_event_context` are kept for the portal and are not predictors. `stability_tier` is the label only.

| Stream | When the model may see it | Examples |
|---|---|---|
| Static | Once, for the whole window | Tenure, age band, local board, NZDep, household type, disability |
| Observed | Only in the 12 history months | Income, rent burden, arrears, local rents, macro series |
| Known future | History and the 12 forecast months | Calendar, holidays, benefit indexation, planned support places, lease countdown |

Past outcome columns (arrears, overcrowding, moves, emergency-housing stay, deprivation category) are allowed in the history, because they already happened. They are not copied into the future. A lease of 99 stays 99 (no fixed term). Any other lease counts down from the last observed month and does not renew. A renewal would tell the model something about the housing outcome.

Numbers are z-scored with the mean and standard deviation of the 240 training whānau. Categories become embedding indices. Index 0 means unknown; the real vocabulary starts at 1.

### Forward pass

Width is small on purpose: each token is 32-dimensional (`d_model=32`), the gated-residual hidden size is 48, and attention has 4 heads.

1. **Each variable gets its own embedding.** A numeric feature is one learned scale-and-shift into 32 dimensions, `x * w + b`, with a separate `w` and `b` per column. A category uses `nn.Embedding`. After this, a month is a stack of 32-d vectors, one per variable, not one flat vector.
2. **Variable selection compresses that stack.** A gated residual network looks at all the embeddings and emits a softmax weight per variable. A second set of gated residuals, one per variable, rewrites each embedding. The weighted sum is a single 32-d vector for that time step. The same block runs three times: once for the static features, once per history month, once per future month. The static vector is turned into a context (`c_s`) that the history and future selectors can see, so who this whānau is changes which monthly variables get weight.
3. **Four static contexts, not one.** From that static summary the model builds `c_s` (the selection context), `c_e` (added back into every time step later), and the LSTM’s starting hidden state and cell state.
4. **One LSTM reads the past, then the future.** It is `nn.LSTM`, one layer, 32 units. The encoder runs over the 12 selected history vectors, starting from the static state. The decoder continues from the encoder’s final state and reads only the 12 selected known-future vectors. The forget-gate bias starts at 1, so a new network does not immediately wipe its memory. `manual_lstm` is the same cell written out gate by gate (input, forget, cell, output) so the JavaScript port can match PyTorch.
5. **Static enrichment, then attention.** A gated residual adds `c_e` to every encoder and decoder step. Interpretable multi-head attention then lets each forecast month look backward. Each head has its own query and key. All heads share one value projection, and the four attention maps are averaged into one. That average is what the portal plots. The mask is causal on the future: forecast month 1 can see the 12 history months plus itself, month 2 can see one step further, and so on. It cannot see later forecast months.
6. **Two gated skips, then two heads.** Attention is added back onto the decoder sequence through a GLU gate and layer norm. A position-wise gated residual runs, then a second gate. `now_head` reads the last enriched history month and classifies the tier at the origin. `head` reads each of the 12 fused decoder steps and classifies the tier for that future month. Class 0 is tier 1 through class 4, tier 5. Both biases start at the log of the training class frequencies, so the untrained model already guesses the common tiers.

The gated residual is the paper’s GRN: a linear layer, ELU, dropout, a second linear layer, then a gate that splits into `a * sigmoid(b)`, plus a skip connection and layer norm. Dropout is 0.1 in training and off at evaluation.

### Training

`src/housing_tft/train.py` fits that module with AdamW (learning rate `1e-3`, weight decay `1e-4`), a cosine schedule, and gradient clipping at 1. The loss is cross-entropy:

`0.5 * (current-month loss) + (average over the 12 future months)`

Future months are not equal. Months 3, 6, and 12 have weights 2, 3, and 2. Rare tiers get a larger class weight, the square root of inverse frequency. Early stopping watches macro-F1 at month 6 on the 60 validation whānau, and restores the best weights. The 200 test whānau are never used to fit or to stop.

The scorecard next to the TFT is not part of the network. It is a “tier stays the same” rule, plus two linear models on the last month (one of them also sees the 3- and 6-month changes). On this panel most whānau do not change tier in six months, so those baselines are the comparison the page reports.

### Browser port

After training, every tensor is written into `web/model/model.json` as base64 float32, along with the scaler, vocabularies, and feature names. `web/tft.js` repeats the embeddings, variable selection, LSTM, attention, gates, and both heads. `node tests/parity.mjs` checks that a browser forward pass matches PyTorch. The architecture is small and explicit enough to reimplement exactly, and the server that serves the portal does not carry Torch.
