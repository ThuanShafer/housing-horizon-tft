"""Fit the housing TFT on 240 whānau, stop on 60, and test on 200.

Run from the project root:

    PYTHONPATH=src python -m housing_tft.train
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from housing_tft.data import (
    HEADLINE_ORIGIN,
    INPUT_KEYS,
    Scaler,
    assign_splits,
    build_windows,
    category_index,
    fit_scaler,
    group_rows,
    load_panel,
    panel_calendar,
)
from housing_tft.forecast import feature_catalog, predict_batch, softmax
from housing_tft.metrics import performance, scores
from housing_tft.model import TemporalFusionTransformer, count_parameters
from housing_tft.simulate import simulate, summarise, write_csv
from housing_tft.spec import KNOWN_CAT, KNOWN_NUM, OBS_CAT, OBS_NUM, STATIC_CAT, STATIC_NUM, TIERS, VOCABS, feature_map

ROOT = Path(__file__).resolve().parents[2]
STEP_WEIGHTS = torch.tensor([1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 2], dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(n: int, batch_size: int, shuffle: bool, seed: int):
    order = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def _loss(logits_now, logits_future, y_now, y_future, class_weight) -> torch.Tensor:
    now = nn.functional.cross_entropy(logits_now, y_now, weight=class_weight)
    flat = nn.functional.cross_entropy(
        logits_future.reshape(-1, 5), y_future.reshape(-1), weight=class_weight, reduction="none"
    ).view_as(y_future)
    future = (flat * STEP_WEIGHTS).mean()
    return 0.5 * now + future


def run_epoch(model, arrays, class_weight, batch_size, train: bool, seed: int, optimizer=None):
    model.train(train)
    total, seen = 0.0, 0
    for sel in _batches(len(arrays["y_now"]), batch_size, train, seed):
        batch = [torch.from_numpy(arrays[key][sel]) for key in INPUT_KEYS]
        y_now = torch.from_numpy(arrays["y_now"][sel])
        y_future = torch.from_numpy(arrays["y_fut"][sel])
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits_now, logits_future, *_ = model(*batch)
            loss = _loss(logits_now, logits_future, y_now, y_future, class_weight)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += float(loss.detach()) * len(sel)
        seen += len(sel)
    return total / max(seen, 1)


@torch.no_grad()
def horizon_f1(model, arrays, horizon_index: int) -> float:
    pred = predict_batch(model, arrays)
    guessed = pred["future"][:, horizon_index].argmax(axis=-1)
    return scores(arrays["y_fut"][:, horizon_index], guessed)["macro_f1"]


def fit_linear(x_fit, y_fit, x_val, y_val, class_weight, seed: int, epochs: int = 40):
    torch.manual_seed(seed)
    model = nn.Linear(x_fit.shape[1], 5)
    counts = np.bincount(y_fit, minlength=5).astype(np.float32)
    freq = counts / counts.sum()
    model.bias.data.copy_(torch.log(torch.tensor(freq).clamp(min=1e-4)))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    x_fit_t, y_fit_t = torch.from_numpy(x_fit), torch.from_numpy(y_fit)
    best, best_f1, wait = None, -1.0, 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_fit_t))
        for start in range(0, len(order), 256):
            sel = order[start:start + 256]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x_fit_t[sel]), y_fit_t[sel], weight=class_weight)
            loss.backward()
            opt.step()
        model.eval()
        guessed = model(torch.from_numpy(x_val)).argmax(-1).numpy()
        f1 = scores(y_val, guessed)["macro_f1"]
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 6:
                break
    model.load_state_dict(best)
    model.eval()
    return model


def design_matrix(arrays, deltas: bool) -> np.ndarray:
    parts = [arrays["static_num"], arrays["hist_num"][:, -1], arrays["known_num_hist"][:, -1]]
    if deltas:
        parts.append(arrays["hist_num"][:, -1] - arrays["hist_num"][:, -4])
        parts.append(arrays["hist_num"][:, -1] - arrays["hist_num"][:, -7])
    specs = (
        (arrays["static_cat"], STATIC_CAT),
        (arrays["hist_cat"][:, -1], OBS_CAT),
        (arrays["known_cat_hist"][:, -1], KNOWN_CAT),
    )
    for values, features in specs:
        for col, feature in enumerate(features):
            card = len(VOCABS[feature.name]) + 1
            eye = np.zeros((len(values), card), np.float32)
            eye[np.arange(len(values)), values[:, col]] = 1.0
            parts.append(eye)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _b64(value) -> dict:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    if array.dtype == np.bool_:
        payload, dtype = array.astype(np.uint8).tobytes(), "bool"
    elif np.issubdtype(array.dtype, np.integer):
        payload, dtype = array.astype(np.int32).tobytes(), "i32"
    else:
        payload, dtype = array.astype(np.float32).tobytes(), "f32"
    return {"dtype": dtype, "shape": list(array.shape), "b64": base64.b64encode(payload).decode("ascii")}


def _tier(cls: int) -> int:
    return int(cls) + 1


def export_bundle(path: Path, model, scaler: Scaler, config: dict, split, history, test_report, cohort, examples, calendar, seed: int):
    path.mkdir(parents=True, exist_ok=True)
    weights = {name: _b64(tensor) for name, tensor in model.state_dict().items()}
    model_doc = {
        "version": 1,
        "config": config,
        "scaler": scaler.to_json(),
        "vocabs": {name: list(values) for name, values in VOCABS.items()},
        "tiers": list(TIERS),
        "features": feature_catalog(),
        "weights": weights,
        "hist_names": model.hist_names,
        "future_names": model.future_names,
        "static_names": model.static_names,
        "national": calendar,
    }
    (path / "model.json").write_text(json.dumps(model_doc), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")
    (path / "cohort.json").write_text(json.dumps(cohort), encoding="utf-8")
    (path / "examples.json").write_text(json.dumps(examples), encoding="utf-8")
    (path / "split.json").write_text(json.dumps({
        "seed": seed,
        "inner_seed": seed + 1,
        "fit": split.fit,
        "val": split.val,
        "test": split.test,
        "epochs": history,
    }, indent=2), encoding="utf-8")


def train_seed(
    seed: int,
    out_dir: Path,
    epochs: int = 24,
    batch_size: int = 128,
    lr: float = 1e-3,
    d_model: int = 32,
    d_hidden: int = 48,
    heads: int = 4,
    patience: int = 5,
    save_checkpoint: bool = False,
) -> dict:
    """Fit and test one seed, and write the browser bundle to out_dir."""
    torch.set_num_threads(4)
    set_seed(seed)
    started = time.perf_counter()
    source = ROOT / "data" / "panel.xlsx"
    print(f"loading {source}")
    rows = load_panel(source)
    calendar = panel_calendar(rows)
    print(f"rows {len(rows)} whānau {len({row['whanau_id'] for row in rows})}")
    grouped = group_rows(rows)
    split = assign_splits(list(grouped), seed=seed)
    print(f"split seed {seed}: fit {len(split.fit)}  val {len(split.val)}  test {len(split.test)}", flush=True)
    scaler = fit_scaler(grouped, split.fit)
    fit_w, _ = build_windows(grouped, split.fit, scaler)
    val_w, _ = build_windows(grouped, split.val, scaler)
    test_all, _ = build_windows(grouped, split.test, scaler)
    test_one, test_meta = build_windows(grouped, split.test, scaler, origins=[HEADLINE_ORIGIN])
    print(f"windows: fit {len(fit_w['y_now'])}  val {len(val_w['y_now'])}  test {len(test_all['y_now'])}")

    y_all = np.concatenate([fit_w["y_now"], fit_w["y_fut"].reshape(-1)])
    counts = np.bincount(y_all, minlength=5).astype(np.float64)
    raw_weight = np.sqrt(counts.sum() / (5.0 * np.maximum(counts, 1.0)))
    class_weight = torch.tensor(raw_weight / raw_weight.mean(), dtype=torch.float32)
    config = {
        "d_model": d_model,
        "d_hidden": d_hidden,
        "n_heads": heads,
        "dropout": 0.1,
        "encoder_len": 12,
        "decoder_len": 12,
    }
    model = TemporalFusionTransformer(**config)
    prior = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32).clamp(min=1e-4))
    model.now_head.bias.data.copy_(prior)
    model.head.bias.data.copy_(prior)
    print(f"parameters {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_state, best_f1, wait = None, -1.0, 0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, fit_w, class_weight, batch_size, True, seed + epoch, optimizer)
        val_loss = run_epoch(model, val_w, class_weight, batch_size, False, seed)
        val_f1 = horizon_f1(model, val_w, 5)
        schedule.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1_h6": val_f1})
        print(f"epoch {epoch:02d}  loss {train_loss:.4f}  val {val_loss:.4f}  val F1@6 {val_f1:.3f}  {time.perf_counter() - t0:.1f}s", flush=True)
        if not np.isfinite(train_loss):
            raise RuntimeError("training loss is not finite")
        if val_f1 > best_f1 + 0.002:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience and epoch >= 8:
                print(f"early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()

    print("fitting linear baselines")
    y_fit, y_val = fit_w["y_fut"][:, 5], val_w["y_fut"][:, 5]
    base_last = fit_linear(design_matrix(fit_w, False), y_fit, design_matrix(val_w, False), y_val, class_weight, seed)
    base_delta = fit_linear(design_matrix(fit_w, True), y_fit, design_matrix(val_w, True), y_val, class_weight, seed)
    majority = int(np.bincount(y_fit, minlength=5).argmax())
    persistence_s = scores(test_one["y_fut"][:, 5], test_one["y_now"])

    def linear_scores(linear, deltas: bool):
        guessed = linear(torch.from_numpy(design_matrix(test_one, deltas))).argmax(-1).numpy()
        return scores(test_one["y_fut"][:, 5], guessed)

    pred = predict_batch(model, test_one)
    pred_all = predict_batch(model, test_all)
    report_horizons = {}
    for name, index in (("now", None), ("h3", 2), ("h6", 5), ("h12", 11)):
        if index is None:
            report_horizons[name] = scores(test_one["y_now"], pred["now"].argmax(-1))
        else:
            report_horizons[name] = scores(test_one["y_fut"][:, index], pred["future"][:, index].argmax(-1))
    all_windows = {
        name: scores(test_all["y_fut"][:, index], pred_all["future"][:, index].argmax(-1))["macro_f1"]
        for name, index in (("h3", 2), ("h6", 5), ("h12", 11))
    }
    ablated = {k: v.copy() for k, v in test_one.items()}
    obs_names = [f.name for f in OBS_NUM]
    ablated["hist_num"][:, :, obs_names.index("gambling_spend_share")] = scaler.apply("gambling_spend_share", 0.0)
    ablated["hist_num"][:, :, obs_names.index("justice_event")] = scaler.apply("justice_event", 0.0)
    ablated["static_num"] = ablated["static_num"].copy()
    ablated["static_num"][:, [f.name for f in STATIC_NUM].index("prior_justice_history")] = scaler.apply("prior_justice_history", 0.0)
    ablated_pred = predict_batch(model, ablated)
    ablation_scores = scores(test_one["y_fut"][:, 5], ablated_pred["future"][:, 5].argmax(-1))

    by_ethnicity = []
    actual_h6 = test_one["y_fut"][:, 5]
    guessed_h6 = pred["future"][:, 5].argmax(-1)
    for ethnicity in sorted(set(test_meta["ethnicity"])):
        mask = test_meta["ethnicity"] == ethnicity
        block = scores(actual_h6[mask], guessed_h6[mask])
        by_ethnicity.append({
            "ethnicity": ethnicity,
            "n": int(mask.sum()),
            "macro_f1": block["macro_f1"],
            "accuracy": block["accuracy"],
            "recall": block["recall"],
            "support": block["support"],
        })

    spec = feature_map()
    importance = {
        "static": _importance(model.static_names, pred["w_static"].mean(axis=0), spec),
        "history": _importance(model.hist_names, pred["w_hist"].mean(axis=(0, 1)), spec),
        "future": _importance(model.future_names, pred["w_future"].mean(axis=(0, 1)), spec),
    }
    attention = pred["attention"][:, 5, :].mean(axis=0)
    cohort = []
    for i, wid in enumerate(test_meta["whanau_id"]):
        series = grouped[wid]
        board = next(row["local_board"] for row in series if int(row["time_idx"]) == HEADLINE_ORIGIN)
        actual = {
            "now": _tier(int(test_one["y_now"][i])),
            "h3": _tier(int(test_one["y_fut"][i, 2])),
            "h6": _tier(int(test_one["y_fut"][i, 5])),
            "h12": _tier(int(test_one["y_fut"][i, 11])),
        }
        predicted = {
            "now": _tier(int(pred["now"][i].argmax())),
            "h3": _tier(int(pred["future"][i, 2].argmax())),
            "h6": _tier(int(pred["future"][i, 5].argmax())),
            "h12": _tier(int(pred["future"][i, 11].argmax())),
        }
        cohort.append({
            "id": wid,
            "ethnicity": str(test_meta["ethnicity"][i]),
            "local_board": board,
            "actual": actual,
            "predicted": predicted,
            "probability_h6": [float(v) for v in softmax(pred["future"][i, 5])],
        })

    examples = _examples(grouped, test_meta, test_one, pred, cohort)
    parity = {
        "inputs": {key: _b64(test_one[key][:1]) for key in INPUT_KEYS},
        "now": pred["now"][:1].astype(float).tolist(),
        "future": pred["future"][:1].astype(float).tolist(),
        "attention": pred["attention"][:1].astype(float).tolist(),
    }
    h6 = report_horizons["h6"]
    base_last_s = linear_scores(base_last, False)
    base_delta_s = linear_scores(base_delta, True)
    majority_s = scores(actual_h6, np.full(len(actual_h6), majority))
    metrics = {
        "task": "Five-tier housing stability, Temporal Fusion Transformer",
        "panel": "Attached workbook tft_synthetic_panel_for_500_whanau.xlsx, 500 whānau, January 2019–December 2023.",
        "seed": seed,
        "inner_seed": seed + 1,
        "families": {"fit": len(split.fit), "validation": len(split.val), "train_pool": 300, "test": len(split.test)},
        "headline_origin": {"time_idx": HEADLINE_ORIGIN, "label": calendar[HEADLINE_ORIGIN]["month"], "history": f"{calendar[HEADLINE_ORIGIN - 11]['month']}–{calendar[HEADLINE_ORIGIN]['month']}", "forecast_through": calendar[HEADLINE_ORIGIN + 12]["month"]},
        "parameters": count_parameters(model),
        "config": config,
        "class_counts_fit_windows": [int(v) for v in counts],
        "best_val_macro_f1_h6": best_f1,
        "epochs_ran": history[-1]["epoch"],
        "history": history,
        "test_origin": report_horizons,
        "test_all_windows_macro_f1": all_windows,
        "baselines_h6": {
            "majority": majority_s,
            "persistence": persistence_s,
            "last_month_linear": base_last_s,
            "last_month_plus_deltas_linear": base_delta_s,
        },
        "ablation_h6": {
            "note": "Gambling share set to 0, justice event set to 0, and prior justice history set to no at evaluation. The model was not retrained.",
            "full": h6,
            "without_gambling_and_justice": ablation_scores,
        },
        "by_ethnicity_h6": by_ethnicity,
        "importance": importance,
        "attention_h6": [float(v) for v in attention],
        "attention_labels": [calendar[t]["month"] for t in range(HEADLINE_ORIGIN - 11, HEADLINE_ORIGIN + 13)],
        "performance_h6": performance(test_one["y_fut"][:, 5], pred["future"][:, 5].argmax(-1), softmax(pred["future"][:, 5])),
        "seconds": round(time.perf_counter() - started, 1),
    }
    if save_checkpoint:
        artifacts = ROOT / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config, "state_dict": best_state, "scaler": scaler.to_json(), "seed": seed, "calendar": calendar}, artifacts / "tft.pt")
        (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export_bundle(out_dir, model, scaler, config, split, history, metrics, cohort, examples, calendar, seed)
    (out_dir / "parity.json").write_text(json.dumps(parity), encoding="utf-8")
    print(
        f"seed {seed}  test macro-F1 {h6['macro_f1']:.3f}  accuracy {h6['accuracy']:.3f}  "
        f"linear {base_last_s['macro_f1']:.3f}  persistence {persistence_s['macro_f1']:.3f}  majority {majority_s['macro_f1']:.3f}",
        flush=True,
    )
    print(f"wrote {out_dir}", flush=True)
    return {"seed": seed, "seconds": metrics["seconds"], "macro_f1": h6["macro_f1"], "accuracy": h6["accuracy"]}


def main():
    parser = argparse.ArgumentParser(description="Train the housing-stability TFT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-hidden", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "model")
    args = parser.parse_args()
    train_seed(
        args.seed,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        heads=args.heads,
        patience=args.patience,
        save_checkpoint=args.out == ROOT / "web" / "model",
    )


def _importance(names, weights, spec) -> list[dict]:
    order = np.argsort(-weights)
    rows = []
    for index in order:
        feature = spec[names[int(index)]]
        rows.append({
            "name": feature.name,
            "domain": feature.domain,
            "role": feature.role,
            "sensitive": feature.sensitive,
            "weight": float(weights[int(index)]),
        })
    return rows


def _jsonable(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


EXAMPLE_COUNT = 20


def _examples(grouped, meta, arrays, pred, cohort) -> list[dict]:
    by_id = {row["id"]: row for row in cohort}
    queues = {tier: [] for tier in range(1, 6)}
    for row in cohort:
        correct = row["predicted"]["h6"] == row["actual"]["h6"]
        confidence = row["probability_h6"][row["predicted"]["h6"] - 1]
        queues[row["actual"]["h6"]].append((correct, confidence, row["id"]))
    for queue in queues.values():
        queue.sort(reverse=True)
    chosen = []
    while len(chosen) < EXAMPLE_COUNT:
        added = False
        for tier in range(1, 6):
            if not queues[tier]:
                continue
            chosen.append(queues[tier].pop(0)[2])
            added = True
            if len(chosen) == EXAMPLE_COUNT:
                break
        if not added:
            break
    # Keep one wrong call in the row when the test set has one.
    has_miss = any(by_id[wid]["predicted"]["h6"] != by_id[wid]["actual"]["h6"] for wid in chosen)
    if chosen and not has_miss:
        misses = [row for row in cohort if row["predicted"]["h6"] != row["actual"]["h6"]]
        if misses:
            misses.sort(key=lambda row: row["probability_h6"][row["predicted"]["h6"] - 1], reverse=True)
            chosen[-1] = misses[0]["id"]
    examples = []
    id_to_index = {wid: i for i, wid in enumerate(meta["whanau_id"])}
    for wid in chosen:
        i = id_to_index[wid]
        history = []
        for row in grouped[wid]:
            if HEADLINE_ORIGIN - 11 <= int(row["time_idx"]) <= HEADLINE_ORIGIN:
                history.append({key: _jsonable(row[key]) for key in ("whanau_id", "time_idx", "month", *feature_map()) if key in row})
        actual_path = [_tier(int(v)) for v in arrays["y_fut"][i]]
        examples.append({
            "id": wid,
            "history": history,
            "actual": by_id[wid]["actual"],
            "actual_path": actual_path,
            "torch_h6": [float(v) for v in softmax(pred["future"][i, 5])],
            "torch_now": [float(v) for v in softmax(pred["now"][i])],
        })
    return examples


def refresh_examples(bundle: Path) -> int:
    """Rewrite examples.json from a fitted bundle without training again."""
    doc = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
    model = TemporalFusionTransformer(**doc["config"])
    state = {}
    for name, spec in doc["weights"].items():
        dtype = {"f32": np.float32, "i32": np.int32, "bool": np.uint8}[spec["dtype"]]
        array = np.frombuffer(base64.b64decode(spec["b64"]), dtype=dtype).reshape(spec["shape"]).copy()
        if spec["dtype"] == "bool":
            array = array.astype(np.bool_)
        state[name] = torch.from_numpy(array)
    model.load_state_dict(state)
    model.eval()
    grouped = group_rows(load_panel(ROOT / "data" / "panel.xlsx"))
    split = json.loads((bundle / "split.json").read_text(encoding="utf-8"))
    cohort = json.loads((bundle / "cohort.json").read_text(encoding="utf-8"))
    scaler = Scaler.from_json(doc["scaler"])
    arrays, meta = build_windows(grouped, split["test"], scaler, origins=[HEADLINE_ORIGIN])
    pred = predict_batch(model, arrays)
    examples = _examples(grouped, meta, arrays, pred, cohort)
    (bundle / "examples.json").write_text(json.dumps(examples), encoding="utf-8")
    return len(examples)


if __name__ == "__main__":
    main()
