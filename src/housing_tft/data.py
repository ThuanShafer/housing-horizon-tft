"""Panel splits, scaling, and the 12-month windows the TFT trains on.

The 300/200 family split uses seed 42. Sixty of the 300 are held back, with
seed 43, for early stopping. The 200 test whānau are not used to fit weights
or to stop training.

Known-future lease months are a countdown from the origin. They are not the
lease that later appears in the panel, because a renewal is an outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from housing_tft.calendar import year_month
from housing_tft.labels import tier_id
from housing_tft.spec import (
    ALIASES,
    EXPECTED_NAMES,
    FEATURES,
    KNOWN_CAT,
    KNOWN_NUM,
    OBS_CAT,
    OBS_NUM,
    STATIC_CAT,
    STATIC_NUM,
    VOCABS,
    feature_map,
)

ENCODER_LEN = 12
DECODER_LEN = 12
HEADLINE_ORIGIN = 48
INPUT_KEYS = (
    "static_cat",
    "static_num",
    "hist_cat",
    "hist_num",
    "known_cat_hist",
    "known_num_hist",
    "known_cat_fut",
    "known_num_fut",
)
NUMERIC_FEATURES = STATIC_NUM + OBS_NUM + KNOWN_NUM


@dataclass
class Split:
    fit: list[str]
    val: list[str]
    test: list[str]

    @property
    def train(self) -> list[str]:
        return self.fit + self.val


@dataclass
class Scaler:
    mean: dict[str, float]
    std: dict[str, float]

    def apply(self, name: str, value: float) -> float:
        return (float(value) - self.mean[name]) / self.std[name]

    def to_json(self) -> dict:
        names = [f.name for f in NUMERIC_FEATURES]
        return {
            "names": names,
            "mean": [self.mean[name] for name in names],
            "std": [self.std[name] for name in names],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Scaler":
        return cls(
            mean=dict(zip(payload["names"], payload["mean"])),
            std=dict(zip(payload["names"], payload["std"])),
        )


def assign_splits(ids: list[str], seed: int = 42, n_train: int = 300, n_val: int = 60) -> Split:
    rng = np.random.default_rng(seed)
    ordered = np.array(sorted(ids))
    rng.shuffle(ordered)
    if len(ordered) < n_train:
        raise ValueError(f"need at least {n_train} whānau, found {len(ordered)}")
    train = ordered[:n_train]
    test = ordered[n_train:]
    inner = np.random.default_rng(seed + 1)
    shuffled = train.copy()
    inner.shuffle(shuffled)
    val = shuffled[:n_val]
    fit = shuffled[n_val:]
    return Split(fit=fit.tolist(), val=val.tolist(), test=test.tolist())


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["whanau_id"]), []).append(row)
    for series in grouped.values():
        series.sort(key=lambda row: int(row["time_idx"]))
    return grouped


def fit_scaler(grouped: dict[str, list[dict]], ids: list[str]) -> Scaler:
    totals = {f.name: 0.0 for f in NUMERIC_FEATURES}
    squares = {f.name: 0.0 for f in NUMERIC_FEATURES}
    count = 0
    for wid in ids:
        for row in grouped[wid]:
            count += 1
            for feature in NUMERIC_FEATURES:
                value = float(row[feature.name])
                totals[feature.name] += value
                squares[feature.name] += value * value
    if count == 0:
        raise ValueError("no rows to fit the scaler")
    mean, std = {}, {}
    for feature in NUMERIC_FEATURES:
        mu = totals[feature.name] / count
        var = max(squares[feature.name] / count - mu * mu, 0.0)
        sigma = math.sqrt(var)
        mean[feature.name] = float(mu)
        std[feature.name] = float(sigma if sigma >= 1e-6 else 1.0)
    return Scaler(mean, std)


def canonical_category(name: str, value) -> str:
    if isinstance(value, (bool, np.bool_)):
        text = "yes" if bool(value) else "no"
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    alias = ALIASES.get(name, {})
    mapped = alias.get(text.lower(), text)
    vocab = VOCABS[name]
    if mapped in vocab:
        return mapped
    for option in vocab:
        if option.lower() == text.lower() or option.lower() == mapped.lower():
            return option
    return mapped


def category_index(name: str, value) -> int:
    key = canonical_category(name, value)
    vocab = VOCABS[name]
    if key not in vocab:
        return 0
    return vocab.index(key) + 1


def _as_float(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite number")
        return number
    text = str(value).strip().replace(",", "").replace("%", "")
    if text.lower() in {"yes", "true", "y"}:
        return 1.0
    if text.lower() in {"no", "false", "n", "none"}:
        return 0.0
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def _numeric_vector(row: dict, features, scaler: Scaler) -> np.ndarray:
    return np.asarray([scaler.apply(f.name, row[f.name]) for f in features], dtype=np.float32)


def _category_vector(row: dict, features) -> np.ndarray:
    return np.asarray([category_index(f.name, row[f.name]) for f in features], dtype=np.int64)


def valid_origins(times: list[int]) -> list[int]:
    have = {int(value) for value in times}
    return [tick for tick in sorted(have) if (tick - ENCODER_LEN + 1) in have and (tick + DECODER_LEN) in have]


def _future_known(history_end: dict, future_rows: list[dict]) -> list[dict]:
    """Known inputs for the forecast months.

    Calendar, holidays, indexation, and planned places come from the workbook
    month, which is the known future. A lease of 99 means no fixed term and
    stays 99. Any other lease counts down and does not renew.
    """
    last_lease = float(history_end["lease_months_remaining"])
    rows = []
    for step, source in enumerate(future_rows, start=1):
        lease = 99.0 if last_lease >= 99 else max(0.0, last_lease - step)
        rows.append({
            "month_of_year": source["month_of_year"],
            "quarter": source["quarter"],
            "school_holiday": source["school_holiday"],
            "public_holidays": source["public_holidays"],
            "benefit_indexation": source["benefit_indexation"],
            "planned_support_places": source["planned_support_places"],
            "lease_months_remaining": lease,
        })
    return rows


def build_windows(grouped: dict[str, list[dict]], ids: list[str], scaler: Scaler, origins: list[int] | None = None):
    chosen = valid_origins([int(row["time_idx"]) for row in grouped[ids[0]]]) if origins is None else list(origins)
    n_windows = len(ids) * len(chosen)
    arrays = {
        "static_num": np.zeros((n_windows, len(STATIC_NUM)), np.float32),
        "static_cat": np.zeros((n_windows, len(STATIC_CAT)), np.int64),
        "hist_num": np.zeros((n_windows, ENCODER_LEN, len(OBS_NUM)), np.float32),
        "hist_cat": np.zeros((n_windows, ENCODER_LEN, len(OBS_CAT)), np.int64),
        "known_num_hist": np.zeros((n_windows, ENCODER_LEN, len(KNOWN_NUM)), np.float32),
        "known_cat_hist": np.zeros((n_windows, ENCODER_LEN, len(KNOWN_CAT)), np.int64),
        "known_num_fut": np.zeros((n_windows, DECODER_LEN, len(KNOWN_NUM)), np.float32),
        "known_cat_fut": np.zeros((n_windows, DECODER_LEN, len(KNOWN_CAT)), np.int64),
        "y_now": np.zeros(n_windows, np.int64),
        "y_fut": np.zeros((n_windows, DECODER_LEN), np.int64),
    }
    whanau_id = []
    origin_out = []
    ethnicity = []
    cursor = 0
    for wid in ids:
        by_time = {int(row["time_idx"]): row for row in grouped[wid]}
        for origin in chosen:
            hist = [by_time[origin - ENCODER_LEN + 1 + step] for step in range(ENCODER_LEN)]
            future_raw = [by_time[origin + 1 + step] for step in range(DECODER_LEN)]
            known_future = _future_known(hist[-1], future_raw)
            arrays["static_num"][cursor] = _numeric_vector(hist[-1], STATIC_NUM, scaler)
            arrays["static_cat"][cursor] = _category_vector(hist[-1], STATIC_CAT)
            for step, row in enumerate(hist):
                arrays["hist_num"][cursor, step] = _numeric_vector(row, OBS_NUM, scaler)
                arrays["hist_cat"][cursor, step] = _category_vector(row, OBS_CAT)
                arrays["known_num_hist"][cursor, step] = _numeric_vector(row, KNOWN_NUM, scaler)
                arrays["known_cat_hist"][cursor, step] = _category_vector(row, KNOWN_CAT)
            for step, row in enumerate(known_future):
                arrays["known_num_fut"][cursor, step] = _numeric_vector(row, KNOWN_NUM, scaler)
                arrays["known_cat_fut"][cursor, step] = _category_vector(row, KNOWN_CAT)
            arrays["y_now"][cursor] = tier_id(hist[-1]) - 1
            arrays["y_fut"][cursor] = [tier_id(row) - 1 for row in future_raw]
            whanau_id.append(wid)
            origin_out.append(origin)
            ethnicity.append(str(hist[-1]["ethnicity"]))
            cursor += 1
    meta = {
        "whanau_id": np.asarray(whanau_id),
        "origin": np.asarray(origin_out, dtype=np.int64),
        "ethnicity": np.asarray(ethnicity),
    }
    return arrays, meta


def parse_payload(payload):
    future = None
    if isinstance(payload, list):
        history = payload
    elif isinstance(payload, dict) and isinstance(payload.get("history"), list):
        history = payload["history"]
        future = payload.get("future_known")
    elif isinstance(payload, dict):
        history = [payload]
    else:
        raise ValueError("JSON must be one month, a list of months, or an object with a history list")
    if not history:
        raise ValueError("history is empty")
    return history, future


def coerce_record(raw: dict, impute: dict[str, float] | None = None) -> tuple[dict, list[str]]:
    if not isinstance(raw, dict):
        raise ValueError("each month must be a JSON object")
    lowered = {}
    for key, value in raw.items():
        name = str(key).strip().lower().replace(" ", "_").replace("-", "_")
        name = name.replace("ā", "a").replace("ē", "e").replace("ī", "i").replace("ō", "o").replace("ū", "u")
        lowered[name] = value
    notes = []
    spec = feature_map()
    missing = [name for name in EXPECTED_NAMES if name not in lowered]
    if missing:
        shown = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" and {len(missing) - 8} more"
        raise ValueError(f"missing features: {shown}{extra}")
    row = {}
    for feature in FEATURES:
        value = lowered[feature.name]
        if value is None or value == "":
            if feature.kind == "numeric":
                fill = 0.0 if impute is None else float(impute.get(feature.name, 0.0))
                row[feature.name] = fill
                notes.append(f"{feature.name} was empty and set to the training average")
            else:
                row[feature.name] = ""
                notes.append(f"{feature.name} was empty and treated as unknown")
            continue
        if feature.kind == "numeric":
            try:
                row[feature.name] = _as_float(value)
            except ValueError as exc:
                raise ValueError(f"{feature.name} is not a number") from exc
        else:
            row[feature.name] = canonical_category(feature.name, value)
            if row[feature.name] not in VOCABS[feature.name]:
                notes.append(f"{feature.name} value {value!r} is outside the training vocabulary and was treated as unknown")
    for optional in ("whanau_id", "time_idx", "year", "month"):
        if optional in lowered and lowered[optional] not in (None, ""):
            row[optional] = lowered[optional]
    if "time_idx" in row:
        row["time_idx"] = int(float(row["time_idx"]))
    if "year" in row:
        row["year"] = int(float(row["year"]))
    if "month" in row:
        row["month"] = int(float(row["month"]))
    return row, notes


def _origin_from_row(row: dict) -> int | None:
    if "time_idx" in row:
        return int(row["time_idx"])
    if "year" in row and "month" in row:
        return (int(row["year"]) - 2021) * 12 + int(row["month"]) - 1
    return None


def load_panel(path) -> list[dict]:
    """Read the attached workbook. Row 3 is the code name; row 4 is the unit line."""
    import pandas as pd

    frame = pd.read_excel(path, sheet_name="Panel", header=2, skiprows=[3])
    if "row_type" in frame.columns:
        kind = frame["row_type"].astype(str).str.strip().str.lower()
        frame = frame.loc[kind != "forecast"].copy()
    rows = frame.to_dict(orient="records")
    blankable = {"emergency_accommodation_type", "main_benefit", "major_event_context"}
    for row in rows:
        for key, value in list(row.items()):
            if _missing_cell(value):
                row[key] = "None" if key in blankable else value
        for key in blankable:
            if _missing_cell(row.get(key)):
                row[key] = "None"
        row["whanau_id"] = str(row["whanau_id"])
        row["stability_tier"] = tier_id(row)
    return rows


def _missing_cell(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() == "nan"


def panel_calendar(rows: list[dict]) -> list:
    """One entry per time index. Index 0 is unused so time_idx can address the list."""
    by_time = {}
    for row in rows:
        by_time.setdefault(int(row["time_idx"]), row)
    calendar = [None]
    for tick in range(1, max(by_time) + 1):
        row = by_time[tick]
        calendar.append({
            "time_idx": tick,
            "month": str(row["month"]),
            "month_of_year": int(row["month_of_year"]),
            "quarter": int(row["quarter"]),
            "school_holiday": int(row["school_holiday"]),
            "public_holidays": int(row["public_holidays"]),
            "benefit_indexation": float(row["benefit_indexation"]),
        })
    return calendar


def project_case_future(last: dict, origin: int | None, calendar: list | None) -> list[dict]:
    lease0 = float(last["lease_months_remaining"])
    places = float(last["planned_support_places"])
    rows = []
    for step in range(1, DECODER_LEN + 1):
        lease = 99.0 if lease0 >= 99 else max(0.0, lease0 - step)
        macro = None
        if calendar is not None and origin is not None and 0 <= origin + step < len(calendar):
            macro = calendar[origin + step]
        if macro:
            month = int(macro["month_of_year"])
            quarter = int(macro["quarter"])
            school = int(macro["school_holiday"])
            holidays = int(macro["public_holidays"])
            benefit = float(macro["benefit_indexation"])
        else:
            month = ((int(last["month_of_year"]) - 1 + step) % 12) + 1
            quarter = (month - 1) // 3 + 1
            school = 1 if month in {1, 4, 7, 9, 10, 12} else 0
            holidays = {1: 3, 2: 1, 4: 3, 6: 1, 10: 1, 12: 2}.get(month, 0)
            benefit = 1.0 if month == 4 else 0.0
        rows.append({
            "month_of_year": month,
            "quarter": quarter,
            "school_holiday": school,
            "public_holidays": holidays,
            "benefit_indexation": benefit,
            "planned_support_places": places,
            "lease_months_remaining": lease,
        })
    return rows


def encode_case(history: list[dict], scaler: Scaler, future_known: list[dict] | None = None, calendar: list | None = None):
    """Encode one whānau history into a batch of size 1."""
    notes: list[str] = []
    coerced = []
    for raw in history:
        row, row_notes = coerce_record(raw, impute=scaler.mean)
        coerced.append(row)
        notes.extend(row_notes)
    if any("time_idx" in row for row in coerced):
        coerced.sort(key=lambda row: int(row.get("time_idx", 0)))
    provided = len(coerced)
    padded = provided < ENCODER_LEN
    if provided > ENCODER_LEN:
        coerced = coerced[-ENCODER_LEN:]
        notes.append(f"used the latest {ENCODER_LEN} months and left the earlier ones out")
    elif padded:
        coerced = [dict(coerced[0]) for _ in range(ENCODER_LEN - provided)] + coerced
        notes.append(
            f"only {provided} month{'s' if provided != 1 else ''} were uploaded, so the earliest month was repeated to fill the 12-month encoder. Treat the forecast with less confidence."
        )
    origin = _origin_from_row(coerced[-1])
    if future_known is not None:
        if len(future_known) != DECODER_LEN:
            raise ValueError(f"future_known must contain {DECODER_LEN} months")
        known_future = []
        for raw in future_known:
            row, row_notes = coerce_record(raw, impute=scaler.mean)
            known_future.append(row)
            notes.extend(row_notes)
    else:
        known_future = project_case_future(coerced[-1], origin, calendar)
        if origin is None:
            notes.append("no time index was supplied, so the future calendar is estimated from the last month")
        notes.append("a lease of 99 stays 99; any shorter lease counts down and is not renewed")
    arrays = {
        "static_num": _numeric_vector(coerced[-1], STATIC_NUM, scaler)[None, :],
        "static_cat": _category_vector(coerced[-1], STATIC_CAT)[None, :],
        "hist_num": np.stack([_numeric_vector(row, OBS_NUM, scaler) for row in coerced])[None, :, :],
        "hist_cat": np.stack([_category_vector(row, OBS_CAT) for row in coerced])[None, :, :],
        "known_num_hist": np.stack([_numeric_vector(row, KNOWN_NUM, scaler) for row in coerced])[None, :, :],
        "known_cat_hist": np.stack([_category_vector(row, KNOWN_CAT) for row in coerced])[None, :, :],
        "known_num_fut": np.stack([_numeric_vector(row, KNOWN_NUM, scaler) for row in known_future])[None, :, :],
        "known_cat_fut": np.stack([_category_vector(row, KNOWN_CAT) for row in known_future])[None, :, :],
    }
    info = {
        "notes": notes,
        "padded": padded,
        "months_provided": provided,
        "origin": origin,
        "origin_label": None if origin is None else _label(origin),
        "history_labels": [_row_label(row, index, provided, padded) for index, row in enumerate(coerced)],
        "future_labels": [_future_label(origin, step) for step in range(DECODER_LEN)],
        "last_row": coerced[-1],
    }
    return arrays, info


def _label(time_idx: int) -> str:
    year, month = year_month(time_idx)
    names = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    return f"{names[month - 1]} {year}"


def _row_label(row: dict, index: int, provided: int, padded: bool) -> str:
    origin = _origin_from_row(row)
    if origin is not None and not (padded and index < ENCODER_LEN - provided):
        return _label(origin)
    if "month" in row and "year" in row and not (padded and index < ENCODER_LEN - provided):
        return _label((int(row["year"]) - 2021) * 12 + int(row["month"]) - 1)
    if padded and index < ENCODER_LEN - provided:
        return "padded"
    return f"month {index + 1}"


def _future_label(origin: int | None, step: int) -> str:
    if origin is None:
        return f"+{step + 1} months"
    return _label(origin + step + 1)
