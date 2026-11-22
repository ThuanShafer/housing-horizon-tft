"""Synthetic 500 × 60 Auckland panel.

The tier is a latent housing-arrangement and stress state. The 50 features are
noisy views of that state, with income and arrears moving before the
arrangement changes. Ethnicity is drawn with the board so the portal can slice
results. It is not a coefficient in the tier equation.

Nothing here is a Census count, a market rent, or an official macro series.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from housing_tft.calendar import N_MONTHS, national_macro
from housing_tft.spec import EXPECTED_NAMES, TIER_NAME

# name, population weight, nzdep, weekly rent index, ethnicity shares
# (Māori, Pacific, European, Asian, MELAA, Other). Shares are normalised.
_BOARD_ROWS = (
    ("Rodney", 7.0, 4.2, 610, (0.11, 0.03, 0.74, 0.07, 0.01, 0.04)),
    ("Hibiscus and Bays", 11.0, 3.0, 740, (0.07, 0.02, 0.72, 0.15, 0.01, 0.03)),
    ("Upper Harbour", 6.5, 3.6, 700, (0.06, 0.03, 0.48, 0.38, 0.02, 0.03)),
    ("Kaipātiki", 8.0, 4.4, 660, (0.08, 0.05, 0.55, 0.27, 0.02, 0.03)),
    ("Devonport-Takapuna", 6.0, 2.6, 820, (0.06, 0.02, 0.62, 0.25, 0.02, 0.03)),
    ("Henderson-Massey", 12.0, 7.4, 560, (0.18, 0.22, 0.32, 0.22, 0.02, 0.04)),
    ("Waitākere Ranges", 5.0, 4.8, 600, (0.14, 0.08, 0.62, 0.10, 0.02, 0.04)),
    ("Waitematā", 8.5, 4.0, 760, (0.08, 0.06, 0.52, 0.28, 0.03, 0.03)),
    ("Whau", 7.5, 6.2, 590, (0.10, 0.16, 0.34, 0.34, 0.03, 0.03)),
    ("Albert-Eden", 9.0, 3.8, 720, (0.08, 0.08, 0.52, 0.26, 0.03, 0.03)),
    ("Puketāpapa", 5.5, 5.4, 640, (0.08, 0.14, 0.36, 0.34, 0.04, 0.04)),
    ("Ōrākei", 7.0, 2.4, 860, (0.07, 0.03, 0.64, 0.20, 0.02, 0.04)),
    ("Maungakiekie-Tāmaki", 7.5, 6.8, 580, (0.16, 0.24, 0.32, 0.22, 0.03, 0.03)),
    ("Howick", 14.0, 3.4, 680, (0.06, 0.06, 0.42, 0.40, 0.03, 0.03)),
    ("Māngere-Ōtāhuhu", 8.0, 8.6, 520, (0.16, 0.52, 0.12, 0.14, 0.02, 0.04)),
    ("Ōtara-Papatoetoe", 9.0, 8.8, 510, (0.18, 0.50, 0.12, 0.14, 0.02, 0.04)),
    ("Manurewa", 10.0, 8.4, 530, (0.24, 0.36, 0.18, 0.16, 0.02, 0.04)),
    ("Papakura", 6.5, 7.8, 540, (0.28, 0.22, 0.32, 0.12, 0.02, 0.04)),
    ("Franklin", 8.0, 4.6, 580, (0.16, 0.08, 0.62, 0.08, 0.02, 0.04)),
    ("Aotea/Great Barrier", 0.4, 6.0, 480, (0.22, 0.04, 0.68, 0.02, 0.01, 0.03)),
    ("Waiheke", 1.1, 4.0, 650, (0.12, 0.04, 0.74, 0.04, 0.02, 0.04)),
)
_ETHNICITIES = ("Māori", "Pacific", "European", "Asian", "MELAA", "Other")
_AGE_WEIGHTS = {
    "0-14": 0.01,
    "15-24": 0.08,
    "25-34": 0.24,
    "35-44": 0.26,
    "45-54": 0.20,
    "55-64": 0.13,
    "65+": 0.08,
}
_AGE_EARN = {
    "0-14": 280.0,
    "15-24": 740.0,
    "25-34": 1080.0,
    "35-44": 1200.0,
    "45-54": 1220.0,
    "55-64": 1100.0,
    "65+": 640.0,
}


def tier_from_state(arrangement: str, stress: float) -> int:
    """Map latent arrangement and stress to tier id 1..5.

    The living situation decides the two crisis tiers. Stress splits a market
    household between stable and at risk, and only a very high stress level
    moves a subsidised household out of assisted stability. Ethnicity, justice
    history, and gambling are not inputs to this label.
    """
    if arrangement == "emergency":
        return 1
    if arrangement == "transitional":
        return 2
    if arrangement == "social":
        return 3 if stress >= 0.92 else 4
    if stress >= 0.55:
        return 3
    return 5


def _boards() -> list[dict]:
    rows = []
    for name, pop, nzdep, rent, shares in _BOARD_ROWS:
        shares = np.asarray(shares, dtype=np.float64)
        shares = shares / shares.sum()
        jitter = ((len(name) % 5) - 2) * 0.35
        rows.append({
            "name": name,
            "pop": float(pop),
            "nzdep": float(np.clip(round(nzdep + ((len(name) % 3) - 1) * 0.15), 1, 10)),
            "rent": float(rent),
            "shares": shares,
            "emp": float(np.clip(1.16 - nzdep / 18.0, 0.78, 1.18)),
            "homeless": float(7.0 + nzdep * 2.7 + jitter),
            "eh": float(80.0 + nzdep * 46.0 + pop * 3.0),
            "spend": float(180.0 + pop * 14.0 + nzdep * 22.0),
            "pti": float(6.2 + rent / 160.0 + nzdep * 0.08),
            "afford": float(rent / (rent + 980.0)),
        })
    return rows


def _draw_arrangement(rng: np.random.Generator, earn: float, nzdep: float, disability: bool) -> str:
    own = np.clip((earn - 850.0) / 1800.0, 0.08, 0.62)
    social = np.clip((nzdep - 3.2) / 8.0, 0.04, 0.40) * np.clip((1400.0 - earn) / 800.0, 0.15, 1.0)
    if disability:
        social = min(0.55, social + 0.08)
        own *= 0.85
    trans = 0.018
    emerg = 0.008
    rent = max(0.05, 1.0 - (own + social + trans + emerg))
    weights = np.array([own, rent, social, trans, emerg], dtype=np.float64)
    weights /= weights.sum()
    return str(rng.choice(["market_own", "market_rent", "social", "transitional", "emergency"], p=weights))


def _benefit_kind(whanau_type: str, disability: bool, age: str) -> str:
    if disability and age in {"45-54", "55-64", "65+"}:
        return "supported_living"
    if whanau_type == "sole_parent":
        return "sole_parent_support"
    if age == "65+":
        return "other"
    return "jobseeker"


def simulate(n: int = 500, months: int = N_MONTHS, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    boards = _boards()
    pop = np.array([b["pop"] for b in boards], dtype=np.float64)
    pop /= pop.sum()
    ages = list(_AGE_WEIGHTS)
    age_p = np.array([_AGE_WEIGHTS[a] for a in ages], dtype=np.float64)
    age_p /= age_p.sum()
    types = ["single_adult", "couple", "sole_parent", "two_parent", "multi_family", "extended"]
    type_p = np.array([0.18, 0.22, 0.16, 0.28, 0.10, 0.06])
    residencies = ["citizen", "permanent_resident", "resident_visa", "temporary", "other"]
    residency_p = np.array([0.62, 0.16, 0.12, 0.07, 0.03])
    macros = [national_macro(t) for t in range(months)]
    place_scale = max(m["planned_support_places"] for m in macros)
    rows: list[dict] = []

    for i in range(n):
        board = boards[int(rng.choice(len(boards), p=pop))]
        age = str(rng.choice(ages, p=age_p))
        whanau_type = str(rng.choice(types, p=type_p))
        disability = bool(rng.random() < 0.17)
        ethnicity = _ETHNICITIES[int(rng.choice(6, p=board["shares"]))]
        residency = str(rng.choice(residencies, p=residency_p))
        prior_justice = bool(rng.random() < (0.11 if age in {"15-24", "25-34"} else 0.06))
        if whanau_type in {"sole_parent", "two_parent"}:
            dependants = int(rng.integers(1, 5))
        elif whanau_type in {"multi_family", "extended"}:
            dependants = int(rng.integers(1, 6))
        else:
            dependants = int(rng.random() < 0.08)
        adults = {"single_adult": 1, "couple": 2, "sole_parent": 1, "two_parent": 2, "multi_family": 3, "extended": 3}[whanau_type]
        household = adults + dependants
        earn = _AGE_EARN[age] * (0.72 if disability else 1.0) * board["emp"] * float(rng.lognormal(0.0, 0.20))
        earn = float(np.clip(earn, 260.0, 3200.0))
        arrangement = _draw_arrangement(rng, earn, board["nzdep"], disability)
        rent_factor = float(rng.uniform(0.62, 0.98))
        stress = float(np.clip(board["rent"] * rent_factor / earn * 0.8 + rng.normal(0.0, 0.05), 0.02, 1.15))
        hidden = float(rng.normal(0.0, 0.05))
        arrears = float(max(0.0, rng.normal(200.0, 150.0) if stress > 0.55 else rng.normal(20.0, 30.0)))
        debt = float(max(0.0, earn * rng.uniform(2.0, 18.0) * (1.4 if stress > 0.6 else 0.7)))
        gamble_prop = float(np.clip(rng.lognormal(-4.4, 0.55), 0.0, 0.18))
        substance_prop = float(rng.random() < 0.08)
        mortgage = float(np.clip(board["rent"] * rent_factor * rng.uniform(0.9, 1.2), 320.0, 1400.0)) if arrangement == "market_own" else 0.0
        if arrangement == "market_rent":
            lease = int(rng.integers(0, 13))
            tenure = int(rng.integers(0, 36))
        elif arrangement == "market_own":
            lease, tenure = 0, int(rng.integers(6, 120))
        elif arrangement == "social":
            lease, tenure = int(rng.integers(24, 72)), int(rng.integers(2, 80))
        elif arrangement == "transitional":
            lease, tenure = int(rng.integers(1, 7)), int(rng.integers(0, 8))
        else:
            lease, tenure = int(rng.integers(0, 2)), int(rng.integers(0, 4))
        bedrooms = max(1, int(round(household / 1.35)))
        move_times: list[int] = []
        high_run = 0
        shock_left = 0
        shock_size = 0.0
        benefit_kind = _benefit_kind(whanau_type, disability, age)
        opening_size = int(household)
        prev_size = household

        for t in range(months):
            macro = macros[t]
            if shock_left == 0 and rng.random() < 0.08:
                shock_left = int(rng.integers(5, 10))
                shock_size = float(rng.uniform(0.35, 0.70))
            employed = 1.0
            if shock_left > 0:
                employed -= shock_size
                shock_left -= 1
            employed = float(np.clip(employed + rng.normal(0.0, 0.03), 0.0, 1.0))
            if age == "65+":
                employed = min(employed, 0.25)
            wage_income = earn * employed
            benefit_amount = (420.0 + 75.0 * dependants) * (1.0 + macro["benefit_indexation"])
            if employed < 0.75:
                benefit_income = benefit_amount * (1.0 - employed)
            else:
                benefit_income = 0.0
            weekly_income = float(np.clip(wage_income + benefit_income, 180.0, 4500.0))
            hours = 38.0 if employed > 0.85 else (24.0 if employed > 0.4 else 0.0)
            hourly = float(wage_income / hours) if hours else 0.0

            if arrangement == "market_own":
                cost = mortgage * float(rng.uniform(0.98, 1.02))
            elif arrangement == "market_rent":
                cost = board["rent"] * rent_factor
            elif arrangement == "social":
                cost = min(board["rent"] * 0.50, weekly_income * 0.25)
            elif arrangement == "transitional":
                cost = board["rent"] * 0.42
            else:
                cost = board["rent"] * 0.12
            burden = float(np.clip(cost / weekly_income, 0.02, 1.5))

            hidden = float(np.clip(0.90 * hidden + rng.normal(0.0, 0.045), -0.4, 0.4))
            disaster_hit = macro["natural_disaster_severity"] * (0.28 if board["nzdep"] >= 7 else 0.10)
            stress = float(np.clip(
                0.70 * stress
                + 0.30 * min(burden, 1.2)
                + 0.45 * max(0.0, 0.55 - employed) * max(shock_size, 0.25)
                + 0.10 * hidden
                + disaster_hit
                + macro["pandemic_restriction_level"] * 0.012
                - 0.01,
                0.01,
                1.45,
            ))
            if burden > 0.32:
                arrears = arrears * 0.94 + (burden - 0.32) * weekly_income * 0.55
            else:
                arrears = arrears * 0.82
            arrears = float(np.clip(arrears + rng.normal(0.0, 25.0), 0.0, 25000.0))
            gamble = float(np.clip(gamble_prop * (0.35 + stress) + rng.normal(0.0, 0.004), 0.0, 0.30))
            debt = float(np.clip(debt * 0.995 + gamble * weekly_income * 8.0 + max(0.0, burden - 0.4) * 40.0 - 25.0, 0.0, 90000.0))

            high_run = high_run + 1 if stress > 0.72 else 0
            support = macro["planned_support_places"] / place_scale
            previous = arrangement
            arrangement = _step_arrangement(rng, arrangement, stress, high_run, support)
            if arrangement != previous:
                move_times.append(t)
                tenure = 0
                high_run = 0
                if arrangement == "market_rent":
                    lease = 12
                    bedrooms = max(1, int(round(household / 1.4)))
                    mortgage = 0.0
                elif arrangement == "market_own":
                    lease = 0
                    mortgage = float(np.clip(board["rent"] * rent_factor, 320.0, 1400.0))
                    bedrooms = max(1, int(round(household / 1.3)))
                elif arrangement == "social":
                    lease = 48
                    mortgage = 0.0
                    bedrooms = max(1, int(round(household / 1.45)))
                elif arrangement == "transitional":
                    lease = int(rng.integers(2, 6))
                    mortgage = 0.0
                    bedrooms = max(1, bedrooms - 1)
                else:
                    lease = 0
                    mortgage = 0.0
                    bedrooms = 1
            else:
                tenure += 1
                if arrangement == "market_rent":
                    lease = max(0, lease - 1)
                    if lease == 0 and rng.random() < (0.82 if stress < 0.55 else 0.22):
                        lease = 12
                elif arrangement == "social":
                    lease = max(0, lease - 1)
                    if lease == 0:
                        lease = 60
                elif arrangement == "transitional":
                    lease = max(0, lease - 1)
                    if lease == 0:
                        lease = int(rng.integers(1, 5))
                else:
                    lease = 0

            if rng.random() < 0.02:
                delta = int(rng.choice([-1, 1]))
                household = max(1, household + delta)
                dependants = int(np.clip(dependants + (delta if dependants or delta > 0 else 0), 0, 8))
            size_change = household - prev_size
            prev_size = household

            moves_12m = sum(1 for mt in move_times if t - 11 <= mt <= t)
            displaced = int(macro["natural_disaster_severity"] > 0.4 and rng.random() < macro["natural_disaster_severity"] * (0.35 if board["nzdep"] >= 7 else 0.12))
            if displaced:
                stress = float(np.clip(stress + 0.08, 0.0, 1.45))

            tier = tier_from_state(arrangement, stress)
            ppb = household / max(1, bedrooms)
            if arrangement == "emergency":
                ppb = max(ppb, 2.2)
            deprivation, accommodation, emergency_stay = _dwelling_record(rng, arrangement, ppb, stress)
            if weekly_income > 0 and benefit_income > wage_income and employed < 0.7:
                benefit = benefit_kind
            elif employed < 0.2:
                benefit = benefit_kind
            else:
                benefit = "none"
            essentials = float(np.clip((160.0 + 65.0 * household) / weekly_income + rng.normal(0.0, 0.02), 0.15, 1.25))
            justice = int(rng.random() < (0.018 if prior_justice else 0.003))
            if justice:
                hidden = float(np.clip(hidden + 0.04, -0.4, 0.4))
            substance = int(rng.random() < (0.012 + (0.05 if substance_prop else 0.0) + 0.02 * min(stress, 1.0)))

            row = {
                "whanau_id": f"W{i + 1:04d}",
                "time_idx": t,
                "year": macro["year"],
                "month": macro["month"],
                "stability_tier": tier,
                "stability_tier_name": TIER_NAME[tier],
                "_arrangement": arrangement,
                "_stress": round(stress, 4),
                "tenure_start": int(tenure),
                "lease_months_remaining": int(lease),
                "rent_arrears": round(arrears, 2),
                "persons_per_bedroom": round(float(ppb), 3),
                "moves_12m": int(moves_12m),
                "emergency_housing_stay": int(emergency_stay),
                "deprivation_category": deprivation,
                "mortgage_payment": round(float(mortgage if arrangement == "market_own" else 0.0), 2),
                "housing_cost_burden": round(burden, 4),
                "disaster_displaced": displaced,
                "month_of_year": macro["month_of_year"],
                "quarter": macro["quarter"],
                "school_holiday": macro["school_holiday"],
                "public_holidays": macro["public_holidays"],
                "age_band": age,
                "residency_status": residency,
                "ethnicity": ethnicity,
                "local_board": board["name"],
                "nzdep_decile": int(board["nzdep"]),
                "local_median_rent": round(board["rent"], 2),
                "local_affordability_index": round(board["afford"], 4),
                "local_price_to_income": round(board["pti"], 3),
                "local_homelessness_rate": round(board["homeless"], 3),
                "local_eh_grants": round(board["eh"], 2),
                "local_service_spend": round(board["spend"], 2),
                "whanau_type": whanau_type,
                "sole_parent": 1 if whanau_type == "sole_parent" else 0,
                "dependants_count": int(dependants),
                "household_size_start": opening_size,
                "household_size_change": int(size_change),
                "weekly_income": round(weekly_income, 2),
                "main_benefit": benefit,
                "hourly_wage": round(hourly, 2),
                "essential_spend_share": round(essentials, 4),
                "debt_balance": round(debt, 2),
                "benefit_indexation": macro["benefit_indexation"],
                "planned_support_places": round(macro["planned_support_places"], 2),
                "emergency_accommodation_type": accommodation,
                "disability": "yes" if disability else "no",
                "substance_service_contact": substance,
                "prior_justice_history": "yes" if prior_justice else "no",
                "justice_event": justice,
                "gambling_spend_share": round(gamble, 4),
                "ocr_rate": macro["ocr_rate"],
                "cpi_inflation_annual": macro["cpi_inflation_annual"],
                "unemployment_rate": macro["unemployment_rate"],
                "petrol_price": macro["petrol_price"],
                "pandemic_restriction_level": macro["pandemic_restriction_level"],
                "natural_disaster_severity": macro["natural_disaster_severity"],
                "major_event_context": macro["major_event_context"],
            }
            rows.append(row)

    _validate(rows, n, months)
    return rows


def _step_arrangement(rng: np.random.Generator, arrangement: str, stress: float, high_run: int, support: float) -> str:
    roll = float(rng.random())
    if arrangement == "emergency":
        if stress < 0.70 and roll < 0.07 + 0.04 * support:
            return "transitional"
        if roll < 0.025:
            return "transitional"
        return arrangement
    if arrangement == "transitional":
        if stress > 0.68 and roll < 0.18:
            return "emergency"
        if stress < 0.58 and roll < 0.08 + 0.06 * support:
            return "social"
        if stress < 0.48 and roll < 0.06:
            return "market_rent"
        return arrangement
    if arrangement == "social":
        if stress > 1.05 and high_run >= 5 and roll < 0.05:
            return "transitional"
        return arrangement
    # Market rent or ownership. The tenure gives way only after stress has stayed high.
    if high_run >= 4 and stress > 0.76 and roll < 0.14:
        return "transitional"
    if high_run >= 7 and stress > 0.98 and roll < 0.06:
        return "emergency"
    if arrangement == "market_own" and stress > 0.85 and high_run >= 5 and roll < 0.06:
        return "market_rent"
    return arrangement


def _dwelling_record(rng: np.random.Generator, arrangement: str, ppb: float, stress: float) -> tuple[str, str, int]:
    if arrangement == "emergency":
        hidden = rng.random() < 0.14
        accommodation = "none" if hidden else str(rng.choice(["motel", "shelter", "refuge", "rough_or_vehicle"], p=[0.46, 0.24, 0.12, 0.18]))
        deprivation = "none" if hidden and rng.random() < 0.4 else "emergency"
        if ppb >= 2.4 and deprivation == "none":
            deprivation = "crowded"
        return deprivation, accommodation, 0 if hidden else 1
    if arrangement == "transitional":
        accommodation = "none" if rng.random() < 0.08 else str(rng.choice(["transitional", "boarding"], p=[0.7, 0.3]))
        deprivation = "crowded" if ppb >= 2.0 else "no_secure_tenure"
        return deprivation, accommodation, 0
    if ppb >= 2.15:
        deprivation = "crowded"
    elif stress > 0.8 and rng.random() < 0.35:
        deprivation = "uninhabitable"
    else:
        deprivation = "none"
    accommodation = "none" if rng.random() > 0.015 else "boarding"
    return deprivation, accommodation, 0


def _validate(rows: list[dict], n: int, months: int) -> None:
    if len(rows) != n * months:
        raise RuntimeError(f"expected {n * months} rows, got {len(rows)}")
    tiers = {row["stability_tier"] for row in rows}
    if not tiers <= {1, 2, 3, 4, 5}:
        raise RuntimeError(f"unexpected tiers {tiers}")
    missing = [name for name in EXPECTED_NAMES if name not in rows[0]]
    if missing:
        raise RuntimeError(f"missing features {missing}")
    for name in EXPECTED_NAMES:
        value = rows[0][name]
        if isinstance(value, float) and not np.isfinite(value):
            raise RuntimeError(f"non-finite {name}")


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["whanau_id", "time_idx", "year", "month", "stability_tier", "stability_tier_name", "_arrangement", "_stress", *EXPECTED_NAMES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict]) -> str:
    tiers = np.array([r["stability_tier"] for r in rows])
    lines = ["month-level tier share"]
    for tier in range(1, 6):
        lines.append(f"  {tier}: {(tiers == tier).mean():.1%}")
    origin = [r for r in rows if r["time_idx"] == 47]
    lines.append(f"at Dec 2024 (n={len(origin)})")
    ot = np.array([r["stability_tier"] for r in origin])
    for tier in range(1, 6):
        lines.append(f"  {tier}: {(ot == tier).mean():.1%} ({int((ot == tier).sum())})")
    return "\n".join(lines)


if __name__ == "__main__":
    panel = simulate()
    out = Path(__file__).resolve().parents[2] / "data" / "panel.csv"
    write_csv(panel, out)
    print(summarise(panel))
    print(f"wrote {out}")
