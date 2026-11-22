"""Simplified synthetic calendar and national scenario paths.

These are not official Reserve Bank, Stats NZ, Ministry of Education, or
MSD series. They exist so every whānau shares a coherent backdrop and so
known-future inputs can be rebuilt from a month index at inference time.
"""

from __future__ import annotations

import math

START_YEAR = 2021
N_MONTHS = 60
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
MONTH_SHORT = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
SCHOOL_HOLIDAY_MONTHS = {1, 4, 7, 10, 12}
PUBLIC_HOLIDAY_COUNTS = {1: 2, 2: 1, 4: 3, 6: 1, 10: 1, 12: 2}
BENEFIT_RATE_BY_YEAR = {2021: 0.03, 2022: 0.04, 2023: 0.06, 2024: 0.04, 2025: 0.03, 2026: 0.03}


def year_month(time_idx: int) -> tuple[int, int]:
    return START_YEAR + int(time_idx) // 12, int(time_idx) % 12 + 1


def month_label(time_idx: int) -> str:
    year, month = year_month(time_idx)
    return f"{MONTH_SHORT[month - 1]} {year}"


def quarter(month: int) -> int:
    return (int(month) - 1) // 3 + 1


def school_holiday(month: int) -> int:
    return 1 if int(month) in SCHOOL_HOLIDAY_MONTHS else 0


def public_holiday_count(month: int) -> int:
    return int(PUBLIC_HOLIDAY_COUNTS.get(int(month), 0))


def benefit_indexation(time_idx: int) -> float:
    year, month = year_month(time_idx)
    if month != 4:
        return 0.0
    return float(BENEFIT_RATE_BY_YEAR.get(year, 0.03))


def benefit_indexation_for_month(month: int, year: int | None = None) -> float:
    if int(month) != 4:
        return 0.0
    if year is None:
        return 0.04
    return float(BENEFIT_RATE_BY_YEAR.get(int(year), 0.03))


def planned_support_places(time_idx: int) -> float:
    """Exogenous published capacity. Not an allocation to a person."""
    t = float(time_idx)
    seasonal = 25.0 * math.sin((t % 12.0) / 12.0 * 2.0 * math.pi)
    after = 80.0 if time_idx >= 26 else 0.0
    return 920.0 + 6.0 * t + seasonal + after


def national_macro(time_idx: int) -> dict:
    t = float(time_idx)
    ocr = 0.25 + 5.2 / (1.0 + math.exp(-(t - 22.0) / 4.5)) - 1.4 / (1.0 + math.exp(-(t - 48.0) / 5.0))
    cpi = 2.0 + 5.2 * math.exp(-((t - 18.0) ** 2) / 180.0)
    unemp = 4.1 + 0.6 * math.sin(t / 9.0) + 0.8 / (1.0 + math.exp(-(t - 40.0) / 6.0))
    petrol = 2.45 + 0.55 * math.sin(t / 5.5) + 0.35 * math.exp(-((t - 16.0) ** 2) / 40.0)
    if time_idx < 8:
        pandemic = 3
    elif time_idx < 14:
        pandemic = 2
    elif time_idx < 20:
        pandemic = 1
    else:
        pandemic = 0
    disaster = 0.0
    event = "none"
    if time_idx == 25:
        disaster, event = 0.92, "national"
    elif time_idx == 26:
        disaster, event = 0.45, "regional"
    elif time_idx == 27:
        disaster, event = 0.15, "local"
    year, month = year_month(time_idx)
    return {
        "year": year,
        "month": month,
        "ocr_rate": float(ocr),
        "cpi_inflation_annual": float(cpi),
        "unemployment_rate": float(unemp),
        "petrol_price": float(petrol),
        "pandemic_restriction_level": int(pandemic),
        "natural_disaster_severity": float(disaster),
        "major_event_context": event,
        "benefit_indexation": benefit_indexation(time_idx),
        "planned_support_places": float(planned_support_places(time_idx)),
        "month_of_year": month,
        "quarter": quarter(month),
        "school_holiday": school_holiday(month),
        "public_holidays": public_holiday_count(month),
    }


def project_known_future(last: dict, horizon: int = 12, origin_time_idx: int | None = None) -> list[dict]:
    """Known inputs for the next ``horizon`` months, as they stand at the origin.

    Lease months count down from the last observed value. They do not include a
    renewal or a new tenure, because that would leak the future housing outcome.
    Calendar fields follow the simplified calendar. Planned places follow the
    synthetic published path when the origin month index is known, and otherwise
    stay at the last uploaded value.
    """
    lease0 = float(last["lease_months_remaining"])
    month0 = int(last["month_of_year"])
    places_hold = float(last["planned_support_places"])
    rows = []
    for step in range(1, horizon + 1):
        month = ((month0 - 1 + step) % 12) + 1
        if origin_time_idx is not None:
            macro = national_macro(int(origin_time_idx) + step)
            benefit = macro["benefit_indexation"]
            places = macro["planned_support_places"]
            month = macro["month_of_year"]
        else:
            year = last.get("year")
            benefit_year = (int(year) + ((month0 - 1 + step) // 12)) if year not in (None, "") else None
            benefit = benefit_indexation_for_month(month, benefit_year)
            places = places_hold
        rows.append({
            "month_of_year": int(month),
            "quarter": quarter(month),
            "school_holiday": school_holiday(month),
            "public_holidays": public_holiday_count(month),
            "benefit_indexation": float(benefit),
            "planned_support_places": float(places),
            "lease_months_remaining": float(max(0.0, lease0 - step)),
        })
    return rows
