"""Five-tier label for the attached workbook.

The workbook has no tier column. Columns marked ``y component`` describe the
housing situation the tier is read from. The rule below is the training
target. It is not a second model.
"""

from __future__ import annotations

import math


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _num(value) -> float:
    if _missing(value):
        return 0.0
    return float(value)


def tier_id(row: dict) -> int:
    """Return tier id 1 (severe) through 5 (stable)."""
    deprivation = str(row.get("deprivation_category") or "")
    if deprivation in {"Without shelter", "Uninhabitable housing"}:
        return 1
    accommodation = "" if _missing(row.get("emergency_accommodation_type")) else str(row.get("emergency_accommodation_type"))
    stay = _num(row.get("emergency_housing_stay"))
    if deprivation in {"Temporary accommodation", "Sharing someone else's dwelling"} or stay == 1 or accommodation not in {"", "None"}:
        return 2
    tenure = str(row.get("tenure_start") or "")
    burden = _num(row.get("housing_cost_burden"))
    arrears = _num(row.get("rent_arrears"))
    crowding = _num(row.get("persons_per_bedroom"))
    if tenure == "Social housing":
        return 3 if burden >= 0.50 else 4
    if burden >= 0.30 or arrears > 0 or crowding >= 2:
        return 3
    return 5
