"""Feature contract, tier legend, and vocabularies for the housing TFT.

The 50 names are the model inputs. ``stability_tier`` is the forecast target
and is never fed back in as a feature. Domain labels group those inputs for
the portal. The source spreadsheet could not be read (HTTP 401), so the
groups follow the feature names and the proposal's static / known-future /
observed split.
"""

from __future__ import annotations

from dataclasses import dataclass


HOUSING = "Housing"
TEMPORAL = "Temporal"
DEMOGRAPHIC = "Demographic"
AREA = "Area"
HOUSEHOLD = "Household"
SOCIO = "Socio-economic"
SERVICE = "Service / Policy"
HEALTH = "Health"
JUSTICE = "Justice"
BEHAVIOUR = "Behavioural"
MACRO = "Macro & Events"


@dataclass(frozen=True)
class Feature:
    name: str
    role: str  # static | known | observed | outcome | audit | context
    kind: str  # numeric | categorical
    domain: str
    description: str
    unit: str = ""
    sensitive: bool = False


FEATURES: tuple[Feature, ...] = (
    Feature("tenure_start", "static", "categorical", HOUSING, "Tenure at the start of the spell.", "category"),
    Feature("lease_months_remaining", "known", "numeric", HOUSING, "Months left on the lease. 99 means there is no fixed lease.", "months"),
    Feature("rent_arrears", "outcome", "numeric", HOUSING, "Rent owing. An outcome component of the tier, not a future input.", "NZD"),
    Feature("persons_per_bedroom", "outcome", "numeric", HOUSING, "People per bedroom. An outcome component of the tier.", "ratio"),
    Feature("moves_12m", "outcome", "numeric", HOUSING, "Moves in the past 12 months. An outcome component of the tier.", "count"),
    Feature("emergency_housing_stay", "outcome", "numeric", HOUSING, "1 if this month is an emergency-housing stay. An outcome component.", "0/1"),
    Feature("deprivation_category", "outcome", "categorical", HOUSING, "Dwelling situation this month. An outcome component of the tier."),
    Feature("mortgage_payment", "observed", "numeric", HOUSING, "Mortgage payment this month.", "NZD per month"),
    Feature("housing_cost_burden", "observed", "numeric", HOUSING, "Housing cost divided by income.", "proportion"),
    Feature("disaster_displaced", "observed", "numeric", HOUSING, "1 if the whānau is displaced by a disaster this month.", "0/1"),
    Feature("month_of_year", "known", "categorical", TEMPORAL, "Calendar month, 1 to 12."),
    Feature("quarter", "known", "categorical", TEMPORAL, "Calendar quarter, 1 to 4."),
    Feature("school_holiday", "known", "numeric", TEMPORAL, "1 if the month includes school holidays in this panel.", "0/1"),
    Feature("public_holidays", "known", "numeric", TEMPORAL, "Public holidays in the month.", "count"),
    Feature("age_band", "static", "categorical", DEMOGRAPHIC, "Age band of the reference adult."),
    Feature("residency_status", "static", "categorical", DEMOGRAPHIC, "Residency status."),
    Feature("ethnicity", "audit", "categorical", DEMOGRAPHIC, "Recorded ethnicity, kept for monitoring. Not a model input.", sensitive=True),
    Feature("local_board", "static", "categorical", AREA, "Auckland local board."),
    Feature("nzdep_decile", "static", "numeric", AREA, "NZDep decile, 1 to 10.", "decile"),
    Feature("local_median_rent", "observed", "numeric", AREA, "Local median rent.", "NZD per week"),
    Feature("local_affordability_index", "observed", "numeric", AREA, "Local affordability index.", "proportion"),
    Feature("local_price_to_income", "observed", "numeric", AREA, "Local price-to-income ratio.", "ratio"),
    Feature("local_homelessness_rate", "observed", "numeric", AREA, "Local homelessness rate.", "per 10,000"),
    Feature("local_eh_grants", "observed", "numeric", AREA, "Local emergency-housing grants.", "grants"),
    Feature("local_service_spend", "observed", "numeric", AREA, "Local service spend.", "NZD per capita"),
    Feature("whanau_type", "static", "categorical", HOUSEHOLD, "Whānau composition."),
    Feature("sole_parent", "static", "numeric", HOUSEHOLD, "1 if the whānau is a sole-parent household.", "0/1"),
    Feature("dependants_count", "static", "numeric", HOUSEHOLD, "Number of dependants.", "children"),
    Feature("household_size_start", "static", "numeric", HOUSEHOLD, "Household size at the start of the panel.", "persons"),
    Feature("household_size_change", "observed", "numeric", HOUSEHOLD, "Change in household size since the previous month.", "persons"),
    Feature("weekly_income", "observed", "numeric", SOCIO, "Weekly income.", "NZD per week"),
    Feature("main_benefit", "observed", "categorical", SOCIO, "Main benefit, or None."),
    Feature("hourly_wage", "observed", "numeric", SOCIO, "Hourly wage.", "NZD per hour"),
    Feature("essential_spend_share", "observed", "numeric", SOCIO, "Share of income spent on essentials.", "proportion"),
    Feature("debt_balance", "observed", "numeric", SOCIO, "Debt balance.", "NZD"),
    Feature("benefit_indexation", "known", "numeric", SERVICE, "1 in the month benefits are indexed.", "0/1"),
    Feature("planned_support_places", "known", "numeric", SERVICE, "Planned support places recorded for the whānau.", "places"),
    Feature("emergency_accommodation_type", "observed", "categorical", SERVICE, "Emergency accommodation type, or None."),
    Feature("disability", "static", "numeric", HEALTH, "1 if a disability is recorded.", "0/1"),
    Feature("substance_service_contact", "observed", "numeric", HEALTH, "1 if there is a substance-service contact this month.", "0/1", True),
    Feature("prior_justice_history", "static", "numeric", JUSTICE, "1 if a justice history was already recorded.", "0/1", True),
    Feature("justice_event", "observed", "numeric", JUSTICE, "1 if a justice event is recorded this month.", "0/1", True),
    Feature("gambling_spend_share", "observed", "numeric", BEHAVIOUR, "Share of income spent on gambling.", "proportion", True),
    Feature("ocr_rate", "observed", "numeric", MACRO, "Official cash rate.", "% per year"),
    Feature("cpi_inflation_annual", "observed", "numeric", MACRO, "Annual CPI inflation.", "% annual change"),
    Feature("unemployment_rate", "observed", "numeric", MACRO, "Unemployment rate.", "% of labour force"),
    Feature("petrol_price", "observed", "numeric", MACRO, "Petrol price.", "NZD per litre"),
    Feature("pandemic_restriction_level", "observed", "numeric", MACRO, "Pandemic restriction level, 0 to 4.", "level"),
    Feature("natural_disaster_severity", "observed", "numeric", MACRO, "Natural-disaster severity, 0 to 3.", "severity"),
    Feature("major_event_context", "context", "categorical", MACRO, "Wider events that month. Not used as a predictor."),
)

EXPECTED_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)

TIERS: tuple[dict, ...] = (
    {
        "id": 1,
        "key": "severe",
        "name": "Severely Deprived (Acute Homelessness)",
        "short": "Severely Deprived",
        "living": "Sleeping rough, cars or tents, emergency shelters, motels, women's refuges, or severely overcrowded couch surfing.",
        "character": "No legal right to stay, little privacy or control, or lacking basic amenities. Needs immediate crisis intervention, such as Housing First.",
    },
    {
        "id": 2,
        "key": "unstable",
        "name": "Unstable / Transitional",
        "short": "Unstable / Transitional",
        "living": "Transitional housing, short-term boarding houses, campgrounds, or a temporary stay with family or whānau.",
        "character": "A roof, but time-limited and without long-term security of tenure. Wrap-around support is aimed at a permanent home.",
    },
    {
        "id": 3,
        "key": "at_risk",
        "name": "At Risk (Housing Insecure)",
        "short": "At Risk",
        "living": "Private rental or mortgaged home under severe financial stress, overcrowding, or poor physical conditions.",
        "character": "Often more than 30–50% of income on housing, with arrears, eviction pressure, or a cold or damp home. Prevention support is the relevant response.",
    },
    {
        "id": 4,
        "key": "subsidised",
        "name": "Subsidised / Assisted Stability",
        "short": "Subsidised / Assisted",
        "living": "Public or social housing, community housing, rent-to-buy, or shared equity.",
        "character": "Longer security of tenure, with a government or non-profit subsidy such as income-related rent keeping the cost workable.",
    },
    {
        "id": 5,
        "key": "stable",
        "name": "Stable / Market Secure",
        "short": "Stable / Market Secure",
        "living": "Unsubsidised homeownership, or a long-term private rental the household can afford.",
        "character": "Security of tenure, a habitable dwelling, and housing costs that are sustainable against income.",
    },
)

TIER_NAME = {t["id"]: t["name"] for t in TIERS}
CLASS_TO_TIER = {i: t["id"] for i, t in enumerate(TIERS)}
TIER_TO_CLASS = {t["id"]: i for i, t in enumerate(TIERS)}

LOCAL_BOARDS: tuple[str, ...] = (
    "Rodney",
    "Hibiscus and Bays",
    "Upper Harbour",
    "Kaipātiki",
    "Devonport-Takapuna",
    "Henderson-Massey",
    "Waitākere Ranges",
    "Waitematā",
    "Whau",
    "Albert-Eden",
    "Puketāpapa",
    "Ōrākei",
    "Maungakiekie-Tāmaki",
    "Howick",
    "Māngere-Ōtāhuhu",
    "Ōtara-Papatoetoe",
    "Manurewa",
    "Papakura",
    "Franklin",
    "Aotea / Great Barrier",
    "Waiheke",
)

VOCABS: dict[str, tuple[str, ...]] = {
    "tenure_start": ("Owner-occupier", "Private rental", "Social housing", "Other"),
    "age_band": ("18-24", "25-34", "35-44", "45-54", "55-64", "65+"),
    "residency_status": ("Citizen", "Permanent resident", "Temporary visa", "Unknown"),
    "ethnicity": ("European", "Asian", "Pacific", "Māori", "Asian;European", "MELAA", "Pacific;European", "Māori;European", "Other", "Māori;Pacific", "MELAA;European"),
    "local_board": LOCAL_BOARDS,
    "whanau_type": ("Couple with children", "Couple only", "One person", "Sole parent", "Other", "Multi-family"),
    "deprivation_category": ("Housed", "Temporary accommodation", "Sharing someone else's dwelling", "Uninhabitable housing", "Without shelter"),
    "main_benefit": ("None", "Jobseeker", "Sole Parent Support", "Supported Living"),
    "emergency_accommodation_type": ("None", "Motel / commercial", "Transitional / temporary", "Supported"),
    "major_event_context": ("None", "Other"),
    "month_of_year": tuple(str(i) for i in range(1, 13)),
    "quarter": ("1", "2", "3", "4"),
}

# Unaccented and boolean spellings accepted on upload.
ALIASES: dict[str, dict[str, str]] = {
    "ethnicity": {"maori": "Māori", "pasifika": "Pacific", "nz european": "European", "pakeha": "European", "pākehā": "European"},
    "local_board": {
        "kaipatiki": "Kaipātiki",
        "waitakere ranges": "Waitākere Ranges",
        "waitemata": "Waitematā",
        "puketapapa": "Puketāpapa",
        "orakei": "Ōrākei",
        "maungakiekie-tamaki": "Maungakiekie-Tāmaki",
        "mangere-otahuhu": "Māngere-Ōtāhuhu",
        "otara-papatoetoe": "Ōtara-Papatoetoe",
        "aotea / great barrier": "Aotea / Great Barrier",
        "great barrier": "Aotea / Great Barrier",
        "aotea/great barrier": "Aotea / Great Barrier",
    },
    "residency_status": {"permanent resident": "Permanent resident", "citizen": "Citizen", "temporary": "Temporary visa", "temporary visa": "Temporary visa"},
    "whanau_type": {"sole parent": "Sole parent", "couple with children": "Couple with children", "one person": "One person"},
    "tenure_start": {"owner": "Owner-occupier", "owner-occupier": "Owner-occupier", "private rental": "Private rental", "social housing": "Social housing"},
    "main_benefit": {"none": "None", "jobseeker": "Jobseeker", "sole parent support": "Sole Parent Support", "supported living": "Supported Living"},
    "emergency_accommodation_type": {"none": "None", "motel": "Motel / commercial", "motel / commercial": "Motel / commercial", "transitional": "Transitional / temporary", "supported": "Supported"},
    "deprivation_category": {"housed": "Housed", "without shelter": "Without shelter", "uninhabitable housing": "Uninhabitable housing"},
}


def features_where(role: str | None = None, kind: str | None = None) -> list[Feature]:
    out = []
    for feature in FEATURES:
        if role is not None and feature.role != role:
            continue
        if kind is not None and feature.kind != kind:
            continue
        out.append(feature)
    return out


def feature_map() -> dict[str, Feature]:
    return {f.name: f for f in FEATURES}


def _model_inputs(kind: str, roles: tuple[str, ...]) -> list[Feature]:
    return [feature for feature in FEATURES if feature.kind == kind and feature.role in roles]


STATIC_CAT = _model_inputs("categorical", ("static",))
STATIC_NUM = _model_inputs("numeric", ("static",))
OBS_CAT = _model_inputs("categorical", ("observed", "outcome"))
OBS_NUM = _model_inputs("numeric", ("observed", "outcome"))
KNOWN_CAT = _model_inputs("categorical", ("known",))
KNOWN_NUM = _model_inputs("numeric", ("known",))

HIST_NAMES = [f.name for f in OBS_NUM + OBS_CAT + KNOWN_NUM + KNOWN_CAT]
FUTURE_NAMES = [f.name for f in KNOWN_NUM + KNOWN_CAT]
STATIC_NAMES = [f.name for f in STATIC_NUM + STATIC_CAT]
