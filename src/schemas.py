"""HealthSentinel data contracts.

Single source of truth for every dataset's columns, logical types, units, nullability,
valid ranges, provenance and meaning. ``docs/data_dictionary.md`` is GENERATED from this
module:  ``python -m src.schemas``  (a test fails if the two drift).

Rules encoded here:
* Every source dataset carries the common observation key: observation_date + site_id + zone_id + shift.
* Every source dataset carries provenance: ``recorded_at`` (occurrence), ``reporting_timestamp``
  (availability) and ``reporting_status``.
* Missing is never zero: measurement and count fields are nullable; ``coerce`` keeps NaN/<NA>.
* No individual identifiers, names, free text or person-level rows exist in any contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
LOGICAL_TO_PANDAS = {
    "date": "datetime64[ns]",      # normalised to midnight
    "datetime": "datetime64[ns]",
    "string": "string",
    "category": "string",          # constrained by ``allowed``
    "int": "Int64",                # nullable integer: missing stays <NA>, never 0
    "float": "float64",            # missing stays NaN
    "bool": "boolean",             # nullable boolean
}


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str                      # logical type, key of LOGICAL_TO_PANDAS
    description: str
    unit: str = "-"
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.dtype not in LOGICAL_TO_PANDAS:
            raise ValueError(f"{self.name}: unknown dtype {self.dtype!r}")


@dataclass(frozen=True)
class DataContract:
    name: str
    description: str
    source: str
    grain: str
    layer: str                      # 'raw' or 'ground_truth'
    file_name: str
    primary_key: tuple[str, ...]
    fields: tuple[Field, ...]
    expected_latency: str
    le_constraints: tuple[tuple[str, str], ...] = ()   # (a, b): a <= b where both present

    @property
    def columns(self) -> list[str]:
        return [f.name for f in self.fields]

    def field(self, name: str) -> Field:
        return next(f for f in self.fields if f.name == name)


# --------------------------------------------------------------------------------------
# Common observation key and provenance
# --------------------------------------------------------------------------------------
SHIFTS = ("day", "night")
OBSERVATION_KEY = ("observation_date", "site_id", "zone_id", "shift")
REPORTING_STATUSES = ("reported", "not_reported")

KEY_FIELDS = (
    Field("observation_date", "date",
          "Site-local calendar date on which the shift STARTS (night shift 18:00 D -> 06:00 D+1 belongs to D).",
          nullable=False),
    Field("site_id", "string", "Synthetic site identifier, e.g. 'S01'.", nullable=False),
    Field("zone_id", "string", "Synthetic zone identifier, globally unique and prefixed by site, e.g. 'S01-Z02'.",
          nullable=False),
    Field("shift", "category", "Work shift. day = 06:00-18:00, night = 18:00-06:00.", nullable=False, allowed=SHIFTS),
)

PROVENANCE_FIELDS = (
    Field("recorded_at", "datetime",
          "When the observed window closed / measurement was taken at source (end of the shift window). "
          "This is OCCURRENCE time.", unit="site-local datetime", nullable=False),
    Field("reporting_timestamp", "datetime",
          "When the record was reported to / became available in PREVENZA. NA when the period was never reported. "
          "Models may only use rows with reporting_timestamp <= decision time.",
          unit="site-local datetime"),
    Field("reporting_status", "category",
          "'reported' = row delivered (individual fields may still be NA); 'not_reported' = whole period missing, "
          "all measurement fields NA (never 0).", nullable=False, allowed=REPORTING_STATUSES),
)
_PROV_CONSTRAINT = (("recorded_at", "reporting_timestamp"),)

# --------------------------------------------------------------------------------------
# Source contracts (model inputs). Every dataset has one row per observation key (1,620 rows).
# --------------------------------------------------------------------------------------
_GRID = "One row per observation key (full 1,620-row grid is preserved; unreported periods are rows with NA values)."

ENVIRONMENTAL = DataContract(
    name="environmental_data",
    description="Zone thermal environment per shift from zone sensors. Heat index is the US NWS heat-index "
                "approximation (Rothfusz regression with Steadman/NWS adjustments) computed from the reported "
                "temperature and humidity. It is NOT WBGT (no radiant-heat or wind component).",
    source="Zone environmental sensors via sensor platform (synthetic)",
    grain=_GRID,
    layer="raw",
    file_name="environmental_data.csv",
    primary_key=OBSERVATION_KEY,
    expected_latency="minutes (5-30 min)",
    fields=KEY_FIELDS + (
        Field("temperature_celsius", "float", "Peak hourly dry-bulb air temperature in the shift.", "degC",
              min_value=-10, max_value=55),
        Field("relative_humidity", "float", "Mean relative humidity over the shift.", "%", min_value=0, max_value=100),
        Field("heat_index_celsius", "float", "NWS heat index from temperature_celsius and relative_humidity. "
              "NA whenever either input is NA.", "degC", min_value=-10, max_value=80),
    ) + PROVENANCE_FIELDS,
    le_constraints=_PROV_CONSTRAINT,
)

WORKFORCE = DataContract(
    name="workforce_data",
    description="Aggregated workforce exposure per zone-shift.",
    source="Rostering / time-and-attendance system, aggregated before export (no person rows; synthetic)",
    grain=_GRID,
    layer="raw",
    file_name="workforce_data.csv",
    primary_key=OBSERVATION_KEY,
    expected_latency="~0.5-3 hours",
    fields=KEY_FIELDS + (
        Field("workforce_count", "int", "Workers PRESENT and working in the zone during the shift (exposed workers). "
              "Equals absenteeism_data.workforce_count - absent_workers.", "count", min_value=0),
        Field("average_exposure_hours", "float", "Mean hours per present worker spent working in the zone's thermal "
              "environment.", "hours", min_value=0, max_value=12),
        Field("total_exposure_hours", "float", "workforce_count x average_exposure_hours.", "person-hours",
              min_value=0),
    ) + PROVENANCE_FIELDS,
    le_constraints=_PROV_CONSTRAINT,
)

CLINICAL = DataContract(
    name="clinical_data",
    description="Aggregated on-site occupational-health / first-aid visits coded to a heat-related syndrome "
                "(syndromic category, not an individual diagnosis), with severity distribution.",
    source="On-site occupational health clinic log, aggregated before export (synthetic)",
    grain=_GRID,
    layer="raw",
    file_name="clinical_data.csv",
    primary_key=OBSERVATION_KEY,
    expected_latency="~12-48 hours (coded after the shift)",
    fields=KEY_FIELDS + (
        Field("heat_related_visits", "int", "Heat-related visits for workers of this zone-shift. "
              "= heat_visits_mild + heat_visits_moderate + heat_visits_severe.", "count", min_value=0),
        Field("heat_visits_mild", "int", "Mild (e.g. heat rash, cramps; returned to work).", "count", min_value=0),
        Field("heat_visits_moderate", "int", "Moderate (e.g. heat exhaustion; removed from work).", "count",
              min_value=0),
        Field("heat_visits_severe", "int", "Severe (e.g. suspected heat stroke; external transfer).", "count",
              min_value=0),
    ) + PROVENANCE_FIELDS,
    le_constraints=(("heat_visits_severe", "heat_related_visits"),) + _PROV_CONSTRAINT,
)

ABSENTEEISM = DataContract(
    name="absenteeism_data",
    description="Aggregated unplanned (sickness) absence among workers rostered to the zone-shift.",
    source="HR / time-and-attendance absence records, aggregated before export (synthetic)",
    grain=_GRID,
    layer="raw",
    file_name="absenteeism_data.csv",
    primary_key=OBSERVATION_KEY,
    expected_latency="~1.5-2.5 days (batch)",
    fields=KEY_FIELDS + (
        Field("workforce_count", "int", "Workers ROSTERED to the zone-shift (absence denominator).", "count",
              min_value=0),
        Field("absent_workers", "int", "Rostered workers with an unplanned (sickness) absence for the shift.",
              "count", min_value=0),
        Field("absence_rate", "float", "absent_workers / workforce_count (NA if either is NA or workforce_count = 0).",
              "fraction 0-1", min_value=0, max_value=1),
        Field("lost_workdays", "float", "Workdays lost to the absences (1 absent shift = 1 workday; equals "
              "absent_workers in this simulation).", "workdays", min_value=0),
    ) + PROVENANCE_FIELDS,
    le_constraints=(("absent_workers", "workforce_count"),) + _PROV_CONSTRAINT,
)

INCIDENTS = DataContract(
    name="incident_data",
    description="Aggregated heat-related workplace incidents (reporter flagged heat/hydration as a factor) by "
                "severity. Counts only - no persons, no free text.",
    source="EHS incident management system, aggregated before export (synthetic)",
    grain=_GRID,
    layer="raw",
    file_name="incident_data.csv",
    primary_key=OBSERVATION_KEY,
    expected_latency="variable, ~1-96 hours (median ~18 h)",
    fields=KEY_FIELDS + (
        Field("heat_incident_count", "int", "= minor_incidents + moderate_incidents + severe_incidents.", "count",
              min_value=0),
        Field("minor_incidents", "int", "First-aid / near-miss level.", "count", min_value=0),
        Field("moderate_incidents", "int", "Medical treatment / restricted work.", "count", min_value=0),
        Field("severe_incidents", "int", "Lost time / regulator-reportable.", "count", min_value=0),
    ) + PROVENANCE_FIELDS,
    le_constraints=(("severe_incidents", "heat_incident_count"),) + _PROV_CONSTRAINT,
)

# --------------------------------------------------------------------------------------
# Ground-truth contracts (simulation only; NEVER a model input)
# --------------------------------------------------------------------------------------
GT_EVENT_TYPES = ("gradual_heatwave_escalation", "isolated_temperature_spike", "heat_escalation_clinical_gap")
LABELS = {0: "normal", 1: "emerging", 2: "escalating"}
SOURCE_DATASETS = ("environmental_data", "workforce_data", "clinical_data", "absenteeism_data", "incident_data")

HEAT_RISK_EVENTS = DataContract(
    name="heat_risk_events",
    description="Scheduled (injected) heat-health events. Used ONLY for evaluation.",
    source="Synthetic generator schedule (config/simulation.yaml -> scenarios)",
    grain="One row per event x affected zone (event_id repeats for multi-zone events).",
    layer="ground_truth",
    file_name="heat_risk_events.csv",
    primary_key=("event_id", "zone_id"),
    expected_latency="n/a (evaluation only)",
    fields=(
        Field("event_id", "string", "Event identifier, e.g. 'EVT-001'.", nullable=False),
        Field("site_id", "string", "Affected site.", nullable=False),
        Field("zone_id", "string", "Affected zone.", nullable=False),
        Field("event_type", "category", "Scenario mechanism. isolated_temperature_spike is a NEGATIVE CONTROL "
              "(no sustained health deterioration; its observations are labelled 0).", nullable=False,
              allowed=GT_EVENT_TYPES),
        Field("onset_timestamp", "datetime", "Start of the first shift in which heat forcing departs from baseline.",
              "site-local datetime", nullable=False),
        Field("escalation_timestamp", "datetime", "Start of the first shift of the scheduled escalation phase "
              "(forcing at full strength; health outcomes rising materially). NA for non-escalating events.",
              "site-local datetime"),
        Field("end_timestamp", "datetime", "End of the last affected shift (exclusive bound).",
              "site-local datetime", nullable=False),
        Field("severity", "category", "Designed severity.", nullable=False,
              allowed=("none", "low", "moderate", "high")),
        Field("description", "string", "Generator description of the scenario.", nullable=False),
    ),
    le_constraints=(("onset_timestamp", "escalation_timestamp"), ("escalation_timestamp", "end_timestamp"),
                    ("onset_timestamp", "end_timestamp")),
)

OBSERVATION_LABELS = DataContract(
    name="observation_labels",
    description="Per-observation evaluation label derived ONLY from the event schedule (never model output). "
                "Rule: shift start in [onset, escalation) -> 1; in [escalation, end) -> 2; otherwise 0. "
                "Negative-control events give 0. Overlaps take the maximum.",
    source="Derived from heat_risk_events.csv",
    grain="One row per observation key (1,620 rows).",
    layer="ground_truth",
    file_name="observation_labels.csv",
    primary_key=OBSERVATION_KEY,
    expected_latency="n/a (evaluation only)",
    fields=KEY_FIELDS + (
        Field("label", "int", "0 normal, 1 emerging, 2 escalating.", nullable=False, min_value=0, max_value=2),
        Field("label_name", "category", "Text form of label.", nullable=False, allowed=tuple(LABELS.values())),
        Field("event_id", "string", "Event that set the label (NA when label = 0 and no event applies)."),
    ),
)

DATA_QUALITY_LOG = DataContract(
    name="data_quality_log",
    description="Structured data-quality issues injected by the generator (reporting gaps, feed delays, sensor "
                "anomalies). Random cell-level missingness is summarised in docs/data_quality_report.md instead.",
    source="Synthetic generator",
    grain="One row per injected issue x zone.",
    layer="ground_truth",
    file_name="data_quality_log.csv",
    primary_key=("issue_id", "zone_id"),
    expected_latency="n/a (evaluation only)",
    fields=(
        Field("issue_id", "string", "Issue identifier, e.g. 'DQ-001'.", nullable=False),
        Field("source_dataset", "category", "Affected dataset.", nullable=False, allowed=SOURCE_DATASETS),
        Field("issue_type", "category", "reporting_gap: rows not_reported; feed_delay: reporting_timestamp later "
              "than usual; sensor_spike: implausible single-shift jump; stuck_sensor: value repeated.",
              nullable=False, allowed=("reporting_gap", "feed_delay", "sensor_spike", "stuck_sensor")),
        Field("site_id", "string", "Affected site.", nullable=False),
        Field("zone_id", "string", "Affected zone.", nullable=False),
        Field("start_timestamp", "datetime", "Start of first affected shift.", "site-local datetime", nullable=False),
        Field("end_timestamp", "datetime", "End of last affected shift.", "site-local datetime", nullable=False),
        Field("details", "string", "Parameters of the injected issue.", nullable=False),
    ),
    le_constraints=(("start_timestamp", "end_timestamp"),),
)

SOURCE_CONTRACTS: dict[str, DataContract] = {
    c.name: c for c in (ENVIRONMENTAL, WORKFORCE, CLINICAL, ABSENTEEISM, INCIDENTS)
}
GROUND_TRUTH_CONTRACTS: dict[str, DataContract] = {
    c.name: c for c in (HEAT_RISK_EVENTS, OBSERVATION_LABELS, DATA_QUALITY_LOG)
}
ALL_CONTRACTS: dict[str, DataContract] = {**SOURCE_CONTRACTS, **GROUND_TRUTH_CONTRACTS}

# Columns that must never appear in a model-input dataset (label leakage guard).
LABEL_COLUMNS = ("label", "label_name", "event_id", "event_type", "onset_timestamp", "escalation_timestamp",
                 "end_timestamp", "severity", "issue_id", "issue_type")

# Field names that must never appear in any dataset (privacy guard).
FORBIDDEN_COLUMN_TOKENS = ("employee", "worker_id", "person_id", "first_name", "last_name", "full_name", "dob", "birth", "address", "email",
                           "phone", "nhs", "ssn", "diagnosis", "free_text", "comment")


# --------------------------------------------------------------------------------------
# Coercion and validation (no ML; purely structural)
# --------------------------------------------------------------------------------------
def coerce(df: pd.DataFrame, contract: DataContract) -> pd.DataFrame:
    """Cast columns to contract dtypes. Missing values stay missing (never filled with 0)."""
    out = df.copy()
    for f in contract.fields:
        if f.name not in out.columns:
            continue
        if f.dtype == "date":
            out[f.name] = pd.to_datetime(out[f.name], errors="coerce").dt.normalize()
        elif f.dtype == "datetime":
            out[f.name] = pd.to_datetime(out[f.name], errors="coerce")
        elif f.dtype == "int":
            out[f.name] = pd.to_numeric(out[f.name], errors="coerce").astype("Int64")
        elif f.dtype == "float":
            out[f.name] = pd.to_numeric(out[f.name], errors="coerce").astype("float64")
        elif f.dtype == "bool":
            out[f.name] = out[f.name].map(
                lambda v: pd.NA if pd.isna(v) else str(v).strip().lower() in ("true", "1", "yes")
            ).astype("boolean")
        else:
            out[f.name] = out[f.name].astype("string")
    return out


def validate(df: pd.DataFrame, contract: DataContract) -> list[str]:
    """Return a list of human-readable contract violations (empty list == valid)."""
    issues: list[str] = []
    cols = set(df.columns)
    missing = [c for c in contract.columns if c not in cols]
    extra = [c for c in df.columns if c not in contract.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
    if extra:
        issues.append(f"unexpected columns: {extra}")
    for c in df.columns:
        if any(tok in c.lower() for tok in FORBIDDEN_COLUMN_TOKENS):
            issues.append(f"forbidden (potentially identifying) column: {c}")

    for f in contract.fields:
        if f.name not in cols:
            continue
        s = df[f.name]
        if not f.nullable and s.isna().any():
            issues.append(f"{f.name}: {int(s.isna().sum())} nulls in non-nullable field")
        present = s.dropna()
        if f.allowed is not None:
            bad = sorted(set(present.astype(str)) - set(f.allowed))
            if bad:
                issues.append(f"{f.name}: values outside {f.allowed}: {bad[:5]}")
        if f.dtype in ("int", "float") and len(present):
            num = pd.to_numeric(present, errors="coerce")
            if num.isna().any():
                issues.append(f"{f.name}: non-numeric values")
            if f.min_value is not None and (num < f.min_value).any():
                issues.append(f"{f.name}: values below {f.min_value}")
            if f.max_value is not None and (num > f.max_value).any():
                issues.append(f"{f.name}: values above {f.max_value}")

    if not missing and df.duplicated(list(contract.primary_key)).any():
        issues.append(f"duplicate primary key {contract.primary_key}")

    for a, b in contract.le_constraints:
        if a in cols and b in cols:
            both = df[[a, b]].dropna()
            if len(both) and (both[a] > both[b]).any():
                issues.append(f"constraint violated: {a} <= {b} ({int((both[a] > both[b]).sum())} rows)")
    return issues


# --------------------------------------------------------------------------------------
# Data dictionary rendering
# --------------------------------------------------------------------------------------
DICTIONARY_PATH = Path(__file__).resolve().parents[1] / "docs" / "data_dictionary.md"


def _fmt_range(f: Field) -> str:
    if f.allowed:
        return ", ".join(f"`{a}`" for a in f.allowed)
    if f.min_value is None and f.max_value is None:
        return ""
    lo = "" if f.min_value is None else f"{f.min_value:g}"
    hi = "" if f.max_value is None else f"{f.max_value:g}"
    return f"{lo} .. {hi}"


def render_data_dictionary() -> str:
    lines = [
        "# PREVENZA (HealthSentinel) Data Dictionary",
        "",
        "> GENERATED from `src/schemas.py` by `python -m src.schemas`. Do not edit by hand.",
        "",
        "All data is **synthetic and aggregated**. No dataset contains individual identifiers, names, free text,",
        "or clinical diagnoses of individuals. Timestamps are site-local and timezone-naive.",
        "",
        "## Conventions",
        "",
        f"- **Common observation key:** `{' + '.join(OBSERVATION_KEY)}`. 90 days x 3 sites x 3 zones x 2 shifts "
        "= 1,620 observations.",
        "- **Night shift** (18:00-06:00) belongs to the date on which it starts.",
        "- **Provenance:** every source row has `recorded_at` (occurrence: end of the shift window), "
        "`reporting_timestamp` (when it became available; NA if never reported) and `reporting_status`. "
        "Models use only rows with `reporting_timestamp <= decision time`.",
        "- **Labels are separate.** Ground-truth files are never joined into model inputs.",
        "- **Missing is never zero.** Logical `int` -> pandas `Int64` (`<NA>`), `float` -> `float64` (`NaN`), "
        "`bool` -> `boolean`. Unreported periods are kept as grid rows with `reporting_status = not_reported` "
        "and NA values, never 0.",
        "- **Nullable = no** means the field must always be present when the row exists.",
        "- **Constraints** `a <= b` are checked only where both values are present.",
        "",
    ]
    sections = [("Source datasets (model inputs)", SOURCE_CONTRACTS),
                ("Ground truth (evaluation only - never a model input)", GROUND_TRUTH_CONTRACTS)]
    for title, contracts in sections:
        lines += [f"## {title}", ""]
        for c in contracts.values():
            lines += [
                f"### `{c.name}`",
                "",
                c.description,
                "",
                f"- **File:** `data/{c.layer}/{c.file_name}`",
                f"- **Source:** {c.source}",
                f"- **Grain:** {c.grain}",
                f"- **Primary key:** `{', '.join(c.primary_key)}`",
                f"- **Expected latency:** {c.expected_latency}",
            ]
            if c.le_constraints:
                lines.append("- **Constraints:** " + "; ".join(f"`{a} <= {b}`" for a, b in c.le_constraints))
            lines += [
                "",
                "| Field | Type | Unit | Nullable | Valid values / range | Meaning |",
                "|---|---|---|---|---|---|",
            ]
            for f in c.fields:
                lines.append(
                    f"| `{f.name}` | {f.dtype} | {f.unit} | {'yes' if f.nullable else 'no'} | {_fmt_range(f)} | "
                    f"{f.description} |"
                )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    DICTIONARY_PATH.write_text(render_data_dictionary(), encoding="utf-8")
    print(f"wrote {DICTIONARY_PATH}")
