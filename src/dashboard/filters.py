"""Pure filtering / selection helpers (no Streamlit; unit-tested)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KEY = ["observation_date", "site_id", "zone_id", "shift"]
SOURCE_LABELS = {"env": "Environmental", "wf": "Workforce", "clin": "Clinical", "abs": "Absenteeism",
                 "inc": "Incidents"}


@dataclass
class Filters:
    sites: list[str] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    shifts: list[str] = field(default_factory=list)
    date_range: tuple | None = None                     # (start date, end date) inclusive, observation dates
    classes: list[str] = field(default_factory=list)


def apply_filters(df: pd.DataFrame, f: Filters, class_col: str | None = "classification") -> pd.DataFrame:
    """Filter by site, zone, shift, observation-date range and (optionally) risk classification.
    Empty selections mean 'all'. Never fills or drops missing values in other columns."""
    m = pd.Series(True, index=df.index)
    if f.sites and "site_id" in df:
        m &= df["site_id"].isin(f.sites)
    if f.zones and "zone_id" in df:
        m &= df["zone_id"].isin(f.zones)
    if f.shifts and "shift" in df:
        m &= df["shift"].isin(f.shifts)
    if f.date_range and "observation_date" in df:
        lo, hi = (pd.Timestamp(d) for d in f.date_range)
        m &= df["observation_date"].between(lo.normalize(), hi.normalize())
    if f.classes and class_col and class_col in df:
        m &= df[class_col].isin(f.classes)
    return df[m]


def latest_per_zone(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Most recent row per site-zone (either shift) with decision_time <= as_of (default: latest available)."""
    d = df if as_of is None else df[df["decision_time"] <= pd.Timestamp(as_of)]
    if d.empty:
        return d
    idx = d.groupby(["site_id", "zone_id"])["decision_time"].idxmax()
    return d.loc[idx].sort_values(["site_id", "zone_id"]).reset_index(drop=True)


def latest_per_series(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    d = df if as_of is None else df[df["decision_time"] <= pd.Timestamp(as_of)]
    if d.empty:
        return d
    idx = d.groupby(["site_id", "zone_id", "shift"])["decision_time"].idxmax()
    return d.loc[idx].sort_values(["site_id", "zone_id", "shift"]).reset_index(drop=True)


def data_flag_text(row: pd.Series) -> str:
    """Human-readable missing / stale indicator for a risk-assessment row ('' when nothing to report)."""
    parts = []
    if isinstance(row.get("missing_sources"), str) and row["missing_sources"]:
        parts.append("Missing: " + row["missing_sources"].replace(";", ", "))
    if isinstance(row.get("stale_sources"), str) and row["stale_sources"]:
        parts.append("Stale: " + row["stale_sources"].replace(";", ", "))
    return " | ".join(parts) if parts else "All sources current"


def measurement_flag_text(row: pd.Series) -> str:
    if row.get("heat_index_validity") == "extrapolated":
        return f"⚠ Heat index {row.get('heat_index_observed', np.nan):.1f} °C outside NWS chart range (extrapolated)"
    if row.get("heat_index_validity") == "unavailable":
        return "Heat index unavailable"
    return "Within range"


def persistence_any(classification: pd.Series, any_classes: list[str]) -> pd.Series:
    """'Any warning' persistence baseline, re-derived from the current class exactly as defined in
    config/forecast.yaml (persistence.any_classes)."""
    return classification.isin(any_classes)


def source_state(unified: pd.DataFrame, prefix: str, value_col: str) -> pd.Series:
    """Per observation: 'not reported' | 'field missing' | 'reported' (missing is never zero)."""
    nr = unified[f"{prefix}_reporting_status"].eq("not_reported")
    miss = unified[f"{prefix}_{value_col}"].isna()
    return pd.Series(np.where(nr, "not reported", np.where(miss, "field missing", "reported")), index=unified.index)


def fmt_score(x, digits: int = 1) -> str:
    return "unavailable" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def fmt_ranking_score(x) -> str:
    """Forecast scores are rankings: show a plain decimal, never a percentage."""
    return "no forecast" if x is None or pd.isna(x) else f"{x:.3f}"
