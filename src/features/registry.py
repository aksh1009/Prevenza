"""Feature registry: names, groups, sources and meanings. Single source for features.parquet columns and
docs/feature_dictionary.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.config import CONFIG_DIR

import yaml

SOURCES = {"env": "environmental_data", "wf": "workforce_data", "clin": "clinical_data",
           "abs": "absenteeism_data", "inc": "incident_data"}

# Unified-table columns read as-of (value matrices). Core fields drive completeness / missing indicators.
BASE_FIELDS = {
    "env_temperature_celsius": "env", "env_relative_humidity": "env", "env_heat_index_celsius": "env",
    "wf_workforce_count": "wf", "wf_average_exposure_hours": "wf", "wf_total_exposure_hours": "wf",
    "clin_heat_related_visits": "clin", "clin_heat_visits_moderate": "clin", "clin_heat_visits_severe": "clin",
    "abs_workforce_count": "abs", "abs_absent_workers": "abs", "abs_absence_rate": "abs",
    "inc_heat_incident_count": "inc", "inc_moderate_incidents": "inc", "inc_severe_incidents": "inc",
}
CORE_FIELDS = ("env_temperature_celsius", "env_relative_humidity", "wf_workforce_count",
               "wf_average_exposure_hours", "clin_heat_related_visits", "abs_workforce_count",
               "abs_absent_workers", "inc_heat_incident_count")


def load_feature_config() -> dict:
    with open(CONFIG_DIR / "features.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Signal:
    name: str
    sources: tuple[str, ...]
    unit: str
    description: str
    fn: Callable[[dict, dict], np.ndarray]   # (base value matrices, feature cfg) -> matrix


def _per100(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 100.0 * num / den, np.nan)


SIGNALS: tuple[Signal, ...] = (
    Signal("temperature_celsius", ("env",), "degC", "Peak shift air temperature.",
           lambda b, c: b["env_temperature_celsius"]),
    Signal("relative_humidity", ("env",), "%", "Mean shift relative humidity.",
           lambda b, c: b["env_relative_humidity"]),
    Signal("heat_index_celsius", ("env",), "degC", "NWS heat index (not WBGT).",
           lambda b, c: b["env_heat_index_celsius"]),
    Signal("heat_index_excess", ("env",), "degC", "max(0, heat index - heat_index_threshold_c).",
           lambda b, c: np.clip(b["env_heat_index_celsius"] - c["heat_index_threshold_c"], 0, None)),
    Signal("workers_present", ("wf",), "count", "Workers present (workforce_data.workforce_count).",
           lambda b, c: b["wf_workforce_count"]),
    Signal("average_exposure_hours", ("wf",), "hours", "Mean hours per present worker in the zone.",
           lambda b, c: b["wf_average_exposure_hours"]),
    Signal("total_exposure_hours", ("wf",), "person-hours", "Total person-hours in the zone.",
           lambda b, c: b["wf_total_exposure_hours"]),
    Signal("heat_exposure_load", ("env", "wf"), "degC x hours",
           "heat_index_excess x average_exposure_hours (needs both sources).",
           lambda b, c: np.clip(b["env_heat_index_celsius"] - c["heat_index_threshold_c"], 0, None)
           * b["wf_average_exposure_hours"]),
    Signal("heat_related_visits", ("clin",), "count", "Heat-related clinic visits.",
           lambda b, c: b["clin_heat_related_visits"]),
    Signal("heat_visits_mod_severe", ("clin",), "count", "Moderate + severe heat-related visits.",
           lambda b, c: b["clin_heat_visits_moderate"] + b["clin_heat_visits_severe"]),
    Signal("heat_visits_per_100", ("clin", "wf"), "per 100 present workers",
           "100 x heat_related_visits / workers present (needs both sources).",
           lambda b, c: _per100(b["clin_heat_related_visits"], b["wf_workforce_count"])),
    Signal("absence_rate", ("abs",), "fraction", "Unplanned absence rate among rostered workers.",
           lambda b, c: b["abs_absence_rate"]),
    Signal("absent_workers", ("abs",), "count", "Rostered workers absent (unplanned).",
           lambda b, c: b["abs_absent_workers"]),
    Signal("heat_incident_count", ("inc",), "count", "Heat-related incidents (all severities).",
           lambda b, c: b["inc_heat_incident_count"]),
    Signal("incidents_mod_severe", ("inc",), "count", "Moderate + severe heat-related incidents.",
           lambda b, c: b["inc_moderate_incidents"] + b["inc_severe_incidents"]),
    Signal("incidents_per_100", ("inc", "wf"), "per 100 present workers",
           "100 x heat_incident_count / workers present (needs both sources).",
           lambda b, c: _per100(b["inc_heat_incident_count"], b["wf_workforce_count"])),
)


def stat_specs(cfg: dict) -> list[tuple[str, str]]:
    s, l = cfg["windows"]["short"], cfg["windows"]["long"]
    b, t = cfg["baseline"], cfg["trend"]
    return [
        ("cur", "Value of the current observation, if reported by decision time (else NA)."),
        ("lag1", "Value of the previous observation in the series, as known at decision time."),
        ("lag2", "Value two observations back, as known at decision time."),
        ("last", "Most recent value known at decision time (any earlier observation of the series)."),
        (f"roll{s['length']}", f"Mean of known values over the last {s['length']} observations incl. current "
                               f"(>= {s['min_periods']} known values, else NA)."),
        (f"roll{l['length']}", f"Mean of known values over the last {l['length']} observations incl. current "
                               f"(>= {l['min_periods']} known values, else NA)."),
        ("hist_median", f"Median of known values over observations t-{b['gap'] + b['length'] - 1}..t-{b['gap']} "
                        f"(baseline excludes the most recent {b['gap']}; >= {b['min_count']} values, else NA)."),
        ("hist_mad", "Median absolute deviation over the same baseline window."),
        ("dev", "last - hist_median (most recent known value vs baseline; age given by <source>_staleness_hours)."),
        ("rz", f"Robust z: (last - hist_median) / max({b['mad_scale']} x MAD, scale_floor)."),
        (f"roll{s['length']}_rz", f"Robust z of the {s['length']}-observation mean vs the baseline."),
        (f"slope{t['length']}", f"OLS slope per observation over known values in the last {t['length']} "
                                f"observations (>= {t['min_points']} points, else NA)."),
    ]


def feature_registry(cfg: dict | None = None) -> list[dict]:
    """Ordered list of every feature column with metadata."""
    cfg = cfg or load_feature_config()
    reg = []
    for p, src in SOURCES.items():
        reg += [
            {"feature": f"{p}_available", "group": "availability", "sources": src, "unit": "bool",
             "description": f"Current observation of {src} reported by decision time."},
            {"feature": f"{p}_window{cfg['windows']['long']['length']}_availability", "group": "availability",
             "sources": src, "unit": "fraction",
             "description": f"Share of the last {cfg['windows']['long']['length']} observations of {src} "
                            "known at decision time."},
            {"feature": f"{p}_staleness_hours", "group": "availability", "sources": src, "unit": "hours",
             "description": f"decision_time - recorded_at of the most recent {src} observation known at "
                            "decision time (NA if none yet)."},
            {"feature": f"{p}_last_delay_hours", "group": "availability", "sources": src, "unit": "hours",
             "description": f"Reporting delay of that most recent known {src} observation."},
        ]
    reg += [
        {"feature": "n_sources_available", "group": "availability", "sources": "all", "unit": "count 0-5",
         "description": "Number of sources whose current observation is reported by decision time."},
        {"feature": "completeness_score", "group": "availability", "sources": "all", "unit": "fraction",
         "description": f"Share of the {len(CORE_FIELDS)} core fields known for the current observation at "
                        f"decision time ({', '.join(CORE_FIELDS)})."},
        {"feature": f"completeness_window{cfg['windows']['long']['length']}", "group": "availability",
         "sources": "all", "unit": "fraction",
         "description": "Mean of the five source window availabilities."},
    ]
    for f in CORE_FIELDS:
        reg.append({"feature": f"{f}_missing", "group": "missingness", "sources": SOURCES[BASE_FIELDS[f]],
                    "unit": "bool", "description": f"{f} unknown for the current observation at decision time "
                                                   "(not yet reported, not reported, or missing in report)."})
    reg += [
        {"feature": "ctx_outdoor", "group": "context", "sources": "zone registry", "unit": "bool",
         "description": "Zone is outdoor or semi-outdoor."},
        {"feature": "ctx_indoor_conditioned", "group": "context", "sources": "zone registry", "unit": "bool",
         "description": "Zone is air-conditioned."},
        {"feature": "ctx_radiant_heat", "group": "context", "sources": "zone registry", "unit": "bool",
         "description": "Zone has a radiant heat source (e.g. furnaces)."},
        {"feature": "ctx_work_intensity", "group": "context", "sources": "zone registry", "unit": "ordinal 1-3",
         "description": "Typical work intensity: 1 light, 2 moderate, 3 heavy."},
        {"feature": "ctx_night_shift", "group": "context", "sources": "key", "unit": "bool",
         "description": "Observation is a night shift."},
    ]
    for sig in SIGNALS:
        src = " + ".join(SOURCES[s] for s in sig.sources)
        for stat, desc in stat_specs(cfg):
            reg.append({"feature": f"{sig.name}_{stat}", "group": f"signal:{sig.name}", "sources": src,
                        "unit": _stat_unit(stat, sig.unit), "description": desc})
    return reg


def _stat_unit(stat: str, unit: str) -> str:
    if stat.endswith("rz"):
        return "robust z"
    if stat.startswith("slope"):
        return f"{unit} per observation"
    return unit


def feature_columns(cfg: dict | None = None) -> list[str]:
    return [r["feature"] for r in feature_registry(cfg)]
