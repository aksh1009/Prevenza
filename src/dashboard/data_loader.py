"""Read-only access to saved pipeline outputs for the dashboard.

* Never writes, regenerates, refits or recomputes model outputs.
* Operational datasets are guarded: label/event columns raise LeakageError (or are stripped where a dataset mixes
  operational and evaluation fields, e.g. alert episodes' match status).
* Cached by file modification time, so regenerated pipeline outputs are picked up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src import config

PROJECT = config.PROJECT_ROOT
PROC = PROJECT / "data" / "processed"
GT = PROJECT / "data" / "ground_truth"


@dataclass(frozen=True)
class Dataset:
    path: Path
    kind: str            # "operational" | "evaluation"
    description: str


DATASETS: dict[str, Dataset] = {
    # operational (no ground truth)
    "unified": Dataset(PROC / "unified_observations.parquet", "operational", "Five sources joined per observation"),
    "features": Dataset(PROC / "features.parquet", "operational", "As-of features at each decision time"),
    "assessments": Dataset(PROC / "risk_assessments.parquet", "operational", "Risk fusion assessments"),
    "explanations": Dataset(PROC / "risk_explanations.parquet", "operational", "Per-component risk explanations"),
    "isolation_forest": Dataset(PROC / "isolation_forest_scores.parquet", "operational", "Isolation Forest scores"),
    "forecasts": Dataset(PROC / "forecast_results.parquet", "operational", "24 h forecasts (no targets)"),
    "episodes": Dataset(PROC / "alert_episodes.parquet", "operational",
                        "Alert episodes (event-matching columns stripped on operational pages)"),
    # evaluation only (contain labels, targets or event matches)
    "backtest_results": Dataset(PROC / "backtest_results.parquet", "evaluation", "Backtest predictions + labels"),
    "backtest_metrics": Dataset(PROC / "backtest_metrics.parquet", "evaluation", "Backtest metrics (Phase 4A)"),
    "event_results": Dataset(PROC / "event_detection_results.parquet", "evaluation", "Event detection results"),
    "episodes_eval": Dataset(PROC / "alert_episodes.parquet", "evaluation", "Alert episodes with event matching"),
    "forecast_evaluation": Dataset(PROC / "forecast_evaluation.parquet", "evaluation", "Forecast targets + scores"),
    "model_summary": Dataset(PROC / "model_comparison_summary.parquet", "evaluation", "Audit model comparison"),
    "events": Dataset(GT / "heat_risk_events.csv", "evaluation", "Ground-truth events"),
    "dq_log": Dataset(GT / "data_quality_log.csv", "evaluation", "Injected data-quality issues (simulation truth)"),
}

# Columns that reveal ground truth / evaluation outcomes; must never reach operational pages.
LABEL_COLUMNS = {"label", "label_name", "event_id", "target", "target_event_id", "matched_event_id",
                 "match_status", "overlap_hours", "event_type", "onset_timestamp", "escalation_timestamp",
                 "end_timestamp", "issue_id", "issue_type", "lr_flag", "persistence_any", "persistence_health"}
STRIP_FOR_OPERATIONAL = {"episodes": {"matched_event_id", "match_status", "overlap_hours"}}


class DataUnavailable(FileNotFoundError):
    """A required pipeline output is missing; message tells the user which command creates it."""


class LeakageError(RuntimeError):
    pass


PRODUCER = {
    "unified": "python -m src.features.build", "features": "python -m src.features.build",
    "assessments": "python -m src.detection.risk_fusion", "explanations": "python -m src.detection.risk_fusion",
    "isolation_forest": "python -m src.detection.isolation_forest", "forecasts": "python -m src.forecasting.forecast",
    "forecast_evaluation": "python -m src.forecasting.forecast", "episodes": "python -m src.evaluation.backtest",
    "episodes_eval": "python -m src.evaluation.backtest", "backtest_results": "python -m src.evaluation.backtest",
    "backtest_metrics": "python -m src.evaluation.backtest", "event_results": "python -m src.evaluation.backtest",
    "model_summary": "python -m src.evaluation.audit", "events": "python -m src.data_generation.generate",
    "dq_log": "python -m src.data_generation.generate",
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _read_cached(path_str: str, mtime: float) -> pd.DataFrame:      # mtime is part of the cache key
    return _read(Path(path_str))


try:                                                             # Streamlit cache when running in the app
    import streamlit as st
    _read_cached = st.cache_data(show_spinner="Loading saved pipeline outputs ...")(_read_cached)
except Exception:                                                # pragma: no cover - tests without runtime
    pass


def load(name: str) -> pd.DataFrame:
    """Load any registered dataset (copy, safe to mutate). Raises DataUnavailable if the file is missing."""
    ds = DATASETS[name]
    if not ds.path.exists():
        raise DataUnavailable(f"`{_rel(ds.path)}` not found. Generate it with "
                              f"`{PRODUCER.get(name, 'the pipeline')}`.")
    df = _read_cached(str(ds.path), ds.path.stat().st_mtime).copy()
    for c in df.columns:
        if c.endswith(("_timestamp", "_time", "_decision", "_start", "_end", "observation_date", "_at")) \
                and df[c].dtype == object:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_operational(name: str) -> pd.DataFrame:
    """Load a dataset for an operational page. Guarantees no ground-truth/evaluation columns."""
    ds = DATASETS[name]
    if ds.kind != "operational":
        raise LeakageError(f"{name} is an evaluation dataset and cannot be used on operational pages")
    df = load(name)
    df = df.drop(columns=[c for c in STRIP_FOR_OPERATIONAL.get(name, ()) if c in df.columns])
    leaked = LABEL_COLUMNS & set(df.columns)
    if leaked:
        raise LeakageError(f"{name} contains evaluation columns {sorted(leaked)}")
    return df


def load_evaluation(name: str) -> pd.DataFrame:
    return load(name)


def config_yaml(name: str) -> dict:
    with open(config.CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def zone_registry() -> pd.DataFrame:
    """Zone metadata from the simulation config (names, environment, intensity)."""
    sim = config.load_simulation()
    rows = []
    for site_id, zone_id, z in config.iter_zones(sim):
        rows.append({"site_id": site_id, "site_name": sim["sites"][site_id]["name"], "zone_id": zone_id,
                     "zone_name": z["name"], "environment": z["environment"].replace("_", " "),
                     "radiant_heat": bool(z["radiant_heat"]), "work_intensity": z["work_intensity"]})
    return pd.DataFrame(rows)


def status() -> pd.DataFrame:
    """Availability of every registered output (for the sidebar / error pages)."""
    return pd.DataFrame([{"dataset": k, "kind": v.kind, "file": _rel(v.path),
                          "available": v.path.exists()} for k, v in DATASETS.items()])


def clear_cache() -> None:
    try:
        _read_cached.clear()
    except AttributeError:
        pass
