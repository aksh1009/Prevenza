"""Unified observation table: five sources joined one-to-one on the observation key.

    python -m src.preprocessing.integrate     -> data/processed/unified_observations.parquet

This table is the RETROSPECTIVE record (every value as eventually reported, with its provenance).
It is NOT a model input by itself: model features must be built as-of a decision time
(src/features/build.py), using a source value only after its reporting_timestamp.
"""
from __future__ import annotations

import pandas as pd

from src import config, schemas

from .load import load_sources

KEY = list(schemas.OBSERVATION_KEY)
PREFIX = {"environmental_data": "env", "workforce_data": "wf", "clinical_data": "clin",
          "absenteeism_data": "abs", "incident_data": "inc"}
PROVENANCE = ("recorded_at", "reporting_timestamp", "reporting_status")
UNIFIED_FILE = "unified_observations.parquet"


class JoinError(ValueError):
    pass


def _prefixed(name: str, df: pd.DataFrame) -> pd.DataFrame:
    p = PREFIX[name]
    out = df.rename(columns={c: f"{p}_{c}" for c in df.columns if c not in KEY})
    out[f"{p}_reporting_delay_hours"] = (
        (out[f"{p}_reporting_timestamp"] - out[f"{p}_recorded_at"]).dt.total_seconds() / 3600)
    return out


def check_keys(sources: dict[str, pd.DataFrame], expected_rows: int | None = None) -> None:
    """Each source must have unique keys and the identical key set (strict one-to-one join)."""
    ref_name, ref = next(iter(sources.items()))
    ref_keys = set(map(tuple, ref[KEY].astype(str).to_numpy()))
    for name, df in sources.items():
        if df[KEY].isna().any().any():
            raise JoinError(f"{name}: null key values")
        dup = df.duplicated(KEY)
        if dup.any():
            raise JoinError(f"{name}: {int(dup.sum())} duplicate keys")
        keys = set(map(tuple, df[KEY].astype(str).to_numpy()))
        if keys != ref_keys:
            raise JoinError(f"{name}: key set differs from {ref_name} "
                            f"(+{len(keys - ref_keys)} / -{len(ref_keys - keys)})")
        if expected_rows is not None and len(df) != expected_rows:
            raise JoinError(f"{name}: {len(df)} rows, expected {expected_rows}")


def build_unified(sources: dict[str, pd.DataFrame], expected_rows: int | None = None) -> pd.DataFrame:
    check_keys(sources, expected_rows)
    out = None
    for name in schemas.SOURCE_DATASETS:
        part = _prefixed(name, sources[name])
        out = part if out is None else out.merge(part, on=KEY, how="inner", validate="one_to_one")
    if len(out) != len(sources[schemas.SOURCE_DATASETS[0]]):
        raise JoinError("row count changed during join")
    # shift window (all sources share recorded_at = shift end; kept per source as provenance)
    out.insert(4, "shift_end", out["env_recorded_at"])
    out.insert(4, "shift_start", out["shift_end"] - pd.Timedelta(hours=12))
    shift_order = out["shift"].map({"day": 0, "night": 1})
    out = out.assign(_o=shift_order).sort_values(["observation_date", "site_id", "zone_id", "_o"]) \
        .drop(columns="_o").reset_index(drop=True)
    return out


def unified_path():
    return config.path_for("processed") / UNIFIED_FILE


def main():
    sim = config.load_simulation()
    unified = build_unified(load_sources(), config.expected_observation_count(sim))
    p = unified_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(p, index=False)
    print(f"unified_observations: {unified.shape[0]} rows x {unified.shape[1]} columns -> "
          f"{p.relative_to(config.PROJECT_ROOT)}")
    return unified


if __name__ == "__main__":
    main()
