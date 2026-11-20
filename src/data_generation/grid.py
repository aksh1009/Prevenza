"""Observation grid: one row per observation_date x site_id x zone_id x shift."""
from __future__ import annotations

import pandas as pd

from src.config import iter_zones, load_settings

SHIFT_HOURS = 12


def shift_start_hours() -> dict[str, int]:
    shifts = load_settings()["observation_key"]["shifts"]
    return {name: int(spec["start"].split(":")[0]) for name, spec in shifts.items()}


def build_grid(sim: dict) -> pd.DataFrame:
    """Full observation grid with zone attributes and shift window timestamps.

    Sorted by zone_id then shift_start, so per-zone time-ordered operations (lags) are contiguous.
    """
    starts = shift_start_hours()
    dates = pd.date_range(sim["start_date"], periods=sim["n_days"], freq="D")
    rows = []
    for site_id, zone_id, z in iter_zones(sim):
        for day_index, d in enumerate(dates):
            for shift in sim["shifts"]:
                start = d + pd.Timedelta(hours=starts[shift])
                rows.append({
                    "observation_date": d, "site_id": site_id, "zone_id": zone_id, "shift": shift,
                    "shift_start": start, "shift_end": start + pd.Timedelta(hours=SHIFT_HOURS),
                    "day_index": day_index, "environment": z["environment"],
                    "radiant_heat": bool(z["radiant_heat"]), "work_intensity": z["work_intensity"],
                    "headcount": int(z["headcount"][shift]),
                })
    grid = pd.DataFrame(rows).sort_values(["zone_id", "shift_start"]).reset_index(drop=True)
    return grid
