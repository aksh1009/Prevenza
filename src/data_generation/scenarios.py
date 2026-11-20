"""Scenario schedule -> heat forcing, ground-truth events and observation labels.

Scenario A (normal operations) is the default everywhere. Scenarios B/C/D are read from
config/simulation.yaml so dates and locations are configurable. Ground truth is derived ONLY from
this schedule, never from generated outcomes or model output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schemas import LABELS

ESCALATING_TYPES = ("gradual_heatwave_escalation", "heat_escalation_clinical_gap")
CONTROL_TYPES = ("isolated_temperature_spike",)
DECLINE_DAYS = 2  # forcing tapers over the last N days of an escalating event


def _zone_site(zone_id: str) -> str:
    return zone_id.split("-")[0]


def event_windows(sim: dict, shift_starts: dict[str, int]) -> list[dict]:
    """Resolve each configured scenario into timestamps (site-local)."""
    out = []
    for sc in sim.get("scenarios", []):
        shifts = sc.get("shifts", list(sim["shifts"]))
        onset_d = pd.Timestamp(sc["onset_date"])
        end_d = pd.Timestamp(sc["end_date"])
        first_shift = min(shifts, key=lambda s: shift_starts[s])
        last_shift = max(shifts, key=lambda s: shift_starts[s])
        onset_ts = onset_d + pd.Timedelta(hours=shift_starts[first_shift])
        end_ts = end_d + pd.Timedelta(hours=shift_starts[last_shift] + 12)
        esc = sc.get("escalation_date")
        esc_ts = pd.Timestamp(esc) + pd.Timedelta(hours=shift_starts[first_shift]) if esc else pd.NaT
        if sc["type"] in ESCALATING_TYPES and pd.isna(esc_ts):
            raise ValueError(f"{sc['id']}: escalating scenario needs escalation_date")
        if sc["type"] in CONTROL_TYPES and not pd.isna(esc_ts):
            raise ValueError(f"{sc['id']}: control scenario must not have escalation_date")
        out.append({**sc, "shifts": shifts, "onset_ts": onset_ts, "escalation_ts": esc_ts, "end_ts": end_ts})
    return out


def forcing(grid: pd.DataFrame, windows: list[dict]) -> pd.DataFrame:
    """Per-observation forcing: temp/RH/exposure deltas (before zone damping) and the event ramp (0..1)."""
    f = pd.DataFrame(0.0, index=grid.index, columns=["temp_delta", "rh_delta", "exposure_delta", "ramp"])
    s = grid["shift_start"]
    for w in windows:
        in_zone = grid["zone_id"].isin(w["zones"]) & grid["shift"].isin(w["shifts"])
        in_win = in_zone & (s >= w["onset_ts"]) & (s < w["end_ts"])
        if not in_win.any():
            continue
        if w["type"] in ESCALATING_TYPES:
            ramp_len = (w["escalation_ts"] - w["onset_ts"]) / pd.Timedelta(hours=1)
            pos = (s - w["onset_ts"]) / pd.Timedelta(hours=1)
            ramp = np.clip(0.15 + 0.85 * pos / ramp_len, 0, 1)            # emerging: 0.15 -> 1
            decline_start = w["end_ts"] - pd.Timedelta(days=DECLINE_DAYS)
            dpos = ((s - decline_start) / (w["end_ts"] - decline_start)).clip(0, 1)
            ramp = np.where(s >= decline_start, 1 - 0.75 * dpos, ramp)    # taper to 0.25
        else:
            ramp = np.ones(len(grid))
        r = pd.Series(ramp, index=grid.index).where(in_win, 0.0)
        f["temp_delta"] += r * w["temp_delta_c"]
        f["rh_delta"] += r * w["rh_delta_pct"]
        f["exposure_delta"] += r * w.get("exposure_delta_hours", 0.0)
        f["ramp"] = np.maximum(f["ramp"], r)
    return f


def build_events(windows: list[dict]) -> pd.DataFrame:
    rows = []
    for w in windows:
        for z in w["zones"]:
            rows.append({
                "event_id": w["id"], "site_id": _zone_site(z), "zone_id": z, "event_type": w["type"],
                "onset_timestamp": w["onset_ts"], "escalation_timestamp": w["escalation_ts"],
                "end_timestamp": w["end_ts"], "severity": w["severity"], "description": w["description"],
            })
    return pd.DataFrame(rows)


def build_labels(grid: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Labels from the event schedule only: [onset, escalation) -> 1, [escalation, end) -> 2, else 0."""
    label = pd.Series(0, index=grid.index)
    event_id = pd.Series(pd.NA, index=grid.index, dtype="string")
    s = grid["shift_start"]
    for ev in events.itertuples():
        if ev.event_type not in ESCALATING_TYPES:
            continue                                   # negative controls stay 0
        z = grid["zone_id"] == ev.zone_id
        emerging = z & (s >= ev.onset_timestamp) & (s < ev.escalation_timestamp)
        escalating = z & (s >= ev.escalation_timestamp) & (s < ev.end_timestamp)
        for mask, lab in ((emerging, 1), (escalating, 2)):
            upd = mask & (label < lab)
            label[upd] = lab
            event_id[upd] = ev.event_id
    out = grid[["observation_date", "site_id", "zone_id", "shift"]].copy()
    out["label"] = label.astype("Int64")
    out["label_name"] = label.map(LABELS)
    out["event_id"] = event_id
    return out
