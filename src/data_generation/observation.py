"""Turn true state into the five reported source datasets.

Adds measurement noise, sensor anomalies, random missing fields, reporting delays, feed delays and
whole-period reporting gaps. Missing values are always NA, never 0. Occurrence time (recorded_at) and
reporting time (reporting_timestamp) are kept separate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import schemas

from .heat_index import heat_index_c

KEY = list(schemas.OBSERVATION_KEY)


class QualityLog:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, source, issue_type, zone_id, start, end, details):
        self.rows.append({"source_dataset": source, "issue_type": issue_type, "site_id": zone_id.split("-")[0],
                          "zone_id": zone_id, "start_timestamp": start, "end_timestamp": end, "details": details})

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=["source_dataset", "issue_type", "site_id", "zone_id",
                                              "start_timestamp", "end_timestamp", "details"])
        df = df.sort_values(["start_timestamp", "source_dataset", "zone_id"], kind="stable").reset_index(drop=True)
        # one issue_id per (source, type, start, end, details) group, spanning its zones
        grp = df.groupby(["source_dataset", "issue_type", "start_timestamp", "end_timestamp", "details"],
                         sort=False).ngroup()
        df.insert(0, "issue_id", [f"DQ-{g + 1:03d}" for g in grp])
        return df


def _date_mask(grid, zones, start_date, end_date):
    d = grid["observation_date"]
    return grid["zone_id"].isin(zones) & (d >= pd.Timestamp(start_date)) & (d <= pd.Timestamp(end_date))


def _window_bounds(grid, mask):
    return grid.loc[mask, "shift_start"].min(), grid.loc[mask, "shift_end"].max()


def _delays(n, spec, rng) -> np.ndarray:
    if spec["dist"] == "uniform":
        d = rng.uniform(spec["low"], spec["high"], n)
    elif spec["dist"] == "lognormal":
        d = rng.lognormal(np.log(spec["median"]), spec["sigma"], n)
    else:
        raise ValueError(spec["dist"])
    return np.clip(d, spec["low"], spec["high"])


def _anomaly_candidates(grid, sim, windows, gap_mask) -> np.ndarray:
    excl = gap_mask.copy()
    for w in windows:
        excl |= _date_mask(grid, w["zones"], pd.Timestamp(w["onset_date"]) - pd.Timedelta(days=1),
                           pd.Timestamp(w["end_date"]) + pd.Timedelta(days=1))
    return ~excl.to_numpy()


def build_sources(grid, sim, windows, env, health, rng) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    dq = sim["data_quality"]
    log = QualityLog()
    n = len(grid)
    base = grid[KEY].copy()
    base["recorded_at"] = grid["shift_end"]

    # ---- reporting gaps (incl. Scenario D clinical gap) ---------------------------------------
    gaps: dict[str, pd.Series] = {s: pd.Series(False, index=grid.index) for s in schemas.SOURCE_DATASETS}
    for g in dq.get("reporting_gaps", []):
        m = _date_mask(grid, g["zones"], g["start_date"], g["end_date"])
        gaps[g["source"]] |= m
        for z in g["zones"]:
            zm = m & (grid["zone_id"] == z)
            log.add(g["source"], "reporting_gap", z, *_window_bounds(grid, zm),
                    f"not reported {g['start_date']}..{g['end_date']}")
    for w in windows:
        cg = w.get("clinical_gap")
        if cg:
            m = _date_mask(grid, w["zones"], cg["start_date"], cg["end_date"])
            gaps["clinical_data"] |= m
            for z in w["zones"]:
                zm = m & (grid["zone_id"] == z)
                log.add("clinical_data", "reporting_gap", z, *_window_bounds(grid, zm),
                        f"scenario {w['id']}: clinic not reporting {cg['start_date']}..{cg['end_date']}")

    # ---- environmental: noise -> anomalies -> random missing ----------------------------------
    noise = dq["measurement_noise"]
    temp = env["true_temp_c"].to_numpy() + rng.normal(0, noise["temperature_sd_c"], n)
    rh = np.clip(env["true_rh_pct"].to_numpy() + rng.normal(0, noise["rh_sd_pct"], n), 0, 100)

    an = dq["anomalies"]
    cand = _anomaly_candidates(grid, sim, windows, gaps["environmental_data"])
    zone = grid["zone_id"].to_numpy()
    spike_idx = rng.choice(np.flatnonzero(cand), size=an["sensor_spikes"], replace=False)
    for i in spike_idx:
        delta = rng.uniform(*an["spike_delta_c"])
        temp[i] += delta
        log.add("environmental_data", "sensor_spike", zone[i], grid.at[i, "shift_start"], grid.at[i, "shift_end"],
                f"temperature +{delta:.1f} C single shift")
    L = an["stuck_length_shifts"]
    used = set(spike_idx.tolist())
    starts = [i for i in np.flatnonzero(cand)
              if i + L < n and zone[i + L] == zone[i] and cand[i:i + L + 1].all()
              and not used.intersection(range(i, i + L + 1))]
    for i in rng.choice(starts, size=an["stuck_sensors"], replace=False):
        temp[i + 1:i + L + 1] = temp[i]
        rh[i + 1:i + L + 1] = rh[i]
        used.update(range(i, i + L + 1))
        log.add("environmental_data", "stuck_sensor", zone[i], grid.at[i + 1, "shift_start"],
                grid.at[i + L, "shift_end"], f"temperature/humidity frozen at previous value for {L} shifts")

    p = dq["random_missing_rate"]["environmental_data"]
    temp[rng.random(n) < p] = np.nan
    rh[rng.random(n) < p] = np.nan
    temp, rh = np.round(temp, 1), np.round(rh, 1)
    envd = base.copy()
    envd["temperature_celsius"] = temp
    envd["relative_humidity"] = rh

    # ---- workforce ------------------------------------------------------------------------------
    p = dq["random_missing_rate"]["workforce_data"]
    wcount = health["present"].astype(float).to_numpy()
    avg = np.clip(health["true_exposure_hours"].to_numpy() + rng.normal(0, noise["exposure_hours_sd"], n), 0, 12)
    wcount[rng.random(n) < p] = np.nan
    avg[rng.random(n) < p] = np.nan
    wfd = base.copy()
    wfd["workforce_count"] = wcount
    wfd["average_exposure_hours"] = np.round(avg, 2)

    # ---- clinical (whole record group missing together so severity reconciles) -----------------
    p = dq["random_missing_rate"]["clinical_data"]
    cl = health[["visits_mild", "visits_moderate", "visits_severe"]].astype(float).to_numpy()
    cl[rng.random(n) < p] = np.nan
    cld = base.copy()
    cld["heat_visits_mild"], cld["heat_visits_moderate"], cld["heat_visits_severe"] = cl.T

    # ---- absenteeism ------------------------------------------------------------------------------
    p = dq["random_missing_rate"]["absenteeism_data"]
    rost = health["rostered"].astype(float).to_numpy()
    absn = health["absent"].astype(float).to_numpy()
    rost[rng.random(n) < p] = np.nan
    absn[rng.random(n) < p] = np.nan
    abd = base.copy()
    abd["workforce_count"] = rost
    abd["absent_workers"] = absn

    # ---- incidents (whole group) ------------------------------------------------------------------
    p = dq["random_missing_rate"]["incident_data"]
    inc = health[["inc_minor", "inc_moderate", "inc_severe"]].astype(float).to_numpy()
    inc[rng.random(n) < p] = np.nan
    ind = base.copy()
    ind["minor_incidents"], ind["moderate_incidents"], ind["severe_incidents"] = inc.T

    frames = {"environmental_data": envd, "workforce_data": wfd, "clinical_data": cld,
              "absenteeism_data": abd, "incident_data": ind}

    # ---- apply gaps (values -> NA), reporting timestamps, feed delays --------------------------
    for name, df in frames.items():
        value_cols = [c for c in df.columns if c not in KEY + ["recorded_at"]]
        gap = gaps[name].to_numpy()
        df.loc[gap, value_cols] = np.nan
        delay_h = _delays(n, dq["reporting_delay_hours"][name], rng)
        for fd in dq.get("feed_delays", []):
            if fd["source"] == name:
                m = _date_mask(grid, fd["zones"], fd["start_date"], fd["end_date"]).to_numpy()
                delay_h = delay_h + m * fd["extra_hours"]
                for z in fd["zones"]:
                    zm = pd.Series(m, index=grid.index) & (grid["zone_id"] == z)
                    log.add(name, "feed_delay", z, *_window_bounds(grid, zm),
                            f"+{fd['extra_hours']} h reporting delay {fd['start_date']}..{fd['end_date']}")
        rep = (df["recorded_at"] + pd.to_timedelta(delay_h, unit="h")).dt.round("min")
        df["reporting_timestamp"] = rep.where(~gap, pd.NaT)
        df["reporting_status"] = np.where(gap, "not_reported", "reported")

    # ---- derived fields (computed AFTER missingness so NA propagates, never 0) ----------------
    envd["heat_index_celsius"] = np.round(heat_index_c(envd["temperature_celsius"], envd["relative_humidity"]), 1)
    wfd["total_exposure_hours"] = np.round(wfd["workforce_count"] * wfd["average_exposure_hours"], 1)
    cld["heat_related_visits"] = cld[["heat_visits_mild", "heat_visits_moderate", "heat_visits_severe"]].sum(
        axis=1, min_count=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = abd["absent_workers"] / abd["workforce_count"]
    abd["absence_rate"] = np.round(rate.where(abd["workforce_count"] > 0), 4)
    abd["lost_workdays"] = abd["absent_workers"] * 1.0
    ind["heat_incident_count"] = ind[["minor_incidents", "moderate_incidents", "severe_incidents"]].sum(
        axis=1, min_count=3)

    out = {}
    for name, df in frames.items():
        c = schemas.SOURCE_CONTRACTS[name]
        df = schemas.coerce(df[c.columns], c)
        out[name] = df.sort_values(KEY, key=_key_sort).reset_index(drop=True)
    return out, log.frame()


def _key_sort(col: pd.Series):
    """Sort shifts day before night; other key columns natural order."""
    if col.name == "shift":
        return col.map({"day": 0, "night": 1})
    return col
