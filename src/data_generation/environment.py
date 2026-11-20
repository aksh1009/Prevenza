"""True (pre-measurement) zone thermal environment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .heat_index import heat_index_c


def _site_weather(sim: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Site-level daily AR(1) temperature anomaly."""
    c = sim["climate"]
    n = sim["n_days"]
    out = {}
    for site_id in sim["sites"]:
        a = np.zeros(n)
        a[0] = rng.normal(0, c["weather_ar1_sd_c"])
        for t in range(1, n):
            a[t] = c["weather_ar1_phi"] * a[t - 1] + rng.normal(0, c["weather_ar1_sd_c"])
        out[site_id] = a
    return out


def simulate_environment(grid: pd.DataFrame, sim: dict, force: pd.DataFrame,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Return true_temp_c, true_rh_pct, true_heat_index_c aligned to ``grid``."""
    c = sim["climate"]
    zenv = sim["zone_environment"]
    n = sim["n_days"]
    weather = _site_weather(sim, rng)
    # Daily RH anomaly per site (hot anomalies slightly drier).
    rh_anom = {s: rng.normal(0, c["rh_daily_sd_pct"], n) - 0.8 * weather[s] for s in sim["sites"]}

    day = grid["day_index"].to_numpy()
    seasonal = c["seasonal_amplitude_c"] * np.sin(np.pi * day / max(n - 1, 1))
    site_temp = grid["site_id"].map({s: v["climate"]["day_temp_c"] for s, v in sim["sites"].items()}).to_numpy()
    site_rh = grid["site_id"].map({s: v["climate"]["rh_day_pct"] for s, v in sim["sites"].items()}).to_numpy()
    w = np.array([weather[s][d] for s, d in zip(grid["site_id"], day)])
    rha = np.array([rh_anom[s][d] for s, d in zip(grid["site_id"], day)])
    night = (grid["shift"] == "night").to_numpy()

    amb_t = site_temp + seasonal + w - night * c["night_temp_drop_c"]
    amb_rh = site_rh + rha + night * c["night_rh_rise_pct"]

    env = grid["environment"].to_numpy()
    temp = np.empty(len(grid))
    rh = np.empty(len(grid))
    factor = np.empty(len(grid))
    for etype, p in zenv.items():
        m = env == etype
        if etype == "indoor_conditioned":
            temp[m] = p["setpoint_c"] + p["coupling"] * (amb_t[m] - p["setpoint_c"])
            rh[m] = p["rh_pct"] + 0.2 * (amb_rh[m] - p["rh_pct"])
        else:
            temp[m] = amb_t[m] + p["temp_offset_c"]
            rh[m] = amb_rh[m] + p["rh_offset_pct"]
        factor[m] = p["forcing_factor"]
    temp += grid["radiant_heat"].to_numpy() * sim["radiant_heat_offset_c"]
    temp += factor * force["temp_delta"].to_numpy()
    rh += factor * force["rh_delta"].to_numpy()
    temp += rng.normal(0, 0.5, len(grid))          # zone micro-variation (true, not sensor noise)
    rh = np.clip(rh + rng.normal(0, 1.5, len(grid)), 5, 100)

    return pd.DataFrame({"true_temp_c": temp, "true_rh_pct": rh,
                         "true_heat_index_c": heat_index_c(temp, rh)}, index=grid.index)
