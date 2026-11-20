"""True workforce exposure and lagged health response (aggregated counts only).

Mechanism (synthetic assumption, documented in docs/simulation_summary.md):
  strain_t = max(0, heat_index_t - threshold) * intensity_factor * exposure_hours_t / base_exposure
  lagged_strain_t = sum_k kernel[k] * strain_{t-k}      (per zone, over consecutive shifts)
Clinical visits respond with a short lag, absence with a longer lag, incidents near-immediately.
Outcomes are drawn from the *true* environment so that sensor noise/missingness do not alter health.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _lagged(x: np.ndarray, zone: np.ndarray, kernel: list[float]) -> np.ndarray:
    """Causal lag filter applied independently within each zone (grid sorted by zone, time)."""
    out = np.zeros_like(x)
    k = np.asarray(kernel, dtype=float)
    for z in np.unique(zone):
        idx = np.flatnonzero(zone == z)
        out[idx] = np.convolve(x[idx], k)[: len(idx)]
    return out


def _split_severity(total: np.ndarray, level: np.ndarray, p_base, p_high, rng) -> np.ndarray:
    """Multinomial split of counts into (minor/mild, moderate, severe), shifting toward severe with strain."""
    pb, ph = np.asarray(p_base), np.asarray(p_high)
    out = np.zeros((len(total), 3), dtype=int)
    for i, (n, a) in enumerate(zip(total, level)):
        if n > 0:
            p = pb + (ph - pb) * a
            out[i] = rng.multinomial(int(n), p / p.sum())
    return out


def simulate_workforce_and_health(grid: pd.DataFrame, sim: dict, env: pd.DataFrame, force: pd.DataFrame,
                                  rng: np.random.Generator) -> pd.DataFrame:
    wf, hm = sim["workforce"], sim["health_model"]
    n = len(grid)
    zone = grid["zone_id"].to_numpy()

    lo, hi = wf["roster_variation"]
    rostered = np.round(grid["headcount"].to_numpy() * rng.uniform(lo, hi, n)).astype(int)
    be = wf["base_exposure_hours"]
    exposure = np.clip(rng.normal(be["mean"], be["sd"], n) + force["exposure_delta"].to_numpy(), 0, be["max"])

    intensity = grid["work_intensity"].map(wf["intensity_factor"]).to_numpy()
    excess = np.clip(env["true_heat_index_c"].to_numpy() - hm["heat_index_threshold_c"], 0, None)
    strain = excess * intensity * exposure / be["mean"]

    l_clin = _lagged(strain, zone, hm["kernel_clinical"])
    l_abs = _lagged(strain, zone, hm["kernel_absence"])
    l_inc = _lagged(strain, zone, hm["kernel_incident"])

    a = hm["absence"]
    p_abs = np.clip(a["base_rate"] + a["strain_coef"] * l_abs, 0, a["max_rate"])
    absent = rng.binomial(rostered, p_abs)
    present = rostered - absent

    c = hm["clinical"]
    visits = rng.poisson(present * (c["base_rate"] + c["strain_coef"] * l_clin))
    sev_level_c = np.clip(l_clin / hm["high_strain_level"], 0, 1)
    vis_split = _split_severity(visits, sev_level_c, c["p_severity_base"], c["p_severity_high"], rng)

    i = hm["incident"]
    incidents = rng.poisson(present * (i["base_rate"] + i["strain_coef"] * l_inc))
    sev_level_i = np.clip(l_inc / hm["high_strain_level"], 0, 1)
    inc_split = _split_severity(incidents, sev_level_i, i["p_severity_base"], i["p_severity_high"], rng)

    return pd.DataFrame({
        "rostered": rostered, "absent": absent, "present": present, "true_exposure_hours": exposure,
        "strain": strain,
        "visits_mild": vis_split[:, 0], "visits_moderate": vis_split[:, 1], "visits_severe": vis_split[:, 2],
        "inc_minor": inc_split[:, 0], "inc_moderate": inc_split[:, 1], "inc_severe": inc_split[:, 2],
    }, index=grid.index)
