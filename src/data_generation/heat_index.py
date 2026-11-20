"""US National Weather Service heat index approximation (NOT WBGT).

Reference algorithm: NWS Weather Prediction Center, "The Heat Index Equation"
(https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml):
  1. Steadman simple formula: HI = 0.5 * (T + 61 + (T - 68) * 1.2 + RH * 0.094)   [T in F]
  2. If the average of HI and T is >= 80 F, use the Rothfusz regression instead, with
     - a low-humidity adjustment (RH < 13 %, 80 <= T <= 112 F), and
     - a high-humidity adjustment (RH > 85 %, 80 <= T <= 87 F).
Heat index accounts for air temperature and humidity only; it ignores radiant heat, wind and
workload, which WBGT would capture. It is used here as a documented approximation of heat stress.
"""
from __future__ import annotations

import numpy as np


def c_to_f(c):
    return np.asarray(c, dtype=float) * 9.0 / 5.0 + 32.0


def f_to_c(f):
    return (np.asarray(f, dtype=float) - 32.0) * 5.0 / 9.0


def heat_index_c(temp_c, rh_pct) -> np.ndarray:
    """Vectorised NWS heat index in degC. NaN in either input -> NaN output."""
    t = c_to_f(temp_c)
    rh = np.asarray(rh_pct, dtype=float)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    roth = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
            - 0.00683783 * t ** 2 - 0.05481717 * rh ** 2 + 0.00122874 * t ** 2 * rh
            + 0.00085282 * t * rh ** 2 - 0.00000199 * t ** 2 * rh ** 2)
    with np.errstate(invalid="ignore"):
        low = (rh < 13) & (t >= 80) & (t <= 112)
        adj_low = ((13 - rh) / 4.0) * np.sqrt(np.clip((17 - np.abs(t - 95.0)) / 17.0, 0, None))
        high = (rh > 85) & (t >= 80) & (t <= 87)
        adj_high = ((rh - 85) / 10.0) * ((87 - t) / 5.0)
        roth = roth - np.where(low, adj_low, 0.0) + np.where(high, adj_high, 0.0)
        use_roth = ((simple + t) / 2.0) >= 80
    hi = np.where(use_roth, roth, simple)
    hi = np.where(np.isnan(t) | np.isnan(rh), np.nan, hi)
    return f_to_c(hi)
