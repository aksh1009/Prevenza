"""As-of feature engineering.

    python -m src.features.build    -> data/processed/unified_observations.parquet
                                       data/processed/features.parquet
                                       docs/feature_dictionary.md

Leakage rule (the only way data enters a feature):
    For an observation t of series (zone_id, shift), decision_time(t) = shift end + decision_lag_hours.
    A source value of observation s of the same series is usable iff
        s <= t  AND  reporting_timestamp(s) <= decision_time(t).
Every statistic (current value, lags, rolling means, baseline median/MAD, trend) is computed from that
as-of view, independently per site-zone-shift series. Ground truth is never read.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import config, schemas
from src.preprocessing.integrate import build_unified, unified_path
from src.preprocessing.load import load_sources

from .registry import (BASE_FIELDS, CORE_FIELDS, SIGNALS, SOURCES, feature_columns, feature_registry,
                       load_feature_config, stat_specs)

KEY = list(schemas.OBSERVATION_KEY)
SERIES = ["site_id", "zone_id", "shift"]
FEATURES_FILE = "features.parquet"
ID_COLUMNS = KEY + ["decision_time"]
_NS_PER_H = 3_600_000_000_000


def _nan_reduce(fn, x, mask, min_count):
    """Apply a nan-aware reduction over masked row entries; NA if fewer than min_count known values."""
    xm = np.where(mask, x, np.nan)
    cnt = np.sum(~np.isnan(xm), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = fn(xm, axis=1)
    return np.where(cnt >= min_count, out, np.nan), xm


def _lag(V, k):
    n = V.shape[0]
    out = np.full(n, np.nan)
    if k < n:
        out[k:] = V[np.arange(k, n), np.arange(n - k)]
    return out


def _last_index(M):
    """Per row, column index of the last True entry (or -1)."""
    n = M.shape[1]
    any_ = M.any(axis=1)
    return np.where(any_, n - 1 - np.argmax(M[:, ::-1], axis=1), -1)


def _slope(V, mask, min_points):
    D = np.arange(V.shape[0])[:, None] - np.arange(V.shape[1])[None, :]
    x = np.where(mask & ~np.isnan(V), -D.astype(float), np.nan)
    y = np.where(np.isnan(x), np.nan, V)
    cnt = np.sum(~np.isnan(x), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xm, ym = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
        sxx = np.nansum((x - xm[:, None]) ** 2, axis=1)
        sxy = np.nansum((x - xm[:, None]) * (y - ym[:, None]), axis=1)
    return np.where((cnt >= min_points) & (sxx > 0), sxy / np.where(sxx > 0, sxx, 1), np.nan)


def _cusum(V, med, scale, mask, c):
    """Upper CUSUM per row over known values in the window (oldest -> newest), vs that row's baseline."""
    n = V.shape[0]
    out = np.full(n, np.nan)
    Z = np.minimum((V - med[:, None]) / scale[:, None], c["z_cap"])
    for t in range(n):
        if np.isnan(med[t]):
            continue
        z = Z[t, mask[t]]
        z = z[~np.isnan(z)]
        if len(z) == 0:
            continue
        s = 0.0
        for v in z:                                    # columns ascending = chronological
            s = max(0.0, s + v - c["k"])
        out[t] = s
    return out


def series_features(g: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Features for one site-zone-shift series (rows sorted by observation_date)."""
    n = len(g)
    t_idx = np.arange(n)
    D = t_idx[:, None] - t_idx[None, :]                       # D[t, s] = t - s
    past = D >= 0
    dt = g["decision_time"].to_numpy("datetime64[ns]").astype("int64")
    out: dict[str, np.ndarray] = {}

    # ---- as-of availability per source ---------------------------------------------------------
    avail, rec_ns, rep_ns = {}, {}, {}
    for p in SOURCES:
        rep = g[f"{p}_reporting_timestamp"]
        rep_i = np.where(rep.isna(), np.iinfo("int64").max, rep.to_numpy("datetime64[ns]").astype("int64"))
        rec_ns[p] = g[f"{p}_recorded_at"].to_numpy("datetime64[ns]").astype("int64")
        rep_ns[p] = rep_i
        avail[p] = past & (rep_i[None, :] <= dt[:, None])

    L = cfg["windows"]["long"]["length"]
    win_long = past & (D < L)
    win_sizes = np.minimum(t_idx + 1, L)
    for p in SOURCES:
        A = avail[p]
        out[f"{p}_available"] = np.diag(A).copy()
        out[f"{p}_window{L}_availability"] = (A & win_long).sum(axis=1) / win_sizes
        last = _last_index(A)
        has = last >= 0
        li = np.where(has, last, 0)
        out[f"{p}_staleness_hours"] = np.where(has, (dt - rec_ns[p][li]) / _NS_PER_H, np.nan)
        out[f"{p}_last_delay_hours"] = np.where(has, (rep_ns[p][li] - rec_ns[p][li]) / _NS_PER_H, np.nan)
    out["n_sources_available"] = sum(out[f"{p}_available"].astype(int) for p in SOURCES)

    # ---- as-of value matrices V[t, s] ----------------------------------------------------------
    base = {}
    for col, p in BASE_FIELDS.items():
        v = pd.to_numeric(g[col], errors="coerce").astype(float).to_numpy()
        base[col] = np.where(avail[p], v[None, :], np.nan)

    core_known = np.column_stack([~np.isnan(np.diag(base[f])) for f in CORE_FIELDS])
    out["completeness_score"] = core_known.mean(axis=1)
    out[f"completeness_window{L}"] = np.mean([out[f"{p}_window{L}_availability"] for p in SOURCES], axis=0)
    for i, f in enumerate(CORE_FIELDS):
        out[f"{f}_missing"] = ~core_known[:, i]

    # ---- per-signal statistics -----------------------------------------------------------------
    s_cfg, l_cfg = cfg["windows"]["short"], cfg["windows"]["long"]
    b, tr = cfg["baseline"], cfg["trend"]
    w_short = past & (D < s_cfg["length"])
    w_hist = (D >= b["gap"]) & (D < b["gap"] + b["length"])
    w_trend = past & (D < tr["length"])
    for sig in SIGNALS:
        V = sig.fn(base, cfg)
        name = sig.name
        cur = np.diag(V).copy()
        known = ~np.isnan(V) & past
        li = _last_index(known)
        last = np.where(li >= 0, V[t_idx, np.where(li >= 0, li, 0)], np.nan)
        r_s, _ = _nan_reduce(np.nanmean, V, w_short, s_cfg["min_periods"])
        r_l, _ = _nan_reduce(np.nanmean, V, win_long, l_cfg["min_periods"])
        med, H = _nan_reduce(np.nanmedian, V, w_hist, b["min_count"])
        mad, _ = _nan_reduce(np.nanmedian, np.abs(H - med[:, None]), w_hist, b["min_count"])
        scale = np.maximum(b["mad_scale"] * mad, cfg["scale_floor"][name])
        vals = {
            "cur": cur, "lag1": _lag(V, 1), "lag2": _lag(V, 2), "last": last,
            f"roll{s_cfg['length']}": r_s, f"roll{l_cfg['length']}": r_l,
            "hist_median": med, "hist_mad": mad, "dev": last - med, "rz": (last - med) / scale,
            f"roll{s_cfg['length']}_rz": (r_s - med) / scale,
            f"slope{tr['length']}": _slope(V, w_trend, tr["min_points"]),
            "cusum": _cusum(V, med, scale, past & (D < cfg["cusum"]["window"]), cfg["cusum"]),
        }
        for stat, _ in stat_specs(cfg):
            out[f"{name}_{stat}"] = vals[stat]
    return out


def _context(df: pd.DataFrame, sim: dict) -> pd.DataFrame:
    zones = {z: a for _, z, a in config.iter_zones(sim)}
    env = df["zone_id"].map(lambda z: zones[z]["environment"])
    return pd.DataFrame({
        "ctx_outdoor": env.isin(["outdoor", "semi_outdoor"]),
        "ctx_indoor_conditioned": env.eq("indoor_conditioned"),
        "ctx_radiant_heat": df["zone_id"].map(lambda z: bool(zones[z]["radiant_heat"])),
        "ctx_work_intensity": df["zone_id"].map(lambda z: {"light": 1, "moderate": 2, "heavy": 3}[
            zones[z]["work_intensity"]]),
        "ctx_night_shift": df["shift"].eq("night"),
    }, index=df.index)


def build_features(unified: pd.DataFrame, cfg: dict | None = None, sim: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_feature_config()
    sim = sim or config.load_simulation()
    u = unified.copy()
    u["decision_time"] = u["shift_end"] + pd.Timedelta(hours=cfg["decision_lag_hours"])
    u = u.sort_values(SERIES + ["observation_date"]).reset_index(drop=True)

    parts = []
    for _, g in u.groupby(SERIES, sort=False):
        feats = series_features(g, cfg)
        parts.append(pd.DataFrame(feats, index=g.index))
    F = pd.concat(parts).loc[u.index]
    F = pd.concat([u[ID_COLUMNS], F, _context(u, sim)], axis=1)

    cols = feature_columns(cfg)
    missing = set(cols) - set(F.columns)
    if missing:
        raise RuntimeError(f"registry/builder mismatch: {sorted(missing)[:5]}")
    F = F[ID_COLUMNS + cols]
    bool_cols = [r["feature"] for r in feature_registry(cfg) if r["unit"] == "bool"]
    F[bool_cols] = F[bool_cols].astype(bool)
    shift_order = F["shift"].map({"day": 0, "night": 1})
    return F.assign(_o=shift_order).sort_values(["observation_date", "site_id", "zone_id", "_o"]) \
        .drop(columns="_o").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------
DICTIONARY_PATH = config.PROJECT_ROOT / "docs" / "feature_dictionary.md"


def render_feature_dictionary(cfg: dict | None = None) -> str:
    cfg = cfg or load_feature_config()
    reg = feature_registry(cfg)
    s, l, b, t = cfg["windows"]["short"], cfg["windows"]["long"], cfg["baseline"], cfg["trend"]
    lines = [
        "# PREVENZA Feature Dictionary",
        "",
        "> GENERATED from `src/features/registry.py` + `config/features.yaml` by `python -m src.features.build`. "
        "Do not edit by hand.",
        "",
        "## Files",
        "",
        "- `data/processed/unified_observations.parquet` - five sources joined one-to-one on "
        "`observation_date + site_id + zone_id + shift`; every value as eventually reported (NA preserved), with "
        "per-source `*_recorded_at`, `*_reporting_timestamp`, `*_reporting_status`, `*_reporting_delay_hours`. "
        "Prefixes: `env_`, `wf_`, `clin_`, `abs_`, `inc_`. **Retrospective record - not a model input.**",
        "- `data/processed/features.parquet` - one row per observation; identifier columns "
        f"`{', '.join(ID_COLUMNS)}` + {len(reg)} as-of features.",
        "",
        "## Leakage rules",
        "",
        f"- **Decision time** = shift end + {cfg['decision_lag_hours']} h. Each row's features use only "
        "information available at that time.",
        "- A source value from observation *s* is used for observation *t* only if *s <= t* in the same "
        "site-zone-shift series **and** `reporting_timestamp(s) <= decision_time(t)`. Never-reported values "
        "are never used.",
        "- Series are independent per `site_id + zone_id + shift`; one observation = one day of that shift.",
        "- No ground-truth labels, event IDs, onset/escalation/end timestamps, or data-quality log entries are "
        "read. Calendar position (date/day index) is deliberately not a feature.",
        "- Consequence: slow sources (clinical ~12-48 h, absenteeism ~36-60 h) are usually **not** known for "
        "the current observation; their `lag*`, `last`, `roll*` and availability features carry the "
        "delayed signal.",
        "",
        "## Statistic definitions (applied to every signal)",
        "",
        f"Windows: short = {s['length']} obs (min {s['min_periods']}), long = {l['length']} obs "
        f"(min {l['min_periods']}); baseline = {b['length']} obs ending {b['gap']} obs before t "
        f"(min {b['min_count']}); trend = {t['length']} obs (min {t['min_points']}).",
        "",
        "| Suffix | Definition |",
        "|---|---|",
    ]
    lines += [f"| `_{k}` | {v} |" for k, v in stat_specs(cfg)]
    lines += ["", "Robust-scale floors (`scale_floor`): " +
              ", ".join(f"`{k}` {v}" for k, v in cfg["scale_floor"].items()) + ".", "",
              "## Signals", "", "| Signal | Sources | Unit | Definition |", "|---|---|---|---|"]
    lines += [f"| `{sg.name}` | {' + '.join(SOURCES[p] for p in sg.sources)} | {sg.unit} | {sg.description} |"
              for sg in SIGNALS]
    lines += ["", f"Signal features: {len(SIGNALS)} signals x {len(stat_specs(cfg))} statistics = "
                  f"{len(SIGNALS) * len(stat_specs(cfg))} columns, named `<signal>_<suffix>`.", "",
              "## Availability, missingness and context features", "",
              "| Feature | Group | Sources | Unit | Meaning |", "|---|---|---|---|---|"]
    lines += [f"| `{r['feature']}` | {r['group']} | {r['sources']} | {r['unit']} | {r['description']} |"
              for r in reg if not r["group"].startswith("signal:")]
    lines += ["", "## Feature count by group", "", "| Group | Columns |", "|---|---|"]
    counts = pd.Series([("signal" if r["group"].startswith("signal:") else r["group"]) for r in reg]).value_counts()
    lines += [f"| {g} | {c} |" for g, c in counts.items()]
    lines += [f"| **total** | **{len(reg)}** |", ""]
    return "\n".join(lines)


def main():
    sim = config.load_simulation()
    cfg = load_feature_config()
    unified = build_unified(load_sources(), config.expected_observation_count(sim))
    features = build_features(unified, cfg, sim)
    proc = config.path_for("processed")
    proc.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(unified_path(), index=False)
    features.to_parquet(proc / FEATURES_FILE, index=False)
    DICTIONARY_PATH.write_text(render_feature_dictionary(cfg), encoding="utf-8")
    print(f"unified_observations.parquet: {unified.shape[0]} rows x {unified.shape[1]} columns")
    print(f"features.parquet: {features.shape[0]} rows x {features.shape[1]} columns "
          f"({len(ID_COLUMNS)} identifiers + {features.shape[1] - len(ID_COLUMNS)} features)")
    print(f"wrote {DICTIONARY_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
