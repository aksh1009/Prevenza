"""Chronological replay: rebuild every decision exactly as it could have been made at the time.

At each decision timestamp T:
  1. information state = unified rows decided by T, with every source value reported after T removed
  2. as-of features (Phase 3A)  ->  Isolation Forest with the PERSISTED model  ->  risk fusion
  3. keep only the rows whose decision_time == T
Concatenating these gives the replayed history. It must equal the saved batch outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import schemas
from src.detection import isolation_forest as iso
from src.detection import risk_fusion as rf
from src.features.build import build_features
from src.features.registry import load_feature_config
from src.preprocessing.integrate import PREFIX, unified_path

KEY = list(schemas.OBSERVATION_KEY)


def information_state(u: pd.DataFrame, T: pd.Timestamp, lag_hours: float) -> pd.DataFrame:
    """Rows whose decision is due by T, with all source values reported after T unknown."""
    v = u[u["shift_end"] + pd.Timedelta(hours=lag_hours) <= T].copy()
    for p in PREFIX.values():
        late = v[f"{p}_reporting_timestamp"].isna() | (v[f"{p}_reporting_timestamp"] > T)
        cols = [c for c in v.columns if c.startswith(f"{p}_") and c not in (f"{p}_recorded_at", f"{p}_reporting_status")]
        for c in cols:
            v.loc[late, c] = np.nan if v[c].dtype.kind == "f" else pd.NA
        v.loc[late, f"{p}_reporting_status"] = "not_reported"
    return v


def decision_times(u: pd.DataFrame, lag_hours: float) -> list[pd.Timestamp]:
    return sorted((u["shift_end"] + pd.Timedelta(hours=lag_hours)).unique())


class Replayer:
    def __init__(self):
        self.u = pd.read_parquet(unified_path())
        self.fcfg = load_feature_config()
        self.icfg = iso.load_if_config()
        self.pipe, self.meta = iso.load_model(iso.model_dir(self.icfg))
        self.rcfg = rf.load_fusion_config()
        self.lag = self.fcfg["decision_lag_hours"]

    def times(self) -> list[pd.Timestamp]:
        return decision_times(self.u, self.lag)

    def step(self, T) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        T = pd.Timestamp(T)
        state = information_state(self.u, T, self.lag)
        F = build_features(state, self.fcfg)
        I = iso.score(F, self.pipe, self.meta, self.icfg)
        A, _ = rf.fuse(F, I, self.rcfg)
        keep = lambda df: df[df["decision_time"] == T].reset_index(drop=True)
        return keep(F), keep(I), keep(A)

    def run(self, times=None, progress: bool = False):
        times = self.times() if times is None else [pd.Timestamp(t) for t in times]
        Fs, Is, As = [], [], []
        for i, T in enumerate(times):
            f, s, a = self.step(T)
            Fs.append(f), Is.append(s), As.append(a)
            if progress and (i % 20 == 0 or i == len(times) - 1):
                print(f"  replay {i + 1}/{len(times)}  {T}")
        cat = lambda xs: pd.concat(xs, ignore_index=True)
        return cat(Fs), cat(Is), cat(As)


def _sort(df):
    o = df["shift"].map({"day": 0, "night": 1})
    return df.assign(_o=o).sort_values(["observation_date", "site_id", "zone_id", "_o"]).drop(columns="_o") \
        .reset_index(drop=True)


def compare(replayed: pd.DataFrame, saved: pd.DataFrame) -> pd.DataFrame:
    """Per-row equality of replayed vs saved outputs (same keys). Returns KEY + replay_match."""
    s = saved.merge(replayed[KEY], on=KEY)                     # only replayed rows
    r = _sort(replayed)
    s = _sort(s)[r.columns]
    eq = pd.Series(True, index=r.index)
    for c in r.columns:
        a, b = r[c], s[c]
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            same = np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            both_na = (a.isna() & b.isna()).to_numpy()
            equal = (a.astype(object) == b.astype(object)).fillna(False).astype(bool).to_numpy()
            same = both_na | (equal & a.notna().to_numpy())
        eq &= same
    out = r[KEY].copy()
    out["replay_match"] = eq.to_numpy()
    return out
