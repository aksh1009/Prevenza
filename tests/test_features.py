"""Phase 3A: integration and as-of feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src import config, schemas
from src.features.build import (DICTIONARY_PATH, FEATURES_FILE, ID_COLUMNS, build_features,
                                render_feature_dictionary)
from src.features.registry import CORE_FIELDS, SOURCES, feature_columns, load_feature_config
from src.preprocessing.integrate import JoinError, PREFIX, build_unified, check_keys, unified_path
from src.preprocessing.load import load_sources

KEY = list(schemas.OBSERVATION_KEY)
SIM = config.load_simulation()
CFG = load_feature_config()
N = config.expected_observation_count(SIM)


@pytest.fixture(scope="module")
def sources():
    return load_sources()


@pytest.fixture(scope="module")
def unified(sources):
    return build_unified(sources, N)


@pytest.fixture(scope="module")
def feats(unified):
    return build_features(unified, CFG, SIM)


def _series(df, zone, shift):
    return df[(df["zone_id"] == zone) & (df["shift"] == shift)].sort_values("observation_date").reset_index(drop=True)


# --- joins -------------------------------------------------------------------------------------------
def test_unified_one_row_per_key(unified):
    assert len(unified) == N == 1620
    assert not unified.duplicated(KEY).any()


@pytest.mark.parametrize("name", schemas.SOURCE_DATASETS)
def test_unified_preserves_values_and_timestamps(sources, unified, name):
    p = PREFIX[name]
    src = sources[name]
    m = unified.merge(src, on=KEY, validate="one_to_one")
    for col in src.columns:
        if col in KEY:
            continue
        a, b = m[f"{p}_{col}"], m[col]
        assert (a.isna() == b.isna()).all(), f"{col}: missingness changed"
        both = a.notna()
        assert (a[both] == b[both]).all(), f"{col}: values changed"
    delay = (m[f"{p}_reporting_timestamp"] - m[f"{p}_recorded_at"]).dt.total_seconds() / 3600
    np.testing.assert_allclose(m[f"{p}_reporting_delay_hours"], delay, equal_nan=True)


def test_join_rejects_duplicate_missing_and_null_keys(sources):
    dup = dict(sources)
    dup["clinical_data"] = pd.concat([sources["clinical_data"], sources["clinical_data"].iloc[[0]]])
    with pytest.raises(JoinError, match="duplicate"):
        check_keys(dup)
    short = dict(sources)
    short["incident_data"] = sources["incident_data"].iloc[1:]
    with pytest.raises(JoinError, match="key set differs"):
        check_keys(short)
    nul = dict(sources)
    df = sources["workforce_data"].copy()
    df.loc[0, "zone_id"] = pd.NA
    nul["workforce_data"] = df
    with pytest.raises(JoinError, match="null key"):
        check_keys(nul)


# --- dimensions and registry ----------------------------------------------------------------------
def test_feature_dimensions_and_columns(feats):
    assert len(feats) == N
    assert list(feats.columns) == ID_COLUMNS + feature_columns(CFG)
    assert not feats.duplicated(KEY).any()


def test_no_ground_truth_in_features(feats):
    gt_cols = set(schemas.LABEL_COLUMNS)
    for c in schemas.GROUND_TRUTH_CONTRACTS.values():
        gt_cols |= set(c.columns) - set(KEY) - {"site_id", "zone_id", "description"}
    assert not gt_cols & set(feats.columns)
    assert not any("label" in c or "event" in c or "escalat" in c or "onset" in c for c in feats.columns)


def test_feature_dictionary_in_sync():
    assert DICTIONARY_PATH.read_text(encoding="utf-8") == render_feature_dictionary(CFG)


# --- availability and missingness -------------------------------------------------------------------
@pytest.mark.parametrize("p", list(SOURCES))
def test_availability_follows_reporting_timestamp(unified, feats, p):
    m = feats[KEY + ["decision_time", f"{p}_available"]].merge(
        unified[KEY + [f"{p}_reporting_timestamp", f"{p}_reporting_status"]], on=KEY)
    expected = m[f"{p}_reporting_timestamp"].notna() & (m[f"{p}_reporting_timestamp"] <= m["decision_time"])
    assert (m[f"{p}_available"] == expected).all()
    assert not m.loc[m[f"{p}_reporting_status"] == "not_reported", f"{p}_available"].any()


def test_decision_time(unified, feats):
    m = feats[KEY + ["decision_time"]].merge(unified[KEY + ["shift_end"]], on=KEY)
    assert (m["decision_time"] == m["shift_end"] + pd.Timedelta(hours=CFG["decision_lag_hours"])).all()


def test_slow_sources_not_known_for_current_shift(feats):
    # clinical min delay (12 h) and absenteeism min delay (36 h) exceed the 1 h decision lag
    assert not feats["clin_available"].any()
    assert not feats["abs_available"].any()
    assert feats["env_available"].mean() > 0.95


def test_availability_aggregates(feats):
    n = sum(feats[f"{p}_available"].astype(int) for p in SOURCES)
    assert (feats["n_sources_available"] == n).all()
    for c in ("completeness_score", "completeness_window7", *[f"{p}_window7_availability" for p in SOURCES]):
        assert feats[c].between(0, 1).all(), c
    miss = feats[[f"{f}_missing" for f in CORE_FIELDS]]
    np.testing.assert_allclose(feats["completeness_score"], 1 - miss.mean(axis=1))


def test_missing_values_not_filled(unified, feats):
    m = feats[KEY + ["heat_index_celsius_cur", "env_available"]].merge(
        unified[KEY + ["env_heat_index_celsius"]], on=KEY)
    should_be_na = m["env_heat_index_celsius"].isna() | ~m["env_available"]
    assert (m["heat_index_celsius_cur"].isna() == should_be_na).all()
    known = ~should_be_na
    assert (m.loc[known, "heat_index_celsius_cur"] == m.loc[known, "env_heat_index_celsius"]).all()
    assert (feats["env_temperature_celsius_missing"] == feats["temperature_celsius_cur"].isna()).all()
    # not_reported periods never appear as 0 in features
    assert feats["heat_related_visits_cur"].isna().all()


def test_scenario_d_clinical_gap_visible_in_availability(feats):
    sc = next(s for s in SIM["scenarios"] if s.get("clinical_gap"))
    z = sc["zones"][0]
    s = _series(feats, z, "day").set_index("observation_date")
    g_end = pd.Timestamp(sc["clinical_gap"]["end_date"])
    before = s.loc[: pd.Timestamp(sc["clinical_gap"]["start_date"]) - pd.Timedelta(days=1)]
    late_gap = s.loc[g_end - pd.Timedelta(days=4): g_end]
    assert (late_gap["clin_window7_availability"] == 0).all()
    assert late_gap["clin_staleness_hours"].min() > before["clin_staleness_hours"].max()
    assert late_gap["env_available"].mean() > 0.75                        # other feeds continue


# --- rolling calculations vs. an independent reference ------------------------------------------
def _reference(u, zone, shift, value_fn, prefixes, floor):
    """Straightforward per-row loop implementing the documented as-of rules."""
    g = _series(u, zone, shift)
    dt = g["shift_end"] + pd.Timedelta(hours=CFG["decision_lag_hours"])
    vals = value_fn(g).astype(float).to_numpy()
    s_len, l_len = CFG["windows"]["short"]["length"], CFG["windows"]["long"]["length"]
    b, tr = CFG["baseline"], CFG["trend"]
    rows = []
    for t in range(len(g)):
        known = {}
        for s in range(t + 1):
            ok = all(pd.notna(g.at[s, f"{p}_reporting_timestamp"]) and g.at[s, f"{p}_reporting_timestamp"] <= dt[t]
                     for p in prefixes)
            if ok and not np.isnan(vals[s]):
                known[s] = vals[s]
        def win(lo, hi):
            return [known[s] for s in range(max(lo, 0), hi + 1) if s in known]
        r = {"cur": known.get(t, np.nan), "lag1": known.get(t - 1, np.nan), "lag2": known.get(t - 2, np.nan)}
        r["last"] = known[max(known)] if known else np.nan
        w = win(t - s_len + 1, t); r[f"roll{s_len}"] = np.mean(w) if len(w) >= 1 else np.nan
        w = win(t - l_len + 1, t); r[f"roll{l_len}"] = np.mean(w) if len(w) >= 3 else np.nan
        h = win(t - b["gap"] - b["length"] + 1, t - b["gap"])
        if len(h) >= b["min_count"]:
            med = np.median(h); mad = np.median(np.abs(np.array(h) - med))
        else:
            med = mad = np.nan
        scale = max(b["mad_scale"] * mad, floor) if not np.isnan(mad) else np.nan
        r.update(hist_median=med, hist_mad=mad, dev=r["last"] - med, rz=(r["last"] - med) / scale,
                 **{f"roll{s_len}_rz": (r[f"roll{s_len}"] - med) / scale})
        pts = [(s - t, known[s]) for s in range(max(t - tr["length"] + 1, 0), t + 1) if s in known]
        xs = np.array([p[0] for p in pts], float)
        r[f"slope{tr['length']}"] = (np.polyfit(xs, [p[1] for p in pts], 1)[0]
                                     if len(pts) >= tr["min_points"] and np.ptp(xs) > 0 else np.nan)
        cu = CFG["cusum"]
        cw = [known[s] for s in range(max(t - cu["window"] + 1, 0), t + 1) if s in known]
        if np.isnan(med) or not cw:
            r["cusum"] = np.nan
        else:
            acc = 0.0
            for v in cw:
                acc = max(0.0, acc + min((v - med) / scale, cu["z_cap"]) - cu["k"])
            r["cusum"] = acc
        rows.append(r)
    return pd.DataFrame(rows)


CASES = [
    ("heat_index_celsius", lambda g: g["env_heat_index_celsius"], ("env",)),
    ("absence_rate", lambda g: g["abs_absence_rate"], ("abs",)),
    ("heat_visits_per_100", lambda g: (100 * g["clin_heat_related_visits"] / g["wf_workforce_count"]
                                        .where(g["wf_workforce_count"] > 0)), ("clin", "wf")),
    ("incidents_mod_severe", lambda g: g["inc_moderate_incidents"] + g["inc_severe_incidents"], ("inc",)),
]


@pytest.mark.parametrize("signal,fn,prefixes", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("zone,shift", [("S02-Z01", "day"), ("S01-Z01", "night")])
def test_rolling_matches_reference(unified, feats, signal, fn, prefixes, zone, shift):
    ref = _reference(unified, zone, shift, fn, prefixes, CFG["scale_floor"][signal])
    got = _series(feats, zone, shift)
    for stat in ref.columns:
        np.testing.assert_allclose(got[f"{signal}_{stat}"].astype(float), ref[stat].astype(float),
                                   rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=f"{signal}_{stat}")


def test_env_rolling_equals_pandas_when_immediately_reported(unified, feats):
    # env delay (<= 0.5 h) < decision lag (1 h): as-of == plain causal rolling over the series
    for (zone, shift), g in unified.groupby(["zone_id", "shift"]):
        g = g.sort_values("observation_date")
        exp = g["env_temperature_celsius"].rolling(3, min_periods=1).mean().to_numpy()
        got = _series(feats, zone, shift)["temperature_celsius_roll3"].to_numpy()
        np.testing.assert_allclose(got, exp, equal_nan=True)


# --- temporal leakage --------------------------------------------------------------------------------
def _truncate(u, cutoff):
    """Information state at `cutoff`: anything reported later is unknown."""
    v = u.copy()
    for name, p in PREFIX.items():
        late = v[f"{p}_reporting_timestamp"].isna() | (v[f"{p}_reporting_timestamp"] > cutoff)
        value_cols = [c for c in v.columns if c.startswith(f"{p}_") and c not in
                      (f"{p}_recorded_at", f"{p}_reporting_status")]
        for c in value_cols:
            v.loc[late, c] = np.nan if v[c].dtype.kind == "f" else pd.NA
        v.loc[late, f"{p}_reporting_status"] = "not_reported"
    return v


@pytest.mark.parametrize("cutoff", ["2025-06-20 07:00", "2025-07-03 19:00", "2025-07-28 07:00"])
def test_no_temporal_leakage_truncation(unified, feats, cutoff):
    cutoff = pd.Timestamp(cutoff)
    trunc = build_features(_truncate(unified, cutoff), CFG, SIM)
    rows = feats["decision_time"] <= cutoff
    assert rows.sum() > 0
    pd.testing.assert_frame_equal(trunc[rows].reset_index(drop=True), feats[rows].reset_index(drop=True))


def test_future_values_do_not_change_past_features(unified, feats):
    cutoff = pd.Timestamp("2025-07-10")
    u = unified.copy()
    future = u["shift_start"] >= cutoff
    for c in ("env_temperature_celsius", "env_heat_index_celsius", "clin_heat_related_visits",
              "abs_absence_rate", "inc_heat_incident_count"):
        u.loc[future, c] = 999
    f2 = build_features(u, CFG, SIM)
    past = feats["decision_time"] < cutoff
    pd.testing.assert_frame_equal(f2[past].reset_index(drop=True), feats[past].reset_index(drop=True))
    assert not f2[~past].equals(feats[~past])                      # the perturbation did take effect


def test_delayed_feed_not_used_before_report(unified, feats):
    """Absenteeism feed delay (+96 h) at S02 2025-07-01..07: values unknown until actually reported."""
    s = _series(feats, "S02-Z01", "day").set_index("observation_date")
    u = _series(unified, "S02-Z01", "day").set_index("observation_date")
    d0 = pd.Timestamp("2025-07-01")
    rep = u.loc[d0, "abs_reporting_timestamp"]
    for k in (1, 2):
        d = d0 + pd.Timedelta(days=k)
        assert s.loc[d, "decision_time"] < rep                       # premise: still unreported
        assert pd.isna(s.loc[d, f"absence_rate_lag{k}"])
    later = s[(s["decision_time"] >= rep)]
    assert later["abs_staleness_hours"].notna().all()
    stale_during = s.loc[d0 + pd.Timedelta(days=1): d0 + pd.Timedelta(days=5), "abs_staleness_hours"]
    assert stale_during.max() > 96                                   # staleness exposes the delay


# --- files ---------------------------------------------------------------------------------------------
def test_files_on_disk_match_rebuild(unified, feats):
    fp = config.path_for("processed") / FEATURES_FILE
    if not fp.exists() or not unified_path().exists():
        pytest.skip("run: python -m src.features.build")
    pd.testing.assert_frame_equal(pd.read_parquet(unified_path()), unified)
    pd.testing.assert_frame_equal(pd.read_parquet(fp), feats)
