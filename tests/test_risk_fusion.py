"""Phase 3B.3: multi-signal risk fusion."""
import copy

import numpy as np
import pandas as pd
import pytest

from src import config, schemas
from src.detection import isolation_forest as iso
from src.detection import risk_fusion as rf
from src.features.build import FEATURES_FILE, build_features
from src.features.registry import load_feature_config
from src.preprocessing.integrate import build_unified, unified_path

KEY = list(schemas.OBSERVATION_KEY)
CFG = rf.load_fusion_config()
NEUTRAL_STALENESS = {"env": 1.0, "wf": 25.0, "clin": 49.0, "abs": 49.0, "inc": 25.0}


# ------------------------------------------------------------------------------- synthetic frames
def frame(n=10, zone="S99-Z01"):
    """One series of `n` days with every fusion input at a neutral (baseline-like) value."""
    dates = pd.date_range("2025-07-01", periods=n, freq="D")
    F = pd.DataFrame({"observation_date": dates, "site_id": zone.split("-")[0], "zone_id": zone, "shift": "day"})
    F["decision_time"] = dates + pd.Timedelta(hours=19)
    for col in rf.required_columns(CFG):
        if col.endswith("_staleness_hours"):
            F[col] = NEUTRAL_STALENESS[col.split("_")[0]]
        elif col.startswith("heat_index_celsius_") and col.endswith(("_last", "_hist_median")):
            F[col] = 30.0
        else:
            F[col] = 0.0
    return F


def set_env(F, rows, hi=45.0, z=6.0, cusum=8.0):
    F.loc[rows, ["heat_index_celsius_last", "heat_index_celsius_rz", "heat_index_celsius_cusum"]] = [hi, z, cusum]


def set_clin(F, rows, z=6.0, cusum=6.0):
    F.loc[rows, ["heat_related_visits_rz", "heat_related_visits_cusum"]] = [z, cusum]


def set_inc(F, rows, z=6.0, cusum=6.0):
    F.loc[rows, ["heat_incident_count_rz", "heat_incident_count_cusum"]] = [z, cusum]


def fuse(F, iso_df=None, cfg=CFG):
    return rf.fuse(F, iso_df, cfg)


# ------------------------------------------------------------------------------------- scenarios
def test_isolated_spike_is_environmental_warning_only():
    F = frame()
    set_env(F, [5], hi=45.0, z=8.0, cusum=2.5)
    A, _ = fuse(F)
    assert A.loc[5, "classification"] == "Environmental warning"
    assert not A.loc[5, "environmental_persistent"]
    assert (A.drop(index=5)["classification"] == "Normal").all()
    assert not A["alert"].any()


def test_sustained_environmental_increase_never_health_class():
    F = frame(12)
    set_env(F, range(3, 12))
    A, _ = fuse(F)
    assert A.loc[4:, "environmental_persistent"].all()
    assert set(A.loc[3:, "classification"]) == {"Environmental warning"}
    assert not A["alert"].any()
    assert A["risk_score"].max() > 0


def test_environmental_plus_clinical_escalation():
    F = frame(12)
    set_env(F, range(3, 12))
    set_clin(F, range(5, 12))
    set_inc(F, range(8, 12))
    A, _ = fuse(F)
    assert A.loc[5, "classification"] == "Emerging health risk"      # clinical + persistent exposure
    assert (A.loc[5:7, "classification"] == "Emerging health risk").all()
    assert A.loc[8, "classification"] == "High health risk"          # 2nd health category + persistence
    assert (A.loc[9:, "classification"] == "High health risk").all()
    assert A.loc[:4, "alert"].sum() == 0
    assert A.loc[8, "risk_score"] > A.loc[5, "risk_score"] > A.loc[4, "risk_score"]


def test_missing_clinical_is_excluded_not_zero():
    F = frame()
    set_env(F, range(10))
    F[["heat_related_visits_last", "heat_related_visits_rz", "clin_staleness_hours"]] = np.nan
    A, E = fuse(F)
    assert (A["clinical_status"] == "missing_data").all()
    assert A["clinical_score"].isna().all()
    assert A["missing_sources"].str.contains("clinical").all()
    assert np.allclose(A["available_weight"], 1 - CFG["weights"]["clinical"])
    clin = E[E["component"] == "clinical"]
    assert (clin["effective_weight"] == 0).all() and (clin["contribution_points"] == 0).all()
    # renormalised: risk equals weighted mean of the remaining categories
    w = {c: v for c, v in CFG["weights"].items() if c != "clinical"}
    exp = sum(w[c] * A.loc[0, f"{c}_score"] for c in w) / sum(w.values())
    assert A.loc[0, "risk_score"] == pytest.approx(exp, abs=0.01)


def test_stale_absenteeism_excluded_and_listed():
    F = frame()
    F["abs_staleness_hours"] = CFG["max_staleness_hours"]["abs"] + 50
    F["absence_rate_rz"] = 10.0                                      # would be elevated if used
    A, E = fuse(F)
    assert (A["absenteeism_status"] == "stale").all()
    assert A["absenteeism_score"].isna().all()
    assert A["stale_sources"].eq("absenteeism").all()
    assert not A["absenteeism_elevated"].any()
    assert E.loc[E["component"] == "absenteeism", "note"].str.contains("old").all()


def test_all_sources_missing():
    F = frame()
    for c in CFG["categories"].values():
        F[c["observed"]] = np.nan
    A, _ = fuse(F)
    assert A["risk_score"].isna().all()
    assert (A["classification"] == "Insufficient evidence").all()
    assert (A["evidence_level"] == "insufficient").all()
    assert (A["reliability_score"] == 0).all()
    assert (A["n_categories_available"] == 0).all()
    assert A["severity_band"].isna().all()


def test_insufficient_baseline():
    F = frame()
    for c in CFG["categories"].values():
        F[c["primary"]] = np.nan                                     # warm-up: no baseline yet
    A, E = fuse(F)
    assert (A["classification"] == "Insufficient evidence").all()
    assert set(E.loc[E["component"] != "isolation_forest", "status"]) == {"no_baseline"}


def test_partial_evidence_cannot_confirm_normal_but_can_warn():
    F = frame()
    for cat in ("workforce_exposure", "clinical", "absenteeism", "incidents"):
        F[CFG["categories"][cat]["observed"]] = np.nan              # only environmental known (0.25 < 0.40)
    A, _ = fuse(F)
    assert (A["classification"] == "Insufficient evidence").all()
    set_env(F, [4], hi=45, z=8, cusum=2.5)
    A2, _ = fuse(F)
    assert A2.loc[4, "classification"] == "Environmental warning"
    assert A2.loc[4, "evidence_level"] == "insufficient"           # severity and evidence kept separate


def test_persistence_required_for_health_alert():
    F = frame()
    set_env(F, [5], hi=45, z=8, cusum=2.5)
    set_clin(F, [5])
    A, _ = fuse(F)
    assert A.loc[5, "clinical_elevated"] and not A.loc[5, "clinical_persistent"]
    assert not A["alert"].any()
    assert "not corroborated" in A.loc[5, "explanation"]
    F2 = frame()
    set_clin(F2, range(2, 10))                                      # health alone, persistent, no exposure
    A2, _ = fuse(F2)
    assert A2["clinical_persistent"].iloc[-1]
    assert not A2["alert"].any()                                    # one health category needs corroboration


def test_persistence_uses_only_past_rows():
    F = frame(12)
    set_env(F, range(3, 12))
    set_clin(F, range(5, 12))
    A, _ = fuse(F)
    F2 = F.copy()
    set_clin(F2, range(9, 12), z=0.0, cusum=0.0)                     # change the future
    A2, _ = fuse(F2)
    pd.testing.assert_frame_equal(A.loc[:8], A2.loc[:8])


def test_weight_normalization_and_contributions():
    F = frame(3)
    set_env(F, [0, 1, 2])
    F[["absence_rate_last", "abs_staleness_hours"]] = np.nan
    F[["heat_incident_count_last"]] = np.nan
    A, E = fuse(F)
    eff = E[E["component"] != "isolation_forest"].groupby(KEY)["effective_weight"].sum()
    assert np.allclose(eff, 1.0, atol=1e-3)
    contrib = E.groupby(KEY)["contribution_points"].sum().reset_index()
    m = A[KEY + ["risk_score"]].merge(contrib, on=KEY)
    assert np.allclose(m["risk_score"], m["contribution_points"], atol=0.05)
    avail = ["environmental", "workforce_exposure", "clinical"]
    w = CFG["weights"]
    exp = sum(w[c] * A.loc[0, f"{c}_score"] for c in avail) / sum(w[c] for c in avail)
    assert A.loc[0, "risk_score"] == pytest.approx(exp, abs=0.01)
    assert A.loc[0, "available_weight"] == pytest.approx(sum(w[c] for c in avail))


def test_weights_are_configurable():
    cfg = copy.deepcopy(CFG)
    cfg["weights"] = {**cfg["weights"], "environmental": 1.0}
    F = frame(2)
    set_env(F, [0, 1])
    A1, _ = fuse(F)
    A2, _ = fuse(F, cfg=cfg)
    assert A2.loc[0, "risk_score"] > A1.loc[0, "risk_score"]


def test_isolation_forest_only_out_of_sample():
    F = frame(4)
    iso_df = F[KEY].copy()
    iso_df["split"] = ["train", "train", "eval", "eval"]
    iso_df["eligibility_status"] = ["eligible", "eligible", "eligible", "warm_up"]
    iso_df["anomaly_score"] = [0.1, 0.1, 0.1, np.nan]
    A, E = fuse(F, iso_df)
    assert A["if_used"].tolist() == [False, False, True, False]
    base, _ = fuse(F)
    w = CFG["isolation_forest"]["weight"]
    assert A.loc[2, "risk_score"] == pytest.approx((1 - w) * base.loc[2, "risk_score"] + w * 100, abs=0.01)
    assert A.loc[0, "risk_score"] == base.loc[0, "risk_score"]
    st = E[E["component"] == "isolation_forest"]["status"].tolist()
    assert st == ["in_sample_not_used", "in_sample_not_used", "available", "not_eligible"]


def test_explanation_contents():
    F = frame(8)
    set_env(F, range(2, 8))
    set_clin(F, range(4, 8))
    F[["absence_rate_last", "abs_staleness_hours"]] = np.nan
    A, E = fuse(F)
    text = A.loc[6, "explanation"]
    for s in ("Emerging health risk", "risk", "evidence", "reliability", "baseline", "CUSUM", "z ",
              "Missing (excluded, not zero): absenteeism", "Isolation Forest", "Rule:"):
        assert s in text, s
    assert len(E) == len(F) * 6
    for col in ("configured_weight", "effective_weight", "signal_score", "observed_value", "baseline_median",
                "robust_z", "cusum", "staleness_hours", "component_scores", "status"):
        assert col in E.columns


def test_config_validation_rejects_bad_weights():
    cfg = copy.deepcopy(CFG)
    cfg["weights"]["clinical"] = 0
    with pytest.raises(ValueError):
        rf.validate_config(cfg)


# ------------------------------------------------------------------------------- real pipeline data
@pytest.fixture(scope="module")
def real():
    proc = config.path_for("processed")
    F = pd.read_parquet(proc / FEATURES_FILE)
    I = pd.read_parquet(proc / "isolation_forest_scores.parquet")
    A, E = rf.fuse(F, I, CFG)
    return F, I, A, E


def test_real_outputs_consistent(real):
    F, I, A, E = real
    assert len(A) == 1620 and len(E) == 1620 * 6
    assert not A.duplicated(KEY).any()
    assert A["risk_score"].dropna().between(0, 100).all()
    assert set(A["classification"]) <= set(rf.CLASSES)
    for c in rf.CATEGORIES:
        assert (A[f"{c}_score"].notna() == (A[f"{c}_status"] == "available")).all()
    contrib = E.groupby(KEY)["contribution_points"].sum().reset_index()
    m = A[KEY + ["risk_score"]].merge(contrib, on=KEY).dropna()
    assert np.allclose(m["risk_score"], m["contribution_points"], atol=0.05)
    assert not (A["alert"] & (A["n_health_elevated"] == 0)).any()        # env alone never alerts
    assert not A.loc[A["if_used"], "decision_time"].lt(pd.Timestamp(
        I.loc[I["split"] == "eval", "decision_time"].min())).any()
    assert not set(A.columns) & set(schemas.LABEL_COLUMNS) - {"severity"}


def test_real_isolated_spikes_not_health_alerts(real):
    _, _, A, _ = real
    sim = config.load_simulation()
    for sc in sim["scenarios"]:
        if sc["type"] != "isolated_temperature_spike":
            continue
        d = pd.Timestamp(sc["onset_date"])
        rows = A[A["zone_id"].isin(sc["zones"]) & A["observation_date"].between(d, d + pd.Timedelta(days=3))]
        assert not (rows["classification"] == "High health risk").any()


def test_real_scenario_d_clinical_gap_handled(real):
    _, _, A, _ = real
    sc = next(s for s in config.load_simulation()["scenarios"] if s.get("clinical_gap"))
    end = pd.Timestamp(sc["clinical_gap"]["end_date"])
    rows = A[A["zone_id"].isin(sc["zones"]) & A["observation_date"].between(end - pd.Timedelta(days=4), end)]
    assert (rows["clinical_status"] != "available").all()
    assert rows["clinical_score"].isna().all()
    assert rows["missing_sources"].str.contains("clinical").any() or rows["stale_sources"].str.contains("clinical").any()
    assert rows["risk_score"].notna().all()                              # other sources still assessed


def _truncate(u, cutoff):
    from src.preprocessing.integrate import PREFIX
    v = u.copy()
    for p in PREFIX.values():
        late = v[f"{p}_reporting_timestamp"].isna() | (v[f"{p}_reporting_timestamp"] > cutoff)
        for c in [c for c in v.columns if c.startswith(f"{p}_") and c not in (f"{p}_recorded_at", f"{p}_reporting_status")]:
            v.loc[late, c] = np.nan if v[c].dtype.kind == "f" else pd.NA
        v.loc[late, f"{p}_reporting_status"] = "not_reported"
    return v


@pytest.mark.parametrize("cutoff", ["2025-07-05 07:00", "2025-07-28 19:00"])
def test_no_temporal_leakage(real, cutoff):
    F, _, A, _ = real
    cutoff = pd.Timestamp(cutoff)
    icfg = iso.load_if_config()
    pipe, meta = iso.load_model(iso.model_dir(icfg))                 # fixed, persisted model
    u = pd.read_parquet(unified_path())
    Ft = build_features(_truncate(u, cutoff), load_feature_config())
    It = iso.score(Ft, pipe, meta, icfg)
    At, _ = rf.fuse(Ft, It, CFG)
    rows = A["decision_time"] <= cutoff
    pd.testing.assert_frame_equal(At[rows].reset_index(drop=True), A[rows].reset_index(drop=True))


def test_files_on_disk_match(real):
    _, _, A, E = real
    proc = config.path_for("processed")
    pa, pe = proc / CFG["output"]["assessments_file"], proc / CFG["output"]["explanations_file"]
    if not pa.exists() or not pe.exists():
        pytest.skip("run: python -m src.detection.risk_fusion")
    pd.testing.assert_frame_equal(pd.read_parquet(pa), A)
    pd.testing.assert_frame_equal(pd.read_parquet(pe), E)
