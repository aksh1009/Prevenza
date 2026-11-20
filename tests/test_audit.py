"""Phase 4C: evaluation-audit safeguards and measurement-validity indicator."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src import config, schemas
from src.detection import risk_fusion as rf
from src.evaluation import audit
from src.features.quality import NWS_MAX_HEAT_INDEX_C, NWS_MAX_TEMPERATURE_C, heat_index_validity
from src.forecasting import forecast as fc
from tests.test_risk_fusion import frame, set_clin, set_env

KEY = list(schemas.OBSERVATION_KEY)
PROC = config.path_for("processed")
RCFG = rf.load_fusion_config()


# ------------------------------------------------------------------------------- validity indicator
def test_heat_index_validity_boundaries():
    t = [43.3, 43.4, 40.0, np.nan, 30.0, 30.0]
    hi = [58.3, 50.0, 58.4, 45.0, np.nan, 35.0]
    assert list(heat_index_validity(t, hi)) == ["within_range", "extrapolated", "extrapolated", "within_range",
                                                "unavailable", "within_range"]
    assert NWS_MAX_TEMPERATURE_C == pytest.approx(43.33, abs=0.01)
    assert NWS_MAX_HEAT_INDEX_C == pytest.approx(58.33, abs=0.01)


def test_validity_never_alters_values():
    hi = np.array([70.0, 30.0])
    before = hi.copy()
    heat_index_validity([45.0, 30.0], hi)
    np.testing.assert_array_equal(hi, before)


def test_fusion_flags_extrapolated_heat_index_without_changing_scores():
    F = frame(8)
    set_env(F, range(2, 8), hi=52.0)
    set_clin(F, range(4, 8))
    A_in, E_in = rf.fuse(F, None, RCFG)
    F2 = F.copy()
    F2.loc[2:7, "heat_index_celsius_last"] = 63.0                  # beyond NWS chart range
    A_out, E_out = rf.fuse(F2, None, RCFG)
    assert (A_out.loc[2:7, "heat_index_validity"] == "extrapolated").all()
    assert (A_in.loc[2:7, "heat_index_validity"] == "within_range").all()
    assert (A_out.loc[2:7, "measurement_flags"] == "heat_index_extrapolated").all()
    # absolute component saturates at 51 C, so the extrapolated magnitude adds no score and no class change
    pd.testing.assert_series_equal(A_out["risk_score"], A_in["risk_score"])
    pd.testing.assert_series_equal(A_out["classification"], A_in["classification"])
    alerts = A_out["alert"] & A_out["environmental_elevated"]
    assert (A_out.loc[alerts, "alert_relies_on_extrapolated_heat_index"]).all() and alerts.any()
    assert A_out.loc[6, "explanation"].count("Measurement caution") == 1
    assert "Measurement caution" not in A_in.loc[6, "explanation"]
    env_notes = E_out[(E_out["component"] == "environmental")].reset_index(drop=True)["note"]
    assert env_notes.iloc[2:8].str.contains("extrapolated").all()


# ----------------------------------------------------------------------------- real outputs preserved
@pytest.fixture(scope="module")
def A():
    return pd.read_parquet(PROC / "risk_assessments.parquet")


def test_raw_heat_index_preserved():
    env = pd.read_csv(config.path_for("raw") / "environmental_data.csv")
    assert env["heat_index_celsius"].max() == pytest.approx(67.4)       # not clipped
    assert (env["heat_index_celsius"] > NWS_MAX_HEAT_INDEX_C).sum() == 9
    assert "wbgt" not in " ".join(env.columns).lower()


def test_fusion_scores_unchanged_by_audit(A):
    P = pd.read_parquet(PROC / "backtest_results.parquet")               # produced before the audit change
    m = A[KEY + ["risk_score", "classification"]].merge(P[KEY + ["risk_score", "classification"]], on=KEY,
                                                         suffixes=("", "_before"))
    assert len(m) == 1620
    np.testing.assert_allclose(m["risk_score"], m["risk_score_before"], equal_nan=True)
    assert (m["classification"] == m["classification_before"]).all()


def test_real_extrapolated_rows_flagged(A):
    x = A[A["heat_index_validity"] == "extrapolated"]
    assert len(x) == 9
    assert (x["heat_index_observed"] > NWS_MAX_HEAT_INDEX_C).all()
    assert x["explanation"].str.contains("Measurement caution").all()
    assert not A.loc[A["heat_index_validity"] != "extrapolated", "explanation"].str.contains("Measurement caution").any()


def test_forecast_limitations_mention_extrapolation_and_model_unchanged():
    f = pd.read_parquet(PROC / "forecast_results.parquet")
    x = f[(f["heat_index_validity"] == "extrapolated") & f["eligible"]]
    assert len(x) > 0 and x["limitations"].str.contains("outside NWS chart range").all()
    meta = json.loads((config.PROJECT_ROOT / "models" / "forecast" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == f["model_version"].iloc[0]
    assert "heat_index_validity" not in meta["features"]                 # indicator is not a model input


# ------------------------------------------------------------------------------------- audit logic
@pytest.fixture(scope="module")
def built():
    model = config.PROJECT_ROOT / "models" / "forecast" / "pipeline.joblib"
    before = hashlib.sha256(model.read_bytes()).hexdigest()
    o = audit.build(write=False)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == before       # audit never refits/saves models
    return o


def test_evt005_spans_validation_and_test(built):
    ev = built["ev_split"].set_index("target_event_id")
    assert ev.loc["EVT-005", "validation"] > 0 and ev.loc["EVT-005", "test"] > 0
    assert ev.loc["EVT-001", ["validation", "test"]].sum() == 0
    assert ev.loc["EVT-002", ["train", "validation"]].sum() == 0


def test_unseen_event_subset_excludes_evt005(built):
    t = built["e5_tab"]
    subsets = list(dict.fromkeys(t["test subset"]))
    assert len(subsets) == 3
    for s in subsets:                                                   # identical rows for every model
        assert t.loc[t["test subset"] == s, "n"].nunique() == 1
    n = [int(t.loc[t["test subset"] == s, "n"].iloc[0]) for s in subsets]
    assert n[0] > n[1] > n[2]
    assert int(t.loc[t["test subset"] == subsets[2], "positives"].iloc[0]) == 27    # EVT-002 only


def test_audit_metrics_match_phase_4b(built):
    overall = fc.run(write=False)[3]
    t = built["e5_tab"]
    allrows = t[t["test subset"].str.startswith("all")].set_index("model")
    for r in overall[overall["split"] == "test"].itertuples():
        assert allrows.loc[r.model, "f1"] == pytest.approx(r.f1, nan_ok=True)
        assert allrows.loc[r.model, "pr_auc"] == pytest.approx(r.pr_auc, nan_ok=True)


def test_score_audit_labels_brier_uncalibrated(built):
    sc = built["scores"]
    assert any("uncalibrated" in c for c in sc.columns)
    assert sc["scores from"].str.contains("IN-SAMPLE").any()


def test_summary_parquet(built):
    p = PROC / "model_comparison_summary.parquet"
    if not p.exists():
        pytest.skip("run: python -m src.evaluation.audit")
    s = pd.read_parquet(p)
    pd.testing.assert_frame_equal(s, built["summary"])
    assert s["synthetic"].all()
    assert {"4B forecast", "4A detection backtest"} <= set(s["phase"])
    assert not s["model"].str.contains("prob", case=False).any()


def test_audit_document_conclusions():
    text = audit.DOC.read_text(encoding="utf-8")
    for phrase in ("not demonstrated", "synthetic", "uncalibrated", "EVT-005", "Hypotheses (not verified)",
                   "Deferred", "Dashboard readiness"):
        assert phrase in text, phrase
