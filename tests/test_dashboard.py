"""Phase 5: dashboard components, data integrity and trust safeguards."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config
from src.dashboard import data_loader as DL
from src.dashboard import filters as FL
from src.dashboard import theme as T
from src.dashboard.pages import OPERATIONAL_PAGES, PAGES
from src.dashboard.pages import evaluation as EVP
from src.dashboard.pages import forecasting as FCP
from src.dashboard.pages import overview as OVP

ROOT = config.PROJECT_ROOT
APP = str(ROOT / "app.py")


def run_page(page: str, **widgets) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["nav"] = page
    at.run()
    for key, value in widgets.items():
        at.selectbox(key=key).set_value(value).run()
    return at


def page_text(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "info", "warning", "error", "success", "subheader", "title", "header"):
        try:
            parts += [str(e.value) for e in getattr(at, kind)]
        except Exception:
            pass
    for df in at.dataframe:
        parts.append(df.value.to_csv(index=False))
    return "\n".join(parts)


# ----------------------------------------------------------------------------------------- loading
@pytest.mark.parametrize("name", list(DL.DATASETS))
def test_every_dataset_loads(name):
    df = DL.load(name)
    assert len(df) > 0


def test_missing_file_raises_friendly_error(monkeypatch, tmp_path):
    monkeypatch.setitem(DL.DATASETS, "assessments",
                        DL.Dataset(tmp_path / "nope.parquet", "operational", "missing on purpose"))
    with pytest.raises(DL.DataUnavailable, match="python -m src.detection.risk_fusion"):
        DL.load("assessments")


def test_app_shows_error_not_crash_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(DL.DATASETS, "forecasts", DL.Dataset(tmp_path / "gone.parquet", "operational", "x"))
    at = run_page("24-hour forecasting")
    assert not at.exception
    assert any("not available" in e.value for e in at.error)


# ------------------------------------------------------------------------------------------ filters
@pytest.fixture(scope="module")
def A():
    return DL.load_operational("assessments")


def test_filter_by_site_zone_shift_date_and_class(A):
    f = FL.Filters(sites=["S02"], zones=["S02-Z01"], shifts=["day"],
                   date_range=(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-10")),
                   classes=["High health risk", "Emerging health risk"])
    out = FL.apply_filters(A, f)
    assert set(out["site_id"]) <= {"S02"} and set(out["zone_id"]) <= {"S02-Z01"} and set(out["shift"]) <= {"day"}
    assert out["observation_date"].between("2025-07-01", "2025-07-10").all()
    assert set(out["classification"]) <= {"High health risk", "Emerging health risk"} and len(out) > 0
    assert len(FL.apply_filters(A, FL.Filters())) == len(A)                 # empty filters = all
    s = FL.apply_filters(A, FL.Filters(date_range=(pd.Timestamp("2025-06-10"), pd.Timestamp("2025-06-10"))))
    assert len(s) == 18                                                     # one day = 9 zones x 2 shifts


def test_filters_preserve_missing_values(A):
    out = FL.apply_filters(A, FL.Filters(sites=["S01"]))
    src = A[A["site_id"] == "S01"]
    for c in ("risk_score", "clinical_score", "absenteeism_score"):
        assert out[c].isna().sum() == src[c].isna().sum()


def test_latest_selection(A):
    L = FL.latest_per_zone(A)
    assert len(L) == 9 and L["decision_time"].nunique() == 1
    assert (L["decision_time"] == A["decision_time"].max()).all()
    t = pd.Timestamp("2025-07-04 19:00")
    L2 = FL.latest_per_zone(A, t)
    assert (L2["decision_time"] <= t).all() and (L2["decision_time"] == t).all()
    assert len(FL.latest_per_series(A)) == 18


# ------------------------------------------------------------------------------------------- leakage
@pytest.mark.parametrize("name", [k for k, v in DL.DATASETS.items() if v.kind == "operational"])
def test_operational_datasets_have_no_labels(name):
    df = DL.load_operational(name)
    assert not DL.LABEL_COLUMNS & set(df.columns)


@pytest.mark.parametrize("name", [k for k, v in DL.DATASETS.items() if v.kind == "evaluation"])
def test_evaluation_datasets_blocked_from_operational_loader(name):
    with pytest.raises(DL.LeakageError):
        DL.load_operational(name)


def test_operational_page_code_uses_operational_loader():
    for name in OPERATIONAL_PAGES:
        src = inspect.getsource(PAGES[name])
        uses = src.count("load_evaluation(")
        if name == "Data quality & reliability":
            assert uses == 1 and 'load_evaluation("dq_log")' in src          # labelled simulation reference only
        else:
            assert uses == 0, name


@pytest.mark.parametrize("page", OPERATIONAL_PAGES)
def test_no_ground_truth_rendered_on_operational_pages(page):
    at = run_page(page)
    assert not at.exception
    text = page_text(at)
    assert "EVT-" not in text
    for col in ("label_name", "target_event_id", "matched_event_id", "match_status"):
        assert col not in text


# ------------------------------------------------------------------------------------------ display
def test_disclaimer_on_every_page():
    for page in PAGES:
        at = run_page(page)
        assert any(T.DISCLAIMER in str(m.value) for m in at.markdown), page


def test_insufficient_evidence_displayed(A):
    first = FL.latest_per_zone(A, A["decision_time"].min())
    tbl = OVP.latest_table(first)
    assert (first["classification"] == "Insufficient evidence").sum() > 0
    assert tbl["Classification"].str.contains("Insufficient evidence").sum() == (first["classification"] == "Insufficient evidence").sum()
    assert tbl["Classification"].str.startswith("◌").sum() > 0              # symbol, not colour alone


def test_missing_and_stale_sources_displayed(A):
    stale = A[A["stale_sources"] != ""].iloc[0]
    miss = A[A["missing_sources"] != ""].iloc[0]
    assert FL.data_flag_text(stale).count("Stale:") == 1
    assert "Missing:" in FL.data_flag_text(miss)
    clean = A[(A["stale_sources"] == "") & (A["missing_sources"] == "")].iloc[0]
    assert FL.data_flag_text(clean) == "All sources current"


def test_missing_measurements_not_zero_filled():
    U = DL.load_operational("unified")
    st = FL.source_state(U, "clin", "heat_related_visits")
    assert (st == "not reported").sum() == U["clin_reporting_status"].eq("not_reported").sum()
    assert U.loc[st != "reported", "clin_heat_related_visits"].isna().all()


def test_heat_index_extrapolation_warning(A):
    row = A[A["heat_index_validity"] == "extrapolated"].iloc[0]
    assert "outside NWS chart range" in FL.measurement_flag_text(row)
    E = DL.load_operational("episodes")
    E = E[(E["model"] == "C") & (E["window"] == "full")]
    hit = A[A["alert_relies_on_extrapolated_heat_index"]].iloc[0]
    ep = E[(E["zone_id"] == hit["zone_id"]) & (E["start_decision"] <= hit["decision_time"])
           & (E["end_decision"] >= hit["decision_time"])]["episode_id"].iloc[0]
    at = run_page("Alert investigation", ai_ep=ep)
    assert not at.exception
    assert "outside the NWS chart range" in page_text(at)


def test_forecast_scores_never_probabilities():
    assert FL.fmt_ranking_score(0.8) == "0.800" and "%" not in FL.fmt_ranking_score(0.8)
    at = run_page("24-hour forecasting")
    text = page_text(at)
    assert T.FORECAST_INTERPRETATION in text
    assert "chance" not in text.lower()
    for chart in at.get("plotly_chart"):
        assert '"indicator"' not in chart.proto.spec                        # no gauges
    fc = DL.load_operational("forecasts")
    assert not any("prob" in c.lower() for c in fc.columns)


def test_persistence_baseline_beside_forecast():
    at = run_page("24-hour forecasting")
    frames = [d.value for d in at.dataframe]
    cmp_ = next(d for d in frames if "Method" in d.columns)
    assert {"Logistic regression forecast (24 h)", "Persistence baseline (any warning)"} <= set(cmp_["Method"])
    latest = next(d for d in frames if "LR forecast flag" in d.columns)
    assert {"LR forecast flag", "Persistence (any warning) flag"} <= set(latest.columns)
    derived = FCP.forecasts_with_baselines()
    fe = DL.load_evaluation("forecast_evaluation")
    m = derived.merge(fe[["observation_date", "zone_id", "shift", "persistence_any"]],
                      on=["observation_date", "zone_id", "shift"])
    assert (m["persistence_any_flag"] == m["persistence_any"]).all()      # derived exactly as in Phase 4B


# ---------------------------------------------------------------------------------- metric provenance
def test_detection_metrics_match_saved():
    bm = DL.load_evaluation("backtest_metrics")
    for window in ("full", "comparison"):
        tbl = EVP.detection_table(bm, window)
        for r in tbl.itertuples(index=False):
            for m, name in EVP.MODEL_NAMES.items():
                if name not in tbl.columns:
                    continue
                v = bm[(bm["window"] == window) & (bm["model"] == m) & (bm["scope"] == r.scope) & (bm["metric"] == r.metric)]
                shown = tbl.loc[(tbl["metric"] == r.metric) & (tbl["scope"] == r.scope), name].iloc[0]
                if len(v):
                    assert shown == pytest.approx(float(v["value"].iloc[0]), nan_ok=True)


def test_forecast_metrics_match_saved_and_confusions_consistent():
    ms = DL.load_evaluation("model_summary")
    tbl = EVP.forecast_table(ms)
    assert len(tbl) == 4 * 3
    for r in tbl.itertuples(index=False):
        assert np.isfinite(r.n) and r.n > 0
    s = ms[(ms["phase"] == "4B forecast") & ms["subset"].str.startswith("all test rows")]
    cm = EVP.test_confusions(DL.load_evaluation("forecast_evaluation"))
    for model, d in cm.items():
        g = s[s["model"] == model].set_index("metric")["value"]
        assert d["tp"] + d["fn"] == g["positives"]
        assert d["tp"] + d["fp"] + d["fn"] + d["tn"] == g["n"]
        if d["tp"] + d["fp"]:
            assert d["tp"] / (d["tp"] + d["fp"]) == pytest.approx(g["precision"])
        assert d["tp"] / (d["tp"] + d["fn"]) == pytest.approx(g["recall"])


def test_in_sample_metrics_match_phase_4b():
    from src.forecasting import forecast as fc
    overall = fc.run(write=False)[3]
    ins = EVP.in_sample_table(DL.load_evaluation("forecast_evaluation")).set_index("model")
    tr = overall[overall["split"] == "train"].set_index("model")
    for m in ins.index:
        assert ins.loc[m, "f1"] == pytest.approx(tr.loc[m, "f1"], nan_ok=True)
        assert ins.loc[m, "pr_auc"] == pytest.approx(tr.loc[m, "pr_auc"], nan_ok=True)


def test_evt002_finding_matches_saved():
    lead = EVP.evt002_lead(DL.load_evaluation("model_summary"))
    assert lead["logistic regression"] == pytest.approx(-61)
    assert lead["persistence (any warning)"] == pytest.approx(71)
    at = run_page("Model evaluation")
    text = page_text(at)
    assert T.EVALUATION_LIMITATION in text
    assert "61 hours after escalation" in text and "71 hours before escalation" in text


# ------------------------------------------------------------------------------------ read-only app
def _fingerprint() -> dict:
    out = {}
    for d in ("data", "models", "config"):
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(ROOT))] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime)
    return out


def test_dashboard_does_not_modify_data_or_models():
    before = _fingerprint()
    for page in PAGES:
        at = run_page(page)
        assert not at.exception, page
    assert _fingerprint() == before
