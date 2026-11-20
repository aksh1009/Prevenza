"""Phase 4A: backtesting and operational validation."""
import numpy as np
import pandas as pd
import pytest

from src import config, schemas
from src.evaluation import backtest as bt
from src.evaluation import metrics as M
from src.evaluation.replay import Replayer, compare, information_state

KEY = list(schemas.OBSERVATION_KEY)
BCFG = bt.load_backtest_config()
MCFG = BCFG["matching"]
PROC = config.path_for("processed")
T0 = pd.Timestamp("2025-07-10 06:00")


# ------------------------------------------------------------------------------------ synthetic helpers
def P_frame(alerts: dict[str, list[int]], n_shifts=20, start=T0):
    """Zone -> list of shift indices (12 h steps) that alert. Returns a prediction-like frame."""
    rows = []
    for zone, idx in alerts.items():
        for i in range(n_shifts):
            s = start + pd.Timedelta(hours=12 * i)
            rows.append({"observation_date": s.normalize(), "site_id": zone.split("-")[0], "zone_id": zone,
                         "shift": "day" if s.hour == 6 else "night", "shift_start": s,
                         "shift_end": s + pd.Timedelta(hours=12), "decision_time": s + pd.Timedelta(hours=13),
                         "alert": i in idx, "risk_score": 50.0, "reliability_score": 80.0,
                         "available_weight": 0.8, "model_c_level": "emerging" if i in idx else "none",
                         "series_position": 20 + i // 2, "label": 0, "event_id": pd.NA})
    return pd.DataFrame(rows)


def events_frame(*evs):
    cols = ["event_id", "site_id", "zone_id", "event_type", "onset_timestamp", "escalation_timestamp",
            "end_timestamp", "severity", "description"]
    return pd.DataFrame([dict(zip(cols, e)) for e in evs])


def esc_event(eid, zone, onset_i, esc_i, end_i, start=T0):
    t = lambda i: start + pd.Timedelta(hours=12 * i)
    return (eid, zone.split("-")[0], zone, "gradual_heatwave_escalation", t(onset_i), t(esc_i), t(end_i), "high", "x")


def ctrl_event(eid, zone, i, start=T0):
    t = lambda k: start + pd.Timedelta(hours=12 * k)
    return (eid, zone.split("-")[0], zone, "isolated_temperature_spike", t(i), pd.NaT, t(i + 1), "none", "x")


def episodes(P, max_gap=24):
    return M.build_episodes(P, "alert", max_gap, "T", "w")


# ------------------------------------------------------------------------------------------- replay
@pytest.fixture(scope="module")
def replayer():
    return Replayer()


def test_information_state_only_contains_reported_data(replayer):
    T = pd.Timestamp("2025-07-05 07:00")
    s = information_state(replayer.u, T, replayer.lag)
    assert (s["shift_end"] + pd.Timedelta(hours=replayer.lag) <= T).all()
    for p in ("env", "wf", "clin", "abs", "inc"):
        rep = s[f"{p}_reporting_timestamp"]
        assert (rep.dropna() <= T).all()
        unknown = s[f"{p}_reporting_status"] == "not_reported"
        val = [c for c in s.columns if c.startswith(f"{p}_") and c not in
               (f"{p}_recorded_at", f"{p}_reporting_status", f"{p}_reporting_timestamp")]
        assert s.loc[unknown, val].isna().all().all()


def test_replay_reproduces_saved_outputs(replayer):
    times = [replayer.times()[i] for i in (30, 95, 170)]
    F, I, A = replayer.run(times)
    for df, name in ((F, "features.parquet"), (I, "isolation_forest_scores.parquet"), (A, "risk_assessments.parquet")):
        c = compare(df, pd.read_parquet(PROC / name))
        assert len(c) == 3 * 9 and c["replay_match"].all(), name


def test_replay_step_ignores_future_data(replayer):
    T = replayer.times()[100]
    base = replayer.step(T)[2]
    saved_u = replayer.u
    try:
        u = saved_u.copy()
        later = u["shift_start"] > T
        u.loc[later, ["env_heat_index_celsius", "clin_heat_related_visits"]] = 999
        u.loc[later, "env_reporting_timestamp"] = u.loc[later, "env_recorded_at"]
        replayer.u = u
        again = replayer.step(T)[2]
    finally:
        replayer.u = saved_u
    pd.testing.assert_frame_equal(base, again)


# ------------------------------------------------------------------------------------------- episodes
def test_episode_grouping_and_gap():
    P = P_frame({"S01-Z01": [2, 3, 4, 8, 9]})
    E = episodes(P, max_gap=24)
    assert len(E) == 2
    assert E["n_alerts"].tolist() == [3, 2]
    assert E["duration_hours"].tolist() == [36.0, 24.0]
    assert len(episodes(P, max_gap=48)) == 1                    # 4 -> 8 is 48 h apart
    assert len(episodes(P_frame({"S01-Z01": [2, 4]}), max_gap=24)) == 1    # one skipped shift bridged
    assert len(episodes(P_frame({"S01-Z01": [2, 5]}), max_gap=24)) == 2


def test_episodes_are_per_zone():
    E = episodes(P_frame({"S01-Z01": [2, 3], "S01-Z02": [2, 3]}))
    assert len(E) == 2 and set(E["zone_id"]) == {"S01-Z01", "S01-Z02"}


# ------------------------------------------------------------------------------------------- matching
def test_matching_by_zone_and_time():
    P = P_frame({"S01-Z01": [6, 7, 15], "S01-Z02": [6, 7]})
    ev = events_frame(esc_event("E1", "S01-Z01", 4, 8, 10))
    E = M.match_episodes(episodes(P), ev, MCFG)
    s = E.set_index(["zone_id", "n_alerts"])["match_status"]
    assert s[("S01-Z01", 2)] == "matched"
    assert s[("S01-Z02", 2)] == "false_alarm"                    # other zone
    assert s[("S01-Z01", 1)] == "false_alarm"                    # 15 is beyond end + 24 h


def test_matching_tolerances():
    ev = events_frame(esc_event("E1", "S01-Z01", 4, 8, 10))
    before = M.match_episodes(episodes(P_frame({"S01-Z01": [2, 3]})), ev, MCFG)
    assert before["match_status"].tolist() == ["false_alarm"]    # ends at onset; pre-tolerance 0
    late = M.match_episodes(episodes(P_frame({"S01-Z01": [11]})), ev, MCFG)
    assert late["match_status"].tolist() == ["matched"]          # within end + 24 h


def test_duplicate_event_matching_prevented():
    ev = events_frame(esc_event("E1", "S01-Z01", 4, 8, 14))
    P = P_frame({"S01-Z01": [5, 6, 11, 12]})
    E = M.match_episodes(episodes(P), ev, MCFG)
    assert E["match_status"].tolist() == ["matched", "matched_duplicate"]
    P["label"] = 0
    P["model_c_health_alert"] = P["alert"]
    R = M.event_results(P, E, ev, "alert", "T", "w", pd.Series(True, index=P.index), MCFG)
    assert R["detected"].tolist() == [True]
    assert R["n_matched_episodes"].tolist() == [2] and R["n_duplicate_episodes"].tolist() == [1]
    s = M.event_summary(R)
    assert s["event_zones_detected"] == 1 and s["event_sensitivity"] == 1.0


def test_one_episode_matches_only_one_event():
    ev = events_frame(esc_event("E1", "S01-Z01", 2, 3, 5), esc_event("E2", "S01-Z01", 6, 8, 12))
    E = M.match_episodes(episodes(P_frame({"S01-Z01": [4, 5, 6, 7, 8, 9]})), ev, MCFG)
    assert len(E) == 1 and E["matched_event_id"].iloc[0] == "E2"      # larger overlap wins
    R = M.event_results(P_frame({"S01-Z01": [4, 5, 6, 7, 8, 9]}), E, ev, "alert", "T", "w",
                        pd.Series(True, index=range(20)), MCFG)
    assert R.set_index("event_id")["detected"].to_dict() == {"E1": False, "E2": True}


# ------------------------------------------------------------------------------------------ lead time
def test_lead_time_and_detection_delay():
    ev = events_frame(esc_event("E1", "S01-Z01", 4, 10, 14))
    P = P_frame({"S01-Z01": [6, 7, 8]})
    E = M.match_episodes(episodes(P), ev, MCFG)
    R = M.event_results(P, E, ev, "alert", "T", "w", pd.Series(True, index=P.index), MCFG)
    first = T0 + pd.Timedelta(hours=12 * 6 + 13)
    assert R["first_alert_decision"].iloc[0] == first
    assert R["detection_delay_hours"].iloc[0] == (first - ev["onset_timestamp"].iloc[0]).total_seconds() / 3600 == 37
    assert R["lead_time_to_escalation_hours"].iloc[0] == 120 - (72 + 13) == 35
    assert bool(R["detected_before_escalation"].iloc[0])


def test_alert_after_escalation_gives_negative_lead():
    ev = events_frame(esc_event("E1", "S01-Z01", 4, 6, 14))
    P = P_frame({"S01-Z01": [9]})
    R = M.event_results(P, M.match_episodes(episodes(P), ev, MCFG), ev, "alert", "T", "w",
                        pd.Series(True, index=P.index), MCFG)
    assert R["lead_time_to_escalation_hours"].iloc[0] < 0
    assert not bool(R["detected_before_escalation"].iloc[0])


# ------------------------------------------------------------------------------------- windows / warm-up
def test_warmup_rows_excluded_from_windows():
    P = P_frame({"S01-Z01": []})
    P["series_position"] = np.arange(len(P))
    wm = M.window_mask(P, "full", BCFG, pd.Timestamp("2030-01-01"))
    assert (~wm[P["series_position"] < BCFG["windows"]["warmup_observations"]]).all()
    assert wm[P["series_position"] >= BCFG["windows"]["warmup_observations"]].all()


def test_event_before_window_excluded_with_reason():
    ev = events_frame(esc_event("E1", "S01-Z01", -10, -5, 3))
    P = P_frame({"S01-Z01": [1]})
    R = M.event_results(P, M.match_episodes(episodes(P), ev, MCFG), ev, "alert", "T", "w",
                        pd.Series(True, index=P.index), MCFG)
    assert not R["included"].iloc[0] and "before evaluation window" in R["excluded_reason"].iloc[0]
    assert M.event_summary(R)["event_zones_included"] == 0


# -------------------------------------------------------------------------------------- false alarms
def test_false_alerts_per_site_week_and_controls():
    ev = events_frame(ctrl_event("C1", "S01-Z01", 4))
    P = P_frame({"S01-Z01": [4, 12], "S02-Z01": []}, n_shifts=28)
    E = M.match_episodes(episodes(P), ev, MCFG)
    assert sorted(E["match_status"]) == ["control_false_alarm", "false_alarm"]
    wm = pd.Series(True, index=P.index)
    em = M.episode_metrics(E, P, wm)
    weeks = P["observation_date"].nunique() / 7
    assert em["false_alarm_episodes"] == 2 and em["episode_ppv"] == 0.0
    assert em["false_alerts_per_site_week"] == pytest.approx(2 / (2 * weeks))
    fs = M.false_alerts_by_site(E, P, wm).set_index("site_id")["false_episodes"].to_dict()
    assert fs == {"S01": 2, "S02": 0}
    R = M.event_results(P, E, ev, "alert", "T", "w", wm, MCFG)
    assert bool(R["control_false_alert"].iloc[0]) and pd.isna(R["detected"].iloc[0])


def test_confusion_counts():
    d = M.confusion(pd.Series([1, 1, 0, 0, 0]).astype(bool), pd.Series([True, pd.NA, True, False, pd.NA], dtype="boolean"))
    assert (d["tp"], d["fn"], d["fp"], d["tn"]) == (1, 1, 1, 2)
    assert d["precision"] == 0.5 and d["recall"] == 0.5 and d["f1"] == 0.5


# ------------------------------------------------------------------------------ real backtest outputs
@pytest.fixture(scope="module")
def outputs():
    o = BCFG["output"]
    paths = [PROC / o[k] for k in ("backtest_results", "event_results", "episodes", "metrics")]
    if not all(p.exists() for p in paths):
        pytest.skip("run: python -m src.evaluation.backtest")
    return [pd.read_parquet(p) for p in paths]


def test_scenario_results_present(outputs):
    P, EV, EP, Mx = outputs
    for m, w in (("A", "full"), ("C", "full"), ("A", "comparison"), ("B", "comparison"), ("C", "comparison")):
        e = EV[(EV["model"] == m) & (EV["window"] == w)]
        assert set(e["event_id"]) == {"EVT-001", "EVT-002", "EVT-003", "EVT-004", "EVT-005"}
    b = EV[EV["model"] == "B"].set_index(["event_id", "zone_id"])
    assert not b.loc[("EVT-001", "S02-Z01"), "included"]        # training period: not evaluable for B
    assert not b.loc[("EVT-004", "S03-Z03"), "included"]
    inc = EV[EV["included"] & (EV["role"] == "escalating_event")]
    assert ((inc["n_matched_episodes"] >= 1) == inc["detected"].astype(bool)).all()
    assert EV.loc[EV["role"] == "negative_control", "detected"].isna().all()


def test_model_b_only_out_of_sample(outputs):
    P = outputs[0]
    assert P.loc[P["split"] != "eval", "model_b_alert"].isna().all()
    assert (P.loc[P["model_b_alert"].notna(), "decision_time"] >= pd.Timestamp(P["version_if_train_cutoff"].iloc[0])).all()
    ep = outputs[2]
    assert set(ep.loc[ep["model"] == "B", "window"]) <= {"comparison"}


def test_comparison_window_identical_rows(outputs):
    Mx = outputs[3]
    n = Mx[(Mx["window"] == "comparison") & (Mx["scope"] == "observation") & (Mx["metric"] == "n")]
    assert n["value"].nunique() == 1 and set(n["model"]) >= {"A", "B", "C"}


def test_missing_data_subsets_defined_from_availability(outputs):
    P = outputs[0]
    masks = M.data_state_masks(P, BCFG)
    assert (P.loc[masks["clinical_missing_or_stale"], "clinical_status"] != "available").all()
    assert (P.loc[masks["all_sources_available"], "n_categories_available"] == 5).all()
    assert masks["clinical_missing_or_stale"].sum() > 0 and masks["absenteeism_delayed_or_stale"].sum() > 0
    wu = P["series_position"] < BCFG["windows"]["warmup_observations"]
    assert not P.loc[wu, "in_full_window"].any() and not P.loc[wu, "in_comparison_window"].any()


def test_replay_verified_all_rows(outputs):
    P = outputs[0]
    cols = [c for c in P.columns if c.startswith("replay_match_")]
    assert len(cols) == 3
    replayed = P[cols].notna().all(axis=1)
    assert replayed.any()
    assert P.loc[replayed, cols].astype(bool).all().all()


def test_evaluation_reproducible_and_labels_not_used_for_predictions(outputs):
    P, EV, EP, Mx = outputs
    from src.preprocessing.load import load_dataset
    labels, events = load_dataset("observation_labels"), load_dataset("heat_risk_events")
    cutoff = pd.Timestamp(P["version_if_train_cutoff"].iloc[0])
    pred_cols = [c for c in P.columns if c not in ("label", "label_name", "event_id", "in_full_window",
                                                     "in_comparison_window") and not c.startswith(("replay_", "version_"))]
    P0 = P[pred_cols]
    _, Mx2, EP2, EV2, *_ = bt.evaluate(P0, labels, events, BCFG, cutoff)
    pd.testing.assert_frame_equal(Mx2.reset_index(drop=True), Mx.reset_index(drop=True))
    pd.testing.assert_frame_equal(EV2.reset_index(drop=True), EV.reset_index(drop=True), check_dtype=False)
    shuffled = labels.copy()
    shuffled["label"] = shuffled["label"].sample(frac=1, random_state=1).to_numpy()
    P3, Mx3, *_ = bt.evaluate(P0, shuffled, events, BCFG, cutoff)
    pd.testing.assert_frame_equal(P3[pred_cols].reset_index(drop=True), P0.reset_index(drop=True))
    assert not Mx3["value"].equals(Mx["value"])                 # metrics depend on labels; predictions do not
