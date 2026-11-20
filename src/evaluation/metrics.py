"""Backtest evaluation: model alerts, alert episodes, event matching and metrics.

Predictions are built from replayed model outputs only. Ground truth (labels/events) is joined AFTER
predictions exist and is used solely to score them - never to set thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import schemas

KEY = list(schemas.OBSERVATION_KEY)
ESCALATING = ("gradual_heatwave_escalation", "heat_escalation_clinical_gap")
MODEL_ALERT = {"A": "model_a_alert", "B": "model_b_alert", "C": "model_c_health_alert"}


# --------------------------------------------------------------------------------------- predictions
def build_predictions(F: pd.DataFrame, I: pd.DataFrame, A: pd.DataFrame, bcfg: dict, lag_hours: float) -> pd.DataFrame:
    """Model A/B/C alerts per observation from replayed outputs (no ground truth)."""
    m = A.merge(I[KEY + ["split", "series_position", "eligibility_status", "anomaly_decision", "anomaly_score"]],
                on=KEY, how="left", validate="one_to_one")
    ma = bcfg["models"]["model_a"]
    rz = F[[f"{s}_rz" for s in ma["signals"]]].astype(float)
    cu = F[[f"{s}_cusum" for s in ma["signals"]]].astype(float)
    a_elig = rz.notna().any(axis=1)
    a_alert = ((rz >= ma["z_threshold"]).any(axis=1) | (cu >= ma["cusum_threshold"]).any(axis=1))
    fa = F[KEY].copy()
    fa["model_a_eligible"] = a_elig.to_numpy()
    fa["model_a_alert"] = pd.array(np.where(a_elig, a_alert, False), dtype="boolean")
    fa.loc[~a_elig.to_numpy(), "model_a_alert"] = pd.NA
    fa["model_a_max_rz"] = rz.max(axis=1).to_numpy()
    fa["model_a_max_cusum"] = cu.max(axis=1).to_numpy()
    m = m.merge(fa, on=KEY, how="left", validate="one_to_one")

    b_ok = m["split"].eq("eval") & m["eligibility_status"].eq("eligible")
    m["model_b_eligible"] = b_ok
    m["model_b_alert"] = m["anomaly_decision"].astype("boolean").where(b_ok, pd.NA)

    mc = bcfg["models"]["model_c"]
    level = np.select([m["classification"].isin(mc["high_classes"]), m["classification"].isin(mc["health_classes"]),
                       m["classification"].isin(mc["warning_classes"]), m["classification"].eq("Insufficient evidence")],
                      ["high", "emerging", "warning", "insufficient"], default="none")
    m["model_c_level"] = level
    m["model_c_health_alert"] = pd.array(m["classification"].isin(mc["health_classes"]), dtype="boolean")
    m["model_c_high_alert"] = pd.array(m["classification"].isin(mc["high_classes"]), dtype="boolean")
    m["model_c_warning"] = pd.array(m["classification"].isin(mc["warning_classes"]), dtype="boolean")
    m["shift_end"] = m["decision_time"] - pd.Timedelta(hours=lag_hours)
    m["shift_start"] = m["shift_end"] - pd.Timedelta(hours=12)
    keep = KEY + ["decision_time", "shift_start", "shift_end", "series_position",
                  "model_a_eligible", "model_a_alert", "model_a_max_rz", "model_a_max_cusum",
                  "model_b_eligible", "model_b_alert", "anomaly_score", "split",
                  "model_c_level", "model_c_health_alert", "model_c_high_alert", "model_c_warning",
                  "risk_score", "classification", "evidence_level", "reliability_score", "available_weight",
                  "n_categories_available", "clinical_status", "absenteeism_status", "incidents_status",
                  "environmental_status", "workforce_exposure_status"]
    return m[keep]


def attach_labels(P: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return P.merge(labels[KEY + ["label", "label_name", "event_id"]], on=KEY, how="left", validate="one_to_one")


def window_mask(P: pd.DataFrame, window: str, bcfg: dict, if_cutoff: pd.Timestamp) -> pd.Series:
    post_warmup = P["series_position"] >= bcfg["windows"]["warmup_observations"]
    if window == "full":
        return post_warmup
    if window == "comparison":
        return post_warmup & (P["decision_time"] >= if_cutoff)
    raise ValueError(window)


# ------------------------------------------------------------------------------------------- episodes
def build_episodes(P: pd.DataFrame, alert_col: str, max_gap_hours: float, model: str, window: str) -> pd.DataFrame:
    """Group alerting observations per site-zone (both shifts) into episodes by decision-time gaps."""
    rows = []
    a = P[P[alert_col].fillna(False).astype(bool)].sort_values(["zone_id", "decision_time"])
    for zone, g in a.groupby("zone_id", sort=True):
        gap = g["decision_time"].diff() > pd.Timedelta(hours=max_gap_hours)
        for _, e in g.groupby(gap.cumsum()):
            rows.append({
                "model": model, "window": window, "site_id": e["site_id"].iloc[0], "zone_id": zone,
                "start_decision": e["decision_time"].min(), "end_decision": e["decision_time"].max(),
                "obs_start": e["shift_start"].min(), "obs_end": e["shift_end"].max(),
                "n_alerts": len(e), "peak_risk_score": e["risk_score"].max(),
                "mean_reliability": e["reliability_score"].mean(), "mean_available_weight": e["available_weight"].mean(),
                "max_level": e["model_c_level"].map({"none": 0, "insufficient": 0, "warning": 1, "emerging": 2,
                                                       "high": 3}).max(),
            })
    cols = ["model", "window", "site_id", "zone_id", "start_decision", "end_decision", "obs_start", "obs_end",
            "n_alerts", "peak_risk_score", "mean_reliability", "mean_available_weight", "max_level"]
    E = pd.DataFrame(rows, columns=cols)
    for c in ("start_decision", "end_decision", "obs_start", "obs_end"):
        E[c] = pd.to_datetime(E[c])                       # keeps datetime dtype when there are no episodes
    E = E.sort_values(["zone_id", "start_decision"]).reset_index(drop=True)
    E.insert(2, "episode_id", [f"{model}-{window}-{i + 1:03d}" for i in range(len(E))])
    E["duration_hours"] = (E["obs_end"] - E["obs_start"]).dt.total_seconds() / 3600
    return E


def _overlap_hours(s1, e1, s2, e2) -> float:
    return max(0.0, (min(e1, e2) - max(s1, s2)).total_seconds() / 3600)


def match_episodes(E: pd.DataFrame, events: pd.DataFrame, mcfg: dict) -> pd.DataFrame:
    """Match each episode to at most one escalating event-zone; each event-zone is credited once.

    Window per event-zone: [onset - pre_tol, end + post_tol). Episodes are processed chronologically; an episode
    overlapping several event-zones takes the largest overlap. Later episodes on an already-credited event are
    `matched_duplicate` (not double-counted, not false). Episodes overlapping only a negative-control event are
    `control_false_alarm`; all other unmatched episodes are `false_alarm`.
    """
    pre = pd.Timedelta(hours=mcfg["pre_onset_tolerance_hours"])
    post = pd.Timedelta(hours=mcfg["post_end_tolerance_hours"])
    ev = events.copy()
    ev["win_start"] = ev["onset_timestamp"] - pre
    ev["win_end"] = ev["end_timestamp"] + post
    credited: set[tuple[str, str]] = set()
    status, matched, overlap = [], [], []
    for e in E.sort_values("start_decision").itertuples():
        cand = ev[(ev["zone_id"] == e.zone_id)]
        best, best_ov, ctrl = None, 0.0, False
        for c in cand.itertuples():
            ov = _overlap_hours(e.obs_start, e.obs_end, c.win_start, c.win_end)
            if ov <= 0:
                continue
            if c.event_type in ESCALATING:
                if ov > best_ov:
                    best, best_ov = c, ov
            else:
                ctrl = True
        if best is not None:
            k = (best.event_id, best.zone_id)
            status.append("matched_duplicate" if k in credited else "matched")
            credited.add(k)
            matched.append(best.event_id)
            overlap.append(best_ov)
        else:
            status.append("control_false_alarm" if ctrl else "false_alarm")
            matched.append(pd.NA)
            overlap.append(0.0)
    order = E.sort_values("start_decision").index
    out = E.copy()
    out.loc[order, "match_status"] = status
    out.loc[order, "matched_event_id"] = matched
    out.loc[order, "overlap_hours"] = overlap
    return out


# ------------------------------------------------------------------------------------- event results
def event_results(P: pd.DataFrame, E: pd.DataFrame, events: pd.DataFrame, alert_col: str, model: str, window: str,
                  wmask: pd.Series, mcfg: dict) -> pd.DataFrame:
    pre = pd.Timedelta(hours=mcfg["pre_onset_tolerance_hours"])
    post = pd.Timedelta(hours=mcfg["post_end_tolerance_hours"])
    W = P[wmask]
    w_start = W["shift_start"].min()
    w_end = W["shift_end"].max()
    rows = []
    for ev in events.itertuples():
        esc = ev.event_type in ESCALATING
        r = {"model": model, "window": window, "event_id": ev.event_id, "site_id": ev.site_id, "zone_id": ev.zone_id,
             "event_type": ev.event_type, "role": "escalating_event" if esc else "negative_control",
             "onset_timestamp": ev.onset_timestamp, "escalation_timestamp": ev.escalation_timestamp,
             "end_timestamp": ev.end_timestamp}
        if ev.onset_timestamp < w_start or ev.onset_timestamp > w_end:
            r.update(included=False, detected=pd.NA, excluded_reason=(
                "onset before evaluation window (warm-up/training period)" if ev.onset_timestamp < w_start
                else "after evaluation window"))
            rows.append(r)
            continue
        z = W[W["zone_id"] == ev.zone_id]
        inwin = z[(z["shift_start"] >= ev.onset_timestamp - pre) & (z["shift_start"] < ev.end_timestamp + post)]
        alerts = inwin[inwin[alert_col].fillna(False).astype(bool)]
        r.update(included=True, excluded_reason=pd.NA, n_alerts_in_window=len(alerts),
                 n_ineligible_in_window=int(inwin[alert_col].isna().sum()))
        if esc:
            eps = E[(E["matched_event_id"] == ev.event_id) & (E["zone_id"] == ev.zone_id)]
            first = alerts["decision_time"].min() if len(alerts) else pd.NaT
            detected = len(eps) > 0 and pd.notna(first)
            lab = z[(z["event_id"] == ev.event_id) & z["label"].isin(mcfg["positive_labels"])]
            r.update(detected=detected, first_alert_decision=first,
                     detection_delay_hours=(first - ev.onset_timestamp).total_seconds() / 3600 if detected else np.nan,
                     lead_time_to_escalation_hours=((ev.escalation_timestamp - first).total_seconds() / 3600
                                                    if detected else np.nan),
                     detected_before_escalation=bool(detected and first <= ev.escalation_timestamp),
                     coverage=float(lab[alert_col].fillna(False).astype(bool).mean()) if len(lab) else np.nan,
                     n_event_observations=len(lab),
                     n_matched_episodes=len(eps),
                     n_duplicate_episodes=int((eps["match_status"] == "matched_duplicate").sum()))
        else:
            r.update(detected=pd.NA, control_false_alert=bool(len(alerts) > 0))
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- observation metrics
def confusion(y: pd.Series, pred: pd.Series) -> dict:
    y = y.astype(bool).to_numpy()
    p = pred.fillna(False).astype(bool).to_numpy()
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    if tp > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0 if (fp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    return {"n": len(y), "positives": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "false_positive_rate": fpr}


def observation_metrics(P: pd.DataFrame, wmask: pd.Series, alert_col: str, positive_labels) -> dict:
    W = P[wmask]
    d = confusion(W["label"].isin(positive_labels), W[alert_col])
    d["n_ineligible"] = int(W[alert_col].isna().sum())
    return d


def episode_metrics(E: pd.DataFrame, P: pd.DataFrame, wmask: pd.Series) -> dict:
    W = P[wmask]
    weeks = W["observation_date"].nunique() / 7.0
    n = len(E)
    false = E["match_status"].isin(["false_alarm", "control_false_alarm"])
    return {"n_episodes": n, "matched": int((E["match_status"] == "matched").sum()),
            "matched_duplicate": int((E["match_status"] == "matched_duplicate").sum()),
            "false_alarm_episodes": int(false.sum()),
            "control_false_alarm_episodes": int((E["match_status"] == "control_false_alarm").sum()),
            "episode_ppv": float((~false).sum() / n) if n else np.nan,
            "false_alerts_per_site_week": float(false.sum() / (W["site_id"].nunique() * weeks)) if weeks else np.nan,
            "median_episode_duration_hours": float(E["duration_hours"].median()) if n else np.nan,
            "mean_episode_duration_hours": float(E["duration_hours"].mean()) if n else np.nan,
            "weeks_in_window": weeks}


def false_alerts_by_site(E: pd.DataFrame, P: pd.DataFrame, wmask: pd.Series) -> pd.DataFrame:
    W = P[wmask]
    weeks = W["observation_date"].nunique() / 7.0
    f = E[E["match_status"].isin(["false_alarm", "control_false_alarm"])]
    out = f.groupby("site_id").size().reindex(sorted(W["site_id"].unique()), fill_value=0).rename("false_episodes")
    out = out.to_frame()
    out["false_alerts_per_week"] = out["false_episodes"] / weeks
    return out.reset_index()


def event_summary(R: pd.DataFrame) -> dict:
    inc = R[R["included"] & (R["role"] == "escalating_event")]
    det = inc[inc["detected"].astype(bool)]
    ctrl = R[R["included"] & (R["role"] == "negative_control")]
    return {"event_zones_included": len(inc), "event_zones_detected": len(det),
            "event_sensitivity": len(det) / len(inc) if len(inc) else np.nan,
            "detected_before_escalation": int(inc["detected_before_escalation"].astype(bool).sum()) if len(inc) else 0,
            "median_lead_time_to_escalation_hours": float(det["lead_time_to_escalation_hours"].median()) if len(det) else np.nan,
            "median_detection_delay_hours": float(det["detection_delay_hours"].median()) if len(det) else np.nan,
            "mean_detection_coverage": float(inc["coverage"].mean()) if len(inc) else np.nan,
            "undetected_event_zones": ", ".join(f"{r.event_id}/{r.zone_id}" for r in inc[~inc["detected"].astype(bool)].itertuples()),
            "excluded_event_zones": ", ".join(f"{r.event_id}/{r.zone_id}" for r in R[~R["included"]].itertuples()),
            "controls_included": len(ctrl),
            "controls_with_false_alert": int(ctrl["control_false_alert"].astype(bool).sum()) if len(ctrl) else 0}


# ------------------------------------------------------------------------------------- data states
def data_state_masks(P: pd.DataFrame, bcfg: dict) -> dict[str, pd.Series]:
    """Subsets defined from data availability only (never from labels)."""
    k = bcfg["data_states"]["multiple_missing_max_categories"]
    return {
        "all_sources_available": P["n_categories_available"] == 5,
        "clinical_missing_or_stale": P["clinical_status"] != "available",
        "absenteeism_delayed_or_stale": P["absenteeism_status"].isin(["stale", "missing_data"]),
        f"multiple_missing (<= {k} of 5 categories)": P["n_categories_available"] <= k,
    }
