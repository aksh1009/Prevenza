"""Phase 4C - pre-dashboard evaluation audit (read-only with respect to models, thresholds, splits and features).

    python -m src.evaluation.audit   -> docs/pre_dashboard_audit.md, data/processed/model_comparison_summary.parquet

Every number in the audit document is computed here from existing outputs. Nothing is refitted or re-tuned:
the persisted forecast model, threshold and splits are reused as-is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src import config, schemas
from src.features.quality import NWS_MAX_HEAT_INDEX_C, NWS_MAX_TEMPERATURE_C, heat_index_validity
from src.forecasting import forecast as fc
from src.preprocessing.load import load_dataset

KEY = list(schemas.OBSERVATION_KEY)
PROC = config.path_for("processed")
DOC = config.PROJECT_ROOT / "docs" / "pre_dashboard_audit.md"
MODELS = {"persistence (health alert)": ("persistence_health", None),
          "persistence (any warning)": ("persistence_any", None),
          "logistic regression": (None, "lr_score")}


# ------------------------------------------------------------------------------------------ helpers
def _metrics(y, flag, score=None) -> dict:
    y = np.asarray(y, int)
    f = np.asarray(flag, bool)
    yb = y.astype(bool)
    tp, fp, fn = int((yb & f).sum()), int((~yb & f).sum()), int((yb & ~f).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if fp + fn else np.nan)
    s = f.astype(float) if score is None else np.asarray(score, float)
    ap = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan
    return {"n": len(y), "positives": int(y.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "pr_auc": ap}


def _model_rows(E: pd.DataFrame, thr: float, score_col: str = "lr_score") -> list[dict]:
    out = []
    for name, (flag_col, sc) in MODELS.items():
        if sc:
            out.append({"model": name, **_metrics(E["target"], E[score_col] >= thr, E[score_col])})
        else:
            out.append({"model": name, **_metrics(E["target"], E[flag_col])})
    return out


def _md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("NA" if np.isnan(v) else fmt.format(v))
            elif isinstance(v, pd.Timestamp):
                cells.append(v.strftime("%Y-%m-%d") if (v.hour, v.minute) == (0, 0) else v.strftime("%Y-%m-%d %H:%M"))
            elif v is None or v is pd.NA or v is pd.NaT:
                cells.append("NA")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- loading
def load_context():
    cfg = fc.load_forecast_config()
    D, final, meta, overall, fct = fc.run(cfg, write=False)        # deterministic; identical to saved model
    saved_pipe, saved_meta = fc.load_model(cfg)
    if saved_meta["model_version"] != meta["model_version"]:
        raise RuntimeError("persisted forecast model differs from a fresh run - outputs are stale")
    events = load_dataset("heat_risk_events")
    labels = load_dataset("observation_labels")
    F = pd.read_parquet(PROC / "features.parquet")
    A = pd.read_parquet(PROC / "risk_assessments.parquet")
    return cfg, D, final, meta, events, labels, F, A


def evaluable(D, split):
    return D[(D["split"] == split) & D["eligible"] & D["target"].notna()]


# ------------------------------------------------------------------------------------ 1 independence
def split_audit(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for s in ("train", "purged", "validation", "test"):
        d = D[D["split"] == s]
        e = d[d["eligible"] & d["target"].notna()]
        rows.append({"split": s, "first shift start": d["shift_start"].min(), "last shift start": d["shift_start"].max(),
                     "rows": len(d), "evaluable rows": len(e), "positive targets": int(e["target"].sum())})
    E = D[D["eligible"] & D["target"].notna() & (D["target"] == 1)]
    ev = pd.crosstab(E["target_event_id"], E["split"]).reindex(columns=["train", "purged", "validation", "test"],
                                                               fill_value=0).reset_index()
    return pd.DataFrame(rows), ev


def evt005_effect(D, events, thr, feature_window_days: int) -> tuple[pd.DataFrame, dict]:
    """Test metrics on all test rows vs subsets that remove the event seen during train+validation refit."""
    T = evaluable(D, "test")
    e5 = events[events["event_id"] == "EVT-005"].iloc[0]
    influenced = (T["zone_id"] == e5.zone_id) & (T["shift_start"] < e5.end_timestamp + pd.Timedelta(days=feature_window_days))
    subsets = {
        "all test rows (as reported in Phase 4B)": T,
        "excluding rows whose target is EVT-005": T[T["target_event_id"].fillna("") != "EVT-005"],
        f"unseen-event subset (also drop {e5.zone_id} rows < EVT-005 end + {feature_window_days} d)": T[~influenced],
    }
    rows = []
    for name, E in subsets.items():
        for r in _model_rows(E, thr):
            rows.append({"test subset": name, **r})
    tail = T[influenced]
    info = {"influenced_rows": int(influenced.sum()), "influenced_positives": int(tail["target"].sum()),
            "lr_flags_in_tail": int((tail["lr_score"] >= thr).sum()),
            "lr_false_flags_in_tail": int(((tail["lr_score"] >= thr) & (tail["target"] == 0)).sum()),
            "lr_false_flags_all_test": int(((T["lr_score"] >= thr) & (T["target"] == 0)).sum()),
            "zone": e5.zone_id, "tail_end": e5.end_timestamp + pd.Timedelta(days=feature_window_days)}
    return pd.DataFrame(rows), info


# --------------------------------------------------------------------------------- 2 vs persistence
def event_level(D, events, thr) -> pd.DataFrame:
    """Per evaluable event: first flag time in [onset - 24 h, end) of the zone, lead times, event recall."""
    rows = []
    for split, score_col, note in (("validation", "lr_score_selection", "train-only model (out-of-sample)"),
                                   ("test", "lr_score", "final refit model (out-of-sample)")):
        E = evaluable(D, split)
        for ev_id in sorted(E.loc[E["target"] == 1, "target_event_id"].dropna().unique()):
            ev = events[events["event_id"] == ev_id].iloc[0]
            z = E[(E["zone_id"] == ev.zone_id) & (E["shift_start"] >= ev.onset_timestamp - pd.Timedelta(hours=24))
                  & (E["shift_start"] < ev.end_timestamp)]
            pos = E[E["target_event_id"] == ev_id]
            for name, (flag_col, sc) in MODELS.items():
                flag = (z[score_col] >= thr) if sc else z[flag_col].astype(bool)
                pflag = (pos[score_col] >= thr) if sc else pos[flag_col].astype(bool)
                first = z.loc[flag, "decision_time"].min() if flag.any() else pd.NaT
                rows.append({
                    "split": split, "event_id": ev_id, "zone_id": ev.zone_id, "model": name,
                    "lr_model": note if sc else "-",
                    "evaluable event rows (positive targets)": len(pos), "event recall": float(pflag.mean()),
                    "detected": bool(pflag.any()),
                    "first flag in window": first,
                    "hours before onset": (ev.onset_timestamp - first).total_seconds() / 3600 if pd.notna(first) else np.nan,
                    "hours before escalation": (ev.escalation_timestamp - first).total_seconds() / 3600 if pd.notna(first) else np.nan,
                    "window starts": z["decision_time"].min(),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ 3 foundry
FOUNDRY_FEATURES = ["heat_index_celsius_last", "heat_index_celsius_hist_median", "heat_index_celsius_rz",
                    "heat_index_celsius_cusum", "heat_exposure_load_roll3", "heat_related_visits_rz",
                    "absence_rate_rz", "heat_incident_count_rz", "clin_staleness_hours", "ctx_outdoor",
                    "ctx_radiant_heat", "risk_score", "environmental_score", "workforce_exposure_score",
                    "clinical_score", "absenteeism_score", "incidents_score", "lr_score"]


def foundry_phases(D, labels, events, F) -> pd.DataFrame:
    extra = [f for f in FOUNDRY_FEATURES if f not in D.columns]
    L = D.merge(labels[KEY + ["label"]], on=KEY).merge(F[KEY + extra], on=KEY)
    rows = []
    for ev_id in ("EVT-001", "EVT-002"):
        ev = events[events["event_id"] == ev_id]
        zones = list(ev["zone_id"])
        onset = ev["onset_timestamp"].iloc[0]
        z = L[L["zone_id"].isin(zones)]
        phases = {"reference (14 d pre-onset)": z[(z["shift_start"] < onset) & (z["shift_start"] >= onset - pd.Timedelta(days=14))],
                  "emerging (label 1)": z[z["label"] == 1], "escalating (label 2)": z[z["label"] == 2]}
        for ph, g in phases.items():
            r = {"event": f"{ev_id} ({', '.join(zones)})", "phase": ph, "rows": len(g)}
            r.update({f: float(pd.to_numeric(g[f], errors="coerce").mean()) for f in FOUNDRY_FEATURES})
            rows.append(r)
    return pd.DataFrame(rows)


def lr_contributions(D, final, cfg) -> pd.DataFrame:
    """Mean logit contribution (coef x transformed value) for positive-target rows of EVT-001 vs EVT-002."""
    feats = fc.feature_names(cfg)
    pre = final[:-1]
    names = pre.get_feature_names_out(feats)
    coef = final.named_steps["model"].coef_[0]
    out = {}
    for ev_id in ("EVT-001", "EVT-002"):
        rows = D[(D["target_event_id"] == ev_id) & D["eligible"]]
        Z = pre.transform(rows[feats].astype(float))
        out[ev_id] = (Z * coef).mean(axis=0)
    df = pd.DataFrame({"feature": names, "coefficient": coef, "EVT-001 positives": out["EVT-001"],
                       "EVT-002 positives": out["EVT-002"]})
    df["difference (002 - 001)"] = df["EVT-002 positives"] - df["EVT-001 positives"]
    df = df.reindex(df["difference (002 - 001)"].abs().sort_values(ascending=False).index)
    intercept = float(final.named_steps["model"].intercept_[0])
    df.attrs["intercept"] = intercept
    return df.head(10).reset_index(drop=True)


# -------------------------------------------------------------------------------------- 4 scores
def score_audit(D, thr) -> tuple[pd.DataFrame, dict]:
    V, T = evaluable(D, "validation"), evaluable(D, "test")
    rows = []
    for name, E, col, note in (("validation", V, "lr_score_selection", "train-only model (threshold source)"),
                               ("validation", V, "lr_score", "final refit model (IN-SAMPLE)"),
                               ("test", T, "lr_score", "final refit model (out-of-sample)")):
        y = E["target"].astype(int)
        s = E[col]
        rows.append({"rows": name, "scores from": note, "prevalence": float(y.mean()), "mean score": float(s.mean()),
                     "mean score (negatives)": float(s[y == 0].mean()), "mean score (positives)": float(s[y == 1].mean()),
                     "flag share at threshold": float((s >= thr).mean()),
                     "negatives above threshold": float((s[y == 0] >= thr).mean()),
                     "brier (uncalibrated score)": float(brier_score_loss(y, s))})
    T_ = T.copy()
    info = {"threshold": thr, "test_mean_score_vs_prevalence": (float(T_["lr_score"].mean()), float(T_["target"].mean()))}
    return pd.DataFrame(rows), info


# ------------------------------------------------------------------------------------ 5 heat index
def heat_index_audit(A: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    env = load_dataset("environmental_data")
    dq = load_dataset("data_quality_log")
    labels = load_dataset("observation_labels")
    env["validity"] = heat_index_validity(env["temperature_celsius"], env["heat_index_celsius"])
    x = env[env["validity"] == "extrapolated"].merge(labels[KEY + ["label", "event_id"]], on=KEY, how="left")
    spikes = dq[dq["issue_type"] == "sensor_spike"]
    shift_start = x["observation_date"] + pd.to_timedelta(x["shift"].map({"day": 6, "night": 18}), unit="h")
    x["context"] = [
        "injected sensor spike (DQ log)" if ((spikes["zone_id"] == z) & (spikes["start_timestamp"] == s)).any()
        else (f"{e} (label {int(l)})" if pd.notna(e) else "no event")
        for z, s, e, l in zip(x["zone_id"], shift_start, x["event_id"], x["label"])]
    x = x.merge(A[KEY + ["classification", "alert_relies_on_extrapolated_heat_index"]], on=KEY, how="left")
    rows = x[["observation_date", "zone_id", "shift", "temperature_celsius", "relative_humidity",
              "heat_index_celsius", "context", "classification", "alert_relies_on_extrapolated_heat_index"]]
    # dew point (Magnus) to show the physical implausibility of the humid-heat combinations
    a, b = 17.62, 243.12
    g = np.log(rows["relative_humidity"] / 100) + a * rows["temperature_celsius"] / (b + rows["temperature_celsius"])
    rows = rows.assign(dew_point_c=(b * g / (a - g)).round(1))
    summary = pd.DataFrame([{
        "rows with heat index": int(env["heat_index_celsius"].notna().sum()),
        "max heat index C": float(env["heat_index_celsius"].max()),
        "max temperature C": float(env["temperature_celsius"].max()),
        f"heat index > {NWS_MAX_HEAT_INDEX_C:.1f} C": int((env["heat_index_celsius"] > NWS_MAX_HEAT_INDEX_C).sum()),
        f"temperature > {NWS_MAX_TEMPERATURE_C:.1f} C": int((env["temperature_celsius"] > NWS_MAX_TEMPERATURE_C).sum()),
        "fusion rows flagged extrapolated": int((A["heat_index_validity"] == "extrapolated").sum()),
        "health alerts coinciding with extrapolated HI": int(A["alert_relies_on_extrapolated_heat_index"].sum()),
    }])
    return summary, rows.reset_index(drop=True), {"raw_max": float(env["heat_index_celsius"].max())}


# ----------------------------------------------------------------------------------------- summary
def summary_table(D, meta, e5_tab, ev_tab) -> pd.DataFrame:
    rows = []
    metrics = ("precision", "recall", "f1", "pr_auc", "n", "positives")
    for r in e5_tab.to_dict("records"):
        for m in metrics:
            rows.append({"phase": "4B forecast", "evaluation": "test", "subset": r["test subset"], "model": r["model"],
                         "metric": m, "value": float(r[m]),
                         "note": "LR = final refit model (uncalibrated score); persistence = current fusion class"})
    V = evaluable(D, "validation")
    for r in _model_rows(V, meta["threshold"], "lr_score_selection"):
        for m in metrics:
            rows.append({"phase": "4B forecast", "evaluation": "validation", "subset": "all validation rows",
                         "model": r["model"], "metric": m, "value": float(r[m]),
                         "note": "LR = train-only selection model (out-of-sample)"})
    for r in ev_tab.to_dict("records"):
        for m, col in (("event_recall", "event recall"), ("hours_before_onset", "hours before onset"),
                       ("hours_before_escalation", "hours before escalation")):
            rows.append({"phase": "4B forecast", "evaluation": r["split"], "subset": f"event {r['event_id']}",
                         "model": r["model"], "metric": m, "value": float(r[col]), "note": "event-level"})
    bm = pd.read_parquet(PROC / "backtest_metrics.parquet")
    keep = bm[bm["metric"].isin(["event_sensitivity", "median_lead_time_to_escalation_hours", "episode_ppv",
                                 "false_alerts_per_site_week", "precision", "recall", "f1"])
              & bm["model"].isin(["A", "B", "C"])]
    for r in keep.itertuples(index=False):
        rows.append({"phase": "4A detection backtest", "evaluation": f"{r.window} window", "subset": r.scope,
                     "model": {"A": "A: MAD + CUSUM", "B": "B: Isolation Forest", "C": "C: Risk fusion"}[r.model],
                     "metric": r.metric, "value": float(r.value), "note": "synthetic backtest (Phase 4A)"})
    out = pd.DataFrame(rows)
    out["synthetic"] = True
    return out


# ---------------------------------------------------------------------------------------- document
def build(write: bool = True) -> dict:
    cfg, D, final, meta, events, labels, F, A = load_context()
    from src.features.registry import load_feature_config
    window = load_feature_config()["windows"]["long"]["length"]
    thr = meta["threshold"]
    split_tab, ev_split = split_audit(D)
    e5_tab, e5 = evt005_effect(D, events, thr, window)
    ev_tab = event_level(D, events, thr)
    phases = foundry_phases(D, labels, events, F)
    contrib = lr_contributions(D, final, cfg)
    sc_tab, sc = score_audit(D, thr)
    hi_sum, hi_rows, hi = heat_index_audit(A)
    bt_ev = pd.read_parquet(PROC / "event_detection_results.parquet")
    bt_ctx = bt_ev[(bt_ev["window"] == "full") & bt_ev["event_id"].isin(["EVT-002", "EVT-005"])
                   & bt_ev["model"].isin(["A", "C"])][["model", "event_id", "detected", "first_alert_decision",
                                                     "lead_time_to_escalation_hours"]]
    summ = summary_table(D, meta, e5_tab, ev_tab)
    out = {"split": split_tab, "ev_split": ev_split, "e5_tab": e5_tab, "e5": e5, "events": ev_tab, "phases": phases,
           "contrib": contrib, "scores": sc_tab, "hi_sum": hi_sum, "hi_rows": hi_rows, "summary": summ, "meta": meta,
           "bt_ctx": bt_ctx}
    if write:
        summ.to_parquet(PROC / "model_comparison_summary.parquet", index=False)
        DOC.write_text(render(out, cfg), encoding="utf-8")
    return out


def _get(tab, subset_prefix, model, metric):
    r = tab[tab["test subset"].str.startswith(subset_prefix) & (tab["model"] == model)]
    return float(r[metric].iloc[0])


def render(o: dict, cfg: dict) -> str:
    e5, meta = o["e5"], o["meta"]
    t = o["e5_tab"]
    lr, pa = "logistic regression", "persistence (any warning)"
    ev = o["events"]
    e2 = ev[(ev["event_id"] == "EVT-002")].set_index("model")
    ph = o["phases"].set_index(["event", "phase"])
    p1 = [i for i in ph.index if i[0].startswith("EVT-001")]
    p2 = [i for i in ph.index if i[0].startswith("EVT-002")]
    ref1, esc1 = ph.loc[p1[0]], ph.loc[p1[2]]
    ref2, esc2 = ph.loc[p2[0]], ph.loc[p2[2]]
    sc = o["scores"]
    return "\n".join([
        "# PREVENZA Pre-Dashboard Evaluation Audit (Phase 4C)", "",
        "> GENERATED by `python -m src.evaluation.audit` from existing outputs. No model, threshold, split or feature "
        "was changed to improve results. **All results are synthetic** (one simulated 90-day dataset) and are not "
        "evidence of real-world clinical or operational performance.", "",
        "## Summary", "",
        f"1. **Evaluation independence:** splits and purges are chronological and correct, but EVT-005 straddles "
        f"validation and test. It contributes {int(o['ev_split'].set_index('target_event_id').loc['EVT-005', 'validation'])} "
        f"validation positives and {int(o['ev_split'].set_index('target_event_id').loc['EVT-005', 'test'])} test positive, "
        f"and its aftermath drives {e5['lr_false_flags_in_tail']} of the {e5['lr_false_flags_all_test']} logistic-regression "
        f"false flags on test. The only fully unseen test event is EVT-002 (one event-zone).",
        f"2. **Forecast vs persistence:** on identical test rows, logistic regression has lower F1 than the "
        f"'any warning' persistence baseline ({_get(t, 'all', lr, 'f1'):.3f} vs {_get(t, 'all', pa, 'f1'):.3f}). On the "
        f"unseen-event subset it ranks better (PR-AUC {_get(t, 'unseen', lr, 'pr_auc'):.3f} vs "
        f"{_get(t, 'unseen', pa, 'pr_auc'):.3f}) but detects less at its operating threshold (recall "
        f"{_get(t, 'unseen', lr, 'recall'):.3f} vs {_get(t, 'unseen', pa, 'recall'):.3f}, F1 "
        f"{_get(t, 'unseen', lr, 'f1'):.3f} vs {_get(t, 'unseen', pa, 'f1'):.3f}) and flags the unseen event later. "
        "With one unseen event, **meaningful added early-warning value is not demonstrated.**",
        "3. **Foundry transfer:** EVT-002 differs from the training event on almost every axis (indoor radiant-heat zone, "
        "higher baseline, smaller relative deviations, different context features). Observed differences and hypotheses "
        "are separated in section 3.",
        "4. **Score interpretation:** scores are uncalibrated; the threshold came from a train-only model and was applied "
        "to a refitted model whose score distribution differs. Brier scores are reported only as properties of an "
        "uncalibrated score.",
        f"5. **Heat-index validity:** {int(o['hi_sum'].iloc[0, 3])} raw readings exceed the NWS chart range (max "
        f"{o['hi_sum'].iloc[0, 1]:.1f} C). Cause: the simulator lets humidity stay high at ~42 C (implied dew points "
        "~29-31 C). Values are preserved; a validity indicator now flags them in fusion and forecast outputs.", "",
        "## 1. Evaluation independence", "",
        "### Split boundaries (Phase 4B, unchanged)", "", _md(o["split"]), "",
        "Purge: rows whose 24 h target window reaches the next split are labelled `purged` and excluded from fitting "
        "and evaluation. Train ends at 2025-07-14 18:00 shift start, validation starts 2025-07-16 06:00; validation ends "
        "2025-08-01 18:00, test starts 2025-08-03 06:00 - no target window crosses a boundary.", "",
        "### Positive targets by event and split", "", _md(o["ev_split"], "{:.0f}"), "",
        "- **EVT-005 appears in validation and test.** Validation holds the bulk of its positive targets; one test row "
        "(2025-08-03 day, S01-Z01) has an EVT-005 target. Because the final model was refitted on train + validation, "
        "the test evaluation is not fully independent of EVT-005.",
        f"- EVT-005's aftermath: {e5['influenced_rows']} test rows in {e5['zone']} fall before EVT-005 end + "
        f"{load_window()} days (the rolling-feature window). The final model raises {e5['lr_flags_in_tail']} flags "
        f"there, {e5['lr_false_flags_in_tail']} of them false - versus {e5['lr_false_flags_all_test']} false flags on the "
        "whole test split.",
        "- Phase 3B / 4A: the Isolation Forest was fitted only on decisions before 2025-07-17 and evaluated after; "
        "MAD+CUSUM and risk fusion have no fitted parameters. Their event results for EVT-005 and EVT-002 are therefore "
        "unseen-event results. Two caveats: (a) the Phase 3B.3 scale-floor change was made after inspecting unlabelled "
        "fused outputs over the whole period; (b) the simulator and detectors were designed by the same team.", "",
        "### Test metrics with and without EVT-005 influence", "", _md(t), "",
        "**Unseen-observation** performance = all test rows (some observations belong to an event seen in refit). "
        "**Unseen-event** performance = the last subset (EVT-002 plus unaffected negatives).", "",
        "## 2. Forecast vs persistence", "",
        "All models are scored on identical eligible, evaluable test rows (table above). Event-level results "
        "(first flag in the event zone from onset - 24 h to end; recall over the event's positive targets):", "",
        _md(ev.drop(columns=["window starts"]), "{:.2f}"), "",
        f"- **EVT-002 (unseen, test):** logistic regression first flags at "
        f"{_fmt_ts(e2.loc[lr, 'first flag in window'])} ({e2.loc[lr, 'hours before escalation']:+.0f} h vs escalation), "
        f"event recall {e2.loc[lr, 'event recall']:.2f}; persistence (any warning) first flags at "
        f"{_fmt_ts(e2.loc[pa, 'first flag in window'])} ({e2.loc[pa, 'hours before escalation']:+.0f} h), recall "
        f"{e2.loc[pa, 'event recall']:.2f}. Neither flags before onset.",
        "- **EVT-005 (validation, train-only model):** logistic regression and persistence (any warning) reach the "
        "same event recall; see table for first-flag times.",
        "- For context, Phase 4A detection (full window): " + "; ".join(
            f"{r.model} {r.event_id}: " + (f"first alert {_fmt_ts(r.first_alert_decision)} ({r.lead_time_to_escalation_hours:+.0f} h)"
                                           if r.detected else "not detected") for r in o["bt_ctx"].itertuples()) + ".",
        f"- **Conclusion:** mixed and weak. Logistic regression's ranking (PR-AUC) is somewhat better on the unseen-event "
        f"subset ({_get(t, 'unseen', lr, 'pr_auc'):.3f} vs {_get(t, 'unseen', pa, 'pr_auc'):.3f}, 27 positives, one event), "
        "but at the operating threshold it has lower recall and F1 and first flags EVT-002 later than persistence. On "
        "validation (EVT-005) it ties persistence on event recall and first-flag time. Added early-warning value over "
        "persistence is **not demonstrated**; a single event cannot establish it either way.", "",
        "## 3. Foundry generalisation (EVT-002 vs training event EVT-001)", "",
        "### Observed: feature means by phase", "", _md(o["phases"], "{:.2f}"), "",
        "### Observed: logistic-regression logit contributions (final model, top 10 differences)", "",
        _md(o["contrib"], "{:.3f}"), "",
        f"Intercept {o['contrib'].attrs.get('intercept', float('nan')):.3f}. Contribution = coefficient x imputed, "
        "standardised value; positive pushes towards a flag.", "",
        "**Observed evidence**", "",
        f"- Baseline heat: the foundry reference heat index is {ref2['heat_index_celsius_last']:.1f} C vs "
        f"{ref1['heat_index_celsius_last']:.1f} C for EVT-001 zones; its robust z in escalation is "
        f"{esc2['heat_index_celsius_rz']:.1f} vs {esc1['heat_index_celsius_rz']:.1f}.",
        f"- Fusion signal: mean escalation risk score {esc2['risk_score']:.1f} (EVT-002) vs {esc1['risk_score']:.1f} "
        f"(EVT-001); environmental score {esc2['environmental_score']:.1f} vs {esc1['environmental_score']:.1f}.",
        f"- Forecast score in escalation: {esc2['lr_score']:.2f} (EVT-002) vs {esc1['lr_score']:.2f} (EVT-001, in-sample).",
        "- Context: every positive in the refit data (EVT-001 at S02, EVT-005 at S01-Z01) is an outdoor or semi-outdoor "
        "zone without radiant heat (`ctx_outdoor` = 1, `ctx_radiant_heat` = 0); EVT-002 is the opposite. In the logit "
        "contributions, these two context features alone move EVT-002 positives about "
        f"{abs(o['contrib'].set_index('feature').loc[['ctx_outdoor', 'ctx_radiant_heat'], 'difference (002 - 001)'].sum()):.2f} "
        "logit units below EVT-001 positives - the largest single difference.", "",
        "**Hypotheses (not verified)**", "",
        "- Relative-deviation features (robust z, CUSUM) are smaller in a zone that is always hot, so the same "
        "physiological load looks less anomalous; absolute exposure would matter more there.",
        "- The model may have learned 'outdoor + large deviation' as the heat-risk pattern from a single training event.",
        "- The heat index ignores radiant heat; a radiant-heat zone's true strain is under-represented by it (WBGT would "
        "capture it). This is a measurement limitation, not fixable by re-weighting.",
        "- No foundry-specific thresholds were introduced; testing these hypotheses needs new, independent scenarios "
        "(e.g. a different simulation seed), not the current test labels.", "",
        "## 4. Score interpretation", "", _md(sc), "",
        f"- Threshold {meta['threshold']:.3f} = max-F1 on validation using the **train-only** model; it was then applied "
        "unchanged to the model refitted on train + validation. The table shows the refit model's score distribution "
        "differs (its validation scores are in-sample), so the operating point on test is not the one selected.",
        f"- On test the mean score is {sc.iloc[2]['mean score']:.3f} against a prevalence of {sc.iloc[2]['prevalence']:.3f} "
        f"(about {sc.iloc[2]['mean score'] / sc.iloc[2]['prevalence']:.1f}x, from balanced class weights), so scores are not probabilities. Brier values "
        "here describe an uncalibrated score and must not be read as probability accuracy. No calibration was fitted "
        "(too few positives to fit and verify one).",
        "- Model, threshold and splits are unchanged.", "",
        "## 5. Environmental measurement validity", "", _md(o["hi_sum"], "{:.1f}"), "", _md(o["hi_rows"], "{:.1f}"), "",
        f"- The NWS heat-index chart covers air temperatures up to {NWS_MAX_TEMPERATURE_C:.1f} C (110 F) and heat-index "
        f"values up to about {NWS_MAX_HEAT_INDEX_C:.1f} C (137 F). The previously reported 63 C value (EVT-001, S02, "
        "2025-07-04) is beyond that range: the Rothfusz regression is being extrapolated.",
        "- Root cause (simulation): heatwave forcing raises humidity together with temperature, and humidity is not "
        "constrained by temperature, giving dew points around 29-31 C at ~41-43 C - a combination that is rarely observed "
        "(the highest dew points ever recorded are about 35 C) and is a simulation artefact here. One flagged reading is an injected sensor spike (intended).",
        "- **Fix applied (non-destructive):** `src/features/quality.py::heat_index_validity`; risk fusion now outputs "
        "`heat_index_observed`, `heat_index_validity`, `measurement_flags`, `alert_relies_on_extrapolated_heat_index`, adds "
        "a caution to explanations and to the environmental explanation note; forecasts add a limitation note. Raw values "
        "are not clipped, heat index is not relabelled, and scores/classes are unchanged (the environmental absolute "
        "component already saturates at 51 C, so extrapolated magnitude adds no extra score).",
        "- **Deferred (simulation correction):** constrain humidity by a dew-point cap in `src/data_generation/environment.py`. "
        "Not applied now because it would regenerate all data and change every downstream result; it should be done "
        "together with a fresh, independent evaluation.", "",
        "## 6. Issues", "",
        "| Issue | Status |", "|---|---|",
        "| EVT-005 spans validation and test; refit saw it | Documented and quantified; split unchanged |",
        "| Forecast adds no demonstrated value over persistence on unseen event | Documented; no tuning |",
        "| Foundry (radiant heat) transfer failure | Documented with evidence vs hypotheses; no foundry thresholds |",
        "| Threshold applied to refitted model without re-selection | Documented; unchanged |",
        "| Brier on uncalibrated scores | Labelled as uncalibrated everywhere |",
        "| Heat index beyond NWS chart range | **Fixed:** validity indicator + explanation/limitation flags (tested) |",
        "| Simulator allows implausible humid heat | Deferred: dew-point cap + full regeneration and re-evaluation |",
        "| Single seed, ~1 event per split | Deferred: multi-seed evaluation |", "",
        "## 7. Dashboard readiness", "",
        "Ready for dashboard integration **as a transparent synthetic demonstrator**, provided the dashboard: shows "
        "evidence level, missing/stale sources and measurement flags beside every risk score; labels forecast scores as "
        "uncalibrated rankings (never probabilities); displays the persistence baseline alongside the forecast; and "
        "carries a 'synthetic data - not clinically validated' notice. It is not ready to be presented as a validated "
        "early-warning or forecasting system.", "",
    ])


def _fmt_ts(v):
    return "none" if pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")


def load_window():
    from src.features.registry import load_feature_config
    return load_feature_config()["windows"]["long"]["length"]


def main():
    o = build()
    print(o["e5_tab"].round(3).to_string(index=False))
    print(f"wrote {DOC.relative_to(config.PROJECT_ROOT)} and data/processed/model_comparison_summary.parquet")


if __name__ == "__main__":
    main()
