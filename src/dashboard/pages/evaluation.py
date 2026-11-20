"""Page 5 - Model evaluation (EVALUATION ONLY: uses ground truth and saved evaluation outputs).

Metric provenance:
* Detection (Phase 4A): backtest_metrics.parquet, event_detection_results.parquet; false alerts per site via
  src.evaluation.metrics.false_alerts_by_site (the same function that produced the backtest report).
* Forecast (Phase 4B/4C): model_comparison_summary.parquet (validation, test subsets, events). Training-period
  (in-sample) metrics and test confusion counts are computed from the saved forecast_evaluation.parquet flags with
  src.forecasting.forecast.binary_metrics - the same definition used in Phase 4B.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.evaluation import metrics as M
from src.forecasting.forecast import binary_metrics

from .. import charts, components as C, theme as T
from ..data_loader import config_yaml, load_evaluation

MODEL_NAMES = {"A": "A: MAD + CUSUM", "B": "B: Isolation Forest", "C": "C: Risk fusion",
               "C-high": "C: Risk fusion (High only)", "C-warning": "C: warnings (not health alerts)"}
DETECTION_METRICS = [("event", "event_zones_included"), ("event", "event_zones_detected"),
                     ("event", "event_sensitivity"), ("event", "mean_detection_coverage"),
                     ("event", "median_lead_time_to_escalation_hours"), ("event", "median_detection_delay_hours"),
                     ("event", "controls_with_false_alert"), ("episode", "n_episodes"), ("episode", "episode_ppv"),
                     ("episode", "false_alerts_per_site_week"), ("episode", "median_episode_duration_hours"),
                     ("observation", "precision"), ("observation", "recall"), ("observation", "f1"),
                     ("observation", "false_positive_rate")]
FC_MODELS = ["persistence (health alert)", "persistence (any warning)", "logistic regression"]
SUBSETS = [
    ("validation", "all validation rows", "Validation (out-of-sample; LR = train-only selection model)"),
    ("test", "all test rows", "Held-out test — all rows"),
    ("test", "excluding rows whose target is EVT-005", "Held-out test — excluding EVT-005 targets"),
    ("test", "unseen-event subset", "Unseen-event test (EVT-002 + unaffected negatives)"),
]


def detection_table(bm: pd.DataFrame, window: str) -> pd.DataFrame:
    sub = bm[bm["window"] == window]
    models = [m for m in ("A", "B", "C", "C-high", "C-warning") if m in set(sub["model"])]
    rows = []
    for scope, metric in DETECTION_METRICS:
        r = {"scope": scope, "metric": metric}
        for m in models:
            v = sub[(sub["model"] == m) & (sub["scope"] == scope) & (sub["metric"] == metric)]["value"]
            r[MODEL_NAMES[m]] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def forecast_table(ms: pd.DataFrame) -> pd.DataFrame:
    f = ms[ms["phase"] == "4B forecast"]
    rows = []
    for ev, prefix, label in SUBSETS:
        s = f[(f["evaluation"] == ev) & f["subset"].str.startswith(prefix)]
        for m in FC_MODELS:
            g = s[s["model"] == m].set_index("metric")["value"]
            if g.empty:
                continue
            rows.append({"evaluation": label, "model": m, **{k: g.get(k, float("nan")) for k in
                                                            ("n", "positives", "precision", "recall", "f1", "pr_auc")}})
    return pd.DataFrame(rows)


def in_sample_table(fe: pd.DataFrame) -> pd.DataFrame:
    tr = fe[(fe["split"] == "train") & fe["eligible"] & fe["target"].notna()]
    y = tr["target"].astype(int)
    rows = []
    for m, flag, score in (("persistence (health alert)", tr["persistence_health"], None),
                           ("persistence (any warning)", tr["persistence_any"], None),
                           ("logistic regression", tr["lr_flag"], tr["lr_score"])):
        d = binary_metrics(y, flag.astype(bool), score)
        rows.append({"evaluation": "Training period — IN-SAMPLE (final model saw these rows)", "model": m,
                     **{k: d[k] for k in ("n", "positives", "precision", "recall", "f1", "pr_auc")}})
    return pd.DataFrame(rows)


def test_confusions(fe: pd.DataFrame) -> dict[str, dict]:
    te = fe[(fe["split"] == "test") & fe["eligible"] & fe["target"].notna()]
    y = te["target"].astype(int)
    out = {}
    for m, flag in (("persistence (health alert)", te["persistence_health"]),
                    ("persistence (any warning)", te["persistence_any"]), ("logistic regression", te["lr_flag"])):
        d = binary_metrics(y, flag.astype(bool))
        out[m] = {"tp": d["tp"], "fp": d["fp"], "fn": d["fn"], "tn": d["n"] - d["tp"] - d["fp"] - d["fn"]}
    return out


def evt002_lead(ms: pd.DataFrame) -> dict[str, float]:
    s = ms[(ms["phase"] == "4B forecast") & (ms["evaluation"] == "test") & (ms["subset"] == "event EVT-002")
           & (ms["metric"] == "hours_before_escalation")]
    return dict(zip(s["model"], s["value"]))


def false_alerts_site(window: str) -> pd.DataFrame:
    P = load_evaluation("backtest_results")
    EP = load_evaluation("episodes_eval")
    bcfg = config_yaml("backtest.yaml")
    cutoff = pd.Timestamp(P["version_if_train_cutoff"].iloc[0])
    wm = M.window_mask(P, window, bcfg, cutoff)
    out = []
    for m in sorted(EP.loc[EP["window"] == window, "model"].unique()):
        fs = M.false_alerts_by_site(EP[(EP["model"] == m) & (EP["window"] == window)], P, wm)
        fs["model"] = MODEL_NAMES[m]
        out.append(fs)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(3)


def render(f) -> None:
    C.header("Model evaluation", "Evaluation-only page: compares detection and forecasting outputs with the "
                                 "simulation's ground truth. Global filters do not apply here.")
    C.limitation(T.EVALUATION_LIMITATION)
    st.caption("Metrics are read from saved evaluation outputs. Thresholds were fixed before evaluation; nothing on "
               "this page can change a model, threshold or score.")
    tab1, tab2, tab3 = st.tabs(["Detection backtest (Phase 4A)", "24 h forecast (Phase 4B)", "Audit findings (Phase 4C)"])

    with tab1:
        bm = load_evaluation("backtest_metrics")
        window = st.radio("Evaluation window", ["full", "comparison"], horizontal=True, key="ev_window",
                          format_func=lambda w: {"full": "Full window (A and C; no fitted parameters)",
                                                 "comparison": "Comparison window (A, B, C on identical out-of-sample rows)"}[w])
        tbl = detection_table(bm, window)
        st.dataframe(_fmt_table(tbl), hide_index=True, key="ev_det")
        C.download(tbl, f"prevenza_detection_metrics_{window}.csv", key="dl_det")
        cols = st.columns(3)
        for col, m in zip(cols, ["A", "B", "C"]):
            s = bm[(bm["window"] == window) & (bm["model"] == m) & (bm["scope"] == "observation")].set_index("metric")["value"]
            if s.empty:
                col.info(f"{MODEL_NAMES[m]}: not evaluable in this window (in-sample period).")
                continue
            col.plotly_chart(charts.confusion(s["tp"], s["fp"], s["fn"], s["tn"], f"{MODEL_NAMES[m]} — observations"),
                             theme=None, key=f"ev_cm_{m}_{window}")
        comp = []
        for metric, scope, label in (("precision", "observation", "precision"), ("recall", "observation", "recall"),
                                     ("f1", "observation", "F1"), ("event_sensitivity", "event", "event sensitivity"),
                                     ("episode_ppv", "episode", "episode PPV")):
            for m in ("A", "B", "C"):
                v = bm[(bm["window"] == window) & (bm["model"] == m) & (bm["metric"] == metric) & (bm["scope"] == scope)]
                if len(v):
                    comp.append({"metric": label, "model": MODEL_NAMES[m], "value": float(v["value"].iloc[0])})
        if comp:
            st.plotly_chart(charts.grouped_bars(pd.DataFrame(comp), "metric", "value", "model",
                                                "Detection model comparison", "value (0-1)", T.MODEL_COLORS),
                            theme=None, key=f"ev_cmp_{window}")
        ev = load_evaluation("event_results")
        ev = ev[ev["window"] == window].copy()
        ev["model"] = ev["model"].map(MODEL_NAMES)
        st.subheader("Event-level detection")
        evt = ev[["model", "event_id", "zone_id", "role", "included", "excluded_reason", "detected", "first_alert_decision",
                  "detection_delay_hours", "lead_time_to_escalation_hours", "coverage", "n_matched_episodes",
                  "control_false_alert"]].sort_values(["event_id", "zone_id", "model"])
        st.dataframe(evt, hide_index=True, key="ev_events")
        C.download(evt, f"prevenza_event_detection_{window}.csv", key="dl_ev")
        lt = ev[ev["included"].astype(bool) & ev["detected"].astype("boolean").fillna(False).astype(bool)]
        if len(lt):
            st.plotly_chart(charts.lead_time_bars(pd.DataFrame({"unit": lt["event_id"] + " " + lt["zone_id"],
                                                                "model": lt["model"],
                                                                "hours": lt["lead_time_to_escalation_hours"]}),
                                                  "Lead time to escalation (detected event-zones)"),
                            theme=None, key=f"ev_lead_{window}")
        fa = false_alerts_site(window)
        if len(fa):
            st.plotly_chart(charts.grouped_bars(fa, "site_id", "false_alerts_per_week", "model",
                                                "False alert episodes per site per week", "false episodes / week",
                                                T.MODEL_COLORS), theme=None, key=f"ev_fa_{window}")

    with tab2:
        ms = load_evaluation("model_summary")
        fe = load_evaluation("forecast_evaluation")
        lead = evt002_lead(ms)
        lr_h, pa_h = lead.get("logistic regression"), lead.get("persistence (any warning)")
        if lr_h is not None and pa_h is not None:
            C.caution(f"Foundry generalisation issue (EVT-002, unseen test event): logistic regression's first alert "
                      f"was {abs(lr_h):.0f} hours {'after' if lr_h < 0 else 'before'} escalation, while any-warning "
                      f"persistence flagged {abs(pa_h):.0f} hours {'before' if pa_h > 0 else 'after'} escalation.")
        C.note("Validation = out-of-sample (train-only model used for selection). Held-out test = final model refitted "
               "on train + validation. EVT-005 spans validation and test, so the unseen-event subset is the cleanest "
               "test. Training-period rows are in-sample and shown only for transparency. Scores are uncalibrated "
               "rankings; PR-AUC uses them only for ranking.")
        tbl = pd.concat([forecast_table(ms), in_sample_table(fe)], ignore_index=True)
        st.dataframe(_fmt_table(tbl), hide_index=True, key="ev_fc")
        C.download(tbl, "prevenza_forecast_metrics.csv", key="dl_fc")
        plot = tbl.melt(id_vars=["evaluation", "model"], value_vars=["precision", "recall", "f1", "pr_auc"])
        sel = st.selectbox("Compare models on", [s[2] for s in SUBSETS], index=3, key="ev_fc_subset")
        p = plot[plot["evaluation"] == sel]
        st.plotly_chart(charts.grouped_bars(p, "variable", "value", "model", f"Forecast vs persistence — {sel}",
                                            "value (0-1)", T.MODEL_COLORS), theme=None, key="ev_fc_bars")
        cols = st.columns(3)
        for col, (m, d) in zip(cols, test_confusions(fe).items()):
            col.plotly_chart(charts.confusion(d["tp"], d["fp"], d["fn"], d["tn"], f"{m} — held-out test"),
                             theme=None, key=f"ev_fccm_{m}")
        st.subheader("Event-level forecast results")
        evr = ms[(ms["phase"] == "4B forecast") & ms["subset"].str.startswith("event ")]
        evr = evr.pivot_table(index=["evaluation", "subset", "model"], columns="metric", values="value").reset_index()
        st.dataframe(_fmt_table(evr), hide_index=True, key="ev_fc_events")

    with tab3:
        doc = config.PROJECT_ROOT / "docs" / "pre_dashboard_audit.md"
        if doc.exists():
            text = doc.read_text(encoding="utf-8")
            summary = text.split("## Summary", 1)[-1].split("## 1.", 1)[0]
            st.markdown("#### Audit summary\n" + summary)
            with st.expander("Full pre-dashboard audit (docs/pre_dashboard_audit.md)"):
                st.markdown(text)
        else:
            C.empty_state("docs/pre_dashboard_audit.md not found. Generate it with `python -m src.evaluation.audit`.")
