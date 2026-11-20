"""Phase 4A - historical backtest and operational validation.

    python -m src.evaluation.backtest            # full chronological replay (all decision times) + evaluation
    python -m src.evaluation.backtest --sample 12  # replay only N evenly spaced decision times (quick check)

Outputs: data/processed/{backtest_results, event_detection_results, alert_episodes, backtest_metrics}.parquet
         docs/backtesting_report.md, docs/model_comparison.md, docs/figures/*.png

SYNTHETIC DATA: every metric describes behaviour on simulated scenarios with known ground truth. It is not
evidence of real-world clinical or operational performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from src import config, schemas
from src.detection import isolation_forest as iso
from src.detection import risk_fusion as rf
from src.features.build import FEATURES_FILE
from src.preprocessing.integrate import PREFIX, unified_path
from src.preprocessing.load import load_dataset

from . import charts
from . import metrics as M
from .replay import Replayer, compare

KEY = list(schemas.OBSERVATION_KEY)
DOCS = config.PROJECT_ROOT / "docs"
WINDOW_MODELS = {"full": ["A", "C"], "comparison": ["A", "B", "C"]}
EXTRA_C = {"C-high": "model_c_high_alert", "C-warning": "model_c_warning"}


def load_backtest_config() -> dict:
    with open(config.CONFIG_DIR / "backtest.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------------------------------ versions
def _sha(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def versions() -> dict:
    cfgs = ["simulation.yaml", "settings.yaml", "features.yaml", "isolation_forest.yaml", "risk_fusion.yaml",
            "backtest.yaml"]
    icfg = iso.load_if_config()
    meta = json.loads((iso.model_dir(icfg) / "metadata.json").read_text(encoding="utf-8"))
    raw = sorted(config.path_for("raw").glob("*.csv"))
    return {"isolation_forest_model": meta["model_version"], "if_train_cutoff": meta["train_cutoff"],
            "config_hash": _sha([config.CONFIG_DIR / c for c in cfgs]), "raw_data_hash": _sha(raw),
            "sklearn": meta["versions"]["sklearn"]}


# ---------------------------------------------------------------------------------------- replay step
def replay_and_verify(sample: int | None):
    R = Replayer()
    times = R.times()
    if sample:
        idx = np.unique(np.linspace(0, len(times) - 1, sample).round().astype(int))
        times = [times[i] for i in idx]
    print(f"replaying {len(times)} decision times ...")
    F, I, A = R.run(times, progress=True)
    proc = config.path_for("processed")
    saved = {"features": pd.read_parquet(proc / FEATURES_FILE),
             "isolation_forest": pd.read_parquet(proc / "isolation_forest_scores.parquet"),
             "assessments": pd.read_parquet(proc / rf.load_fusion_config()["output"]["assessments_file"])}
    checks = {}
    match = None
    for name, df in (("features", F), ("isolation_forest", I), ("assessments", A)):
        c = compare(df, saved[name]).rename(columns={"replay_match": f"replay_match_{name}"})
        checks[name] = {"rows": len(c), "matching": int(c.iloc[:, -1].sum())}
        match = c if match is None else match.merge(c, on=KEY)
    full = len(times) == len(R.times())
    if not full:      # evaluate on saved outputs, verified on the replayed sample
        F, I, A = saved["features"], saved["isolation_forest"], saved["assessments"]
    return F, I, A, match, {"decision_times_replayed": len(times), "decision_times_total": len(R.times()),
                            "full_replay": full, "checks": checks}, R.lag


# ------------------------------------------------------------------------------------------ evaluation
def evaluate(P0: pd.DataFrame, labels: pd.DataFrame, events: pd.DataFrame, bcfg: dict, if_cutoff: pd.Timestamp):
    P = M.attach_labels(P0, labels)
    pos = bcfg["matching"]["positive_labels"]
    rows, ep_all, ev_all, site_rows, state_rows = [], [], [], [], []

    def add(model, window, scope, d):
        for k, v in d.items():
            if isinstance(v, (int, float, np.integer, np.floating, bool)) or v is None:
                rows.append({"model": model, "window": window, "scope": scope, "metric": k,
                             "value": float(v) if v is not None else np.nan})

    for window, models in WINDOW_MODELS.items():
        wm = M.window_mask(P, window, bcfg, if_cutoff)
        variants = [(m, M.MODEL_ALERT[m]) for m in models] + [(k, c) for k, c in EXTRA_C.items()]
        for model, col in variants:
            E = M.build_episodes(P[wm], col, bcfg["episodes"]["max_gap_hours"], model, window)
            E = M.match_episodes(E, events, bcfg["matching"])
            R = M.event_results(P, E, events, col, model, window, wm, bcfg["matching"])
            add(model, window, "observation", M.observation_metrics(P, wm, col, pos))
            add(model, window, "episode", M.episode_metrics(E, P, wm))
            add(model, window, "event", M.event_summary(R))
            if model in M.MODEL_ALERT:
                ep_all.append(E)
                ev_all.append(R)
                fs = M.false_alerts_by_site(E, P, wm)
                fs.insert(0, "window", window), fs.insert(0, "model", model)
                site_rows.append(fs)
                for state, sm in M.data_state_masks(P, bcfg).items():
                    d = M.confusion(P.loc[wm & sm, "label"].isin(pos), P.loc[wm & sm, col])
                    state_rows.append({"model": model, "window": window, "data_state": state, **d})
    # insufficient baseline (warm-up) rows: excluded from metrics, described here
    wu = P["series_position"] < bcfg["windows"]["warmup_observations"]
    warm = {"rows": int(wu.sum()), "positives": int(P.loc[wu, "label"].isin(pos).sum()),
            "model_a_ineligible": int(P.loc[wu, "model_a_alert"].isna().sum()),
            "model_b_ineligible": int(P.loc[wu, "model_b_alert"].isna().sum()),
            "model_c_health_alerts": int(P.loc[wu, "model_c_health_alert"].sum()),
            "classification_counts": P.loc[wu, "classification"].value_counts().to_dict()}
    return (P, pd.DataFrame(rows), pd.concat([e for e in ep_all if len(e)] or ep_all[:1], ignore_index=True), pd.concat(ev_all, ignore_index=True),
            pd.concat(site_rows, ignore_index=True), pd.DataFrame(state_rows), warm)


def latency_table(u: pd.DataFrame, P: pd.DataFrame, wm: pd.Series) -> pd.DataFrame:
    keys = P.loc[wm, KEY]
    w = u.merge(keys, on=KEY)
    rows = []
    for name, p in PREFIX.items():
        d = w[f"{p}_reporting_delay_hours"].dropna()
        rows.append({"source": name, "reported share": float(w[f"{p}_reporting_status"].eq("reported").mean()),
                     "median delay h": float(d.median()), "p95 delay h": float(d.quantile(0.95)),
                     "max delay h": float(d.max())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    df = df.copy()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("NA" if np.isnan(v) else floatfmt.format(v))
            elif isinstance(v, pd.Timestamp):
                cells.append(v.strftime("%Y-%m-%d %H:%M"))
            elif v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                cells.append("NA")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _metric_table(Mx: pd.DataFrame, window: str, models: list[str], wanted: list[tuple[str, str]]) -> pd.DataFrame:
    sub = Mx[Mx["window"] == window]
    out = []
    for scope, metric in wanted:
        r = {"metric": metric}
        for m in models:
            v = sub[(sub["model"] == m) & (sub["scope"] == scope) & (sub["metric"] == metric)]["value"]
            r[m] = float(v.iloc[0]) if len(v) else np.nan
        out.append(r)
    return pd.DataFrame(out)


HEADLINE = [("event", "event_zones_included"), ("event", "event_zones_detected"), ("event", "event_sensitivity"),
            ("event", "detected_before_escalation"), ("event", "median_lead_time_to_escalation_hours"),
            ("event", "median_detection_delay_hours"), ("event", "mean_detection_coverage"),
            ("event", "controls_included"), ("event", "controls_with_false_alert"),
            ("episode", "n_episodes"), ("episode", "episode_ppv"), ("episode", "false_alarm_episodes"),
            ("episode", "false_alerts_per_site_week"), ("episode", "median_episode_duration_hours"),
            ("observation", "precision"), ("observation", "recall"), ("observation", "f1"),
            ("observation", "false_positive_rate"), ("observation", "tp"), ("observation", "fp"),
            ("observation", "fn"), ("observation", "tn"), ("observation", "n_ineligible")]

DISCLAIMER = ("> **Synthetic validation only.** All results come from a simulated 90-day dataset with injected, "
              "known scenarios. They show how the pipeline behaves under those simulated conditions; they are "
              "**not** evidence of real-world clinical, epidemiological or operational performance, which would "
              "require validation on real, independently labelled data. Thresholds were set before evaluation and "
              "were not tuned on these events. Scores are not probabilities and not individual clinical risk.")


def _event_table(EV: pd.DataFrame, window: str) -> pd.DataFrame:
    e = EV[EV["window"] == window].copy()
    e["result"] = np.where(~e["included"], "excluded: " + e["excluded_reason"].astype(str),
                  np.where(e["role"] == "negative_control",
                           np.where(e["control_false_alert"].astype("boolean").fillna(False), "false alert on control",
                                    "no alert (correct)"),
                           np.where(e["detected"].astype("boolean").fillna(False), "detected", "NOT detected")))
    cols = ["model", "event_id", "zone_id", "role", "result", "first_alert_decision", "detection_delay_hours",
            "lead_time_to_escalation_hours", "coverage", "n_matched_episodes", "n_duplicate_episodes"]
    for c in cols:
        if c not in e:
            e[c] = np.nan
    return e[cols].sort_values(["event_id", "zone_id", "model"])


def _scenario_text(EV: pd.DataFrame) -> list[str]:
    def res(eid, model, window):
        e = EV[(EV["event_id"] == eid) & (EV["model"] == model) & (EV["window"] == window)]
        if e.empty or not e["included"].all():
            return "not evaluable in this window"
        if e["role"].iloc[0] == "negative_control":
            n = int(e["n_alerts_in_window"].sum())
            return f"{n} alert(s) on spike window" + (" (false alert)" if n else " (correct: no alert)")
        parts = []
        for r in e.itertuples():
            if r.detected:
                parts.append(f"{r.zone_id}: detected, first alert {r.first_alert_decision:%m-%d %H:%M}, "
                             f"{r.lead_time_to_escalation_hours / 24:+.1f} d vs escalation, coverage {r.coverage:.0%}")
            else:
                parts.append(f"{r.zone_id}: not detected")
        return "; ".join(parts)
    lines = []
    for eid, title in [("EVT-001", "EVT-001 - Scenario B gradual heatwave, construction site (3 zones)"),
                       ("EVT-002", "EVT-002 - Scenario B heat build-up, foundry"),
                       ("EVT-003", "EVT-003 - Scenario C isolated spike, warehouse (negative control)"),
                       ("EVT-004", "EVT-004 - Scenario C isolated spike, outdoor yard (negative control)"),
                       ("EVT-005", "EVT-005 - Scenario D escalation with missing clinical reporting, loading dock")]:
        lines += [f"**{title}**", ""]
        for m, w in [("A", "full"), ("C", "full"), ("A", "comparison"), ("B", "comparison"), ("C", "comparison")]:
            lines.append(f"- {charts.MODEL_NAMES[m]} ({w} window): {res(eid, m, w)}")
        lines.append("")
    return lines


def write_reports(ctx: dict) -> None:
    Mx, EV, EP, SITES, STATES, warm = ctx["metrics"], ctx["events"], ctx["episodes"], ctx["sites"], ctx["states"], ctx["warm"]
    ver, rep, fig = ctx["versions"], ctx["replay"], ctx["figures"]
    full_tab = _metric_table(Mx, "full", ["A", "C", "C-high", "C-warning"], HEADLINE)
    comp_tab = _metric_table(Mx, "comparison", ["A", "B", "C"], HEADLINE)
    chk = pd.DataFrame([{"output": k, "rows replayed": v["rows"], "rows identical": v["matching"]}
                        for k, v in rep["checks"].items()])
    win = pd.DataFrame(ctx["windows"])
    states = STATES.copy()
    states = states[["window", "model", "data_state", "n", "positives", "recall", "precision", "false_positive_rate"]]
    rel = ctx["alert_evidence"]

    (DOCS / "backtesting_report.md").write_text("\n".join([
        "# PREVENZA Backtesting and Operational Validation (Phase 4A)", "",
        "> GENERATED by `python -m src.evaluation.backtest`. Do not edit by hand.", "", DISCLAIMER, "",
        "## 1. Chronological replay", "",
        f"Decision timestamps replayed: **{rep['decision_times_replayed']} of {rep['decision_times_total']}** "
        f"({'full replay' if rep['full_replay'] else 'sample; evaluation uses saved outputs verified on the sample'}). "
        "At each timestamp the information state contains only source values with `reporting_timestamp <= T`; "
        "features, the persisted Isolation Forest and risk fusion are recomputed and only that timestamp's rows kept.",
        "", _md(chk), "",
        f"Model versions: Isolation Forest `{ver['isolation_forest_model']}` (trained on decisions before "
        f"{ver['if_train_cutoff']}), config hash `{ver['config_hash']}`, raw data hash `{ver['raw_data_hash']}`, "
        f"scikit-learn {ver['sklearn']}.", "",
        "## 2. Definitions", "",
        "- **Models.** A = any of six signals with robust z >= 5 or CUSUM >= 5 (no fusion). B = Isolation Forest "
        "anomaly decision, out-of-sample rows only. C = risk fusion health alert (Emerging or High health risk). "
        "C-high = High health risk only; C-warning = Environmental warning / Elevated occupational exposure (not health alerts).",
        "- **Episodes.** Alerting observations of a site-zone (both shifts) with consecutive decisions <= "
        f"{ctx['bcfg']['episodes']['max_gap_hours']} h apart.",
        f"- **Matching.** An episode matches an escalating event-zone if its observation span overlaps "
        f"[onset - {ctx['bcfg']['matching']['pre_onset_tolerance_hours']} h, end + "
        f"{ctx['bcfg']['matching']['post_end_tolerance_hours']} h). Each episode matches at most one event-zone (largest "
        "overlap); each event-zone is credited once (later episodes = `matched_duplicate`). Episodes overlapping only a "
        "negative-control spike are `control_false_alarm`.",
        "- **Event sensitivity** = detected / included escalating event-zones. **Episode PPV** = non-false episodes / all "
        "episodes. **Detection delay** = first alert decision - onset. **Lead time** = escalation - first alert decision "
        "(positive = warning before escalation). **Coverage** = share of the event's labelled observations (label 1-2) with an alert.",
        "- **Observation metrics** use label >= 1 as positive; warm-up rows are excluded; ineligible predictions count as no alert "
        "(reported as `n_ineligible`). **False alerts per site-week** = false episodes / (sites x weeks in window).", "",
        "## 3. Evaluation windows", "", _md(win), "",
        "## 4. Results - full window (models A and C; no fitted parameters)", "", _md(full_tab), "",
        "## 5. Results by event", "", _md(_event_table(EV, "full"), "{:.1f}"), "",
        "### Scenario summaries", "", *_scenario_text(EV),
        "## 6. Missing-data conditions (defined from data availability, not labels)", "", _md(states), "",
        "Small subsets (few positives) give unstable rates; read `n` and `positives` before the ratios.", "",
        "## 7. Insufficient baseline (warm-up)", "",
        f"- {warm['rows']} warm-up observations (first {ctx['bcfg']['windows']['warmup_observations']} per series) are "
        f"excluded from all metrics; {warm['positives']} of them carry a positive label.",
        f"- Model A ineligible on {warm['model_a_ineligible']}, Model B on {warm['model_b_ineligible']}; Model C raised "
        f"{warm['model_c_health_alerts']} health alerts there. C classifications: "
        + ", ".join(f"{k} {v}" for k, v in warm["classification_counts"].items()) + ".", "",
        "## 8. Reporting latency, availability and evidence at alerts (full window)", "",
        _md(ctx["latency"], "{:.2f}"), "", _md(rel, "{:.2f}"), "",
        "## 9. False alerts by site", "", _md(SITES, "{:.2f}"), "",
        "## 10. Figures", "",
        *[f"![{k}]({v})" for k, v in fig.items()], "",
        "## 11. Limitations", "",
        "- One simulated realisation (seed 42) with 5 escalating event-zones in the full window and 2 in the "
        "out-of-sample comparison window: metrics have very wide uncertainty and must not be generalised.",
        "- The simulator and the detectors were designed by the same team; scenario mechanics (lags, heat-strain "
        "model) favour signal shapes the features were built to see. Real data will be noisier and less aligned.",
        "- Model B cannot be evaluated before its training cutoff, so EVT-001 and EVT-004 only test A and C.",
        "- Labels follow the injected schedule (onset/escalation), not observed outcomes; 'emerging' observations "
        "may legitimately show little signal yet.", "",
    ]), encoding="utf-8")

    comp_states = states[states["window"] == "comparison"]
    (DOCS / "model_comparison.md").write_text("\n".join([
        "# PREVENZA Model Comparison (Phase 4A)", "",
        "> GENERATED by `python -m src.evaluation.backtest`. Do not edit by hand.", "", DISCLAIMER, "",
        f"All three models are scored on the **identical out-of-sample window**: decisions at or after the Isolation "
        f"Forest training cutoff ({ver['if_train_cutoff']}), after warm-up. Only valid out-of-sample Isolation Forest "
        "predictions are used.", "",
        "- **A: MAD + CUSUM** - univariate robust z / CUSUM on six signals, any-signal rule.",
        "- **B: Isolation Forest** - multivariate isolation, contamination 0.05 on the training window.",
        "- **C: Risk fusion** - evidence-weighted categories, corroboration + persistence (Emerging/High health risk).", "",
        "## Headline metrics (comparison window)", "", _md(comp_tab), "",
        "## Events in the comparison window", "", _md(_event_table(EV, "comparison"), "{:.1f}"), "",
        "## False alerts by site (comparison window)", "", _md(SITES[SITES["window"] == "comparison"], "{:.2f}"), "",
        "## Missing-data conditions (comparison window)", "", _md(comp_states), "",
        "## Figures", "",
        *[f"![{k}]({v})" for k, v in fig.items() if k != "event_timelines"], "",
        "## Reading the comparison", "",
        "- The comparison window contains only 2 escalating event-zones and 1 negative control; differences of one "
        "event change sensitivity by 50 percentage points.",
        "- Observation-level recall rewards alerting on every labelled shift; event-level metrics reward early, "
        "single detections. Read both.",
        "- These are synthetic results and do not establish real-world superiority of any model.", "",
    ]), encoding="utf-8")


# --------------------------------------------------------------------------------------------- main
def make_figures(P, EV, Mx, SITES, STATES, events, bcfg, if_cutoff) -> dict:
    d = config.PROJECT_ROOT / bcfg["output"]["figures_dir"]
    out = {}
    out["event_timelines"] = charts.event_timelines(P, events, d / "event_timelines.png")
    lt = pd.concat([EV[(EV["model"] == "A") & (EV["window"] == "full")], EV[(EV["model"] == "C") & (EV["window"] == "full")],
                    EV[(EV["model"] == "B") & (EV["window"] == "comparison")]])
    out["lead_times"] = charts.lead_times(lt, d / "lead_times.png")
    wmf = M.window_mask(P, "full", bcfg, if_cutoff)
    multi = pd.crosstab(P.loc[wmf, "label_name"], P.loc[wmf, "model_c_level"]).reindex(
        index=["normal", "emerging", "escalating"], columns=["none", "insufficient", "warning", "emerging", "high"],
        fill_value=0)
    binary = {}
    for m in ["A", "B", "C"]:
        s = Mx[(Mx["model"] == m) & (Mx["window"] == "comparison") & (Mx["scope"] == "observation")].set_index("metric")["value"]
        binary[m] = {k: int(s[k]) for k in ("tp", "fp", "fn", "tn")}
    out["confusion_matrix"] = charts.confusion_matrices(multi, binary, d / "confusion_matrix.png")
    pr = _metric_table(Mx, "comparison", ["A", "B", "C"],
                       [("observation", "precision"), ("observation", "recall"), ("observation", "f1"),
                        ("event", "event_sensitivity"), ("episode", "episode_ppv")]).set_index("metric")
    out["precision_recall"] = charts.grouped_bars(pr, d / "precision_recall.png",
                                                  "Precision / recall comparison (comparison window)", "value (0-1)", (0, 1.05))
    fs = SITES[SITES["window"] == "comparison"].pivot(index="site_id", columns="model", values="false_alerts_per_week")
    out["false_alerts_by_site"] = charts.grouped_bars(fs[["A", "B", "C"]], d / "false_alerts_by_site.png",
                                                      "False alert episodes per week by site (comparison window)",
                                                      "false episodes / week")
    st = STATES[STATES["window"] == "comparison"]
    rec = st.pivot(index="data_state", columns="model", values="recall")[["A", "B", "C"]]
    rec.index = [f"{i}\n(pos={int(st[(st.data_state == i)].positives.iloc[0])})" for i in rec.index]
    out["missing_data_recall"] = charts.grouped_bars(rec, d / "missing_data_recall.png",
                                                     "Observation recall by data condition (comparison window)", "recall", (0, 1.05))
    fpr = st.pivot(index="data_state", columns="model", values="false_positive_rate")[["A", "B", "C"]]
    fpr.index = rec.index
    out["missing_data_false_positive_rate"] = charts.grouped_bars(
        fpr, d / "missing_data_false_positive_rate.png",
        "False-positive rate by data condition (comparison window)", "false-positive rate")
    return {k: f"figures/{v.name}" for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="PREVENZA Phase 4A backtest")
    ap.add_argument("--sample", type=int, default=None, help="replay only N decision times (default: all)")
    args = ap.parse_args(argv)
    bcfg = load_backtest_config()
    sample = args.sample if args.sample is not None else bcfg["replay"]["sample"]

    F, I, A, match, rep, lag = replay_and_verify(sample)
    ver = versions()
    if_cutoff = pd.Timestamp(ver["if_train_cutoff"])
    P0 = M.build_predictions(F, I, A, bcfg, lag)
    labels = load_dataset("observation_labels")                     # ground truth: scoring only
    events = load_dataset("heat_risk_events")
    P, Mx, EP, EV, SITES, STATES, warm = evaluate(P0, labels, events, bcfg, if_cutoff)

    P = P.merge(match, on=KEY, how="left")
    P["in_full_window"] = M.window_mask(P, "full", bcfg, if_cutoff)
    P["in_comparison_window"] = M.window_mask(P, "comparison", bcfg, if_cutoff)
    for k, v in ver.items():
        P[f"version_{k}"] = v
    proc = config.path_for("processed")
    o = bcfg["output"]
    P.to_parquet(proc / o["backtest_results"], index=False)
    EV.to_parquet(proc / o["event_results"], index=False)
    EP.to_parquet(proc / o["episodes"], index=False)
    Mx.to_parquet(proc / o["metrics"], index=False)

    u = pd.read_parquet(unified_path())
    wmf = P["in_full_window"]
    windows = []
    for w in ("full", "comparison"):
        wm = P[f"in_{w}_window"]
        windows.append({"window": w, "first decision": P.loc[wm, "decision_time"].min(),
                        "last decision": P.loc[wm, "decision_time"].max(), "observations": int(wm.sum()),
                        "models": ", ".join(WINDOW_MODELS[w])})
    alerts = P[wmf & P["model_c_health_alert"].fillna(False).astype(bool)]
    alert_evidence = pd.DataFrame([{
        "C health-alert observations": len(alerts),
        "mean available weight": alerts["available_weight"].mean(),
        "mean reliability": alerts["reliability_score"].mean(),
        "share evidence adequate": alerts["evidence_level"].eq("adequate").mean(),
        "mean categories available": alerts["n_categories_available"].mean(),
        "all-row mean available weight": P.loc[wmf, "available_weight"].mean()}])
    fig = make_figures(P, EV, Mx, SITES, STATES, events, bcfg, if_cutoff)
    write_reports({"metrics": Mx, "events": EV, "episodes": EP, "sites": SITES, "states": STATES, "warm": warm,
                   "versions": ver, "replay": rep, "figures": fig, "bcfg": bcfg, "windows": windows,
                   "latency": latency_table(u, P, wmf), "alert_evidence": alert_evidence})
    print(json.dumps(rep["checks"]))
    print("wrote backtest outputs, docs/backtesting_report.md, docs/model_comparison.md, docs/figures/")


if __name__ == "__main__":
    main()
