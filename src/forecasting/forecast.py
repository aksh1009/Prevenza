"""Phase 4B - 24-hour heat-risk state forecasting.

    python -m src.forecasting.forecast

Question: at decision time t for site-zone-shift (z, s), will site-zone z be in an emerging/escalating labelled
state in any observation that starts within the next 24 h?

* Targets come from FUTURE ground-truth labels and are used only to train/evaluate.
* Inputs are the row's own as-of features (Phase 3A) and risk-fusion outputs (Phase 3B.3) - both verified
  leakage-free by the Phase 4A chronological replay. Isolation Forest scores are excluded (in-sample in training).
* Splits are chronological with a purge so no training target window reaches into a later split.
* Scores are UNCALIBRATED forecast scores (class-weighted logistic regression), not probabilities.
"""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, schemas
from src.features.build import FEATURES_FILE
from src.preprocessing.load import load_dataset

KEY = list(schemas.OBSERVATION_KEY)
SERIES = ["site_id", "zone_id", "shift"]
SHIFT_START_H = {"day": 6, "night": 18}
REPORT = config.PROJECT_ROOT / "docs" / "forecast_report.md"
FIG = config.PROJECT_ROOT / "docs" / "figures" / "forecast_pr_curve.png"


def load_forecast_config() -> dict:
    with open(config.CONFIG_DIR / "forecast.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def feature_names(cfg) -> list[str]:
    return list(cfg["features"]["as_of"]) + list(cfg["features"]["fusion"])


def _shift_start(df: pd.DataFrame) -> pd.Series:
    return df["observation_date"] + pd.to_timedelta(df["shift"].map(SHIFT_START_H), unit="h")


# ---------------------------------------------------------------------------------------------- targets
def build_targets(labels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Forward-looking target per observation from FUTURE labels of the same site-zone (both shifts).

    target = 1 if any observation with shift_start in (s, s + horizon] has a positive label; NA if the horizon
    extends past the end of the labelled data (cannot be evaluated).
    """
    h = pd.Timedelta(hours=cfg["target"]["horizon_hours"])
    pos = set(cfg["target"]["positive_labels"])
    L = labels.copy()
    L["shift_start"] = _shift_start(L)
    out = []
    for zone, g in L.sort_values("shift_start").groupby("zone_id", sort=False):
        s = g["shift_start"].to_numpy()
        lab = g["label"].astype(int).to_numpy()
        ev = g["event_id"].astype(object).to_numpy()
        end = s.max()
        tgt = np.full(len(g), np.nan)
        tev = np.full(len(g), None, dtype=object)
        for i in range(len(g)):
            if s[i] + h > end:
                continue                                   # horizon not fully observed
            fut = (s > s[i]) & (s <= s[i] + h)
            hit = fut & np.isin(lab, list(pos))
            tgt[i] = float(hit.any())
            if hit.any():
                tev[i] = ev[np.flatnonzero(hit)[0]]
        part = g[KEY].copy()
        part["target"] = pd.array(np.where(np.isnan(tgt), pd.NA, tgt), dtype="Int64")
        part["target_event_id"] = pd.array(tev, dtype="string")
        out.append(part)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------- inputs
def build_frame(F: pd.DataFrame, A: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Decision-time inputs + eligibility + persistence baselines. Contains NO labels or targets."""
    extra = [c for c in ("heat_index_validity",) if c in A.columns]
    fus = A[KEY + cfg["features"]["fusion"] + ["classification", "missing_sources", "stale_sources",
                                                "evidence_level"] + extra]
    X = F[KEY + ["decision_time"] + cfg["features"]["as_of"]].merge(fus, on=KEY, how="left", validate="one_to_one")
    X["shift_start"] = _shift_start(X)
    order = X.sort_values(SERIES + ["observation_date"]).index
    X.loc[order, "series_position"] = X.loc[order].groupby(SERIES, sort=False).cumcount().to_numpy()
    X["series_position"] = X["series_position"].astype(int)
    el = cfg["eligibility"]
    warm = X["series_position"] < el["warmup_observations"]
    low = X["available_weight"].fillna(0) < el["min_available_weight"]
    X["eligible"] = ~warm & ~low
    X["ineligibility_reason"] = np.where(warm, "warm_up (no baseline)",
                                         np.where(low, "insufficient evidence (available weight < "
                                                  f"{el['min_available_weight']})", ""))
    p = cfg["persistence"]
    X["persistence_health"] = X["classification"].isin(p["health_classes"])
    X["persistence_any"] = X["classification"].isin(p["any_classes"])
    feats = feature_names(cfg)
    miss = X[feats].isna()
    X["n_inputs_missing"] = miss.sum(axis=1)
    X["missing_inputs"] = miss.apply(lambda r: ";".join(c for c, v in r.items() if v), axis=1)
    for c in feats:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
    return X


def assign_splits(X: pd.DataFrame, cfg: dict) -> pd.Series:
    sp = cfg["split"]
    dates = np.sort(X["observation_date"].unique())
    n_tr = int(round(sp["train_fraction"] * len(dates)))
    n_va = int(round(sp["validation_fraction"] * len(dates)))
    d = X["observation_date"]
    split = pd.Series(np.where(d < dates[n_tr], "train", np.where(d < dates[n_tr + n_va], "validation", "test")),
                      index=X.index)
    purge = pd.Timedelta(hours=sp["purge_hours"])
    for cur, nxt in (("train", "validation"), ("validation", "test")):
        nxt_start = X.loc[split == nxt, "shift_start"].min()
        split[(split == cur) & (X["shift_start"] + purge >= nxt_start)] = "purged"
    return split


# ---------------------------------------------------------------------------------------------- model
def make_pipeline(C: float, cfg: dict) -> Pipeline:
    lr = cfg["logistic_regression"]
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, class_weight=lr["class_weight"], max_iter=lr["max_iter"],
                                     random_state=lr["random_state"])),
    ])


def _xy(D, mask, feats):
    m = mask & D["eligible"] & D["target"].notna()
    return D.loc[m, feats].astype(float), D.loc[m, "target"].astype(int)


def best_f1_threshold(y, s) -> float:
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = np.where(prec[:-1] + rec[:-1] > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]), 0)
    return float(thr[int(np.argmax(f1))])


def fit(D: pd.DataFrame, cfg: dict):
    """Select C and threshold on validation (train-only fits), then optionally refit on train+validation."""
    feats = feature_names(cfg)
    Xtr, ytr = _xy(D, D["split"] == "train", feats)
    Xva, yva = _xy(D, D["split"] == "validation", feats)
    if ytr.sum() == 0 or yva.sum() == 0:
        raise RuntimeError("no positive targets in train or validation")
    grid = []
    for C in cfg["logistic_regression"]["C_grid"]:
        p = make_pipeline(C, cfg).fit(Xtr, ytr)
        grid.append({"C": C, "validation_average_precision": float(average_precision_score(yva, p.predict_proba(Xva)[:, 1]))})
    best = max(grid, key=lambda g: (g["validation_average_precision"], -g["C"]))
    p_tr = make_pipeline(best["C"], cfg).fit(Xtr, ytr)
    thr = best_f1_threshold(yva, p_tr.predict_proba(Xva)[:, 1])
    if cfg["logistic_regression"]["refit_on_train_validation"]:
        Xf, yf = _xy(D, D["split"].isin(["train", "validation"]), feats)
        final = make_pipeline(best["C"], cfg).fit(Xf, yf)
        fit_rows = len(Xf)
    else:
        final, fit_rows = p_tr, len(Xtr)
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    h.update(np.ascontiguousarray(Xtr.to_numpy(float)).tobytes())
    meta = {"model_version": h.hexdigest()[:12], "features": feats, "selected_C": best["C"], "C_grid": grid,
            "threshold": thr, "threshold_source": "max F1 on validation (train-only model)",
            "train_rows": len(Xtr), "train_positives": int(ytr.sum()), "validation_rows": len(Xva),
            "validation_positives": int(yva.sum()), "final_fit_rows": fit_rows,
            "refit_on_train_validation": cfg["logistic_regression"]["refit_on_train_validation"],
            "config": cfg, "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__}}
    return final, p_tr, meta


def save_model(pipe, meta, cfg):
    d = config.PROJECT_ROOT / cfg["output"]["model_dir"]
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, d / "pipeline.joblib")
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return d


def load_model(cfg):
    d = config.PROJECT_ROOT / cfg["output"]["model_dir"]
    return joblib.load(d / "pipeline.joblib"), json.loads((d / "metadata.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------- evaluation
def scores(D: pd.DataFrame, pipe, cfg) -> np.ndarray:
    s = np.full(len(D), np.nan)
    m = D["eligible"].to_numpy()
    if m.any():
        s[m] = pipe.predict_proba(D.loc[m, feature_names(cfg)].astype(float))[:, 1]
    return s


def binary_metrics(y, flag, score=None) -> dict:
    y = np.asarray(y, int)
    f = np.asarray(flag, bool)
    yb = y.astype(bool)
    tp, fp, fn = int((yb & f).sum()), int((~yb & f).sum()), int((yb & ~f).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if fp + fn else np.nan)
    d = {"n": len(y), "positives": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
         "f1": f1}
    s = f.astype(float) if score is None else np.asarray(score, float)
    d["pr_auc"] = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan
    d["brier"] = float(brier_score_loss(y, s)) if score is not None and len(y) else np.nan
    return d


def evaluate(D: pd.DataFrame, cfg: dict, thr: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, by_site, by_event = [], [], []
    for split in ("train", "validation", "test"):
        m = (D["split"] == split) & D["eligible"] & D["target"].notna()
        E = D[m].copy()
        y = E["target"].astype(int)
        if split == "validation":                 # out-of-sample: train-only selection model
            E["lr_score"] = E["lr_score_selection"]
        lr_note = {"train": "in-sample (fitted on these rows)",
                   "validation": "out-of-sample (train-only model used for selection)",
                   "test": "out-of-sample (final model)"}[split]
        models = {"persistence (health alert)": (E["persistence_health"], None, "no fitting"),
                  "persistence (any warning)": (E["persistence_any"], None, "no fitting"),
                  "logistic regression": (E["lr_score"] >= thr, E["lr_score"], lr_note)}
        for name, (flag, sc, note) in models.items():
            rows.append({"split": split, "model": name, "sample": note, **binary_metrics(y, flag, sc)})
            if split == "test":
                for site, g in E.groupby("site_id"):
                    f = flag.loc[g.index]
                    by_site.append({"model": name, "site_id": site,
                                    **binary_metrics(g["target"].astype(int), f, None if sc is None else sc.loc[g.index])})
        if split in ("validation", "test"):
            for ev_id, g in E[E["target"] == 1].groupby("target_event_id"):
                r = {"split": split, "event_id": ev_id, "zones": ", ".join(sorted(g["zone_id"].unique())),
                     "positive rows": len(g)}
                for name, col in (("persistence (health alert)", "persistence_health"),
                                  ("persistence (any warning)", "persistence_any")):
                    r[f"recall {name}"] = float(g[col].mean())
                r["recall logistic regression"] = float((g["lr_score"] >= thr).mean())
                first = g.loc[g["lr_score"] >= thr, "decision_time"].min()
                r["first LR positive forecast"] = first
                by_event.append(r)
            neg = E[E["target"] == 0]
            by_event.append({"split": split, "event_id": "(no event: negatives)", "zones": "all",
                             "positive rows": 0, "recall persistence (health alert)": np.nan,
                             "recall persistence (any warning)": np.nan, "recall logistic regression": np.nan,
                             "false-positive rate LR": float((neg["lr_score"] >= thr).mean()),
                             "false-positive rate persistence (any)": float(neg["persistence_any"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(by_site), pd.DataFrame(by_event)


def reliability_table(y, s, bins=5) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, int), "s": np.asarray(s, float)})
    df["bin"] = pd.cut(df["s"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=False).agg(n=("y", "size"), mean_score=("s", "mean"), observed_rate=("y", "mean"))
    return t.reset_index().assign(bin=lambda x: x["bin"].astype(str))


# ------------------------------------------------------------------------------------------- outputs
def forecast_table(D: pd.DataFrame, meta: dict, cfg: dict) -> pd.DataFrame:
    """Operational 24 h forecasts. No labels or targets in this table."""
    thr = meta["threshold"]
    out = D[KEY + ["decision_time"]].copy()
    out["horizon_end"] = D["shift_start"] + pd.Timedelta(hours=cfg["target"]["horizon_hours"] + 12)
    out["split"] = D["split"]
    out["eligible"] = D["eligible"]
    out["ineligibility_reason"] = D["ineligibility_reason"]
    out["forecast_score"] = D["lr_score"].round(4)
    out["forecast_flag"] = pd.array(np.where(D["eligible"], D["lr_score"] >= thr, False), dtype="boolean")
    out.loc[~D["eligible"], "forecast_flag"] = pd.NA
    out["threshold"] = thr
    out["persistence_health_flag"] = D["persistence_health"]
    out["current_classification"] = D["classification"]
    out["evidence_level"] = D["evidence_level"]
    out["heat_index_validity"] = D["heat_index_validity"] if "heat_index_validity" in D else "unavailable"
    out["n_inputs_missing"] = D["n_inputs_missing"]
    out["missing_inputs"] = D["missing_inputs"]
    out["missing_sources"] = D["missing_sources"]
    out["stale_sources"] = D["stale_sources"]
    out["in_sample"] = D["split"].isin(["train", "validation", "purged"]) if meta["refit_on_train_validation"] \
        else D["split"].isin(["train", "purged"])
    lim = []
    for r in out.itertuples():
        notes = ["uncalibrated score (not a probability)", "synthetic-data model"]
        if not r.eligible:
            notes = [f"no forecast: {r.ineligibility_reason}"]
        else:
            if r.n_inputs_missing:
                notes.append(f"{r.n_inputs_missing} inputs imputed (training medians)")
            if r.missing_sources:
                notes.append(f"missing: {r.missing_sources.replace(';', ', ')}")
            if r.stale_sources:
                notes.append(f"stale: {r.stale_sources.replace(';', ', ')}")
            if r.heat_index_validity == "extrapolated":
                notes.append("heat index outside NWS chart range (extrapolated input, magnitude unreliable)")
            if r.evidence_level != "adequate":
                notes.append(f"evidence {r.evidence_level}")
            if r.in_sample:
                notes.append("in-sample row (model fitted on this period)")
        lim.append("; ".join(notes))
    out["limitations"] = lim
    last = out.groupby(SERIES)["decision_time"].transform("max")
    out["is_latest"] = out["decision_time"] == last
    out["model_version"] = meta["model_version"]
    return out


def pr_figure(Dt: pd.DataFrame, thr: float):
    from src.evaluation import charts  # shared palette/style
    import matplotlib.pyplot as plt
    y = Dt["target"].astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    p, r, _ = precision_recall_curve(y, Dt["lr_score"])
    ax.plot(r, p, color=charts.MODEL_COLORS["C"], lw=2, label="logistic regression (test)")
    for name, col, mk in (("persistence (health alert)", "persistence_health", "o"),
                          ("persistence (any warning)", "persistence_any", "s")):
        d = binary_metrics(y, Dt[col])
        ax.scatter(d["recall"], d["precision"], marker=mk, s=60, color=charts.MODEL_COLORS["A" if mk == "o" else "B"],
                   label=name, zorder=3)
    d = binary_metrics(y, Dt["lr_score"] >= thr)
    ax.scatter(d["recall"], d["precision"], marker="D", s=50, color=charts.INK, label="LR at validation threshold",
               zorder=4)
    ax.axhline(y.mean(), color=charts.INK2, lw=1, ls="--", label=f"prevalence {y.mean():.2f}")
    ax.set_xlim(0, 1.02), ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall"), ax.set_ylabel("precision")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("24 h forecast: precision-recall on test (synthetic)", loc="left")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("NA" if np.isnan(v) else f"{v:.3f}")
            elif isinstance(v, pd.Timestamp):
                cells.append(v.strftime("%Y-%m-%d %H:%M"))
            elif v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)) or (v is pd.NaT):
                cells.append("NA")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(D, meta, overall, by_site, by_event, rel, cfg) -> str:
    sp = D.groupby("split").agg(rows=("eligible", "size"), eligible=("eligible", "sum"),
                                first_date=("observation_date", "min"), last_date=("observation_date", "max"))
    ev = D[D["eligible"] & D["target"].notna()].groupby("split")["target"].agg(["size", "sum"]).rename(
        columns={"size": "evaluable rows", "sum": "positive targets"})
    sp = sp.join(ev).reset_index()
    events = D[D["target"] == 1].groupby("split")["target_event_id"].agg(lambda s: ", ".join(sorted(s.dropna().unique())))
    sp["events behind positives"] = sp["split"].map(events).fillna("-")
    return "\n".join([
        "# PREVENZA 24-hour Heat-Risk Forecast (Phase 4B)", "",
        "> GENERATED by `python -m src.forecasting.forecast`. Do not edit by hand.", "",
        "> **Synthetic, small-sample results.** The forecaster is trained and tested on one simulated 90-day dataset "
        "with three escalating events (five event-zones). Each split contains essentially one event, so metrics "
        "describe behaviour on those simulated scenarios only and carry very large uncertainty. Scores are "
        "**uncalibrated** (class-weighted logistic regression) - they rank risk; they are not probabilities and "
        "not individual clinical risk. No real-world validity is claimed.", "",
        "## Task", "",
        f"- Target: site-zone in an emerging/escalating labelled state in any observation starting within "
        f"{cfg['target']['horizon_hours']} h after the current shift start (next two shifts, both shift types). "
        "Built from future labels for training/evaluation only; rows whose horizon passes the end of data have no target.",
        "- Inputs: the row's own as-of features and risk-fusion outputs at its decision timestamp (replay-verified in "
        "Phase 4A). Isolation Forest scores are excluded (in-sample during training). Labels are never inputs.",
        f"- Eligibility: post warm-up ({cfg['eligibility']['warmup_observations']} observations) and fusion available "
        f"weight >= {cfg['eligibility']['min_available_weight']}; otherwise no forecast is issued.", "",
        "## Chronological splits", "",
        f"Fractions of dates {cfg['split']['train_fraction']}/{cfg['split']['validation_fraction']}/remainder, fixed "
        f"a priori; rows whose {cfg['split']['purge_hours']} h target window reaches the next split are purged.", "",
        _md(sp), "",
        "## Models", "",
        "- **Persistence (health alert)**: forecast positive if the current risk-fusion class is Emerging/High health risk.",
        "- **Persistence (any warning)**: forecast positive if the current class is any non-normal warning.",
        f"- **Logistic regression**: median imputation + missing-indicators, standardisation, L2 logistic regression "
        f"with balanced class weights; C selected on validation average precision from {cfg['logistic_regression']['C_grid']} "
        f"-> **C = {meta['selected_C']}**; threshold = max-F1 on validation -> **{meta['threshold']:.3f}**; final model "
        f"{'refitted on train + validation' if meta['refit_on_train_validation'] else 'trained on train only'} "
        f"({meta['final_fit_rows']} rows). Version `{meta['model_version']}`.",
        "- **XGBoost: not used.** The training split contains one escalating event (EVT-001, 3 zones of the same "
        "site); a boosted-tree model with many interactions would memorise it, and validation has a single event to "
        "detect that. A linear model with few inputs is the defensible choice at this sample size.", "",
        "Validation C grid:", "", _md(pd.DataFrame(meta["C_grid"])), "",
        "## Results", "", _md(overall), "",
        "PR-AUC = average precision. For persistence (binary flags) it reduces to a single operating point. Brier is "
        "shown for the logistic score but, with balanced class weights and no calibration step, it measures an "
        "uncalibrated score and should not be read as probability accuracy.", "",
        "## Test results by site", "", _md(by_site), "",
        "## Validation / test results by event", "", _md(by_event), "",
        "## Score reliability on test (descriptive, NOT a calibration claim)", "", _md(rel), "",
        "Too few test positives exist to fit or validate a calibration map; the table shows why scores must not be "
        "read as probabilities.", "",
        "![PR curve](figures/forecast_pr_curve.png)", "",
        "## Outputs", "",
        "- `data/processed/forecast_results.parquet` - operational forecasts (no labels): score, flag, threshold, "
        "eligibility, missing/stale inputs, limitations text, in-sample flag, `is_latest` per site-zone-shift.",
        "- `data/processed/forecast_evaluation.parquet` - targets + model outputs for evaluation only.",
        f"- `{cfg['output']['model_dir']}/` - fitted pipeline (imputer, scaler, model) and metadata (config, features, C, "
        "threshold, versions).", "",
        "## Limitations", "",
        "- One simulated realisation; ~1 event per split; results can flip with a different seed or split point.",
        "- Targets follow the injected schedule (labels), so 'emerging' periods may have little observable signal.",
        "- Validation threshold transferred to a model refitted on train + validation (standard practice, unverified here).",
        "- Scores are uncalibrated; no claim of probability accuracy is made.",
        "- Real-world use would require prospective validation on real, independently labelled data.", "",
    ])


# --------------------------------------------------------------------------------------------- main
def run(cfg: dict | None = None, write: bool = True):
    cfg = cfg or load_forecast_config()
    proc = config.path_for("processed")
    F = pd.read_parquet(proc / FEATURES_FILE)
    A = pd.read_parquet(proc / "risk_assessments.parquet")
    X = build_frame(F, A, cfg)
    T = build_targets(load_dataset("observation_labels"), cfg)
    D = X.merge(T, on=KEY, how="left", validate="one_to_one")
    D["split"] = assign_splits(D, cfg)
    final, p_tr, meta = fit(D, cfg)
    D["lr_score"] = scores(D, final, cfg)
    D["lr_score_selection"] = scores(D, p_tr, cfg)
    overall, by_site, by_event = evaluate(D, cfg, meta["threshold"])
    Dt = D[(D["split"] == "test") & D["eligible"] & D["target"].notna()]
    rel = reliability_table(Dt["target"], Dt["lr_score"])
    fc = forecast_table(D, meta, cfg)
    ev = D[KEY + ["decision_time", "split", "eligible", "target", "target_event_id", "lr_score",
                  "persistence_health", "persistence_any"]].copy()
    ev["lr_flag"] = ev["lr_score"] >= meta["threshold"]
    if write:
        save_model(final, meta, cfg)
        fc.to_parquet(proc / cfg["output"]["forecasts_file"], index=False)
        ev.to_parquet(proc / cfg["output"]["evaluation_file"], index=False)
        pr_figure(Dt, meta["threshold"])
        REPORT.write_text(render_report(D, meta, overall, by_site, by_event, rel, cfg), encoding="utf-8")
    return D, final, meta, overall, fc


def main():
    D, _, meta, overall, fc = run()
    print(f"model {meta['model_version']}: C={meta['selected_C']} threshold={meta['threshold']:.3f}")
    print(overall[["split", "model", "n", "positives", "precision", "recall", "f1", "pr_auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
