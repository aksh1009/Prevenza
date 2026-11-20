"""Phase 3B.3 - evidence-aware multi-signal risk fusion.

    python -m src.detection.risk_fusion

Inputs : data/processed/features.parquet (as-of features incl. robust z and CUSUM)
         data/processed/isolation_forest_scores.parquet
Outputs: data/processed/risk_assessments.parquet   (one row per observation)
         data/processed/risk_explanations.parquet  (one row per observation x evidence component)
         docs/risk_fusion_report.md

Principles
* Five category scores (0-100): environmental, workforce_exposure, clinical, absenteeism, incidents.
* Missing, stale or baseline-less categories are EXCLUDED and listed - never scored 0. Weights are explicitly
  renormalised over available categories (`available_weight` reports the share of evidence present).
* Severity (risk_score / severity_band) and evidence sufficiency (evidence_level / reliability_score) are separate.
* Environmental (or exposure) evidence alone can never yield a health-risk class; health classes need a health
  category plus independent corroboration and persistence.
* Isolation Forest contributes only out-of-sample (evaluation-period) rows.
* Ground truth is never read. Scores are not probabilities and not individual clinical risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from src import config, schemas
from src.features.build import FEATURES_FILE
from src.features.quality import heat_index_validity

KEY = list(schemas.OBSERVATION_KEY)
SERIES = ["site_id", "zone_id", "shift"]
CATEGORIES = ("environmental", "workforce_exposure", "clinical", "absenteeism", "incidents")
SOURCE_NAMES = {"env": "environmental_data", "wf": "workforce_data", "clin": "clinical_data",
                "abs": "absenteeism_data", "inc": "incident_data"}
CLASSES = ("Normal", "Environmental warning", "Elevated occupational exposure", "Emerging health risk",
           "High health risk", "Insufficient evidence")
ALERT_CLASSES = ("Emerging health risk", "High health risk")
REPORT_PATH = config.PROJECT_ROOT / "docs" / "risk_fusion_report.md"


def load_fusion_config() -> dict:
    with open(config.CONFIG_DIR / "risk_fusion.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def required_columns(cfg: dict) -> list[str]:
    cols = set()
    for c in cfg["categories"].values():
        cols |= {c["primary"], c["observed"], c["baseline"], c["cusum"], f"{c['source']}_staleness_hours"}
        cols |= {comp["feature"] for comp in c["components"]}
    mv = cfg.get("measurement_validity", {}).get("heat_index")
    if mv:
        cols |= {mv["temperature_feature"], mv["heat_index_feature"]}
    return sorted(cols)


def validate_config(cfg: dict) -> None:
    if set(cfg["weights"]) != set(CATEGORIES) or set(cfg["categories"]) != set(CATEGORIES):
        raise ValueError(f"weights/categories must be exactly {CATEGORIES}")
    if any(w <= 0 for w in cfg["weights"].values()):
        raise ValueError("weights must be positive")
    bad = [c for c in required_columns(cfg) if c in schemas.LABEL_COLUMNS]
    if bad:
        raise ValueError(f"label columns cannot be inputs: {bad}")
    cl = cfg["classification"]
    if not 1 <= cl["persistence"]["min_elevated"] <= cl["persistence"]["window"]:
        raise ValueError("persistence.min_elevated must be in [1, window]")


def _ramp(x: pd.Series, low: float, high: float) -> pd.Series:
    return (100.0 * (x.astype(float) - low) / (high - low)).clip(0, 100)


# --------------------------------------------------------------------------------- category scoring
def category_scores(F: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Per category: score (NA unless available), status, component coverage and component detail."""
    out = {}
    for cat in CATEGORIES:
        c = cfg["categories"][cat]
        src = c["source"]
        comp_scores = {}
        num = pd.Series(0.0, index=F.index)
        den = pd.Series(0.0, index=F.index)
        for comp in c["components"]:
            s = _ramp(F[comp["feature"]], comp["low"], comp["high"])
            comp_scores[comp["feature"]] = s
            known = s.notna()
            num = num + s.fillna(0.0) * comp["weight"] * known     # unknown component: excluded, not 0
            den = den + comp["weight"] * known
        total_w = sum(comp["weight"] for comp in c["components"])
        raw = (num / den).where(den > 0)

        stale_h = F[f"{src}_staleness_hours"].astype(float)
        observed = F[c["observed"]].astype(float)
        primary = F[c["primary"]].astype(float)
        status = np.select(
            [observed.isna() | stale_h.isna(), stale_h > cfg["max_staleness_hours"][src], primary.isna()],
            ["missing_data", "stale", "no_baseline"], default="available")
        status = pd.Series(status, index=F.index)
        avail = status == "available"
        detail = pd.DataFrame(comp_scores).round(1).apply(
            lambda r: "; ".join(f"{k}={'NA' if pd.isna(v) else v}" for k, v in r.items()), axis=1)
        out[cat] = pd.DataFrame({
            "score": raw.where(avail),
            "status": status,
            "coverage": (den / total_w).where(avail, 0.0),
            "observed": observed, "baseline": F[c["baseline"]].astype(float), "robust_z": primary,
            "cusum": F[c["cusum"]].astype(float), "staleness_hours": stale_h, "components": detail,
        }, index=F.index)
    return out


def _persistence(elevated: pd.DataFrame, F: pd.DataFrame, window: int, min_elev: int) -> pd.DataFrame:
    """Elevated now AND elevated in >= min_elev of the last `window` observations of the series (causal)."""
    order = F.sort_values(SERIES + ["observation_date"]).index
    e = elevated.loc[order].astype(int)
    grp = [F.loc[order, s] for s in SERIES]
    counts = e.groupby(grp, sort=False).transform(lambda x: x.rolling(window, min_periods=1).sum())
    return ((counts >= min_elev) & elevated.loc[order]).reindex(elevated.index)


# ------------------------------------------------------------------------------------------- fusion
def fuse(F: pd.DataFrame, iso: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_config(cfg)
    F = F.reset_index(drop=True)
    cats = category_scores(F, cfg)
    w = cfg["weights"]
    w_total = sum(w.values())

    scores = pd.DataFrame({c: cats[c]["score"] for c in CATEGORIES})
    status = pd.DataFrame({c: cats[c]["status"] for c in CATEGORIES})
    avail = status == "available"
    wvec = pd.Series(w)
    avail_w = avail.mul(wvec, axis=1)
    avail_weight = avail_w.sum(axis=1) / w_total
    eff_w = avail_w.div(avail_w.sum(axis=1).where(lambda s: s > 0), axis=0)          # renormalised
    fused = (scores.fillna(0.0) * eff_w).sum(axis=1, min_count=1).where(avail.any(axis=1))

    # Isolation Forest: out-of-sample, eligible rows only
    ic = cfg["isolation_forest"]
    if iso is not None:
        m = F[KEY].merge(iso[KEY + ["split", "eligibility_status", "anomaly_score"]], on=KEY, how="left")
        if_used = (m["split"].eq("eval") & m["eligibility_status"].eq("eligible") & m["anomaly_score"].notna())
        if_score = _ramp(m["anomaly_score"], ic["score_low"], ic["score_high"])
        if_split, if_raw, if_elig = m["split"], m["anomaly_score"], m["eligibility_status"]
    else:
        if_used = pd.Series(False, index=F.index)
        if_score = if_raw = pd.Series(np.nan, index=F.index)
        if_split = if_elig = pd.Series(pd.NA, index=F.index, dtype="string")
    if_used = if_used & fused.notna()
    w_if = np.where(if_used, ic["weight"], 0.0)
    risk = (1 - w_if) * fused + w_if * if_score.fillna(0.0)

    # classification
    cl = cfg["classification"]
    elevated = avail & (scores >= cl["elevated_threshold"])
    persistent = _persistence(elevated, F, cl["persistence"]["window"], cl["persistence"]["min_elevated"])
    H, X = cl["health_categories"], cl["exposure_categories"]
    n_h_elev, n_h_pers = elevated[H].sum(axis=1), persistent[H].sum(axis=1)
    exp_elev, exp_pers = elevated[X].any(axis=1), persistent[X].any(axis=1)
    high = (n_h_pers >= 2) | ((n_h_elev >= 2) & (n_h_pers >= 1) & exp_elev)
    emerging = (n_h_elev >= 1) & (exp_elev | (n_h_elev >= 2)) & ((n_h_pers >= 1) | exp_pers)
    occ = elevated["workforce_exposure"]
    envw = elevated["environmental"]
    ev = cfg["evidence"]
    insufficient = (avail_weight < ev["min_available_weight"]) | ~avail.any(axis=1)
    cls = np.select([high, emerging, occ, envw], list(CLASSES[4:0:-1]), default="Normal")
    cls = np.where((cls == "Normal") & insufficient, "Insufficient evidence", cls)
    evidence = np.select([insufficient, avail_weight < ev["adequate_available_weight"]],
                         ["insufficient", "limited"], default="adequate")
    coverage = pd.DataFrame({c: cats[c]["coverage"] for c in CATEGORIES})
    reliability = 100 * (coverage * wvec).sum(axis=1) / w_total
    bands = cfg["severity_bands"]
    sev = pd.Series(pd.NA, index=F.index, dtype="string")
    lo = -np.inf
    for b in bands:
        sev[(risk > lo) & (risk <= b["max"])] = b["name"]
        lo = b["max"]

    def _join(mask):
        return mask.apply(lambda r: ";".join(c for c, v in r.items() if v), axis=1)

    A = F[KEY + ["decision_time"]].copy()
    A["risk_score"] = risk.round(2)
    A["severity_band"] = sev
    A["classification"] = cls
    A["alert"] = A["classification"].isin(ALERT_CLASSES)
    A["evidence_level"] = evidence
    A["reliability_score"] = reliability.round(1)
    A["available_weight"] = avail_weight.round(4)
    A["n_categories_available"] = avail.sum(axis=1)
    A["contributing_sources"] = _join(avail)
    A["missing_sources"] = _join(status.isin(["missing_data", "no_baseline"]))
    A["stale_sources"] = _join(status == "stale")
    for c in CATEGORIES:
        A[f"{c}_score"] = scores[c].round(2)
        A[f"{c}_status"] = status[c]
        A[f"{c}_elevated"] = elevated[c]
        A[f"{c}_persistent"] = persistent[c]
    A["n_health_elevated"] = n_h_elev
    A["n_health_persistent"] = n_h_pers
    A["exposure_corroborated"] = exp_elev
    A["if_used"] = if_used
    A["if_signal_score"] = if_score.round(2)
    A["if_contribution"] = pd.Series(np.where(if_used, w_if * if_score, 0.0)).round(2)

    # measurement validity: qualifies (never alters) the environmental evidence
    mv = cfg.get("measurement_validity", {}).get("heat_index")
    if mv:
        A["heat_index_observed"] = F[mv["heat_index_feature"]].astype(float)
        A["heat_index_validity"] = heat_index_validity(F[mv["temperature_feature"]], F[mv["heat_index_feature"]],
                                                       mv["max_temperature_c"], mv["max_heat_index_c"])
    else:
        A["heat_index_observed"] = np.nan
        A["heat_index_validity"] = "unavailable"
    extrap = A["heat_index_validity"].eq("extrapolated")
    A["measurement_flags"] = np.where(extrap, "heat_index_extrapolated", "")
    A["alert_relies_on_extrapolated_heat_index"] = A["alert"] & extrap & A["environmental_elevated"]

    # explanations (long)
    rows = []
    for c in CATEGORIES:
        d = cats[c]
        rows.append(pd.DataFrame({
            **{k: F[k] for k in KEY}, "decision_time": F["decision_time"], "component": c,
            "source_dataset": SOURCE_NAMES[cfg["categories"][c]["source"]], "status": d["status"],
            "configured_weight": w[c], "effective_weight": eff_w[c].fillna(0.0).round(4),
            "signal_score": d["score"].round(2),
            "contribution_points": ((1 - w_if) * eff_w[c].fillna(0.0) * d["score"].fillna(0.0)).round(2),
            "observed_value": d["observed"], "baseline_median": d["baseline"], "robust_z": d["robust_z"].round(2),
            "cusum": d["cusum"].round(2), "staleness_hours": d["staleness_hours"],
            "elevated": elevated[c], "persistent": persistent[c], "component_scores": d["components"],
            "note": _category_note(c, d, cfg) + (np.where(
                extrap, "; heat index outside NWS chart range (extrapolated) - magnitude unreliable", "")
                if c == "environmental" else ""),
        }))
    rows.append(pd.DataFrame({
        **{k: F[k] for k in KEY}, "decision_time": F["decision_time"], "component": "isolation_forest",
        "source_dataset": "isolation_forest_scores", "status": np.where(if_used, "available", np.where(
            if_split.eq("train").fillna(False), "in_sample_not_used", "not_eligible")),
        "configured_weight": ic["weight"], "effective_weight": w_if, "signal_score": if_score.round(2),
        "contribution_points": A["if_contribution"], "observed_value": if_raw.astype(float),
        "baseline_median": np.nan, "robust_z": np.nan, "cusum": np.nan, "staleness_hours": np.nan,
        "elevated": False, "persistent": False, "component_scores": "",
        "note": np.where(if_used, "out-of-sample isolation score (not a probability)",
                         np.where(if_split.eq("train").fillna(False),
                                  "training-period row: model fitted on later data, excluded",
                                  "isolation forest not eligible for this row (warm-up/insufficient)")),
    }))
    E = pd.concat(rows, ignore_index=True)
    A["explanation"] = _explain(A, cats, cfg)
    return A, E


def _category_note(cat: str, d: pd.DataFrame, cfg: dict) -> pd.Series:
    c = cfg["categories"][cat]
    lim = cfg["max_staleness_hours"][c["source"]]
    msgs = {"available": "", "missing_data": "no data known at decision time (excluded, not zero)",
            "no_baseline": "insufficient baseline history (excluded)"}
    return d.apply(lambda r: f"last report {r['staleness_hours']:.0f} h old > {lim} h limit (excluded)"
                   if r["status"] == "stale" else msgs[r["status"]], axis=1)


def _fmt(v, nd=1):
    return "NA" if pd.isna(v) else f"{v:.{nd}f}"


def _explain(A: pd.DataFrame, cats: dict, cfg: dict) -> pd.Series:
    labels = {"environmental": "environmental", "workforce_exposure": "workforce exposure",
              "clinical": "clinical", "absenteeism": "absenteeism", "incidents": "incidents"}
    rule = {
        "High health risk": "two or more health categories elevated with persistence (and/or exposure corroboration)",
        "Emerging health risk": "a health category elevated with independent corroboration and persistence",
        "Elevated occupational exposure": "workforce heat-exposure load elevated; no corroborated health signal",
        "Environmental warning": "environmental heat elevated; environmental evidence alone cannot indicate a "
                                 "health cluster",
        "Normal": "no category elevated beyond threshold",
        "Insufficient evidence": "too little current evidence to confirm normal conditions",
    }
    out = []
    for i, r in A.iterrows():
        drivers, others = [], []
        for c in CATEGORIES:
            d = cats[c].loc[i]
            if d["status"] == "available":
                txt = (f"{labels[c]} {_fmt(d['score'], 0)} (observed {_fmt(d['observed'], 2)} vs baseline "
                       f"{_fmt(d['baseline'], 2)}, z {_fmt(d['robust_z'])}, CUSUM {_fmt(d['cusum'])})")
                (drivers if r[f"{c}_elevated"] else others).append(txt + (" [persistent]" if r[f"{c}_persistent"]
                                                                           else ""))
        head = (f"{r['classification']}: risk {_fmt(r['risk_score'], 0)}/100 ({r['severity_band'] if pd.notna(r['severity_band']) else 'no score'}); "
                f"evidence {r['evidence_level']}, reliability {_fmt(r['reliability_score'], 0)}%, "
                f"{_fmt(100 * r['available_weight'], 0)}% of signal weight available.")
        parts = [head]
        if drivers:
            parts.append("Elevated: " + "; ".join(drivers) + ".")
        if others:
            parts.append("Not elevated: " + "; ".join(others) + ".")
        if r["missing_sources"]:
            parts.append(f"Missing (excluded, not zero): {r['missing_sources'].replace(';', ', ')}.")
        if r["stale_sources"]:
            parts.append(f"Stale (excluded): {r['stale_sources'].replace(';', ', ')}.")
        parts.append(f"Isolation Forest: {'+' + _fmt(r['if_contribution']) + ' points (out-of-sample)' if r['if_used'] else 'not used'}.")
        if r["n_health_elevated"] >= 1 and r["classification"] not in ALERT_CLASSES:
            parts.append("Note: health signal elevated but not corroborated/persistent - monitor.")
        if r["heat_index_validity"] == "extrapolated":
            parts.append(f"Measurement caution: heat index {_fmt(r['heat_index_observed'])} C is outside the NWS chart "
                         "range (temperature <= 43.3 C, heat index <= 58.3 C); the value is an extrapolation and its "
                         "magnitude is unreliable (environmental score saturates; classification is not based on it alone).")
        parts.append(f"Rule: {rule[r['classification']]}.")
        out.append(" ".join(parts))
    return pd.Series(out, index=A.index)


# ------------------------------------------------------------------------------------------- report
def render_report(A: pd.DataFrame, cfg: dict) -> str:
    def table(df):
        df = df.reset_index()
        lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
        return "\n".join(lines)
    cls = A["classification"].value_counts().reindex(CLASSES, fill_value=0).rename("observations")
    ct = pd.crosstab(A["severity_band"].fillna("no score"), A["evidence_level"])
    st = pd.DataFrame({c: A[f"{c}_status"].value_counts() for c in CATEGORIES}).fillna(0).astype(int)
    return "\n".join([
        "# PREVENZA Risk Fusion Run (Phase 3B.3)", "",
        "> GENERATED by `python -m src.detection.risk_fusion`. Synthetic data. Descriptive counts only - risk "
        "scores are evidence-weighted 0-100 indices, **not probabilities and not individual clinical risk**. "
        "No ground-truth labels were used; no performance metrics are reported here.", "",
        "## Configuration", "",
        "- Weights: " + ", ".join(f"{k} {v}" for k, v in cfg["weights"].items()) + " (renormalised over available "
        "categories).",
        f"- Elevated threshold {cfg['classification']['elevated_threshold']}; persistence = elevated now and in >= "
        f"{cfg['classification']['persistence']['min_elevated']} of last {cfg['classification']['persistence']['window']} "
        "observations of the series.",
        f"- Evidence: insufficient < {cfg['evidence']['min_available_weight']:.0%} available weight; adequate >= "
        f"{cfg['evidence']['adequate_available_weight']:.0%}.",
        f"- Isolation Forest weight {cfg['isolation_forest']['weight']} (evaluation-period rows only).", "",
        "## Classification counts", "", table(cls.to_frame()), "",
        "## Severity x evidence (kept separate by design)", "", table(ct), "",
        "## Category status counts", "", table(st), "",
    ])


def main():
    cfg = load_fusion_config()
    proc = config.path_for("processed")
    F = pd.read_parquet(proc / FEATURES_FILE)
    iso_path = proc / "isolation_forest_scores.parquet"
    iso = pd.read_parquet(iso_path) if iso_path.exists() else None
    A, E = fuse(F, iso, cfg)
    A.to_parquet(proc / cfg["output"]["assessments_file"], index=False)
    E.to_parquet(proc / cfg["output"]["explanations_file"], index=False)
    REPORT_PATH.write_text(render_report(A, cfg), encoding="utf-8")
    print(f"risk_assessments: {A.shape}; risk_explanations: {E.shape}")
    print(A["classification"].value_counts().to_string())


if __name__ == "__main__":
    main()
