"""Phase 3B.2 - Isolation Forest anomaly detection on as-of features.

    python -m src.detection.isolation_forest              # fit on training window, score all rows, persist
    python -m src.detection.isolation_forest --use-saved  # score with the persisted pipeline (no refit)

Inputs : data/processed/features.parquet (Phase 3A, as-of features only). Ground truth is never read.
Outputs: data/processed/isolation_forest_scores.parquet
         models/isolation_forest/{pipeline.joblib, metadata.json}
         docs/isolation_forest_report.md

Interpretation: `anomaly_score` is a relative isolation score (higher = more unusual than training data;
> 0 means flagged at the configured contamination). It is NOT a probability and NOT a clinical risk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src import config, schemas
from src.features.build import FEATURES_FILE

KEY = list(schemas.OBSERVATION_KEY)
SERIES = ["site_id", "zone_id", "shift"]
STATUS = ("eligible", "warm_up", "insufficient_features")
REPORT_PATH = config.PROJECT_ROOT / "docs" / "isolation_forest_report.md"
_FORBIDDEN_TOKENS = ("label", "event", "onset", "escalat", "severity", "issue_")


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- config
def load_if_config() -> dict:
    with open(config.CONFIG_DIR / "isolation_forest.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_features(cfg: dict) -> list[str]:
    return list(cfg["features"]["signal"]) + list(cfg["features"].get("data_quality", []))


def validate_config(cfg: dict, available_columns) -> None:
    feats = model_features(cfg)
    if len(set(feats)) != len(feats):
        raise ConfigError("duplicate feature names")
    missing = [f for f in feats if f not in set(available_columns)]
    if missing:
        raise ConfigError(f"features not in features.parquet: {missing}")
    bad = [f for f in feats if f in schemas.LABEL_COLUMNS or any(t in f.lower() for t in _FORBIDDEN_TOKENS)]
    if bad:
        raise ConfigError(f"ground-truth-like columns may not be features: {bad}")
    if cfg["split"]["method"] != "chronological":
        raise ConfigError("only chronological splits are allowed (no random temporal splits)")
    if not 0 < cfg["split"]["train_fraction"] < 1:
        raise ConfigError("train_fraction must be in (0, 1)")
    el = cfg["eligibility"]
    if not set(el["required_features"]) <= set(feats):
        raise ConfigError("required_features must be model features")
    if not 1 <= el["min_signal_features"] <= len(cfg["features"]["signal"]):
        raise ConfigError("min_signal_features out of range")
    c = cfg["model"]["contamination"]
    if c != "auto" and not 0 < float(c) <= 0.5:
        raise ConfigError("contamination must be 'auto' or in (0, 0.5]")


# ------------------------------------------------------------------------------- eligibility / split
def series_position(F: pd.DataFrame) -> pd.Series:
    order = F.sort_values(SERIES + ["observation_date"])
    return order.groupby(SERIES, sort=False).cumcount().reindex(F.index)


def eligibility(F: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feats, sig = model_features(cfg), cfg["features"]["signal"]
    el = cfg["eligibility"]
    known = F[feats].notna()
    pos = series_position(F)
    n_sig = known[sig].sum(axis=1)
    req_ok = known[el["required_features"]].all(axis=1)
    status = np.where(pos < el["warmup_observations"], "warm_up",
                      np.where((n_sig >= el["min_signal_features"]) & req_ok, "eligible", "insufficient_features"))
    missing = known.apply(lambda r: ";".join(c for c, ok in r.items() if not ok), axis=1)
    return pd.DataFrame({"series_position": pos.astype(int), "eligibility_status": status,
                         "n_model_features": len(feats), "n_features_available": known.sum(axis=1).astype(int),
                         "n_signal_features_available": n_sig.astype(int), "missing_features": missing},
                        index=F.index)


def chronological_split(F: pd.DataFrame, cfg: dict) -> tuple[pd.Timestamp, pd.Series]:
    """Train = decision_time < cutoff; eval = decision_time >= cutoff + embargo. Purely time-based."""
    dates = np.sort(F["decision_time"].dt.normalize().unique())
    k = int(np.ceil(cfg["split"]["train_fraction"] * len(dates))) - 1
    cutoff = pd.Timestamp(dates[k]) + pd.Timedelta(days=1)            # end of the k-th decision date
    eval_start = cutoff + pd.Timedelta(hours=cfg["split"].get("embargo_hours", 0))
    dt = F["decision_time"]
    split = pd.Series(np.where(dt < cutoff, "train", np.where(dt >= eval_start, "eval", "embargo")),
                      index=F.index)
    return cutoff, split


# ------------------------------------------------------------------------------------------- model
def build_pipeline(cfg: dict) -> Pipeline:
    pp, m = cfg["preprocessing"], cfg["model"]
    scaler = {"robust": RobustScaler(), "standard": StandardScaler()}[pp["scaler"]]
    contamination = m["contamination"] if m["contamination"] == "auto" else float(m["contamination"])
    return Pipeline([
        ("impute", SimpleImputer(strategy=pp["imputer_strategy"])),
        ("scale", scaler),
        ("iforest", IsolationForest(n_estimators=m["n_estimators"], max_samples=m["max_samples"],
                                    contamination=contamination, max_features=m["max_features"],
                                    bootstrap=m["bootstrap"], random_state=m["random_state"])),
    ])


def _fingerprint(cfg: dict, X: pd.DataFrame) -> str:
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(X.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


def fit(F: pd.DataFrame, cfg: dict) -> tuple[Pipeline, dict]:
    """Fit on eligible TRAINING rows only (imputer, scaler and forest all see training data only)."""
    validate_config(cfg, F.columns)
    feats = model_features(cfg)
    el = eligibility(F, cfg)
    cutoff, split = chronological_split(F, cfg)
    train_mask = (split == "train") & (el["eligibility_status"] == "eligible")
    X = F.loc[train_mask, feats].astype(float)
    if len(X) < 50:
        raise RuntimeError(f"only {len(X)} eligible training rows")
    empty = [c for c in feats if X[c].isna().all()]
    if empty:
        raise RuntimeError(f"features with no training values: {empty}")
    pipe = build_pipeline(cfg).fit(X)
    meta = {
        "model_version": _fingerprint(cfg, X),
        "features": feats,
        "train_cutoff": str(cutoff),
        "train_rows": int(len(X)),
        "train_decision_time_max": str(F.loc[train_mask, "decision_time"].max()),
        "imputer_medians": dict(zip(feats, map(float, pipe.named_steps["impute"].statistics_))),
        "threshold_offset": float(pipe.named_steps["iforest"].offset_),
        "config": cfg,
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    return pipe, meta


def score(F: pd.DataFrame, pipe: Pipeline, meta: dict, cfg: dict) -> pd.DataFrame:
    """Score eligible rows; ineligible rows get NA decision/score (no forced predictions)."""
    feats = meta["features"]
    el = eligibility(F, cfg)
    cutoff = pd.Timestamp(meta["train_cutoff"])
    eval_start = cutoff + pd.Timedelta(hours=cfg["split"].get("embargo_hours", 0))
    dt = F["decision_time"]
    split = np.where(dt < cutoff, "train", np.where(dt >= eval_start, "eval", "embargo"))
    ok = (el["eligibility_status"] == "eligible").to_numpy()

    out = F[KEY + ["decision_time"]].copy()
    out["split"] = split
    out = pd.concat([out, el], axis=1)
    score_ = np.full(len(F), np.nan)
    raw = np.full(len(F), np.nan)
    if ok.any():
        X = F.loc[ok, feats].astype(float)
        score_[ok] = -pipe.decision_function(X)            # > 0  <=>  predicted anomaly
        raw[ok] = pipe.score_samples(X)
    out["anomaly_score"] = score_
    out["raw_score_samples"] = raw
    out["anomaly_decision"] = pd.array(np.where(ok, score_ > 0, False), dtype="boolean")
    out.loc[~ok, "anomaly_decision"] = pd.NA
    out["n_imputed"] = pd.array(np.where(ok, len(feats) - out["n_features_available"], 0), dtype="Int64")
    out.loc[~ok, "n_imputed"] = pd.NA
    out["model_version"] = meta["model_version"]
    cols = KEY + ["decision_time", "split", "series_position", "eligibility_status", "anomaly_decision",
                  "anomaly_score", "raw_score_samples", "n_model_features", "n_features_available",
                  "n_signal_features_available", "n_imputed", "missing_features", "model_version"]
    return out[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------------------- persist
def model_dir(cfg: dict) -> Path:
    return config.PROJECT_ROOT / cfg["output"]["model_dir"]


def save_model(pipe: Pipeline, meta: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, directory / "pipeline.joblib")
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_model(directory: Path) -> tuple[Pipeline, dict]:
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if meta["versions"]["sklearn"] != sklearn.__version__:
        print(f"warning: model saved with sklearn {meta['versions']['sklearn']}, running {sklearn.__version__}")
    return joblib.load(directory / "pipeline.joblib"), meta


# ------------------------------------------------------------------------------------------ report
def _table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _range_table(scores: pd.DataFrame, F: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    el = (scores["eligibility_status"] == "eligible").to_numpy()
    rows = []
    for split in ("train", "eval"):
        m = el & (scores["split"] == split).to_numpy()
        r = {"split": split, "max anomaly_score": float(scores.loc[m, "anomaly_score"].max())}
        r.update({f"p99 {f}": float(F.loc[m, f].quantile(0.99)) for f in feats})
        rows.append(r)
    return pd.DataFrame(rows).set_index("split").T.reset_index().rename(columns={"index": "statistic"})


def render_report(scores: pd.DataFrame, F: pd.DataFrame, meta: dict, cfg: dict) -> str:
    feats = meta["features"]
    counts = (scores.groupby(["split", "eligibility_status"]).size().unstack(fill_value=0)
              .reindex(columns=list(STATUS), fill_value=0).reset_index())
    el = scores[scores["eligibility_status"] == "eligible"]
    flag = (el.groupby("split")["anomaly_decision"].agg(["size", "sum"])
            .rename(columns={"size": "eligible rows", "sum": "flagged"}).reset_index())
    flag["flagged share"] = (flag["flagged"] / flag["eligible rows"]).astype(float)
    elig_rows = scores["eligibility_status"].eq("eligible").to_numpy()
    miss = pd.DataFrame({
        "feature": feats,
        "train NA share (eligible)": [float(F.loc[elig_rows & (scores["split"] == "train").to_numpy(), f]
                                            .isna().mean()) for f in feats],
        "eval NA share (eligible)": [float(F.loc[elig_rows & (scores["split"] == "eval").to_numpy(), f]
                                           .isna().mean()) for f in feats],
        "training median (imputation)": [meta["imputer_medians"][f] for f in feats],
    })
    imp = el["n_imputed"].astype(int).value_counts().sort_index().rename_axis("features imputed") \
        .reset_index(name="eligible rows")
    m = cfg["model"]
    return "\n".join([
        "# PREVENZA Isolation Forest Run (Phase 3B.2)",
        "",
        "> GENERATED by `python -m src.detection.isolation_forest`. Synthetic data. Descriptive run summary only:",
        "> **anomaly scores are relative isolation scores, not probabilities and not clinical risk.** No "
        "ground-truth labels were used for features, splitting or tuning; no performance metrics are reported "
        "here (evaluation is a later phase).",
        "",
        "## Model",
        "",
        f"- Pipeline: `SimpleImputer({cfg['preprocessing']['imputer_strategy']})` -> "
        f"`{cfg['preprocessing']['scaler'].title()}Scaler` -> `IsolationForest(n_estimators={m['n_estimators']}, "
        f"max_samples={m['max_samples']}, contamination={m['contamination']}, max_features={m['max_features']}, "
        f"random_state={m['random_state']})`; all steps fitted on training rows only.",
        f"- Model version (config + training-data fingerprint): `{meta['model_version']}`; sklearn "
        f"{meta['versions']['sklearn']}.",
        f"- Features ({len(feats)}): " + ", ".join(f"`{f}`" for f in feats) + ".",
        "- Isolation Forest is insensitive to monotone per-feature rescaling; the scaler is kept for a consistent, "
        "inspectable preprocessing step.",
        "- Because data-quality features (completeness, staleness) are inputs, a flag can reflect unusual data "
        "availability as well as unusual heat/health signals. Attribution is a later phase.",
        "",
        "## Chronological split",
        "",
        f"- Training: `decision_time < {meta['train_cutoff']}` (first {cfg['split']['train_fraction']:.0%} of "
        f"decision dates); evaluation: `decision_time >= cutoff + {cfg['split'].get('embargo_hours', 0)} h`.",
        f"- Model fitted on **{meta['train_rows']}** eligible training rows (last training decision "
        f"{meta['train_decision_time_max']}).",
        "",
        "## Eligibility (no forced predictions)",
        "",
        f"- `warm_up`: first {cfg['eligibility']['warmup_observations']} observations of a series (no baseline yet).",
        f"- `insufficient_features`: fewer than {cfg['eligibility']['min_signal_features']} of "
        f"{len(cfg['features']['signal'])} signal features known, or a required feature "
        f"({', '.join(cfg['eligibility']['required_features'])}) missing.",
        "- Ineligible rows have NA `anomaly_decision` / `anomaly_score`.",
        "",
        _table(counts),
        "",
        "## Flag counts (descriptive)",
        "",
        _table(flag),
        "",
        "Training flagged share is set by `contamination`; evaluation share reflects how evaluation data differs "
        "from the training window.",
        "",
        _table(_range_table(scores, F, feats)),
        "",
        "Score and feature ranges by split (eligible rows). If the training window contains more extreme "
        "conditions than the evaluation window, the contamination-based threshold can sit above every evaluation "
        "score; the continuous score still ranks evaluation rows. Changing the threshold after inspecting "
        "evaluation data would be tuning on evaluation data.",
        "",
        "## Feature availability and imputation",
        "",
        _table(miss),
        "",
        _table(imp),
        "",
    ])


# --------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="PREVENZA Isolation Forest (Phase 3B.2)")
    ap.add_argument("--use-saved", action="store_true", help="score with persisted pipeline; do not refit")
    args = ap.parse_args(argv)
    cfg = load_if_config()
    F = pd.read_parquet(config.path_for("processed") / FEATURES_FILE)
    validate_config(cfg, F.columns)
    if args.use_saved:
        pipe, meta = load_model(model_dir(cfg))
    else:
        pipe, meta = fit(F, cfg)
        save_model(pipe, meta, model_dir(cfg))
    scores = score(F, pipe, meta, cfg)
    out = config.path_for("processed") / cfg["output"]["scores_file"]
    scores.to_parquet(out, index=False)
    REPORT_PATH.write_text(render_report(scores, F, meta, cfg), encoding="utf-8")
    print(f"model {meta['model_version']}: trained on {meta['train_rows']} rows (cutoff {meta['train_cutoff']})")
    print(scores.groupby(["split", "eligibility_status"]).size().to_string())
    print(f"wrote {out.relative_to(config.PROJECT_ROOT)}, {model_dir(cfg).relative_to(config.PROJECT_ROOT)}/, "
          f"{REPORT_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
