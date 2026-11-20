"""Phase 3B.2: Isolation Forest."""
import copy

import numpy as np
import pandas as pd
import pytest

from src import config, schemas
from src.detection import isolation_forest as iso
from src.features.build import FEATURES_FILE

KEY = list(schemas.OBSERVATION_KEY)
CFG = iso.load_if_config()


@pytest.fixture(scope="module")
def F():
    return pd.read_parquet(config.path_for("processed") / FEATURES_FILE)


@pytest.fixture(scope="module")
def fitted(F):
    return iso.fit(F, CFG)


@pytest.fixture(scope="module")
def scores(F, fitted):
    pipe, meta = fitted
    return iso.score(F, pipe, meta, CFG)


def _cfg(**overrides):
    c = copy.deepcopy(CFG)
    for path, v in overrides.items():
        d = c
        *head, last = path.split(".")
        for h in head:
            d = d[h]
        d[last] = v
    return c


# --- configuration ---------------------------------------------------------------------------------
def test_config_rejects_label_features_and_random_split(F):
    with pytest.raises(iso.ConfigError, match="ground-truth"):
        iso.validate_config(_cfg(**{"features.data_quality": ["label"]}), list(F.columns) + ["label"])
    with pytest.raises(iso.ConfigError, match="chronological"):
        iso.validate_config(_cfg(**{"split.method": "random"}), F.columns)
    with pytest.raises(iso.ConfigError, match="not in features"):
        iso.validate_config(_cfg(**{"features.signal": ["nonexistent_rz"]}), F.columns)
    iso.validate_config(CFG, F.columns)


def test_model_features_are_configured_subset(F, fitted):
    _, meta = fitted
    assert meta["features"] == CFG["features"]["signal"] + CFG["features"]["data_quality"]
    assert not set(meta["features"]) & set(schemas.LABEL_COLUMNS)


# --- temporal split ----------------------------------------------------------------------------------
def test_split_is_chronological(F):
    cutoff, split = iso.chronological_split(F, CFG)
    tr, ev = F.loc[split == "train", "decision_time"], F.loc[split == "eval", "decision_time"]
    assert tr.max() < cutoff <= ev.min()
    assert set(split) == {"train", "eval"}
    dates = np.sort(F["decision_time"].dt.normalize().unique())
    assert (tr.dt.normalize().nunique() / len(dates)) == pytest.approx(CFG["split"]["train_fraction"], abs=0.02)


def test_split_does_not_depend_on_row_order(F):
    shuffled = F.sample(frac=1.0, random_state=0)
    _, s1 = iso.chronological_split(F, CFG)
    _, s2 = iso.chronological_split(shuffled, CFG)
    pd.testing.assert_series_equal(s1.sort_index(), s2.sort_index())


def test_embargo_excludes_gap(F):
    cutoff, split = iso.chronological_split(F, _cfg(**{"split.embargo_hours": 48}))
    emb = F.loc[split == "embargo", "decision_time"]
    assert len(emb) > 0 and emb.min() >= cutoff and emb.max() < cutoff + pd.Timedelta(hours=48)


def test_training_uses_only_training_rows(F, fitted, scores):
    pipe, meta = fitted
    F2 = F.copy()
    ev = F2["decision_time"] >= pd.Timestamp(meta["train_cutoff"])
    for c in meta["features"]:
        F2.loc[ev, c] = 1e6                                 # wildly perturb evaluation-period features
    pipe2, meta2 = iso.fit(F2, CFG)
    assert meta2["model_version"] == meta["model_version"]
    assert meta2["imputer_medians"] == meta["imputer_medians"]
    tr = (scores["split"] == "train") & (scores["eligibility_status"] == "eligible")
    s2 = iso.score(F2, pipe2, meta2, CFG)
    np.testing.assert_allclose(s2.loc[tr, "anomaly_score"], scores.loc[tr, "anomaly_score"])


def test_imputer_and_scaler_fit_on_training_only(F, fitted, scores):
    pipe, meta = fitted
    tr = ((scores["split"] == "train") & (scores["eligibility_status"] == "eligible")).to_numpy()
    X = F.loc[tr, meta["features"]].astype(float)
    np.testing.assert_allclose(pipe.named_steps["impute"].statistics_, X.median().to_numpy())
    assert meta["train_rows"] == int(tr.sum())
    assert pd.Timestamp(meta["train_decision_time_max"]) < pd.Timestamp(meta["train_cutoff"])


# --- eligibility / warm-up / missingness -----------------------------------------------------------
def test_warmup_rows_not_scored(F, scores):
    wu = scores["series_position"] < CFG["eligibility"]["warmup_observations"]
    assert (scores.loc[wu, "eligibility_status"] == "warm_up").all()
    assert wu.sum() == 18 * CFG["eligibility"]["warmup_observations"]
    assert scores.loc[wu, ["anomaly_score", "raw_score_samples"]].isna().all().all()
    assert scores.loc[wu, "anomaly_decision"].isna().all()


def test_ineligible_rows_have_no_forced_prediction(scores):
    inel = scores["eligibility_status"] != "eligible"
    assert scores.loc[inel, "anomaly_decision"].isna().all()
    assert scores.loc[inel, "anomaly_score"].isna().all()
    assert scores.loc[inel, "n_imputed"].isna().all()
    el = ~inel
    assert scores.loc[el, ["anomaly_decision", "anomaly_score", "n_imputed"]].notna().all().all()


def test_missing_features_imputed_or_ineligible(F, fitted):
    pipe, meta = fitted
    F2 = F.copy()
    el = iso.eligibility(F, CFG)
    rows = el.index[(el["eligibility_status"] == "eligible")][:6]
    a, b = rows[:3], rows[3:]
    F2.loc[a, ["absence_rate_rz", "clin_staleness_hours"]] = np.nan     # optional features -> imputed
    F2.loc[b, "heat_index_celsius_rz"] = np.nan                         # required -> insufficient
    s = iso.score(F2, pipe, meta, CFG)
    assert (s.loc[a, "eligibility_status"] == "eligible").all()
    assert (s.loc[a, "n_imputed"] == 2).all()
    assert s.loc[a, "missing_features"].str.contains("absence_rate_rz").all()
    assert s.loc[a, "anomaly_score"].notna().all()
    assert (s.loc[b, "eligibility_status"] == "insufficient_features").all()
    assert s.loc[b, "anomaly_score"].isna().all()
    # too few signal features -> insufficient
    F3 = F.copy()
    F3.loc[a, CFG["features"]["signal"][1:4]] = np.nan                  # 3 of 6 left (< 4)
    assert (iso.eligibility(F3, CFG).loc[a, "eligibility_status"] == "insufficient_features").all()


def test_availability_summary_consistent(F, scores):
    feats = CFG["features"]["signal"] + CFG["features"]["data_quality"]
    known = F[feats].notna().sum(axis=1).to_numpy()
    assert (scores["n_features_available"].to_numpy() == known).all()
    assert (scores["n_model_features"] == len(feats)).all()
    el = scores["eligibility_status"] == "eligible"
    assert (scores.loc[el, "n_imputed"] == len(feats) - scores.loc[el, "n_features_available"]).all()
    none_missing = scores["n_features_available"] == len(feats)
    assert (scores.loc[none_missing, "missing_features"] == "").all()


# --- decisions and scores ------------------------------------------------------------------------
def test_decision_matches_score_and_contamination(scores):
    el = scores[scores["eligibility_status"] == "eligible"]
    assert (el["anomaly_decision"] == (el["anomaly_score"] > 0)).all()
    tr = el[el["split"] == "train"]
    assert tr["anomaly_decision"].mean() == pytest.approx(CFG["model"]["contamination"], abs=0.02)


def test_scores_not_presented_as_probabilities(scores):
    assert not any("prob" in c or "risk" in c for c in scores.columns)
    assert (scores["anomaly_score"].dropna() < 0).any()                  # not bounded in [0, 1]


# --- reproducibility / persistence ---------------------------------------------------------------
def test_reproducible_with_seed(F, scores):
    pipe, meta = iso.fit(F, CFG)
    pd.testing.assert_frame_equal(iso.score(F, pipe, meta, CFG), scores)
    pipe2, meta2 = iso.fit(F, _cfg(**{"model.random_state": 7}))
    s2 = iso.score(F, pipe2, meta2, _cfg(**{"model.random_state": 7}))
    assert not np.allclose(s2["anomaly_score"].dropna(), scores["anomaly_score"].dropna())


def test_persistence_roundtrip(F, fitted, scores, tmp_path):
    pipe, meta = fitted
    iso.save_model(pipe, meta, tmp_path)
    pipe2, meta2 = iso.load_model(tmp_path)
    assert meta2["features"] == meta["features"] and meta2["model_version"] == meta["model_version"]
    pd.testing.assert_frame_equal(iso.score(F, pipe2, meta2, CFG), scores)


def test_saved_artifacts_match_current_fit(F, scores):
    d = iso.model_dir(CFG)
    out = config.path_for("processed") / CFG["output"]["scores_file"]
    if not (d / "pipeline.joblib").exists() or not out.exists():
        pytest.skip("run: python -m src.detection.isolation_forest")
    pipe, meta = iso.load_model(d)
    pd.testing.assert_frame_equal(iso.score(F, pipe, meta, CFG), scores)
    pd.testing.assert_frame_equal(pd.read_parquet(out), scores)


# --- leakage ----------------------------------------------------------------------------------------
def test_no_ground_truth_read(F, monkeypatch):
    import src.preprocessing.load as load
    def boom(*a, **k):
        raise AssertionError("ground truth / raw data must not be read by the detector")
    monkeypatch.setattr(load, "load_dataset", boom)
    monkeypatch.setattr(pd, "read_csv", boom)
    pipe, meta = iso.fit(F, CFG)
    iso.score(F, pipe, meta, CFG)


def test_scores_contain_no_label_columns(scores):
    assert not set(scores.columns) & set(schemas.LABEL_COLUMNS)
