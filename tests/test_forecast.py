"""Phase 4B: 24-hour forecasting."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import config, schemas
from src.forecasting import forecast as fc

KEY = list(schemas.OBSERVATION_KEY)
CFG = fc.load_forecast_config()
PROC = config.path_for("processed")


@pytest.fixture(scope="module")
def run():
    return fc.run(CFG, write=False)


@pytest.fixture(scope="module")
def inputs():
    F = pd.read_parquet(PROC / "features.parquet")
    A = pd.read_parquet(PROC / "risk_assessments.parquet")
    from src.preprocessing.load import load_dataset
    return F, A, load_dataset("observation_labels")


def labels_frame(pattern: list[int], zone="S01-Z01"):
    rows = []
    for i, lab in enumerate(pattern):
        d = pd.Timestamp("2025-07-01") + pd.Timedelta(days=i // 2)
        rows.append({"observation_date": d, "site_id": "S01", "zone_id": zone,
                     "shift": "day" if i % 2 == 0 else "night", "label": lab,
                     "event_id": "E1" if lab else pd.NA})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- targets
def test_target_looks_only_at_next_24h():
    T = fc.build_targets(labels_frame([0, 0, 0, 1, 2, 0, 0, 0, 0, 0]), CFG)
    # shift i looks at i+1 and i+2 (12 h steps, both shift types)
    assert T["target"].tolist()[:8] == [0, 1, 1, 1, 0, 0, 0, 0]
    assert T["target"].iloc[-2:].isna().all()                  # horizon passes end of data
    assert T.loc[T["target"] == 1, "target_event_id"].eq("E1").all()


def test_target_ignores_current_label():
    T = fc.build_targets(labels_frame([1, 0, 0, 0, 0]), CFG)
    assert T["target"].iloc[0] == 0                           # current positive, future negative


def test_targets_per_zone(inputs):
    _, _, L = inputs
    T = fc.build_targets(L, CFG)
    assert len(T) == len(L) and not T.duplicated(KEY).any()
    assert T["target"].isna().sum() == 9 * 2                   # last 2 observations per zone lack a full horizon


# ---------------------------------------------------------------------------------------------- inputs
def test_inputs_contain_no_labels_or_targets(inputs):
    F, A, L = inputs
    X = fc.build_frame(F, A, CFG)
    bad = set(schemas.LABEL_COLUMNS) | {"target", "target_event_id"}
    assert not bad & set(X.columns)
    assert not bad & set(fc.feature_names(CFG))


def test_inputs_independent_of_labels(inputs, run):
    F, A, L = inputs
    D = run[0]
    L2 = L.copy()
    L2["label"] = L2["label"].sample(frac=1, random_state=0).to_numpy()
    X1 = fc.build_frame(F, A, CFG)
    pd.testing.assert_frame_equal(X1, fc.build_frame(F, A, CFG))
    assert not fc.build_targets(L2, CFG)["target"].equals(fc.build_targets(L, CFG)["target"])


def test_eligibility_and_missing_indicators(run):
    D = run[0]
    assert not D.loc[D["series_position"] < CFG["eligibility"]["warmup_observations"], "eligible"].any()
    assert (D.loc[~D["eligible"], "ineligibility_reason"] != "").all()
    assert D.loc[~D["eligible"], "lr_score"].isna().all()
    assert D.loc[D["eligible"], "lr_score"].notna().all()
    feats = fc.feature_names(CFG)
    assert (D["n_inputs_missing"] == D[feats].isna().sum(axis=1)).all()
    assert D.loc[D["eligible"], "n_inputs_missing"].max() > 0  # missing inputs are scored via imputation


def test_persistence_uses_current_classification_only(run):
    D = run[0]
    assert (D["persistence_health"] == D["classification"].isin(CFG["persistence"]["health_classes"])).all()
    assert (D["persistence_any"] == D["classification"].isin(CFG["persistence"]["any_classes"])).all()


# ----------------------------------------------------------------------------------------------- splits
def test_chronological_splits_with_purge(run):
    D = run[0]
    h = pd.Timedelta(hours=CFG["target"]["horizon_hours"])
    tr, va, te = (D[D["split"] == s] for s in ("train", "validation", "test"))
    assert tr["shift_start"].max() < va["shift_start"].min() <= va["shift_start"].max() < te["shift_start"].min()
    assert (tr["shift_start"] + h < va["shift_start"].min()).all()   # no target window reaches validation
    assert (va["shift_start"] + h < te["shift_start"].min()).all()
    assert (D["split"] == "purged").sum() > 0


def test_split_independent_of_row_order(run):
    D = run[0]
    s1 = fc.assign_splits(D, CFG)
    sh = D.sample(frac=1, random_state=3)
    pd.testing.assert_series_equal(fc.assign_splits(sh, CFG).sort_index(), s1.sort_index())


# ------------------------------------------------------------------------------------- training / leakage
def test_test_data_never_influences_model(run):
    D, final, meta = run[0], run[1], run[2]
    D2 = D.copy()
    te = D2["split"] == "test"
    D2.loc[te, "target"] = 1 - D2.loc[te, "target"].fillna(0)     # flip test targets
    for c in fc.feature_names(CFG):
        D2.loc[te, c] = 1e6                                         # wreck test features
    final2, _, meta2 = fc.fit(D2, CFG)
    assert meta2["selected_C"] == meta["selected_C"] and meta2["threshold"] == meta["threshold"]
    assert meta2["model_version"] == meta["model_version"]
    ok = D["eligible"] & ~te
    np.testing.assert_allclose(fc.scores(D[ok], final2, CFG), fc.scores(D[ok], final, CFG))


def test_threshold_selected_on_validation_only(run):
    D, _, meta = run[0], run[1], run[2]
    va = (D["split"] == "validation") & D["eligible"] & D["target"].notna()
    thr = fc.best_f1_threshold(D.loc[va, "target"].astype(int), D.loc[va, "lr_score_selection"])
    assert thr == pytest.approx(meta["threshold"])


def test_class_imbalance_handled(run):
    final = run[1]
    assert final.named_steps["model"].class_weight == "balanced"
    assert final.named_steps["impute"].add_indicator


# -------------------------------------------------------------------------------------------- metrics
def test_metrics_match_sklearn(run):
    D, _, meta, overall = run[0], run[1], run[2], run[3]
    t = D[(D["split"] == "test") & D["eligible"] & D["target"].notna()]
    row = overall[(overall["split"] == "test") & (overall["model"] == "logistic regression")].iloc[0]
    assert row["pr_auc"] == pytest.approx(average_precision_score(t["target"].astype(int), t["lr_score"]))
    flag = t["lr_score"] >= meta["threshold"]
    y = t["target"].astype(int).astype(bool)
    assert row["tp"] == int((flag & y).sum()) and row["fp"] == int((flag & ~y).sum())
    assert set(overall["split"]) == {"train", "validation", "test"}
    lr_train = overall[(overall["split"] == "train") & (overall["model"] == "logistic regression")]
    assert lr_train["sample"].str.contains("in-sample").all()


def test_validation_reported_out_of_sample(run):
    overall = run[3]
    v = overall[(overall["split"] == "validation") & (overall["model"] == "logistic regression")]
    assert v["sample"].str.contains("out-of-sample").all()


# --------------------------------------------------------------------------------------------- outputs
def test_forecast_table(run):
    fct = run[4]
    assert not ({"target", "target_event_id", "label"} & set(fct.columns))
    assert not any("prob" in c for c in fct.columns)
    assert fct.loc[~fct["eligible"], ["forecast_score"]].isna().all().all()
    assert fct.loc[~fct["eligible"], "forecast_flag"].isna().all()
    el = fct[fct["eligible"]]
    assert (el["forecast_flag"] == (el["forecast_score"] >= el["threshold"].iloc[0] - 5e-5)).mean() > 0.99
    assert el["limitations"].str.contains("uncalibrated").all()
    assert fct.groupby(["site_id", "zone_id", "shift"])["is_latest"].sum().eq(1).all()
    assert (fct["horizon_end"] > fct["decision_time"]).all()
    assert fct.loc[fct["split"] == "train", "in_sample"].all() and not fct.loc[fct["split"] == "test", "in_sample"].any()


def test_reproducible_and_persisted(run, tmp_path):
    D, final, meta = run[0], run[1], run[2]
    D2, final2, meta2, *_ = fc.run(CFG, write=False)
    np.testing.assert_array_equal(D2["lr_score"].to_numpy(), D["lr_score"].to_numpy())
    cfg = copy.deepcopy(CFG)
    cfg["output"]["model_dir"] = str(tmp_path)
    fc.save_model(final, meta, cfg)
    p, m = fc.load_model(cfg)
    np.testing.assert_array_equal(fc.scores(D, p, CFG), D["lr_score"].to_numpy())
    assert m["threshold"] == meta["threshold"] and m["features"] == fc.feature_names(CFG)


def test_saved_outputs_match(run):
    D, final, meta = run[0], run[1], run[2]
    path = PROC / CFG["output"]["forecasts_file"]
    if not path.exists():
        pytest.skip("run: python -m src.forecasting.forecast")
    saved = pd.read_parquet(path)
    pd.testing.assert_frame_equal(saved, run[4])
    p, m = fc.load_model(CFG)
    assert m["model_version"] == meta["model_version"]
