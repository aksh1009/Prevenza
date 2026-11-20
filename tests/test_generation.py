"""Phase 2: synthetic generator checks."""
import numpy as np
import pandas as pd
import pytest

from src import config, schemas
from src.data_generation.generate import generate, read_dataset, validate_all
from src.data_generation.grid import build_grid, shift_start_hours
from src.data_generation.heat_index import heat_index_c
from src.data_generation.scenarios import ESCALATING_TYPES, build_labels

KEY = list(schemas.OBSERVATION_KEY)
SIM = config.load_simulation()
N_OBS = config.expected_observation_count(SIM)


@pytest.fixture(scope="module")
def data():
    return generate(SIM)


def _value_cols(name):
    return [c for c in schemas.SOURCE_CONTRACTS[name].columns
            if c not in KEY + ["recorded_at", "reporting_timestamp", "reporting_status"]]


# --- dimensions, keys, ids, dates --------------------------------------------------------------
@pytest.mark.parametrize("name", [*schemas.SOURCE_DATASETS, "observation_labels"])
def test_dimensions_and_unique_keys(data, name):
    df = data[name]
    assert len(df) == N_OBS == 1620
    assert not df.duplicated(KEY).any()
    assert list(df.columns) == schemas.ALL_CONTRACTS[name].columns


def test_all_sources_share_identical_key_grid(data):
    ref = data["environmental_data"][KEY]
    for name in [*schemas.SOURCE_DATASETS[1:], "observation_labels"]:
        pd.testing.assert_frame_equal(data[name][KEY], ref)


def test_date_range(data):
    d = data["environmental_data"]["observation_date"]
    assert d.min() == pd.Timestamp(SIM["start_date"])
    assert d.nunique() == SIM["n_days"] == 90
    assert d.max() == pd.Timestamp(SIM["start_date"]) + pd.Timedelta(days=89)


def test_valid_ids(data):
    valid = {(s, z) for s, z, _ in config.iter_zones(SIM)}
    for name in [*schemas.SOURCE_DATASETS, "observation_labels", "heat_risk_events", "data_quality_log"]:
        pairs = set(map(tuple, data[name][["site_id", "zone_id"]].astype(str).to_numpy()))
        assert pairs <= valid, name


def test_all_datasets_conform_to_contracts(data):
    # covers non-negative counts, ranges, allowed values, nullability, a <= b constraints
    assert validate_all(data) == {}


# --- plausibility and internal consistency --------------------------------------------------------
def test_plausible_environment(data):
    e = data["environmental_data"]
    assert e["temperature_celsius"].dropna().between(10, 55).all()
    assert e["relative_humidity"].dropna().between(0, 100).all()
    # heat index recomputes from reported inputs, and is NA exactly when an input is NA
    hi = np.round(heat_index_c(e["temperature_celsius"], e["relative_humidity"]), 1)
    np.testing.assert_allclose(e["heat_index_celsius"], hi, atol=0.051, equal_nan=True)
    inputs_na = e["temperature_celsius"].isna() | e["relative_humidity"].isna()
    assert (e["heat_index_celsius"].isna() == inputs_na).all()


def test_heat_index_reference_values():
    # NWS chart: 90 F / 50 % RH -> ~95 F; 80 F / 40 % -> ~80 F (simple formula region)
    assert abs(heat_index_c(32.22, 50) - (95 - 32) * 5 / 9) < 0.6
    assert abs(heat_index_c(26.67, 40) - 26.7) < 0.6
    assert np.isnan(heat_index_c(np.nan, 50))


def test_workforce_consistency(data):
    w = data["workforce_data"]
    both = w["workforce_count"].notna() & w["average_exposure_hours"].notna()
    np.testing.assert_allclose(w.loc[both, "total_exposure_hours"],
                               (w.loc[both, "workforce_count"] * w.loc[both, "average_exposure_hours"]).astype(float),
                               atol=0.051)
    assert w.loc[~both, "total_exposure_hours"].isna().all()
    assert w["average_exposure_hours"].dropna().between(0, 12).all()


def test_absence_consistency(data):
    a = data["absenteeism_data"]
    ok = a["absent_workers"].notna() & a["workforce_count"].notna()
    assert (a.loc[ok, "absent_workers"] <= a.loc[ok, "workforce_count"]).all()
    np.testing.assert_allclose(a.loc[ok, "absence_rate"],
                               (a.loc[ok, "absent_workers"] / a.loc[ok, "workforce_count"]).astype(float), atol=1e-4)
    assert a.loc[~ok, "absence_rate"].isna().all()
    pd.testing.assert_series_equal(a["lost_workdays"], a["absent_workers"].astype(float), check_names=False)


def test_present_plus_absent_equals_rostered(data):
    w, a = data["workforce_data"], data["absenteeism_data"]
    m = w[KEY + ["workforce_count"]].merge(a[KEY + ["workforce_count", "absent_workers"]], on=KEY,
                                           suffixes=("_present", "_rostered")).dropna()
    assert len(m) > 1400
    assert (m["workforce_count_present"] + m["absent_workers"] == m["workforce_count_rostered"]).all()


def test_incident_and_clinical_severity_reconcile(data):
    i = data["incident_data"]
    s = i["minor_incidents"] + i["moderate_incidents"] + i["severe_incidents"]
    ok = i["heat_incident_count"].notna()
    assert (s[ok] == i.loc[ok, "heat_incident_count"]).all()
    assert s[~ok].isna().all()
    c = data["clinical_data"]
    s = c["heat_visits_mild"] + c["heat_visits_moderate"] + c["heat_visits_severe"]
    ok = c["heat_related_visits"].notna()
    assert (s[ok] == c.loc[ok, "heat_related_visits"]).all()


# --- missingness ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", schemas.SOURCE_DATASETS)
def test_missing_values_present_and_never_zero_filled(data, name):
    df = data[name]
    vals = df[_value_cols(name)]
    assert vals.isna().any().any(), f"{name}: expected some missing values"
    nr = df["reporting_status"] == "not_reported"
    assert vals[nr].isna().all().all(), "unreported rows must be NA, not 0"
    assert df.loc[nr, "reporting_timestamp"].isna().all()
    assert df.loc[~nr, "reporting_timestamp"].notna().all()


def test_scenario_d_clinical_gap_while_other_sources_report(data):
    sc = next(s for s in SIM["scenarios"] if s.get("clinical_gap"))
    g = sc["clinical_gap"]
    def in_gap(df):
        d = df["observation_date"]
        return df["zone_id"].isin(sc["zones"]) & (d >= g["start_date"]) & (d <= g["end_date"])
    c = data["clinical_data"]
    assert (c.loc[in_gap(c), "reporting_status"] == "not_reported").all()
    for name in ("environmental_data", "workforce_data", "absenteeism_data", "incident_data"):
        df = data[name]
        assert (df.loc[in_gap(df), "reporting_status"] == "reported").mean() > 0.9, name
    # environmental risk rises during the gap compared with the 14 days before onset
    e = data["environmental_data"]
    zone = e["zone_id"].isin(sc["zones"]) & (e["shift"] == "day")
    pre = zone & (e["observation_date"] < sc["onset_date"]) & \
        (e["observation_date"] >= pd.Timestamp(sc["onset_date"]) - pd.Timedelta(days=14))
    assert e.loc[zone & in_gap(e), "heat_index_celsius"].mean() > e.loc[pre, "heat_index_celsius"].mean() + 5


# --- timestamps -------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", schemas.SOURCE_DATASETS)
def test_reporting_timestamps(data, name):
    df = data[name]
    starts = shift_start_hours()
    expected_end = df["observation_date"] + pd.to_timedelta(df["shift"].map(starts).astype(int) + 12, unit="h")
    assert (df["recorded_at"] == expected_end).all(), "recorded_at must be the shift end (occurrence)"
    delay = (df["reporting_timestamp"] - df["recorded_at"]).dt.total_seconds() / 3600
    spec = SIM["data_quality"]["reporting_delay_hours"][name]
    extra = max([fd["extra_hours"] for fd in SIM["data_quality"]["feed_delays"] if fd["source"] == name], default=0)
    assert delay.dropna().min() >= spec["low"] - 1 / 60
    assert delay.dropna().max() <= spec["high"] + extra + 1 / 60


def test_source_latency_ordering(data):
    med = {n: (data[n]["reporting_timestamp"] - data[n]["recorded_at"]).median() for n in schemas.SOURCE_DATASETS}
    assert med["environmental_data"] < med["workforce_data"] < med["clinical_data"] < med["absenteeism_data"]


# --- ground truth / labels -----------------------------------------------------------------------
def test_events_match_config(data):
    ev = data["heat_risk_events"]
    assert set(ev["event_id"]) == {s["id"] for s in SIM["scenarios"]}
    assert len(ev) == sum(len(s["zones"]) for s in SIM["scenarios"])
    assert set(ev["event_type"]) == set(schemas.GT_EVENT_TYPES)       # scenarios B, C, D all present


def test_labels_derived_from_schedule(data):
    grid = build_grid(SIM)
    relabel = build_labels(grid, data["heat_risk_events"])
    m = data["observation_labels"].merge(relabel, on=KEY, suffixes=("", "_re"))
    assert (m["label"] == m["label_re"]).all()
    lab = data["observation_labels"]
    assert set(lab["label"].unique()) == {0, 1, 2}
    assert (lab["label_name"] == lab["label"].map(schemas.LABELS)).all()
    assert lab.loc[lab["label"] == 0, "event_id"].isna().all()


def test_label_event_consistency(data):
    ev, lab = data["heat_risk_events"], data["observation_labels"]
    shift_start = lab["observation_date"] + pd.to_timedelta(lab["shift"].map(shift_start_hours()).astype(int), unit="h")
    for e in ev.itertuples():
        z = lab["zone_id"] == e.zone_id
        inside = z & (shift_start >= e.onset_timestamp) & (shift_start < e.end_timestamp)
        if e.event_type in ESCALATING_TYPES:
            assert set(lab.loc[inside, "label"]) == {1, 2}
            assert (lab.loc[z & (shift_start < e.escalation_timestamp) & inside, "label"] == 1).all()
            assert (lab.loc[z & (shift_start >= e.escalation_timestamp) & inside, "label"] == 2).all()
        else:
            assert (lab.loc[inside, "label"] == 0).all(), "negative-control spike must be labelled 0"
    # every non-zero label lies inside an escalating event
    nz = lab["label"] > 0
    assert lab.loc[nz, "event_id"].isin(ev.loc[ev["event_type"].isin(ESCALATING_TYPES), "event_id"]).all()


def test_escalation_signal_exceeds_reference(data):
    """Scenario mechanics: outcomes during escalation exceed the same zones' pre-event reference."""
    lab, c = data["observation_labels"], data["clinical_data"]
    a = data["absenteeism_data"]
    for e in data["heat_risk_events"].query("event_type == 'gradual_heatwave_escalation'").itertuples():
        z = lab["zone_id"] == e.zone_id
        esc = z & (lab["label"] == 2)
        ref = z & (lab["observation_date"] < e.onset_timestamp.normalize()) & \
            (lab["observation_date"] >= e.onset_timestamp.normalize() - pd.Timedelta(days=14))
        assert a.loc[esc, "absence_rate"].mean() > a.loc[ref, "absence_rate"].mean()
        assert c.loc[esc, "heat_related_visits"].mean() > c.loc[ref, "heat_related_visits"].mean()


def test_quality_log_covers_configured_issues(data):
    log = data["data_quality_log"]
    dq = SIM["data_quality"]
    assert (log["issue_type"] == "sensor_spike").sum() == dq["anomalies"]["sensor_spikes"]
    assert (log["issue_type"] == "stuck_sensor").sum() == dq["anomalies"]["stuck_sensors"]
    assert {"reporting_gap", "feed_delay"} <= set(log["issue_type"])


# --- leakage ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", schemas.SOURCE_DATASETS)
def test_no_label_columns_in_sources(data, name):
    assert not set(data[name].columns) & set(schemas.LABEL_COLUMNS)


# --- reproducibility --------------------------------------------------------------------------------
def test_seed_reproducibility(data):
    again = generate(SIM, seed=SIM["random_seed"])
    for name in data:
        pd.testing.assert_frame_equal(data[name], again[name])
    other = generate(SIM, seed=SIM["random_seed"] + 1)
    assert not data["environmental_data"].equals(other["environmental_data"])
    pd.testing.assert_frame_equal(data["heat_risk_events"], other["heat_risk_events"])  # schedule is config-driven


def test_files_on_disk_match_generator(data):
    for name in schemas.ALL_CONTRACTS:
        path = config.path_for(schemas.ALL_CONTRACTS[name].layer) / schemas.ALL_CONTRACTS[name].file_name
        if not path.exists():
            pytest.skip("run: python -m src.data_generation.generate")
        pd.testing.assert_frame_equal(read_dataset(name), data[name], check_exact=False, atol=1e-6)
