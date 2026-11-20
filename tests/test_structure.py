"""Phase 1 structural checks: layout, config, contracts, data dictionary."""
import numpy as np
import pandas as pd
import pytest

from src import config, schemas

ROOT = config.PROJECT_ROOT

REQUIRED_DIRS = [
    "config", "data/raw", "data/processed", "data/ground_truth", "docs", "tests",
    *[f"src/{m}" for m in ("data_generation", "preprocessing", "features", "detection", "forecasting",
                            "explainability", "recommendations", "evaluation")],
]
REQUIRED_FILES = ["README.md", "requirements.txt", "docs/architecture.md", "docs/data_dictionary.md",
                  "config/settings.yaml", "config/simulation.yaml"]


@pytest.mark.parametrize("d", REQUIRED_DIRS)
def test_dirs_exist(d):
    assert (ROOT / d).is_dir()


@pytest.mark.parametrize("f", REQUIRED_FILES)
def test_files_exist(f):
    assert (ROOT / f).is_file() and (ROOT / f).stat().st_size > 0


def test_src_packages_importable():
    for d in REQUIRED_DIRS:
        if d.startswith("src/"):
            assert (ROOT / d / "__init__.py").is_file()


# --- config -----------------------------------------------------------------------------
def test_simulation_design_is_1620_observations():
    sim = config.load_simulation()
    assert sim["n_days"] == 90
    assert len(sim["sites"]) == 3
    assert all(len(s["zones"]) == 3 for s in sim["sites"].values())
    assert tuple(sim["shifts"]) == schemas.SHIFTS
    assert config.expected_observation_count(sim) == 1620


def test_zone_ids_unique_and_prefixed_by_site():
    sim = config.load_simulation()
    zones = list(config.iter_zones(sim))
    assert len({z for _, z, _ in zones}) == len(zones)
    assert all(z.startswith(f"{s}-") for s, z, _ in zones)


def test_settings_key_matches_schema_key():
    assert tuple(config.load_settings()["observation_key"]["fields"]) == schemas.OBSERVATION_KEY


def test_latency_configured_for_every_source():
    assert set(config.load_settings()["expected_latency_hours"]) == set(schemas.SOURCE_CONTRACTS)


# --- contracts --------------------------------------------------------------------------
@pytest.mark.parametrize("c", schemas.SOURCE_CONTRACTS.values(), ids=lambda c: c.name)
def test_source_contracts_have_key_and_provenance(c):
    for col in (*schemas.OBSERVATION_KEY, "recorded_at", "reporting_timestamp", "reporting_status"):
        assert col in c.columns, f"{c.name} missing {col}"
    for col in (*schemas.OBSERVATION_KEY, "recorded_at", "reporting_status"):
        assert not c.field(col).nullable
    assert c.field("reporting_timestamp").nullable           # NA when never reported
    assert tuple(c.primary_key) == schemas.OBSERVATION_KEY


@pytest.mark.parametrize("c", schemas.ALL_CONTRACTS.values(), ids=lambda c: c.name)
def test_contract_integrity(c):
    assert len(c.columns) == len(set(c.columns)), "duplicate columns"
    assert set(c.primary_key) <= set(c.columns)
    for a, b in c.le_constraints:
        assert a in c.columns and b in c.columns
    for col in c.columns:
        assert not any(tok in col.lower() for tok in schemas.FORBIDDEN_COLUMN_TOKENS), col


def test_source_contracts_have_no_label_columns():
    for c in schemas.SOURCE_CONTRACTS.values():
        assert not set(c.columns) & set(schemas.LABEL_COLUMNS), c.name


def test_ground_truth_has_lifecycle_timestamps():
    gt = schemas.HEAT_RISK_EVENTS
    for col in ("onset_timestamp", "escalation_timestamp", "end_timestamp"):
        assert gt.field(col).dtype == "datetime"
    assert gt.field("escalation_timestamp").nullable         # non-escalating (control) events allowed
    assert not gt.field("onset_timestamp").nullable


# --- validator / coercion behaviour (on tiny hand-built frames, not model outputs) ----
def _env_rows():
    return pd.DataFrame({
        "observation_date": ["2025-06-01", "2025-06-01"], "site_id": ["S01", "S01"],
        "zone_id": ["S01-Z01", "S01-Z01"], "shift": ["day", "night"],
        "temperature_celsius": [33.0, None], "relative_humidity": [50.0, None], "heat_index_celsius": [37.0, None],
        "recorded_at": ["2025-06-01 18:00", "2025-06-02 06:00"],
        "reporting_timestamp": ["2025-06-01 18:15", None],
        "reporting_status": ["reported", "not_reported"],
    })


def test_valid_frame_passes():
    df = schemas.coerce(_env_rows(), schemas.ENVIRONMENTAL)
    assert schemas.validate(df, schemas.ENVIRONMENTAL) == []


def test_missing_stays_missing_not_zero():
    df = schemas.coerce(_env_rows(), schemas.ENVIRONMENTAL)
    assert np.isnan(df.loc[1, "heat_index_celsius"])
    clin = schemas.coerce(pd.DataFrame({"heat_related_visits": [3, None]}), schemas.CLINICAL)
    assert clin["heat_related_visits"].dtype == "Int64"
    assert pd.isna(clin.loc[1, "heat_related_visits"])


def test_validator_catches_violations():
    df = _env_rows()
    df.loc[0, "shift"] = "evening"
    df.loc[0, "reporting_timestamp"] = "2025-06-01 12:00"     # before recorded_at
    df.loc[1, "site_id"] = None                               # non-nullable
    df["employee_id"] = ["a", "b"]                            # forbidden
    issues = " | ".join(schemas.validate(schemas.coerce(df, schemas.ENVIRONMENTAL), schemas.ENVIRONMENTAL))
    for expected in ("shift", "recorded_at <= reporting_timestamp", "site_id", "forbidden", "unexpected"):
        assert expected in issues


def test_validator_catches_duplicate_key():
    df = pd.concat([_env_rows().iloc[[0]]] * 2)
    assert any("duplicate" in i for i in schemas.validate(df, schemas.ENVIRONMENTAL))


def test_ground_truth_order_enforced():
    row = dict(event_id="EVT-001", site_id="S03", zone_id="S03-Z01", event_type="gradual_heatwave_escalation",
               onset_timestamp="2025-07-10 06:00", escalation_timestamp="2025-07-08 06:00",
               end_timestamp="2025-07-15 06:00", severity="high", description="x")
    df = schemas.coerce(pd.DataFrame([row]), schemas.HEAT_RISK_EVENTS)
    assert any("onset_timestamp <= escalation_timestamp" in i for i in schemas.validate(df, schemas.HEAT_RISK_EVENTS))
    df.loc[0, "escalation_timestamp"] = pd.NaT                # non-escalating event is valid
    assert schemas.validate(df, schemas.HEAT_RISK_EVENTS) == []


# --- docs -------------------------------------------------------------------------------
def test_data_dictionary_in_sync_with_schemas():
    on_disk = schemas.DICTIONARY_PATH.read_text(encoding="utf-8")
    assert on_disk == schemas.render_data_dictionary(), "run: python -m src.schemas"

