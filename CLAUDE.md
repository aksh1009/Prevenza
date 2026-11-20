# CLAUDE.md — HealthSentinel

## Objective
HealthSentinel is an explainable, privacy-preserving **population and workforce health early-warning system**.
First use case: **occupational heat-related health risk** across workplace zones. It fuses environmental,
workforce-exposure, aggregated clinical, aggregated absenteeism and incident signals to detect rising risk
early, explain which signals/zones/periods drove an alert, and recommend a proportionate preventive action
with an accountable role. Built for a hackathon (<24 h, solo), so prefer simple, transparent methods that
work end-to-end over sophisticated ones that don't.

## Architecture (see docs/architecture.md)
```
data_generation -> raw CSV -> preprocessing -> processed Parquet (full key grid + availability flags)
  -> features -> detection (+ forecasting) -> explainability -> recommendations -> Streamlit app
                                     \-> evaluation (back-test vs data/ground_truth)
```

## Module responsibilities
| Module | Responsibility | Must NOT |
|---|---|---|
| `src/schemas.py` | Data contracts, `coerce`, `validate`, data-dictionary rendering | contain business logic |
| `src/config.py` | Load `config/*.yaml`; paths; expected observation count | hold hardcoded parameters |
| `src/data_generation/` | Synthetic, aggregated source feeds + ground truth from `config/simulation.yaml` (seeded) | write person-level rows |
| `src/preprocessing/` | Validate, coerce, keep full `observation_date+site_id+zone_id+shift` grid, availability flags, as-of filtering on `reporting_timestamp` | fill missing with 0 |
| `src/features/` | Per-key features: WBGT exceedance, rolling baselines, rates per worker, data-coverage features | use future data (leakage) |
| `src/detection/` | Anomaly/risk scoring vs baseline, alert rules with configurable thresholds | read ground truth |
| `src/forecasting/` | Short-horizon (1-3 day) risk projection | read ground truth |
| `src/explainability/` | Per-alert contributions by signal, zone and time window | invent attributions |
| `src/recommendations/` | Map risk level + drivers to proportionate action, responsible role, evidence, assumptions | make individual-level recommendations |
| `src/evaluation/` | Back-test: lead time, sensitivity, PPV, false alarms, latency, missing-input behaviour | tune on the test window silently |

## Coding conventions
- Python 3.11, type hints, small pure functions; pandas DataFrames in, DataFrames out.
- Every function that loads source data validates against its contract (`schemas.validate`).
- Common observation key is `OBSERVATION_KEY` from `src/schemas.py` — never re-type it.
- Parameters (thresholds, windows, seeds, k for suppression) live in `config/`, not in code.
- Raw = CSV, processed = Parquet, ground truth = CSV. Never overwrite raw data in later stages.
- Run modules as `python -m src.<module>` from the project root.
- When a contract changes, regenerate the dictionary: `python -m src.schemas`.

## Data and missingness rules
- **Missing is never zero.** Use `Int64`/`float64`/`boolean` with `<NA>`/`NaN`. Carry explicit availability
  flags (e.g. from `reporting_status`, or derived `clinical_available`) instead of imputing silently. Any imputation must be
  explicit, flagged and documented.
- Preserve `recorded_at` (occurrence) and `reporting_timestamp` (availability; NA if never reported).
  Back-tests evaluate decisions **as-of** a decision time, using only rows with
  `reporting_timestamp <= decision_time`.
- Ground truth (`data/ground_truth/`) is used **only** by `src/evaluation/` and the data generator.

## Data generation
- Regenerate all synthetic data and reports: `python -m src.data_generation.generate [--seed N]`.
  Scenario dates/locations, data-quality injection and model parameters live in `config/simulation.yaml`.
- Build unified table + as-of features: `python -m src.features.build` (settings in `config/features.yaml`).
  Features for an observation use a source value only if `reporting_timestamp <= decision_time`
  (shift end + `decision_lag_hours`). `unified_observations.parquet` is retrospective - never a model input.
- Isolation Forest: `python -m src.detection.isolation_forest [--use-saved]` (settings in
  `config/isolation_forest.yaml`; model persisted in `models/isolation_forest/`). Chronological split only;
  preprocessing fitted on training rows only; warm-up/insufficient rows get NA, never a forced score.
  `anomaly_score` is a relative isolation score - never call it a probability or clinical risk.
- Risk fusion: `python -m src.detection.risk_fusion` (config `config/risk_fusion.yaml`). Missing/stale/no-baseline
  categories are excluded and listed, never scored 0; environmental evidence alone never yields a health class;
  Isolation Forest contributes only on out-of-sample (eval) rows. Pipeline order: generate -> features.build ->
  isolation_forest -> risk_fusion -> evaluation.backtest (replays every decision time; config/backtest.yaml).
- Forecasting: `python -m src.forecasting.forecast` (config/forecast.yaml). Targets come from future labels;
  inputs only from decision-time features/fusion. Scores are uncalibrated - never call them probabilities.
- Heat index beyond the NWS chart range (T > 43.3 C or HI > 58.3 C) is flagged `heat_index_validity = extrapolated`;
  never clip it, never relabel it WBGT, and always surface the flag wherever the value is shown.
- Audit: `python -m src.evaluation.audit` (read-only; never refits models). EVT-005 spans validation and test -
  report unseen-event results separately from all-test results.
- Backtest metrics are SYNTHETIC validation only. Never tune thresholds on ground-truth events; never quote a
  metric not present in docs/backtesting_report.md.
- Generated docs (`docs/simulation_summary.md`, `docs/isolation_forest_report.md`, `docs/risk_fusion_report.md`, `docs/data_quality_report.md`, `docs/data_dictionary.md`,
  `docs/feature_dictionary.md`) must not be edited by hand.
- Environment: `.venv` (Python 3.11). Activate with `.venv\Scriptsctivate` (Windows).

## Privacy constraints
- All health/workforce data is **synthetic and aggregated** to zone-shift (or coarser). No employee IDs,
  names, demographics, free text or individual diagnoses — `schemas.FORBIDDEN_COLUMN_TOKENS` guards this.
- Role-based views (`config/settings.yaml: privacy.roles`); small-cell suppression: counts 1..4 shown as `<5`
  in any user-facing view or export. Operations supervisors see risk levels and actions, not health counts.
- Outputs must state: **not for individual employment, disciplinary or performance decisions; not a
  clinical diagnosis.** Recommendations target zones, shifts and work conditions, never named people.

## No mock ML outputs — honesty rules
- Never hardcode risk scores, alerts, explanations, or metrics. Every number shown must be computed from
  data by code in this repo.
- Never invent or quote performance metrics (sensitivity, PPV, lead time, etc.) that weren't produced by
  `src/evaluation/` on the current data. If something wasn't measured, say "not measured".
- Results are from **simulated** data; label them as such and do not claim real-world effectiveness.
- If a model is a placeholder or rule-based baseline, name it that way.

## Testing
- `pytest -q` from the project root. Structural tests in `tests/` cover folder layout, contracts, config
  and the generated data dictionary.
- Each new module gets tests for: contract conformance of outputs, missing-input behaviour (NaN stays NaN,
  no crash), no leakage of ground truth, and determinism under the configured seed.
