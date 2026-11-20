# HealthSentinel Architecture

Status: **Phase 3B.3 (multi-signal risk fusion).** Components marked *planned* are not implemented yet.

## 1. Purpose and scope
An explainable, privacy-preserving early warning for **occupational heat-related health risk**. It combines
materially different signal types (environmental sensors, workforce exposure, aggregated clinic visits,
aggregated absenteeism, incident reports) per workplace zone and shift. It detects deviations from baseline
early enough to act on, and recommends a proportionate preventive action with an accountable role.

All data is **synthetic** (90 days, 3 sites x 3 zones, 2 shifts). Results describe the simulation, not
real-world performance.

## 2. Data flow
```
config/simulation.yaml
        |
        v
src/data_generation  ──►  data/raw/*.csv  (5 source feeds, recorded_at + reporting_timestamp)
        |                 data/ground_truth/*.csv  (events, data-quality scenarios; evaluation only)
        v
src/preprocessing    validate + coerce (src/schemas.py), full key grid,
                     availability flags, as-of filter (reporting_timestamp <= t)
        |                                     ──► data/processed/observations.parquet
        v
src/features         heat exceedance (WBGT vs work-intensity limits), rolling zone baselines,
                     rates per worker / per person-hour, cross-signal lags, coverage features
        v
src/detection        baseline-deviation scoring per signal + fused risk score + alert rules   (planned)
src/forecasting      1-3 day risk projection from heat trend + exposure                         (planned)
        v
src/explainability   per-alert contributions: which signal, zone, shift, time window             (planned)
        v
src/recommendations  risk tier + drivers ──► action, responsible role, evidence, assumptions      (planned)
        v
Streamlit app        role-based, small-cell-suppressed views                                     (planned)

src/evaluation       as-of back-test vs ground truth ──► lead time, sensitivity, PPV, false alarms,
                     data latency, behaviour under missing inputs                                (planned)
```

## 3. Common observation key and grid
- Key: **`observation_date + site_id + zone_id + shift`** (`schemas.OBSERVATION_KEY`).
- `zone_id` is globally unique (`S01-Z02`); `site_id` is kept in the key for grouping and access control.
- Night shift (18:00-06:00) belongs to the date it starts.
- Full grid = 90 x 9 zones x 2 shifts = **1,620 observations**. Every source dataset is delivered on
  this grid. A row absent from a source becomes a missing observation with an availability flag — never 0.
- All five sources (including incidents) are aggregated to the grid. A count of 0 is a reported zero; an
  unreported period has `reporting_status = not_reported` and NA values.

## 4. Time, latency and as-of evaluation
Each source row has `recorded_at` (occurrence: shift end) and `reporting_timestamp` (available to us; NA if never reported). Expected
latencies are set in `config/settings.yaml`: environment ~15 min, workforce ~1 h, clinic ~24 h,
absenteeism ~48 h, incidents variable.

Detection runs at a **decision time** (e.g. the start of each shift). It may use only rows with
`reporting_timestamp <= decision_time`. This keeps the back-test honest: slow signals (absence) cannot explain an
alert before they would really have arrived, and the system's measured lead time includes data latency.

## 5. Missing-data behaviour
| Situation | Representation | Expected system behaviour |
|---|---|---|
| Sensor fault (spike / stuck) | plausible-looking but wrong values (logged in `data_quality_log.csv`) | robust baselines; require cross-signal corroboration before alerting |
| Random missing field | NA in a reported row; derived fields NA | detection uses remaining signals; confidence lowered and shown |
| Partial coverage | `readings_received < readings_expected` | aggregates used if coverage >= `min_env_coverage_fraction`, else unavailable |
| Feed delayed | `reporting_timestamp` later than usual | as-of view lacks the rows; alert explains "clinic data not yet received" |
| Reporting gap | row kept, `not_reported`, values NA | treated as unavailable, never 0 |

Behaviour under each `data_quality_log.csv` issue type is reported in evaluation.

## 6. Detection, explainability and recommendations (planned design)
- **Detection:** transparent baseline-deviation scores per signal (e.g. rolling zone/shift baseline with
  robust z-scores, or Poisson excess for small counts), heat-exposure scores against work-intensity
  WBGT limits, combined into a fused zone-shift risk score. Thresholds live in config and are calibrated on an
  early training window, not on the evaluation window.
- **Explainability:** every alert stores its per-signal contributions, the zones and shifts involved, the
  time window, the data available at decision time, and the missing inputs.
- **Recommendations:** a rule table mapping risk tier + dominant drivers to a proportionate action
  (e.g. hydration and buddy checks -> adjusted work/rest regime -> rescheduling heavy tasks to cooler shift
  -> ventilation/engineering fix -> stop-work and regulatory notification). Each action names a
  responsible role (operations supervisor, site HSE manager, occupational health lead, regulator/public
  health) and lists its evidence and assumptions.

## 7. Privacy model
- **Tier A — analytic store:** synthetic zone-shift aggregates, no identifiers. Used by the pipeline only.
- **Tier B — role views:** small-cell suppression (counts 1..4 -> `<5`), role-based field visibility
  (`config/settings.yaml: privacy.roles`). Supervisors see risk tier and actions only.
- **Tier C — external/regulatory:** site-week aggregates, alert log and evaluation report.
- Guards: `schemas.FORBIDDEN_COLUMN_TOKENS` rejects identifying columns; no free-text fields; incidents
  carry only counts of workers involved. Every view carries the notice: *not for individual employment,
  disciplinary or performance decisions; not a clinical diagnosis.*

## 8. Evaluation definitions (fixed now so they cannot be tuned later)
Ground truth: `data/ground_truth/heat_risk_events.csv` (one row per event x zone) and per-observation labels in
`observation_labels.csv` (0 normal, 1 emerging, 2 escalating; derived from the schedule only).
- **Match:** an alert for zone *z* on date *d* matches an event row if `zone_id = z` and
  `onset_timestamp <= shift start < end_timestamp`. Negative-control spike events never match.
- **Event detected:** at least one matching alert before `escalation_timestamp` (or before `end_timestamp` for
  non-escalating events).
- **Sensitivity:** detected event-zones / all event-zones.
- **PPV:** matching alerts / all alerts.
- **False alarms:** non-matching alerts, reported per 100 zone-weeks.
- **Lead time:** `escalation_timestamp - first matching alert time` (hours/days); reported as median and range over
  escalating events. Not defined for non-escalating events.
- **Data latency:** distribution of `reporting_timestamp - recorded_at` per source, and which sources were
  available at each alert.
- **Missing-input behaviour:** sensitivity, PPV and false-alarm counts inside vs outside
  `data_quality_log.csv` windows.

No metric is reported unless `src/evaluation/` computed it on the current data.

## 9. Phases
1. Skeleton, contracts, data dictionary, structural tests. **(done)**
2. Synthetic generator for the 5 feeds + ground truth + data-quality log. **(done - `python -m src.data_generation.generate`)**
3. Preprocessing (grid, availability, as-of) and features. **(3A done - `python -m src.features.build`; decision time = shift end + 1 h; see docs/feature_dictionary.md)**
4. Detection + explainability + recommendations. **(3B.2 Isolation Forest done - `python -m src.detection.isolation_forest`; chronological split, train-only preprocessing, eligibility gating; see docs/isolation_forest_report.md. 3B.3 risk fusion done - `python -m src.detection.risk_fusion` (config/risk_fusion.yaml): robust-z + CUSUM + absolute levels -> five 0-100 category scores, available-weight renormalisation, staleness exclusion, persistence + independent-category rules, separate severity and evidence; see docs/risk_fusion_report.md. The robust z-scores/CUSUM in features.parquet serve as the statistical detector (no separate 3B.1 module).)**
5. Back-test and evaluation report.
6. Streamlit app with role-based, suppressed views.
