# PREVENZA Feature Dictionary

> GENERATED from `src/features/registry.py` + `config/features.yaml` by `python -m src.features.build`. Do not edit by hand.

## Files

- `data/processed/unified_observations.parquet` - five sources joined one-to-one on `observation_date + site_id + zone_id + shift`; every value as eventually reported (NA preserved), with per-source `*_recorded_at`, `*_reporting_timestamp`, `*_reporting_status`, `*_reporting_delay_hours`. Prefixes: `env_`, `wf_`, `clin_`, `abs_`, `inc_`. **Retrospective record - not a model input.**
- `data/processed/features.parquet` - one row per observation; identifier columns `observation_date, site_id, zone_id, shift, decision_time` + 244 as-of features.

## Leakage rules

- **Decision time** = shift end + 1.0 h. Each row's features use only information available at that time.
- A source value from observation *s* is used for observation *t* only if *s <= t* in the same site-zone-shift series **and** `reporting_timestamp(s) <= decision_time(t)`. Never-reported values are never used.
- Series are independent per `site_id + zone_id + shift`; one observation = one day of that shift.
- No ground-truth labels, event IDs, onset/escalation/end timestamps, or data-quality log entries are read. Calendar position (date/day index) is deliberately not a feature.
- Consequence: slow sources (clinical ~12-48 h, absenteeism ~36-60 h) are usually **not** known for the current observation; their `lag*`, `last`, `roll*` and availability features carry the delayed signal.

## Statistic definitions (applied to every signal)

Windows: short = 3 obs (min 1), long = 7 obs (min 3); baseline = 28 obs ending 7 obs before t (min 7); trend = 7 obs (min 3).

| Suffix | Definition |
|---|---|
| `_cur` | Value of the current observation, if reported by decision time (else NA). |
| `_lag1` | Value of the previous observation in the series, as known at decision time. |
| `_lag2` | Value two observations back, as known at decision time. |
| `_last` | Most recent value known at decision time (any earlier observation of the series). |
| `_roll3` | Mean of known values over the last 3 observations incl. current (>= 1 known values, else NA). |
| `_roll7` | Mean of known values over the last 7 observations incl. current (>= 3 known values, else NA). |
| `_hist_median` | Median of known values over observations t-34..t-7 (baseline excludes the most recent 7; >= 7 values, else NA). |
| `_hist_mad` | Median absolute deviation over the same baseline window. |
| `_dev` | last - hist_median (most recent known value vs baseline; age given by <source>_staleness_hours). |
| `_rz` | Robust z: (last - hist_median) / max(1.4826 x MAD, scale_floor). |
| `_roll3_rz` | Robust z of the 3-observation mean vs the baseline. |
| `_slope7` | OLS slope per observation over known values in the last 7 observations (>= 3 points, else NA). |
| `_cusum` | One-sided upper CUSUM over known values of the last 7 observations, z vs current baseline capped at 3.0, k = 0.5: S = max(0, S + min(z, cap) - k). NA without baseline or known values. |

Robust-scale floors (`scale_floor`): `temperature_celsius` 0.5, `relative_humidity` 2.0, `heat_index_celsius` 0.5, `heat_index_excess` 1.0, `workers_present` 1.0, `average_exposure_hours` 0.25, `total_exposure_hours` 5.0, `heat_exposure_load` 10.0, `heat_related_visits` 0.5, `heat_visits_mod_severe` 0.5, `heat_visits_per_100` 1.0, `absence_rate` 0.01, `absent_workers` 0.5, `heat_incident_count` 0.5, `incidents_mod_severe` 0.5, `incidents_per_100` 0.5.

## Signals

| Signal | Sources | Unit | Definition |
|---|---|---|---|
| `temperature_celsius` | environmental_data | degC | Peak shift air temperature. |
| `relative_humidity` | environmental_data | % | Mean shift relative humidity. |
| `heat_index_celsius` | environmental_data | degC | NWS heat index (not WBGT). |
| `heat_index_excess` | environmental_data | degC | max(0, heat index - heat_index_threshold_c). |
| `workers_present` | workforce_data | count | Workers present (workforce_data.workforce_count). |
| `average_exposure_hours` | workforce_data | hours | Mean hours per present worker in the zone. |
| `total_exposure_hours` | workforce_data | person-hours | Total person-hours in the zone. |
| `heat_exposure_load` | environmental_data + workforce_data | degC x hours | heat_index_excess x average_exposure_hours (needs both sources). |
| `heat_related_visits` | clinical_data | count | Heat-related clinic visits. |
| `heat_visits_mod_severe` | clinical_data | count | Moderate + severe heat-related visits. |
| `heat_visits_per_100` | clinical_data + workforce_data | per 100 present workers | 100 x heat_related_visits / workers present (needs both sources). |
| `absence_rate` | absenteeism_data | fraction | Unplanned absence rate among rostered workers. |
| `absent_workers` | absenteeism_data | count | Rostered workers absent (unplanned). |
| `heat_incident_count` | incident_data | count | Heat-related incidents (all severities). |
| `incidents_mod_severe` | incident_data | count | Moderate + severe heat-related incidents. |
| `incidents_per_100` | incident_data + workforce_data | per 100 present workers | 100 x heat_incident_count / workers present (needs both sources). |

Signal features: 16 signals x 13 statistics = 208 columns, named `<signal>_<suffix>`.

## Availability, missingness and context features

| Feature | Group | Sources | Unit | Meaning |
|---|---|---|---|---|
| `env_available` | availability | environmental_data | bool | Current observation of environmental_data reported by decision time. |
| `env_window7_availability` | availability | environmental_data | fraction | Share of the last 7 observations of environmental_data known at decision time. |
| `env_staleness_hours` | availability | environmental_data | hours | decision_time - recorded_at of the most recent environmental_data observation known at decision time (NA if none yet). |
| `env_last_delay_hours` | availability | environmental_data | hours | Reporting delay of that most recent known environmental_data observation. |
| `wf_available` | availability | workforce_data | bool | Current observation of workforce_data reported by decision time. |
| `wf_window7_availability` | availability | workforce_data | fraction | Share of the last 7 observations of workforce_data known at decision time. |
| `wf_staleness_hours` | availability | workforce_data | hours | decision_time - recorded_at of the most recent workforce_data observation known at decision time (NA if none yet). |
| `wf_last_delay_hours` | availability | workforce_data | hours | Reporting delay of that most recent known workforce_data observation. |
| `clin_available` | availability | clinical_data | bool | Current observation of clinical_data reported by decision time. |
| `clin_window7_availability` | availability | clinical_data | fraction | Share of the last 7 observations of clinical_data known at decision time. |
| `clin_staleness_hours` | availability | clinical_data | hours | decision_time - recorded_at of the most recent clinical_data observation known at decision time (NA if none yet). |
| `clin_last_delay_hours` | availability | clinical_data | hours | Reporting delay of that most recent known clinical_data observation. |
| `abs_available` | availability | absenteeism_data | bool | Current observation of absenteeism_data reported by decision time. |
| `abs_window7_availability` | availability | absenteeism_data | fraction | Share of the last 7 observations of absenteeism_data known at decision time. |
| `abs_staleness_hours` | availability | absenteeism_data | hours | decision_time - recorded_at of the most recent absenteeism_data observation known at decision time (NA if none yet). |
| `abs_last_delay_hours` | availability | absenteeism_data | hours | Reporting delay of that most recent known absenteeism_data observation. |
| `inc_available` | availability | incident_data | bool | Current observation of incident_data reported by decision time. |
| `inc_window7_availability` | availability | incident_data | fraction | Share of the last 7 observations of incident_data known at decision time. |
| `inc_staleness_hours` | availability | incident_data | hours | decision_time - recorded_at of the most recent incident_data observation known at decision time (NA if none yet). |
| `inc_last_delay_hours` | availability | incident_data | hours | Reporting delay of that most recent known incident_data observation. |
| `n_sources_available` | availability | all | count 0-5 | Number of sources whose current observation is reported by decision time. |
| `completeness_score` | availability | all | fraction | Share of the 8 core fields known for the current observation at decision time (env_temperature_celsius, env_relative_humidity, wf_workforce_count, wf_average_exposure_hours, clin_heat_related_visits, abs_workforce_count, abs_absent_workers, inc_heat_incident_count). |
| `completeness_window7` | availability | all | fraction | Mean of the five source window availabilities. |
| `env_temperature_celsius_missing` | missingness | environmental_data | bool | env_temperature_celsius unknown for the current observation at decision time (not yet reported, not reported, or missing in report). |
| `env_relative_humidity_missing` | missingness | environmental_data | bool | env_relative_humidity unknown for the current observation at decision time (not yet reported, not reported, or missing in report). |
| `wf_workforce_count_missing` | missingness | workforce_data | bool | wf_workforce_count unknown for the current observation at decision time (not yet reported, not reported, or missing in report). |
| `wf_average_exposure_hours_missing` | missingness | workforce_data | bool | wf_average_exposure_hours unknown for the current observation at decision time (not yet reported, not reported, or missing in report). |
| `clin_heat_related_visits_missing` | missingness | clinical_data | bool | clin_heat_related_visits unknown for the current observation at decision time (not yet reported, not reported, or missing in report). |
| `abs_workforce_count_missing` | missingness | absenteeism_data | bool | abs_workforce_count unknown for the current observation at decision time (not yet reported, not reported, or missing in report). |
| `abs_absent_workers_missing` | missingness | absenteeism_data | bool | abs_absent_workers unknown for the current observation at decision time (not yet reported, not reported, or missing in report). |
| `inc_heat_incident_count_missing` | missingness | incident_data | bool | inc_heat_incident_count unknown for the current observation at decision time (not yet reported, not reported, or missing in report). |
| `ctx_outdoor` | context | zone registry | bool | Zone is outdoor or semi-outdoor. |
| `ctx_indoor_conditioned` | context | zone registry | bool | Zone is air-conditioned. |
| `ctx_radiant_heat` | context | zone registry | bool | Zone has a radiant heat source (e.g. furnaces). |
| `ctx_work_intensity` | context | zone registry | ordinal 1-3 | Typical work intensity: 1 light, 2 moderate, 3 heavy. |
| `ctx_night_shift` | context | key | bool | Observation is a night shift. |

## Feature count by group

| Group | Columns |
|---|---|
| signal | 208 |
| availability | 23 |
| missingness | 8 |
| context | 5 |
| **total** | **244** |
