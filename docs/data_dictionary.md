# PREVENZA (HealthSentinel) Data Dictionary

> GENERATED from `src/schemas.py` by `python -m src.schemas`. Do not edit by hand.

All data is **synthetic and aggregated**. No dataset contains individual identifiers, names, free text,
or clinical diagnoses of individuals. Timestamps are site-local and timezone-naive.

## Conventions

- **Common observation key:** `observation_date + site_id + zone_id + shift`. 90 days x 3 sites x 3 zones x 2 shifts = 1,620 observations.
- **Night shift** (18:00-06:00) belongs to the date on which it starts.
- **Provenance:** every source row has `recorded_at` (occurrence: end of the shift window), `reporting_timestamp` (when it became available; NA if never reported) and `reporting_status`. Models use only rows with `reporting_timestamp <= decision time`.
- **Labels are separate.** Ground-truth files are never joined into model inputs.
- **Missing is never zero.** Logical `int` -> pandas `Int64` (`<NA>`), `float` -> `float64` (`NaN`), `bool` -> `boolean`. Unreported periods are kept as grid rows with `reporting_status = not_reported` and NA values, never 0.
- **Nullable = no** means the field must always be present when the row exists.
- **Constraints** `a <= b` are checked only where both values are present.

## Source datasets (model inputs)

### `environmental_data`

Zone thermal environment per shift from zone sensors. Heat index is the US NWS heat-index approximation (Rothfusz regression with Steadman/NWS adjustments) computed from the reported temperature and humidity. It is NOT WBGT (no radiant-heat or wind component).

- **File:** `data/raw/environmental_data.csv`
- **Source:** Zone environmental sensors via sensor platform (synthetic)
- **Grain:** One row per observation key (full 1,620-row grid is preserved; unreported periods are rows with NA values).
- **Primary key:** `observation_date, site_id, zone_id, shift`
- **Expected latency:** minutes (5-30 min)
- **Constraints:** `recorded_at <= reporting_timestamp`

| Field | Type | Unit | Nullable | Valid values / range | Meaning |
|---|---|---|---|---|---|
| `observation_date` | date | - | no |  | Site-local calendar date on which the shift STARTS (night shift 18:00 D -> 06:00 D+1 belongs to D). |
| `site_id` | string | - | no |  | Synthetic site identifier, e.g. 'S01'. |
| `zone_id` | string | - | no |  | Synthetic zone identifier, globally unique and prefixed by site, e.g. 'S01-Z02'. |
| `shift` | category | - | no | `day`, `night` | Work shift. day = 06:00-18:00, night = 18:00-06:00. |
| `temperature_celsius` | float | degC | yes | -10 .. 55 | Peak hourly dry-bulb air temperature in the shift. |
| `relative_humidity` | float | % | yes | 0 .. 100 | Mean relative humidity over the shift. |
| `heat_index_celsius` | float | degC | yes | -10 .. 80 | NWS heat index from temperature_celsius and relative_humidity. NA whenever either input is NA. |
| `recorded_at` | datetime | site-local datetime | no |  | When the observed window closed / measurement was taken at source (end of the shift window). This is OCCURRENCE time. |
| `reporting_timestamp` | datetime | site-local datetime | yes |  | When the record was reported to / became available in PREVENZA. NA when the period was never reported. Models may only use rows with reporting_timestamp <= decision time. |
| `reporting_status` | category | - | no | `reported`, `not_reported` | 'reported' = row delivered (individual fields may still be NA); 'not_reported' = whole period missing, all measurement fields NA (never 0). |

### `workforce_data`

Aggregated workforce exposure per zone-shift.

- **File:** `data/raw/workforce_data.csv`
- **Source:** Rostering / time-and-attendance system, aggregated before export (no person rows; synthetic)
- **Grain:** One row per observation key (full 1,620-row grid is preserved; unreported periods are rows with NA values).
- **Primary key:** `observation_date, site_id, zone_id, shift`
- **Expected latency:** ~0.5-3 hours
- **Constraints:** `recorded_at <= reporting_timestamp`

| Field | Type | Unit | Nullable | Valid values / range | Meaning |
|---|---|---|---|---|---|
| `observation_date` | date | - | no |  | Site-local calendar date on which the shift STARTS (night shift 18:00 D -> 06:00 D+1 belongs to D). |
| `site_id` | string | - | no |  | Synthetic site identifier, e.g. 'S01'. |
| `zone_id` | string | - | no |  | Synthetic zone identifier, globally unique and prefixed by site, e.g. 'S01-Z02'. |
| `shift` | category | - | no | `day`, `night` | Work shift. day = 06:00-18:00, night = 18:00-06:00. |
| `workforce_count` | int | count | yes | 0 ..  | Workers PRESENT and working in the zone during the shift (exposed workers). Equals absenteeism_data.workforce_count - absent_workers. |
| `average_exposure_hours` | float | hours | yes | 0 .. 12 | Mean hours per present worker spent working in the zone's thermal environment. |
| `total_exposure_hours` | float | person-hours | yes | 0 ..  | workforce_count x average_exposure_hours. |
| `recorded_at` | datetime | site-local datetime | no |  | When the observed window closed / measurement was taken at source (end of the shift window). This is OCCURRENCE time. |
| `reporting_timestamp` | datetime | site-local datetime | yes |  | When the record was reported to / became available in PREVENZA. NA when the period was never reported. Models may only use rows with reporting_timestamp <= decision time. |
| `reporting_status` | category | - | no | `reported`, `not_reported` | 'reported' = row delivered (individual fields may still be NA); 'not_reported' = whole period missing, all measurement fields NA (never 0). |

### `clinical_data`

Aggregated on-site occupational-health / first-aid visits coded to a heat-related syndrome (syndromic category, not an individual diagnosis), with severity distribution.

- **File:** `data/raw/clinical_data.csv`
- **Source:** On-site occupational health clinic log, aggregated before export (synthetic)
- **Grain:** One row per observation key (full 1,620-row grid is preserved; unreported periods are rows with NA values).
- **Primary key:** `observation_date, site_id, zone_id, shift`
- **Expected latency:** ~12-48 hours (coded after the shift)
- **Constraints:** `heat_visits_severe <= heat_related_visits`; `recorded_at <= reporting_timestamp`

| Field | Type | Unit | Nullable | Valid values / range | Meaning |
|---|---|---|---|---|---|
| `observation_date` | date | - | no |  | Site-local calendar date on which the shift STARTS (night shift 18:00 D -> 06:00 D+1 belongs to D). |
| `site_id` | string | - | no |  | Synthetic site identifier, e.g. 'S01'. |
| `zone_id` | string | - | no |  | Synthetic zone identifier, globally unique and prefixed by site, e.g. 'S01-Z02'. |
| `shift` | category | - | no | `day`, `night` | Work shift. day = 06:00-18:00, night = 18:00-06:00. |
| `heat_related_visits` | int | count | yes | 0 ..  | Heat-related visits for workers of this zone-shift. = heat_visits_mild + heat_visits_moderate + heat_visits_severe. |
| `heat_visits_mild` | int | count | yes | 0 ..  | Mild (e.g. heat rash, cramps; returned to work). |
| `heat_visits_moderate` | int | count | yes | 0 ..  | Moderate (e.g. heat exhaustion; removed from work). |
| `heat_visits_severe` | int | count | yes | 0 ..  | Severe (e.g. suspected heat stroke; external transfer). |
| `recorded_at` | datetime | site-local datetime | no |  | When the observed window closed / measurement was taken at source (end of the shift window). This is OCCURRENCE time. |
| `reporting_timestamp` | datetime | site-local datetime | yes |  | When the record was reported to / became available in PREVENZA. NA when the period was never reported. Models may only use rows with reporting_timestamp <= decision time. |
| `reporting_status` | category | - | no | `reported`, `not_reported` | 'reported' = row delivered (individual fields may still be NA); 'not_reported' = whole period missing, all measurement fields NA (never 0). |

### `absenteeism_data`

Aggregated unplanned (sickness) absence among workers rostered to the zone-shift.

- **File:** `data/raw/absenteeism_data.csv`
- **Source:** HR / time-and-attendance absence records, aggregated before export (synthetic)
- **Grain:** One row per observation key (full 1,620-row grid is preserved; unreported periods are rows with NA values).
- **Primary key:** `observation_date, site_id, zone_id, shift`
- **Expected latency:** ~1.5-2.5 days (batch)
- **Constraints:** `absent_workers <= workforce_count`; `recorded_at <= reporting_timestamp`

| Field | Type | Unit | Nullable | Valid values / range | Meaning |
|---|---|---|---|---|---|
| `observation_date` | date | - | no |  | Site-local calendar date on which the shift STARTS (night shift 18:00 D -> 06:00 D+1 belongs to D). |
| `site_id` | string | - | no |  | Synthetic site identifier, e.g. 'S01'. |
| `zone_id` | string | - | no |  | Synthetic zone identifier, globally unique and prefixed by site, e.g. 'S01-Z02'. |
| `shift` | category | - | no | `day`, `night` | Work shift. day = 06:00-18:00, night = 18:00-06:00. |
| `workforce_count` | int | count | yes | 0 ..  | Workers ROSTERED to the zone-shift (absence denominator). |
| `absent_workers` | int | count | yes | 0 ..  | Rostered workers with an unplanned (sickness) absence for the shift. |
| `absence_rate` | float | fraction 0-1 | yes | 0 .. 1 | absent_workers / workforce_count (NA if either is NA or workforce_count = 0). |
| `lost_workdays` | float | workdays | yes | 0 ..  | Workdays lost to the absences (1 absent shift = 1 workday; equals absent_workers in this simulation). |
| `recorded_at` | datetime | site-local datetime | no |  | When the observed window closed / measurement was taken at source (end of the shift window). This is OCCURRENCE time. |
| `reporting_timestamp` | datetime | site-local datetime | yes |  | When the record was reported to / became available in PREVENZA. NA when the period was never reported. Models may only use rows with reporting_timestamp <= decision time. |
| `reporting_status` | category | - | no | `reported`, `not_reported` | 'reported' = row delivered (individual fields may still be NA); 'not_reported' = whole period missing, all measurement fields NA (never 0). |

### `incident_data`

Aggregated heat-related workplace incidents (reporter flagged heat/hydration as a factor) by severity. Counts only - no persons, no free text.

- **File:** `data/raw/incident_data.csv`
- **Source:** EHS incident management system, aggregated before export (synthetic)
- **Grain:** One row per observation key (full 1,620-row grid is preserved; unreported periods are rows with NA values).
- **Primary key:** `observation_date, site_id, zone_id, shift`
- **Expected latency:** variable, ~1-96 hours (median ~18 h)
- **Constraints:** `severe_incidents <= heat_incident_count`; `recorded_at <= reporting_timestamp`

| Field | Type | Unit | Nullable | Valid values / range | Meaning |
|---|---|---|---|---|---|
| `observation_date` | date | - | no |  | Site-local calendar date on which the shift STARTS (night shift 18:00 D -> 06:00 D+1 belongs to D). |
| `site_id` | string | - | no |  | Synthetic site identifier, e.g. 'S01'. |
| `zone_id` | string | - | no |  | Synthetic zone identifier, globally unique and prefixed by site, e.g. 'S01-Z02'. |
| `shift` | category | - | no | `day`, `night` | Work shift. day = 06:00-18:00, night = 18:00-06:00. |
| `heat_incident_count` | int | count | yes | 0 ..  | = minor_incidents + moderate_incidents + severe_incidents. |
| `minor_incidents` | int | count | yes | 0 ..  | First-aid / near-miss level. |
| `moderate_incidents` | int | count | yes | 0 ..  | Medical treatment / restricted work. |
| `severe_incidents` | int | count | yes | 0 ..  | Lost time / regulator-reportable. |
| `recorded_at` | datetime | site-local datetime | no |  | When the observed window closed / measurement was taken at source (end of the shift window). This is OCCURRENCE time. |
| `reporting_timestamp` | datetime | site-local datetime | yes |  | When the record was reported to / became available in PREVENZA. NA when the period was never reported. Models may only use rows with reporting_timestamp <= decision time. |
| `reporting_status` | category | - | no | `reported`, `not_reported` | 'reported' = row delivered (individual fields may still be NA); 'not_reported' = whole period missing, all measurement fields NA (never 0). |

## Ground truth (evaluation only - never a model input)

### `heat_risk_events`

Scheduled (injected) heat-health events. Used ONLY for evaluation.

- **File:** `data/ground_truth/heat_risk_events.csv`
- **Source:** Synthetic generator schedule (config/simulation.yaml -> scenarios)
- **Grain:** One row per event x affected zone (event_id repeats for multi-zone events).
- **Primary key:** `event_id, zone_id`
- **Expected latency:** n/a (evaluation only)
- **Constraints:** `onset_timestamp <= escalation_timestamp`; `escalation_timestamp <= end_timestamp`; `onset_timestamp <= end_timestamp`

| Field | Type | Unit | Nullable | Valid values / range | Meaning |
|---|---|---|---|---|---|
| `event_id` | string | - | no |  | Event identifier, e.g. 'EVT-001'. |
| `site_id` | string | - | no |  | Affected site. |
| `zone_id` | string | - | no |  | Affected zone. |
| `event_type` | category | - | no | `gradual_heatwave_escalation`, `isolated_temperature_spike`, `heat_escalation_clinical_gap` | Scenario mechanism. isolated_temperature_spike is a NEGATIVE CONTROL (no sustained health deterioration; its observations are labelled 0). |
| `onset_timestamp` | datetime | site-local datetime | no |  | Start of the first shift in which heat forcing departs from baseline. |
| `escalation_timestamp` | datetime | site-local datetime | yes |  | Start of the first shift of the scheduled escalation phase (forcing at full strength; health outcomes rising materially). NA for non-escalating events. |
| `end_timestamp` | datetime | site-local datetime | no |  | End of the last affected shift (exclusive bound). |
| `severity` | category | - | no | `none`, `low`, `moderate`, `high` | Designed severity. |
| `description` | string | - | no |  | Generator description of the scenario. |

### `observation_labels`

Per-observation evaluation label derived ONLY from the event schedule (never model output). Rule: shift start in [onset, escalation) -> 1; in [escalation, end) -> 2; otherwise 0. Negative-control events give 0. Overlaps take the maximum.

- **File:** `data/ground_truth/observation_labels.csv`
- **Source:** Derived from heat_risk_events.csv
- **Grain:** One row per observation key (1,620 rows).
- **Primary key:** `observation_date, site_id, zone_id, shift`
- **Expected latency:** n/a (evaluation only)

| Field | Type | Unit | Nullable | Valid values / range | Meaning |
|---|---|---|---|---|---|
| `observation_date` | date | - | no |  | Site-local calendar date on which the shift STARTS (night shift 18:00 D -> 06:00 D+1 belongs to D). |
| `site_id` | string | - | no |  | Synthetic site identifier, e.g. 'S01'. |
| `zone_id` | string | - | no |  | Synthetic zone identifier, globally unique and prefixed by site, e.g. 'S01-Z02'. |
| `shift` | category | - | no | `day`, `night` | Work shift. day = 06:00-18:00, night = 18:00-06:00. |
| `label` | int | - | no | 0 .. 2 | 0 normal, 1 emerging, 2 escalating. |
| `label_name` | category | - | no | `normal`, `emerging`, `escalating` | Text form of label. |
| `event_id` | string | - | yes |  | Event that set the label (NA when label = 0 and no event applies). |

### `data_quality_log`

Structured data-quality issues injected by the generator (reporting gaps, feed delays, sensor anomalies). Random cell-level missingness is summarised in docs/data_quality_report.md instead.

- **File:** `data/ground_truth/data_quality_log.csv`
- **Source:** Synthetic generator
- **Grain:** One row per injected issue x zone.
- **Primary key:** `issue_id, zone_id`
- **Expected latency:** n/a (evaluation only)
- **Constraints:** `start_timestamp <= end_timestamp`

| Field | Type | Unit | Nullable | Valid values / range | Meaning |
|---|---|---|---|---|---|
| `issue_id` | string | - | no |  | Issue identifier, e.g. 'DQ-001'. |
| `source_dataset` | category | - | no | `environmental_data`, `workforce_data`, `clinical_data`, `absenteeism_data`, `incident_data` | Affected dataset. |
| `issue_type` | category | - | no | `reporting_gap`, `feed_delay`, `sensor_spike`, `stuck_sensor` | reporting_gap: rows not_reported; feed_delay: reporting_timestamp later than usual; sensor_spike: implausible single-shift jump; stuck_sensor: value repeated. |
| `site_id` | string | - | no |  | Affected site. |
| `zone_id` | string | - | no |  | Affected zone. |
| `start_timestamp` | datetime | site-local datetime | no |  | Start of first affected shift. |
| `end_timestamp` | datetime | site-local datetime | no |  | End of last affected shift. |
| `details` | string | - | no |  | Parameters of the injected issue. |
