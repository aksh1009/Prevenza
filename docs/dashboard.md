# Prevenza Dashboard (Phase 5)

`streamlit run app.py` — a **read-only** view over saved pipeline outputs. Launching the dashboard never regenerates
data, refits models, recomputes risk fusion, or writes files (tested).

## Visual design — clinical-industrial light workspace

- **Frame:** obsidian-navy navigation rail (248 px: brand lockup, 6-page nav, dataset snapshot, reload/data files);
  alabaster canvas (#F8FAFC) with white hairline-bordered cards (#E2E8F0, 8 px radius, no shadows); sticky amber
  governance bar carrying the synthetic-demonstrator disclaimer; page header with metadata pills and a monospace
  "As of" badge; a horizontal **global filter ribbon** (Site, Zone, Shift, Observation period, Risk classification,
  Reset) — locked with an explicit "Evaluation-only view · Global filters disabled" badge on Model evaluation.
  Streamlit chrome (toolbar, menu, footer, decoration) is hidden.
- **Type:** IBM Plex Sans (UI/headings, 600 max) + IBM Plex Mono (timestamps, IDs, numbers); tabular numerals.
- **Tokens:** `src/dashboard/theme.py` — every semantic state (6 risk classes + missing, stale, invalid,
  forecast) has an ink / fill / border / series colour and a glyph; evidence streams use a fixed muted palette.
- **Components** (`components.py`): KPI cards with eyebrow + monochrome icon + optional semantic left border; status
  badges (glyph + text); segmented evidence bars; reliability bars; callouts (method / limitation / invalid);
  dashed empty states; humanised table headers (`humanize`) and Yes/No text instead of checkboxes.
- **Charts** (`charts.py`, Plotly template `prevenza`, mode bar hidden): thin-ring donut; 100 % stacked site bars;
  zone × day matrix (pale → amber → crimson, pale = no score); day/night classification tile strip; **five
  synchronised category small multiples** (no spaghetti); elevated-stream highlighting around alert episodes;
  sorted contribution bars ("+23.0 pts · score 73"); step-line availability; horizontal latency/staleness boxes;
  category-status matrix; monochrome confusion matrices with row %; lead-time **dumbbell** plot.
- **Heat index:** the rainbow gauge was replaced by a thermal readout card with a linear NWS bullet bar
  (caution 27 · extreme caution 32 · danger 39 · extreme danger 51 °C) and an "Outside NWS chart range" badge.
- **Deliberate deviations:** the forecast threshold line shows the saved model's fixed threshold (0.666), not a
  designed value; navigation uses a styled sidebar radio (keeps tests/deep links); row navigation uses an
  "Investigate zone" selector because Streamlit tables cannot be made clickable without checkboxes; no PR curve is
  shown because detection models have no saved curve data.

## Architecture

```
app.py                          sidebar navigation, global filters, permanent disclaimer, error handling
src/dashboard/
  theme.py                      colours, class symbols, fixed wording (disclaimer, forecast interpretation)
  data_loader.py                dataset registry, mtime-keyed cache, operational vs evaluation guard
  filters.py                    pure filter / latest-selection / display-text helpers (unit-tested)
  components.py                 CSS, KPI cards, badges, notes, cautions, downloads
  charts.py                     Plotly builders (gaps for missing data, no zero-filling, no gauges)
  pages/{overview, site_monitoring, alert_investigation, forecasting, evaluation, data_quality}.py
.streamlit/config.toml          light theme, headless server
```

- **Caching:** files are cached by path + modification time, so regenerated outputs appear automatically; the
  sidebar "Reload saved outputs" button clears the cache explicitly.
- **Missing files:** pages show a friendly error naming the command that produces the file.
- **Deep links:** `?page=overview|monitoring|alerts|forecasting|evaluation|data-quality`.

## Operational vs evaluation data

| Dataset | Kind | Used on |
|---|---|---|
| unified_observations, features, risk_assessments, risk_explanations, isolation_forest_scores, forecast_results | operational | operational pages |
| alert_episodes | operational **after stripping** `match_status`, `matched_event_id`, `overlap_hours` | Alert investigation |
| backtest_results, backtest_metrics, event_detection_results, alert_episodes (full), forecast_evaluation, model_comparison_summary, heat_risk_events | evaluation | Model evaluation only |
| data_quality_log (simulation truth of injected faults) | evaluation | Data quality page, in a clearly labelled "simulation reference" expander with event identifiers removed |

`load_operational()` raises `LeakageError` for evaluation datasets or any label/event column. Tests also render every
operational page and assert that no event identifier or label column appears.

## Page notes

- **Executive overview:** "Assessment as of" slider replays any historical simulated decision time (default: latest,
  2025-08-30 07:00). Nothing is labelled live or real-time. Selecting a table row sets the focus zone for other pages.
- **Site & zone monitoring:** charts use `connectgaps=False`, so unavailable data are gaps; an availability strip shows
  reported / field missing / not reported per source. Raw heat-index values outside the NWS chart range are marked ✕.
- **Alert investigation:** risk-fusion (model C) episodes from the full backtest window. Shows evidence at first
  detection, category contributions, saved explanation text, and a prominent caution whenever
  `alert_relies_on_extrapolated_heat_index` is true. Suggested actions are labelled illustrative, not medical.
- **24-hour forecasting:** shows the exact interpretation statement, ranking score as a decimal (never a percentage,
  never a gauge), fixed threshold, eligibility, imputed-input count, missing/stale sources, latest marker, and a
  side-by-side table of forecast flag vs persistence (any warning) vs persistence (health alert) vs current state.
  The any-warning persistence flag is re-derived from `current_classification` using `config/forecast.yaml`
  `persistence.any_classes` — identical to Phase 4B's saved values (tested).
- **Model evaluation:** detection metrics from `backtest_metrics.parquet`; event results from
  `event_detection_results.parquet`; false alerts per site via `src.evaluation.metrics.false_alerts_by_site` (the same
  function used in Phase 4A). Forecast metrics from `model_comparison_summary.parquet` (validation, all-test,
  excluding-EVT-005, unseen-event, events). **Documented additions:** training-period (in-sample) metrics and test
  confusion counts are computed from the saved `forecast_evaluation.parquet` flags with
  `src.forecasting.forecast.binary_metrics` — the Phase 4B definition (tests confirm they match Phase 4B).
- **Data quality & reliability:** separates valid, missing, stale, invalid measurement and insufficient evidence;
  reliability = share of configured evidence weight that was available and current.

## Unavailable or limited fields

- No live data source exists; all timestamps are historical simulated decision times.
- The operational forecast file has no any-warning persistence column; it is derived as described above.
- The Phase 4A per-site false-alert table was not saved as a file; it is recomputed with the original function.
- Training-period forecast metrics were not saved in the summary file; they are recomputed (in-sample, labelled).

## Screenshots

`docs/screenshots/{overview, monitoring, alerts, forecasting, evaluation, data-quality}.png` (captured headless with
Playwright against a local server; first viewport of each page).
