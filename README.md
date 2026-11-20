# Prevenza

**AI-powered Population & Workforce Health Early Warning System** — *Detect Early. Understand Risk. Prevent Harm.*

> **SYNTHETIC DEMONSTRATOR** — All data is simulated. Prevenza has not been clinically validated and must not be
> used for real-world medical, occupational-health or emergency decisions.

Prevenza fuses environmental, workforce-exposure, aggregated clinical, absenteeism and incident signals per workplace
zone and shift to flag emerging occupational heat-health risk, explain each assessment, and forecast the next 24 hours.
Everything runs on a 90-day synthetic dataset (3 sites × 3 zones × 2 shifts).

## Quick start (Windows, Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# (optional) rebuild the whole pipeline from scratch
python -m src.data_generation.generate        # synthetic data + ground truth
python -m src.features.build                  # unified observations + as-of features
python -m src.detection.isolation_forest      # Isolation Forest (chronological split)
python -m src.detection.risk_fusion           # evidence-aware risk fusion
python -m src.evaluation.backtest             # chronological replay backtest (~3 min)
python -m src.forecasting.forecast            # 24 h forecast vs persistence
python -m src.evaluation.audit                # pre-dashboard audit

# launch the dashboard (read-only over the saved outputs above)
streamlit run app.py

# tests
python -m pytest -q
```

The repository already contains the generated outputs, so `streamlit run app.py` works immediately.

## Dashboard

Six pages (sidebar navigation; deep links such as `?page=alerts`):

| Page | Purpose | Data |
|---|---|---|
| Executive overview | Zone states at any historical decision time, class distribution, site comparison | operational |
| Site & zone monitoring | Risk trend, classification timeline, category scores, source measurements, availability | operational |
| Alert investigation | Risk-fusion alert episodes, evidence at detection, explanations, illustrative actions | operational |
| 24-hour forecasting | Uncalibrated forecast ranking vs persistence baseline vs current state | operational |
| Model evaluation | Detection backtest, forecast evaluation, audit findings | **evaluation only** |
| Data quality & reliability | Missing / stale / invalid / insufficient evidence, latency, heat-index validity | operational |

See [docs/dashboard.md](docs/dashboard.md) for details and [docs/screenshots/](docs/screenshots/) for screenshots.

## Key documents

- [CLAUDE.md](CLAUDE.md) — project rules (privacy, missing data, honesty)
- [docs/architecture.md](docs/architecture.md) — pipeline and phases
- [docs/data_dictionary.md](docs/data_dictionary.md), [docs/feature_dictionary.md](docs/feature_dictionary.md)
- [docs/risk_fusion_report.md](docs/risk_fusion_report.md), [docs/backtesting_report.md](docs/backtesting_report.md),
  [docs/model_comparison.md](docs/model_comparison.md), [docs/forecast_report.md](docs/forecast_report.md)
- [docs/pre_dashboard_audit.md](docs/pre_dashboard_audit.md) — evaluation-independence audit and limitations

## Honest status

Risk fusion avoids false alarms on isolated spikes but detects late; the 24 h logistic-regression forecast shows no
demonstrated early-warning improvement over a simple persistence baseline and transfers poorly to the foundry
scenario. Forecast scores are uncalibrated rankings, not probabilities. All results come from a small number of
simulated events and require prospective validation on real data.
