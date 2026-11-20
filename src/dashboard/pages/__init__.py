"""Dashboard pages. Operational pages use data_loader.load_operational only; evaluation.py is evaluation-only."""
from . import alert_investigation, data_quality, evaluation, forecasting, overview, site_monitoring

PAGES = {
    "Executive overview": overview,
    "Site & zone monitoring": site_monitoring,
    "Alert investigation": alert_investigation,
    "24-hour forecasting": forecasting,
    "Model evaluation": evaluation,
    "Data quality & reliability": data_quality,
}
SLUGS = {"overview": "Executive overview", "monitoring": "Site & zone monitoring", "alerts": "Alert investigation",
         "forecasting": "24-hour forecasting", "evaluation": "Model evaluation", "data-quality": "Data quality & reliability"}
OPERATIONAL_PAGES = ["Executive overview", "Site & zone monitoring", "Alert investigation", "24-hour forecasting",
                     "Data quality & reliability"]
