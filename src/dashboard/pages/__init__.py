"""Dashboard pages. Operational pages use data_loader.load_operational only; evaluation.py is evaluation-only."""
from . import alert_investigation, data_quality, evaluation, forecasting, home, overview, site_monitoring

PAGES = {
    "Home": home,
    "Executive overview": overview,
    "Site & zone monitoring": site_monitoring,
    "Alert investigation": alert_investigation,
    "24-hour forecasting": forecasting,
    "Model evaluation": evaluation,
    "Data quality & reliability": data_quality,
}
SLUGS = {"home": "Home", "overview": "Executive overview", "monitoring": "Site & zone monitoring", "alerts": "Alert investigation",
         "forecasting": "24-hour forecasting", "evaluation": "Model evaluation", "data-quality": "Data quality & reliability"}
OPERATIONAL_PAGES = ["Executive overview", "Site & zone monitoring", "Alert investigation", "24-hour forecasting",
                     "Data quality & reliability"]

# Page header metadata: (subtitle, pills, as-of session key or None, global filters enabled)
META = {
    "Home": (None, [], None, False),                     # front page renders its own hero, no header/ribbon
    "Executive overview": ("Simulated operational heat-health risk across sites, zones and shifts",
                           ["Historical replay · not live data"], "ov_asof", True),
    "Site & zone monitoring": ("Fused risk assessments and the source measurements behind them",
                               ["Gaps = unavailable data, never zero"], None, True),
    "Alert investigation": ("Why the risk-fusion system raised a health-risk alert, and on what evidence",
                            ["Risk fusion (model C) episodes", "Not clinical diagnoses"], None, True),
    "24-hour forecasting": ("Will a site-zone enter an emerging or escalating heat-risk state within 24 hours?",
                            ["Logistic regression vs persistence", "Uncalibrated rankings"], "fc_t", True),
    "Model evaluation": ("Detection and forecasting outputs compared with the simulation's ground truth",
                         ["Evaluation only", "Synthetic events"], None, False),
    "Data quality & reliability": ("How complete, timely and valid is the evidence behind each assessment?",
                                   ["Missing ≠ zero", "Missing ≠ normal"], None, True),
}
