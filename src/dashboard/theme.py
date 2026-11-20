"""Visual constants shared by every dashboard page (colours always paired with text labels)."""

DISCLAIMER = ("SYNTHETIC DEMONSTRATOR — All data is simulated. Prevenza has not been clinically validated and must "
              "not be used for real-world medical, occupational-health or emergency decisions.")
FORECAST_INTERPRETATION = ("Forecast scores are uncalibrated model rankings, not probabilities. They must not be "
                           "interpreted as the probability of a health event.")
EVALUATION_LIMITATION = ("These results are based on a small number of simulated events. They are exploratory and do "
                         "not establish real-world predictive performance.")
TAGLINE = "Detect Early. Understand Risk. Prevent Harm."

# Surfaces and ink
NAVY, NAVY_2 = "#0f1b2d", "#1b2b44"
SURFACE, PANEL, BORDER = "#f6f7f9", "#ffffff", "#e4e3df"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"

# Ordered risk classes (severity) - colour + short code + label; never colour alone.
CLASS_ORDER = ["Normal", "Environmental warning", "Elevated occupational exposure", "Emerging health risk",
               "High health risk", "Insufficient evidence"]
CLASS_COLORS = {
    "Normal": "#0ca30c",
    "Environmental warning": "#fab219",
    "Elevated occupational exposure": "#ec835a",
    "Emerging health risk": "#d03b3b",
    "High health risk": "#7a1414",
    "Insufficient evidence": "#8a8984",
}
CLASS_ICONS = {"Normal": "●", "Environmental warning": "▲", "Elevated occupational exposure": "◆",
               "Emerging health risk": "■", "High health risk": "✖", "Insufficient evidence": "◌"}

# Five risk categories: reference categorical slots 1-5 (fixed order, never cycled)
CATEGORIES = ["environmental", "workforce_exposure", "clinical", "absenteeism", "incidents"]
CATEGORY_LABELS = {"environmental": "Environmental", "workforce_exposure": "Workforce exposure",
                   "clinical": "Clinical", "absenteeism": "Absenteeism", "incidents": "Incidents"}
CATEGORY_COLORS = {"environmental": "#2a78d6", "workforce_exposure": "#eb6834", "clinical": "#1baf7a",
                   "absenteeism": "#eda100", "incidents": "#e87ba4"}

# Models on the evaluation page
MODEL_COLORS = {"A: MAD + CUSUM": "#2a78d6", "B: Isolation Forest": "#eb6834", "C: Risk fusion": "#1baf7a",
                "persistence (health alert)": "#eda100", "persistence (any warning)": "#e87ba4",
                "logistic regression": "#6250d6"}

# Data-state taxonomy (data quality page)
DATA_STATES = {
    "Valid observation": ("#0ca30c", "Reported, within range, usable"),
    "Missing": ("#8a8984", "Not reported or field empty - never treated as zero"),
    "Stale": ("#fab219", "Last report older than the source's staleness limit - excluded from fusion"),
    "Invalid measurement": ("#d03b3b", "Heat index outside the NWS chart range (extrapolated) - magnitude unreliable"),
    "Insufficient evidence": ("#52514e", "Too little current evidence to confirm normal conditions"),
}

PLOT_LAYOUT = dict(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color=INK_2),
    paper_bgcolor=PANEL, plot_bgcolor=PANEL, margin=dict(l=10, r=20, t=48, b=10),
    hoverlabel=dict(bgcolor=PANEL, font_color=INK, bordercolor=BORDER),
    legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, title_text=""),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, automargin=True),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, automargin=True),
)
