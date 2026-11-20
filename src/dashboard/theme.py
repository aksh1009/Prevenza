"""Visual constants shared by every dashboard page - dark operations-console theme.
Colours are always paired with text labels and symbols (never colour alone)."""

DISCLAIMER = ("SYNTHETIC DEMONSTRATOR — All data is simulated. Prevenza has not been clinically validated and must "
              "not be used for real-world medical, occupational-health or emergency decisions.")
FORECAST_INTERPRETATION = ("Forecast scores are uncalibrated model rankings, not probabilities. They must not be "
                           "interpreted as the probability of a health event.")
EVALUATION_LIMITATION = ("These results are based on a small number of simulated events. They are exploratory and do "
                         "not establish real-world predictive performance.")
TAGLINE = "Detect Early. Understand Risk. Prevent Harm."

# Surfaces and ink (dark)
NAVY, NAVY_2 = "#070d1a", "#122038"
SURFACE, PANEL, PANEL_2, BORDER = "#0a1122", "#0f1a2e", "#14223b", "#1f2e4a"
INK, INK_2, MUTED = "#e8eef8", "#a7b4c9", "#6f7f99"
ACCENT = "#3b82f6"

# Ordered risk classes (severity ramp) - colour + symbol + label
CLASS_ORDER = ["Normal", "Environmental warning", "Elevated occupational exposure", "Emerging health risk",
               "High health risk", "Insufficient evidence"]
CLASS_COLORS = {
    "Normal": "#22c55e",
    "Environmental warning": "#facc15",
    "Elevated occupational exposure": "#fb923c",
    "Emerging health risk": "#f87171",
    "High health risk": "#e11d48",
    "Insufficient evidence": "#64748b",
}
CLASS_ICONS = {"Normal": "●", "Environmental warning": "▲", "Elevated occupational exposure": "◆",
               "Emerging health risk": "■", "High health risk": "✖", "Insufficient evidence": "◌"}

# Five risk categories: reference categorical palette, dark-mode steps, fixed order
CATEGORIES = ["environmental", "workforce_exposure", "clinical", "absenteeism", "incidents"]
CATEGORY_LABELS = {"environmental": "Environmental", "workforce_exposure": "Workforce exposure",
                   "clinical": "Clinical", "absenteeism": "Absenteeism", "incidents": "Incidents"}
CATEGORY_COLORS = {"environmental": "#3987e5", "workforce_exposure": "#d95926", "clinical": "#199e70",
                   "absenteeism": "#c98500", "incidents": "#d55181"}

MODEL_COLORS = {"A: MAD + CUSUM": "#3987e5", "B: Isolation Forest": "#d95926", "C: Risk fusion": "#199e70",
                "persistence (health alert)": "#c98500", "persistence (any warning)": "#d55181",
                "logistic regression": "#9085e9"}

DATA_STATES = {
    "Valid observation": ("#22c55e", "Reported, within range, usable"),
    "Missing": ("#94a3b8", "Not reported or field empty - never treated as zero"),
    "Stale": ("#facc15", "Last report older than the source's staleness limit - excluded from fusion"),
    "Invalid measurement": ("#f87171", "Heat index outside the NWS chart range (extrapolated) - magnitude unreliable"),
    "Insufficient evidence": ("#64748b", "Too little current evidence to confirm normal conditions"),
}

# Sequential magnitude ramp (one warm hue, dark -> bright) for risk heatmaps
RISK_RAMP = [[0.0, "#16233c"], [0.25, "#4a2a1c"], [0.5, "#9a3f12"], [0.75, "#ea6a1f"], [1.0, "#ffc18a"]]
# Sequential blue ramp for counts (confusion matrices)
COUNT_RAMP = [[0.0, "#14223b"], [0.35, "#1c4b8a"], [0.7, "#2a78d6"], [1.0, "#6da7ec"]]

PLOT_LAYOUT = dict(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color=INK_2),
    paper_bgcolor=PANEL, plot_bgcolor=PANEL, margin=dict(l=10, r=20, t=44, b=10),
    hoverlabel=dict(bgcolor=PANEL_2, font_color=INK, bordercolor=BORDER),
    legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, title_text="", font=dict(color=INK_2)),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, automargin=True, tickfont=dict(color=INK_2)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, automargin=True, tickfont=dict(color=INK_2)),
)
