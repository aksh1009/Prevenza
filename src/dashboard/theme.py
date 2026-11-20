"""Prevenza design tokens - control-room console (dark, tinted navy neutrals, one cyan signal accent).
Warm hues are reserved for risk semantics only. Every semantic state has an ink / fill / border triad, a chart-series
colour and a glyph, so colour never carries meaning alone."""

DISCLAIMER = ("SYNTHETIC DEMONSTRATOR — All data is simulated. Prevenza has not been clinically validated and must "
              "not be used for real-world medical, occupational-health or emergency decisions.")
FORECAST_INTERPRETATION = ("Forecast scores are uncalibrated model rankings, not probabilities. They must not be "
                           "interpreted as the probability of a health event.")
EVALUATION_LIMITATION = ("These results are based on a small number of simulated events. They are exploratory and do "
                         "not establish real-world predictive performance.")
TAGLINE = "Detect Early. Understand Risk. Prevent Harm."

# ---- neutrals (one tinted-navy family) + one accent ----------------------------------------------------
CANVAS, SURFACE, SURFACE_MUTED = "#0C1220", "#121A2A", "#172133"
SIDEBAR, SIDEBAR_HOVER = "#080D18", "#141E30"
BRAND_PRIMARY = SIDEBAR
INK, INK_2, MUTED = "#E6ECF5", "#A9B6CA", "#7D8BA3"          # contrast on SURFACE: 14.8 / 8.3 / 5.0 : 1
ON_DARK, ON_DARK_MUTED = INK, "#93A1B8"
BORDER, BORDER_STRONG, BORDER_SIDEBAR = "#222D42", "#33405A", "#182235"
ACCENT, ACCENT_SUBTLE = "#67C1D6", "#12303A"                 # signal cyan - interaction & model C only
FONT_SANS = "IBM Plex Sans, Segoe UI, sans-serif"
FONT_COND = "IBM Plex Sans Condensed, IBM Plex Sans, sans-serif"
FONT_MONO = "IBM Plex Mono, ui-monospace, monospace"

# ---- semantic states: (ink, fill, border, chart series, glyph) ----------------------------------------
STATES = {
    "Normal": ("#86EFAC", "#0F2419", "#1F5135", "#22C55E", "●"),
    "Environmental warning": ("#FCD34D", "#251F0B", "#5C4813", "#EAB308", "▲"),
    "Elevated occupational exposure": ("#FDBA74", "#28180C", "#6B3A12", "#F97316", "◆"),
    "Emerging health risk": ("#FCA5A5", "#2B1215", "#6B2228", "#F87171", "■"),
    "High health risk": ("#FFFFFF", "#B91C1C", "#DC2626", "#E11D48", "✕"),
    "Insufficient evidence": ("#CBD5E1", "#1A2333", "#3A475E", "#64748B", "◌"),
    "Missing": ("#94A3B8", "#111827", "#475569", "#475569", "○"),
    "Stale": ("#FDE047", "#24200A", "#5B4E12", "#CA8A04", "◷"),
    "Invalid measurement": ("#F0ABFC", "#28102E", "#6B2A75", "#D946EF", "⊘"),
    "Valid observation": ("#86EFAC", "#0F2419", "#1F5135", "#22C55E", "●"),
    "Forecast": ("#93C5FD", "#0F1F3A", "#1E3A6B", "#60A5FA", "↗"),
}
CLASS_ORDER = ["Normal", "Environmental warning", "Elevated occupational exposure", "Emerging health risk",
               "High health risk", "Insufficient evidence"]
CLASS_COLORS = {c: STATES[c][3] for c in CLASS_ORDER}
CLASS_ICONS = {c: STATES[c][4] for c in CLASS_ORDER}

# ---- evidence streams (fixed order, dark-mode steps) -----------------------------------------------------
CATEGORIES = ["environmental", "workforce_exposure", "clinical", "absenteeism", "incidents"]
CATEGORY_LABELS = {"environmental": "Environmental", "workforce_exposure": "Workforce exposure",
                   "clinical": "Clinical", "absenteeism": "Absenteeism", "incidents": "Incidents"}
CATEGORY_COLORS = {"environmental": "#60A5FA", "workforce_exposure": "#F59E0B", "clinical": "#2DD4BF",
                   "absenteeism": "#A78BFA", "incidents": "#FB7185"}
SOURCE_COLORS = {"Environmental": "#60A5FA", "Workforce": "#F59E0B", "Clinical": "#2DD4BF",
                 "Absenteeism": "#A78BFA", "Incidents": "#FB7185"}

MODEL_COLORS = {"A: MAD + CUSUM": "#94A3B8", "B: Isolation Forest": "#F59E0B", "C: Risk fusion": ACCENT,
                "C: Risk fusion (High only)": "#E11D48", "C: warnings (not health alerts)": "#F97316",
                "persistence (health alert)": "#2DD4BF", "persistence (any warning)": "#A78BFA",
                "logistic regression": "#60A5FA"}

DATA_STATES = {k: (STATES[k][3], d) for k, d in {
    "Valid observation": "Reported, within range, usable",
    "Missing": "Not reported or field empty — never treated as zero",
    "Stale": "Older than the source's staleness limit — excluded from fusion",
    "Invalid measurement": "Heat index beyond the NWS chart range — magnitude unreliable",
    "Insufficient evidence": "Too little current evidence to call conditions normal",
}.items()}

RISK_RAMP = [[0.0, "#26344F"], [0.25, "#3D3312"], [0.6, "#C2410C"], [1.0, "#FECACA"]]   # dark -> ember -> pale
COUNT_RAMP = [[0.0, "#141D2F"], [0.45, "#1E4A5A"], [1.0, ACCENT]]

PLOT_LAYOUT = dict(
    font=dict(family=FONT_SANS, size=14, color=INK_2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
    margin=dict(l=44, r=16, t=32, b=36),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#E6ECF5", bordercolor="#E6ECF5", font=dict(family=FONT_MONO, size=13, color="#0C1220")),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="",
                font=dict(size=13.5, color=INK_2)),
    xaxis=dict(showgrid=False, linecolor=BORDER_STRONG, linewidth=1, ticks="outside", tickcolor=BORDER_STRONG,
               tickfont=dict(family=FONT_MONO, size=12.5, color=MUTED), zeroline=False, automargin=True),
    yaxis=dict(showgrid=True, gridcolor=BORDER, gridwidth=1, linecolor=BORDER_STRONG, linewidth=1,
               ticks="outside", tickcolor=BORDER_STRONG, tickfont=dict(family=FONT_MONO, size=12.5, color=MUTED),
               zeroline=False, automargin=True),
)
PLOT_CONFIG = {"displayModeBar": False}
