"""Plotly builders - 'prevenza' enterprise template. Titles live in the surrounding card header, not the figure.
Missing values stay gaps (connectgaps=False); nothing is zero-filled; no gauges."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import theme as T

pio.templates["prevenza"] = go.layout.Template(layout=T.PLOT_LAYOUT)
CLASS_CODE = {c: i for i, c in enumerate(T.CLASS_ORDER)}


def _fig(height: int = 300, **layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="prevenza", height=height, **layout)
    return fig


def _threshold(fig, y=40, label="Elevated threshold 40"):
    fig.add_hline(y=y, line=dict(color="#94A3B8", dash="dash", width=1), annotation_text=label,
                  annotation_position="top left", annotation_font=dict(size=12.5, color=T.MUTED))


# ------------------------------------------------------------------------------------------ overview
def class_donut(df: pd.DataFrame, center: str) -> go.Figure:
    counts = df["classification"].value_counts().reindex(T.CLASS_ORDER, fill_value=0)
    counts = counts[counts > 0]
    fig = _fig(300, hovermode="closest", margin=dict(l=8, r=8, t=8, b=8))
    fig.add_pie(labels=[f"{T.CLASS_ICONS[c]} {c}" for c in counts.index], values=counts.values, hole=0.72,
                marker=dict(colors=[T.CLASS_COLORS[c] for c in counts.index], line=dict(color=T.SURFACE, width=2)),
                textinfo="none", sort=False, direction="clockwise",
                hovertemplate="%{label}<br>%{value} assessments · %{percent}<extra></extra>")
    fig.add_annotation(text=center, showarrow=False, font=dict(size=13, color=T.INK, family=T.FONT_SANS))
    fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle", xanchor="left",
                                  font=dict(size=11.5)))
    return fig


def site_comparison(df: pd.DataFrame) -> go.Figure:
    ct = pd.crosstab(df["site_id"], df["classification"]).reindex(columns=T.CLASS_ORDER, fill_value=0)
    share = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = _fig(300, hovermode="closest", bargap=0.38, barmode="stack")
    for c in T.CLASS_ORDER:
        if ct[c].sum() == 0:
            continue
        fig.add_bar(x=share.index, y=share[c], name=f"{T.CLASS_ICONS[c]} {c}",
                    marker=dict(color=T.CLASS_COLORS[c], line=dict(color=T.SURFACE, width=2)),
                    text=[f"{v:.0f}%" if v >= 8 else "" for v in share[c]], textposition="inside",
                    insidetextfont=dict(family=T.FONT_MONO, size=11, color="#0C1220"),
                    customdata=ct[c], hovertemplate=f"%{{x}} · {c}<br>%{{y:.1f}}% (%{{customdata}} obs)<extra></extra>")
    fig.update_layout(yaxis=dict(range=[0, 100], ticksuffix="%"), showlegend=False)
    return fig


def site_risk_trend(df: pd.DataFrame) -> go.Figure:
    d = df.groupby(["site_id", "observation_date"])["risk_score"].max().reset_index()
    fig = _fig(300)
    for (site, g), c in zip(d.groupby("site_id"), ["#1D4ED8", "#D97706", "#0D9488"]):
        fig.add_scatter(x=g["observation_date"], y=g["risk_score"], mode="lines", name=site,
                        line=dict(color=c, width=2), connectgaps=False, hovertemplate="%{y:.1f}")
    _threshold(fig)
    fig.update_layout(yaxis=dict(range=[0, 100]))
    return fig


def zone_heatmap(df: pd.DataFrame) -> go.Figure:
    p = df.pivot_table(index="zone_id", columns="observation_date", values="risk_score", aggfunc="max")
    fig = _fig(300, hovermode="closest", plot_bgcolor=T.CANVAS)      # no score = canvas, darker than risk 0
    fig.add_heatmap(z=p.to_numpy(), x=p.columns, y=p.index, zmin=0, zmax=100, colorscale=T.RISK_RAMP, xgap=1, ygap=2,
                    colorbar=dict(thickness=8, outlinewidth=0, tickfont=dict(family=T.FONT_MONO, size=12, color=T.MUTED),
                                  len=0.9),
                    hovertemplate="%{y} · %{x|%d %b}<br>daily max risk %{z:.1f}<extra></extra>")
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False))
    return fig


def category_bars(means: pd.Series, statuses: dict) -> go.Figure:
    fig = _fig(300, hovermode="closest", bargap=0.32, margin=dict(l=120, r=16, t=8, b=30))
    rows = [(T.CATEGORY_LABELS[c], means.get(c), T.CATEGORY_COLORS[c], statuses.get(c, "")) for c in T.CATEGORIES]
    fig.add_bar(y=[r[0] for r in rows], x=[0 if pd.isna(r[1]) else r[1] for r in rows], orientation="h",
                marker=dict(color=[r[2] if pd.notna(r[1]) else T.BORDER for r in rows], cornerradius=3),
                text=["unavailable" if pd.isna(r[1]) else f"{r[1]:.0f} · {r[3]}" for r in rows],
                textposition="outside", cliponaxis=False, textfont=dict(family=T.FONT_MONO, size=13, color=T.INK_2),
                hovertemplate="%{y}: %{text}<extra></extra>")
    fig.add_vline(x=40, line=dict(color="#94A3B8", dash="dash", width=1))
    fig.update_layout(xaxis=dict(range=[0, 130], showgrid=True, gridcolor=T.SURFACE_MUTED),
                      yaxis=dict(autorange="reversed", showgrid=False), showlegend=False)
    return fig


# ------------------------------------------------------------------------------------------ zone views
def risk_trend(df: pd.DataFrame, shade: tuple | None = None) -> go.Figure:
    fig = _fig(300)
    for shift, dash in (("day", "solid"), ("night", "dot")):
        d = df[df["shift"] == shift].sort_values("decision_time")
        if d.empty:
            continue
        fig.add_scatter(x=d["decision_time"], y=d["risk_score"], mode="lines", name=f"{shift} shift",
                        line=dict(color=T.INK if shift == "day" else T.MUTED, width=2, dash=dash),
                        connectgaps=False, customdata=d["classification"],
                        hovertemplate="%{y:.1f} · %{customdata}")
    if shade:
        fig.add_vrect(x0=shade[0], x1=shade[1], fillcolor="#7F1D1D", opacity=0.28, line_width=0, layer="below")
    _threshold(fig)
    fig.update_layout(yaxis=dict(range=[0, 100]))
    return fig


def classification_tiles(df: pd.DataFrame) -> go.Figure:
    """Day/night tile strip; cells without an assessment stay pale (missing)."""
    d = df.copy()
    d["row"] = d["shift"].map({"day": "Day", "night": "Night"})
    p = d.pivot_table(index="row", columns="observation_date", values="classification", aggfunc="first")
    p = p.reindex(["Day", "Night"])
    z = p.apply(lambda s: s.map(CLASS_CODE)).to_numpy(dtype=float)
    n = len(T.CLASS_ORDER)
    scale = []
    for i, c in enumerate(T.CLASS_ORDER):
        scale += [[i / n, T.CLASS_COLORS[c]], [(i + 1) / n, T.CLASS_COLORS[c]]]
    fig = _fig(140, hovermode="closest", plot_bgcolor=T.SURFACE_MUTED, margin=dict(l=48, r=16, t=8, b=30))
    fig.add_heatmap(z=z, x=p.columns, y=p.index, zmin=-0.5, zmax=n - 0.5, colorscale=scale, showscale=False,
                    xgap=1.5, ygap=3, customdata=p.fillna("no assessment").to_numpy(),
                    hovertemplate="%{y} · %{x|%d %b}<br>%{customdata}<extra></extra>")
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig


def category_small_multiples(df: pd.DataFrame) -> go.Figure:
    """Five synchronised 85 px tracks (shared x) - no overlapping spaghetti."""
    d = df.sort_values("decision_time")
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035)
    for i, c in enumerate(T.CATEGORIES, start=1):
        fig.add_scatter(x=d["decision_time"], y=d[f"{c}_score"], mode="lines", name=T.CATEGORY_LABELS[c],
                        line=dict(color=T.CATEGORY_COLORS[c], width=1.6), connectgaps=False, row=i, col=1,
                        customdata=d[f"{c}_status"], hovertemplate="%{y:.0f} (%{customdata})")
        fig.add_hline(y=40, line=dict(color="#94A3B8", dash="dash", width=1), row=i, col=1)
        fig.update_yaxes(range=[0, 100], tickvals=[0, 50, 100], row=i, col=1)
    fig.update_layout(template="prevenza", height=5 * 85 + 50, showlegend=False, margin=dict(l=185, r=16, t=12, b=30))
    for i, c in enumerate(T.CATEGORIES, start=1):
        dom = fig.layout[f"yaxis{'' if i == 1 else i}"].domain
        fig.add_annotation(xref="paper", yref="paper", x=-0.045, y=(dom[0] + dom[1]) / 2, xanchor="right",
                           showarrow=False, text=T.CATEGORY_LABELS[c],
                           font=dict(size=13.5, color=T.CATEGORY_COLORS[c], family=T.FONT_SANS))
    fig.update_xaxes(showgrid=False)
    return fig


def category_highlight(df: pd.DataFrame, elevated: set[str]) -> go.Figure:
    """Elevated streams full colour; others muted at 30% opacity."""
    fig = _fig(300)
    d = df.sort_values("decision_time")
    for c in T.CATEGORIES:
        hot = c in elevated
        fig.add_scatter(x=d["decision_time"], y=d[f"{c}_score"], mode="lines", name=T.CATEGORY_LABELS[c],
                        line=dict(color=T.CATEGORY_COLORS[c], width=2.2 if hot else 1.2), opacity=1.0 if hot else 0.3,
                        connectgaps=False, hovertemplate="%{y:.0f}")
    _threshold(fig)
    fig.update_layout(yaxis=dict(range=[0, 100]))
    return fig


def measurement_lines(df: pd.DataFrame, series: list[tuple[str, str, str]], ytitle: str,
                      flag_col: str | None = None, height: int = 260) -> go.Figure:
    fig = _fig(height)
    d = df.sort_values("shift_start")
    for col, label, color in series:
        fig.add_scatter(x=d["shift_start"], y=d[col], mode="lines", name=label, connectgaps=False,
                        line=dict(color=color, width=1.8), hovertemplate="%{y}")
    if flag_col and flag_col in d:
        x = d[d[flag_col]]
        if len(x):
            fig.add_scatter(x=x["shift_start"], y=x[series[-1][0]], mode="markers", name="⊘ extrapolated heat index",
                            marker=dict(symbol="x-thin", size=12, color="#A21CAF", line=dict(width=2, color="#A21CAF")),
                            hovertemplate="outside NWS chart range: %{y}")
    fig.update_layout(yaxis_title=dict(text=ytitle, font=dict(size=13)))
    return fig


def availability_strip(states: pd.DataFrame) -> go.Figure:
    code = {"reported": 0, "field missing": 1, "not reported": 2}
    z = states.apply(lambda s: s.map(code)).T.to_numpy(dtype=float)
    fig = _fig(60 + 30 * len(states.columns), hovermode="closest", margin=dict(l=96, r=16, t=8, b=30))
    fig.add_heatmap(z=z, x=states.index, y=list(states.columns), zmin=0, zmax=2, showscale=False,
                    colorscale=[[0, "#1F6B43"], [0.33, "#1F6B43"], [0.34, "#8A6A12"], [0.66, "#8A6A12"],
                                [0.67, "#334155"], [1, "#334155"]],
                    customdata=states.T.to_numpy(), hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>",
                    xgap=1, ygap=3)
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False))
    return fig


# ------------------------------------------------------------------------------------------ alerts
def contribution_bars(X: pd.DataFrame) -> go.Figure:
    x = X[X["component"] != "isolation_forest"].copy()
    x["pts"] = x["contribution_points"].astype(float)
    x = x.sort_values("pts", ascending=True)
    labels = [T.CATEGORY_LABELS.get(c, c) for c in x["component"]]
    text = [f"+{p:.1f} pts · score {s:.0f}" if pd.notna(s) else f"{st_} · excluded"
            for p, s, st_ in zip(x["pts"], x["signal_score"], x["status"])]
    fig = _fig(260, hovermode="closest", bargap=0.32, margin=dict(l=120, r=16, t=8, b=36))
    fig.add_bar(y=labels, x=x["pts"], orientation="h", text=text, textposition="outside", cliponaxis=False,
                textfont=dict(family=T.FONT_MONO, size=13, color=T.INK_2),
                marker=dict(color=[T.CATEGORY_COLORS.get(c, T.MUTED) for c in x["component"]], cornerradius=3),
                hovertemplate="%{y}: %{x:.1f} points<extra></extra>")
    fig.update_layout(xaxis=dict(range=[0, max(10, float(x["pts"].max()) * 1.7)], showgrid=True,
                                 gridcolor=T.SURFACE_MUTED, title=dict(text="points of risk score", font=dict(size=13))),
                      yaxis=dict(showgrid=False), showlegend=False)
    return fig


# ------------------------------------------------------------------------------------------ forecast
def forecast_timeline(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = _fig(300)
    d = df.sort_values("decision_time")
    fig.add_scatter(x=d["decision_time"], y=d["forecast_score"], mode="lines", name="LR ranking score",
                    line=dict(color=T.STATES["Forecast"][3], width=2), connectgaps=False, hovertemplate="%{y:.3f}")
    fig.add_hline(y=threshold, line=dict(color="#94A3B8", dash="dash", width=1),
                  annotation_text=f"Fixed flag threshold {threshold:.3f}", annotation_position="top left",
                  annotation_font=dict(size=12.5, color=T.MUTED))
    p = d[d["persistence_any_flag"]]
    fig.add_scatter(x=p["decision_time"], y=np.full(len(p), 1.03), mode="markers", name="Persistence (any warning)",
                    marker=dict(symbol="diamond", size=7, color=T.MODEL_COLORS["persistence (any warning)"]),
                    hovertemplate="persistence flag")
    fig.update_layout(yaxis=dict(range=[0, 1.08], title=dict(text="ranking score (not a probability)",
                                                              font=dict(size=13))))
    return fig


# ------------------------------------------------------------------------------------------ evaluation
def confusion(tp, fp, fn, tn) -> go.Figure:
    z = np.array([[tn, fp], [fn, tp]], dtype=float)
    rows = z.sum(axis=1, keepdims=True)
    pct = np.divide(z, rows, out=np.zeros_like(z), where=rows > 0) * 100
    xs, ys = ["No alert", "Alert"], ["Negative", "Positive"]
    fig = _fig(230, hovermode="closest", margin=dict(l=76, r=12, t=8, b=40))
    fig.add_heatmap(z=z, x=xs, y=ys, showscale=False, colorscale=T.COUNT_RAMP, xgap=2, ygap=2,
                    hovertemplate="%{y} · %{x}: %{z:.0f}<extra></extra>")
    for i in range(2):
        for j in range(2):
            dark = z.max() > 0 and z[i, j] / z.max() > 0.55
            fig.add_annotation(x=xs[j], y=ys[i], showarrow=False,
                               text=f"<b>{int(z[i, j])}</b><br><span style='font-size:11px'>{pct[i, j]:.0f}% of row</span>",
                               font=dict(family=T.FONT_MONO, size=16, color="#0C1220" if dark else T.INK))
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False, title=dict(text="label", font=dict(size=13))),
                      xaxis=dict(showgrid=False, title=dict(text="prediction", font=dict(size=13))))
    return fig


def grouped_bars(df: pd.DataFrame, x: str, y: str, color: str, ytitle: str, colors: dict) -> go.Figure:
    fig = _fig(320, hovermode="closest", barmode="group", bargap=0.32)
    for name, g in df.groupby(color, sort=False):
        fig.add_bar(x=g[x], y=g[y], name=name, marker=dict(color=colors.get(name, T.MUTED), cornerradius=2,
                    line=dict(color=T.SURFACE, width=1)), hovertemplate=f"{name}<br>%{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_layout(yaxis_title=dict(text=ytitle, font=dict(size=13)))
    return fig


def lead_time_dumbbell(df: pd.DataFrame) -> go.Figure:
    """df: unit, model, hours (positive = before escalation). Missing = not detected (no marker)."""
    units = list(dict.fromkeys(df["unit"]))
    fig = _fig(80 + 46 * len(units), hovermode="closest", margin=dict(l=130, r=16, t=24, b=44))
    wide = df.pivot_table(index="unit", columns="model", values="hours").reindex(units)
    for u in units:
        vals = wide.loc[u].dropna()
        if len(vals) >= 2:
            fig.add_scatter(x=[vals.min(), vals.max()], y=[u, u], mode="lines", line=dict(color=T.BORDER_STRONG, width=2),
                            showlegend=False, hoverinfo="skip")
    for m in df["model"].unique():
        g = df[df["model"] == m]
        fig.add_scatter(x=g["hours"], y=g["unit"], mode="markers", name=m,
                        marker=dict(size=11, color=T.MODEL_COLORS.get(m, T.MUTED), line=dict(color=T.SURFACE, width=1.5)),
                        hovertemplate=f"{m}<br>%{{y}}: %{{x:+.0f}} h vs escalation<extra></extra>")
    fig.add_vline(x=0, line=dict(color=T.INK_2, width=1))
    fig.update_layout(xaxis=dict(title=dict(text="hours before escalation (0 = escalation; negative = after)",
                                            font=dict(size=13)), showgrid=True, gridcolor=T.SURFACE_MUTED),
                      yaxis=dict(autorange="reversed", showgrid=False))
    return fig


# ------------------------------------------------------------------------------------------ data quality
def availability_steps(daily: pd.DataFrame) -> go.Figure:
    fig = _fig(300)
    for label in daily.columns:
        fig.add_scatter(x=daily.index, y=daily[label] * 100, mode="lines", name=label, line_shape="hv",
                        line=dict(color=T.SOURCE_COLORS.get(label, T.MUTED), width=1.8), hovertemplate="%{y:.0f}%")
    fig.update_layout(yaxis=dict(range=[0, 104], ticksuffix="%"))
    return fig


def missingness_bars(mt: pd.DataFrame) -> go.Figure:
    fig = _fig(220, hovermode="closest", barmode="stack", bargap=0.35, margin=dict(l=96, r=16, t=8, b=36))
    for col, color, name in (("reported", "#2F8F5B", "Reported"), ("field missing (reported row)", "#D97706", "Field missing"),
                             ("not reported (whole period)", "#475569", "Not reported")):
        fig.add_bar(y=mt["source"], x=mt[col], name=name, orientation="h", marker=dict(color=color),
                    hovertemplate=f"{name}: %{{x}}<extra>%{{y}}</extra>")
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=True,
                      gridcolor=T.SURFACE_MUTED, title=dict(text="observations", font=dict(size=13))))
    return fig


def box_h(df: pd.DataFrame, cat: str, val: str, xtitle: str, log: bool = False) -> go.Figure:
    fig = _fig(260, hovermode="closest", margin=dict(l=96, r=16, t=8, b=40))
    for name, g in df.groupby(cat, sort=False):
        fig.add_box(x=g[val].dropna(), name=str(name), orientation="h",
                    marker=dict(color=T.SOURCE_COLORS.get(str(name), T.MUTED), size=3),
                    line=dict(width=1.2), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed", showgrid=False),
                      xaxis=dict(type="log" if log else "linear", showgrid=True, gridcolor=T.SURFACE_MUTED,
                                 title=dict(text=xtitle, font=dict(size=13))))
    return fig


def status_matrix(status: pd.DataFrame) -> go.Figure:
    fig = _fig(220, hovermode="closest", margin=dict(l=150, r=16, t=30, b=10))
    fig.add_heatmap(z=status.to_numpy(dtype=float), x=list(status.columns), y=list(status.index), showscale=False,
                    colorscale=T.COUNT_RAMP, xgap=2, ygap=2, hovertemplate="%{y} · %{x}: %{z:.0f}<extra></extra>")
    zmax = float(status.to_numpy().max()) or 1.0
    for i, r in enumerate(status.index):
        for j, c in enumerate(status.columns):
            v = status.loc[r, c]
            fig.add_annotation(x=c, y=r, text=str(int(v)), showarrow=False,
                               font=dict(family=T.FONT_MONO, size=12, color="#0C1220" if v / zmax > 0.55 else T.INK))
    fig.update_layout(xaxis=dict(showgrid=False, side="top"), yaxis=dict(showgrid=False, autorange="reversed"))
    return fig


def simple_bars(x, y, color: str, xtitle: str, ytitle: str, height: int = 240) -> go.Figure:
    fig = _fig(height, hovermode="closest", bargap=0.3)
    fig.add_bar(x=x, y=y, marker=dict(color=color, cornerradius=2), hovertemplate="%{x}: %{y}<extra></extra>")
    fig.update_layout(xaxis_title=dict(text=xtitle, font=dict(size=13)), yaxis_title=dict(text=ytitle, font=dict(size=13)))
    return fig


def histogram(x, color: str, xtitle: str, height: int = 240) -> go.Figure:
    fig = _fig(height, hovermode="closest", bargap=0.08)
    fig.add_histogram(x=x, nbinsx=20, marker=dict(color=color))
    fig.update_layout(xaxis_title=dict(text=xtitle, font=dict(size=13)),
                      yaxis_title=dict(text="assessments", font=dict(size=13)))
    return fig


# NWS heat-index bands (degC) for the thermal bullet bar (a measurement, not a probability)
HI_BANDS = [(15, 27, "#1E293B", "Below caution"), (27, 32, "#3D3312", "Caution"), (32, 39, "#6B4E0E", "Extreme caution"),
            (39, 51, "#9A3412", "Danger"), (51, 60, "#991B1B", "Extreme danger")]


def nws_band(v: float) -> str:
    for lo, hi, _, name in HI_BANDS:
        if v < hi:
            return name if v >= lo else HI_BANDS[0][3]
    return "Extreme danger"
