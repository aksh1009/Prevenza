"""Plotly chart builders. Missing values stay gaps (connectgaps=False); nothing is zero-filled."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme as T


def _fig(title: str = "", height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**T.PLOT_LAYOUT, height=height, title=dict(text=title, x=0, font=dict(size=14, color=T.INK)))
    return fig


def class_distribution(df: pd.DataFrame, title: str) -> go.Figure:
    counts = df["classification"].value_counts().reindex(T.CLASS_ORDER, fill_value=0)
    fig = _fig(title, 300)
    fig.add_bar(y=[f"{T.CLASS_ICONS[c]} {c}" for c in counts.index], x=counts.values, orientation="h",
                marker=dict(color=[T.CLASS_COLORS[c] for c in counts.index], line=dict(color=T.PANEL, width=2)),
                text=counts.values, textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x}<extra></extra>")
    fig.update_layout(yaxis=dict(autorange="reversed", gridcolor=T.BORDER, automargin=True),
                      xaxis=dict(title="observations", range=[0, max(counts.max(), 1) * 1.15], gridcolor=T.BORDER),
                      showlegend=False)
    return fig


def site_comparison(df: pd.DataFrame, title: str) -> go.Figure:
    """Stacked share of classifications per site."""
    ct = pd.crosstab(df["site_id"], df["classification"]).reindex(columns=T.CLASS_ORDER, fill_value=0)
    fig = _fig(title, 300)
    for c in T.CLASS_ORDER:
        if ct[c].sum() == 0:
            continue
        fig.add_bar(x=ct.index, y=ct[c], name=f"{T.CLASS_ICONS[c]} {c}", marker=dict(color=T.CLASS_COLORS[c],
                    line=dict(color=T.PANEL, width=2)), hovertemplate=f"{c}: %{{y}}<extra>%{{x}}</extra>")
    fig.update_layout(barmode="stack", yaxis=dict(title="observations", automargin=True, gridcolor=T.BORDER),
                      height=360)
    return fig


def risk_trend(df: pd.DataFrame, title: str = "Risk score (0-100, evidence-weighted index)") -> go.Figure:
    fig = _fig(title, 300)
    for shift, dash in (("day", "solid"), ("night", "dot")):
        d = df[df["shift"] == shift].sort_values("decision_time")
        if d.empty:
            continue
        fig.add_scatter(x=d["decision_time"], y=d["risk_score"], mode="lines+markers", name=f"{shift} shift",
                        line=dict(color=T.INK_2, width=2, dash=dash), marker=dict(size=5), connectgaps=False,
                        customdata=np.stack([d["classification"], d["evidence_level"]], axis=1),
                        hovertemplate="%{x}<br>risk %{y:.1f}<br>%{customdata[0]}<br>evidence %{customdata[1]}<extra></extra>")
    fig.update_layout(yaxis=dict(range=[0, 105], gridcolor=T.BORDER))
    return fig


def classification_timeline(df: pd.DataFrame, title: str = "Risk classification timeline") -> go.Figure:
    fig = _fig(title, 260)
    for c in T.CLASS_ORDER:
        d = df[df["classification"] == c]
        if d.empty:
            continue
        fig.add_scatter(x=d["decision_time"], y=d["shift"], mode="markers", name=f"{T.CLASS_ICONS[c]} {c}",
                        marker=dict(color=T.CLASS_COLORS[c], size=11, symbol="square", line=dict(color=T.PANEL, width=1)),
                        hovertemplate=f"%{{x}}<br>{c}<extra></extra>")
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=["night", "day"], gridcolor=T.BORDER))
    return fig


def category_scores(df: pd.DataFrame, title: str = "Category scores (0-100; gaps = unavailable)") -> go.Figure:
    fig = _fig(title, 320)
    d = df.sort_values("decision_time")
    for c in T.CATEGORIES:
        fig.add_scatter(x=d["decision_time"], y=d[f"{c}_score"], mode="lines", name=T.CATEGORY_LABELS[c],
                        line=dict(color=T.CATEGORY_COLORS[c], width=2), connectgaps=False,
                        customdata=d[f"{c}_status"],
                        hovertemplate=f"{T.CATEGORY_LABELS[c]}: %{{y:.1f}} (%{{customdata}})<extra></extra>")
    fig.add_hline(y=40, line=dict(color=T.MUTED, dash="dash", width=1),
                  annotation_text="elevated threshold (40, fixed)", annotation_position="top left")
    fig.update_layout(yaxis=dict(range=[0, 105], gridcolor=T.BORDER))
    return fig


def measurement_lines(df: pd.DataFrame, series: list[tuple[str, str, str]], title: str, ytitle: str,
                      flag_col: str | None = None) -> go.Figure:
    """series: (column, label, colour). Rows where the value is NA are left as gaps."""
    fig = _fig(title, 280)
    d = df.sort_values("shift_start")
    for col, label, color in series:
        fig.add_scatter(x=d["shift_start"], y=d[col], mode="lines+markers", name=label, connectgaps=False,
                        line=dict(color=color, width=2), marker=dict(size=4),
                        hovertemplate=f"{label}: %{{y}}<extra>%{{x}}</extra>")
    if flag_col and flag_col in d:
        x = d[d[flag_col]]
        if len(x):
            fig.add_scatter(x=x["shift_start"], y=x[series[-1][0]], mode="markers", name="⚠ extrapolated heat index",
                            marker=dict(symbol="x", size=12, color="#d03b3b", line=dict(width=2)),
                            hovertemplate="outside NWS chart range: %{y}<extra></extra>")
    fig.update_layout(yaxis_title=ytitle)
    return fig


def availability_strip(states: pd.DataFrame, title: str) -> go.Figure:
    """states: index = shift_start, columns = sources, values in {'reported','field missing','not reported'}."""
    code = {"reported": 0, "field missing": 1, "not reported": 2}
    z = states.apply(lambda s: s.map(code)).T.to_numpy(dtype=float)
    fig = _fig(title, 60 + 34 * len(states.columns))
    fig.add_heatmap(z=z, x=states.index, y=list(states.columns), zmin=0, zmax=2, showscale=False,
                    colorscale=[[0, "#cfe8cf"], [0.33, "#cfe8cf"], [0.34, "#fbd38d"], [0.66, "#fbd38d"],
                                [0.67, "#9a9893"], [1, "#9a9893"]],
                    customdata=states.T.to_numpy(), hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
                    xgap=1, ygap=2)
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(showgrid=False))
    return fig


def confusion(tp, fp, fn, tn, title: str) -> go.Figure:
    z = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig = _fig(title, 260)
    fig.add_heatmap(z=z, x=["no alert", "alert"], y=["negative", "positive"], showscale=False,
                    colorscale=[[0, "#fcfcfb"], [0.3, "#cde2fb"], [0.6, "#3987e5"], [1, "#0d366b"]],
                    text=z.astype(int), texttemplate="%{text}", textfont=dict(size=14),
                    hovertemplate="label %{y}<br>prediction %{x}<br>%{z:.0f}<extra></extra>")
    fig.update_layout(xaxis=dict(side="bottom", showgrid=False), yaxis=dict(autorange="reversed", showgrid=False))
    return fig


def grouped_bars(df: pd.DataFrame, x: str, y: str, color: str, title: str, ytitle: str, colors: dict) -> go.Figure:
    fig = _fig(title, 320)
    for name, g in df.groupby(color, sort=False):
        fig.add_bar(x=g[x], y=g[y], name=name, marker=dict(color=colors.get(name, T.MUTED), line=dict(color=T.PANEL, width=2)),
                    hovertemplate=f"{name}<br>%{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_layout(barmode="group", yaxis_title=ytitle)
    return fig


def forecast_timeline(df: pd.DataFrame, threshold: float, title: str) -> go.Figure:
    fig = _fig(title, 320)
    d = df.sort_values("decision_time")
    fig.add_scatter(x=d["decision_time"], y=d["forecast_score"], mode="lines+markers", name="LR forecast score (ranking)",
                    line=dict(color=T.MODEL_COLORS["logistic regression"], width=2), connectgaps=False,
                    hovertemplate="%{x}<br>ranking score %{y:.3f}<extra></extra>")
    fig.add_hline(y=threshold, line=dict(color=T.INK_2, dash="dash", width=1),
                  annotation_text=f"fixed flag threshold {threshold:.3f}", annotation_position="top left")
    p = d[d["persistence_any_flag"]]
    fig.add_scatter(x=p["decision_time"], y=np.full(len(p), 1.04), mode="markers", name="persistence (any warning) flag",
                    marker=dict(symbol="triangle-down", size=10, color=T.MODEL_COLORS["persistence (any warning)"]),
                    hovertemplate="persistence flag<extra>%{x}</extra>")
    fig.update_layout(yaxis=dict(range=[0, 1.08], title="ranking score (not a probability)", gridcolor=T.BORDER))
    return fig


def lead_time_bars(df: pd.DataFrame, title: str) -> go.Figure:
    """df: unit, model, hours (positive = before escalation)."""
    fig = _fig(title, 340)
    for model, g in df.groupby("model", sort=False):
        fig.add_bar(y=g["unit"], x=g["hours"], orientation="h", name=model,
                    marker=dict(color=T.MODEL_COLORS.get(model, T.MUTED), line=dict(color=T.PANEL, width=2)),
                    hovertemplate=f"{model}<br>%{{y}}: %{{x:+.0f}} h vs escalation<extra></extra>")
    fig.add_vline(x=0, line=dict(color=T.INK, width=1))
    fig.update_layout(barmode="group", xaxis_title="hours before escalation (negative = after)",
                      yaxis=dict(autorange="reversed"))
    return fig


def box_by(df: pd.DataFrame, x: str, y: str, title: str, ytitle: str) -> go.Figure:
    fig = _fig(title, 300)
    for name, g in df.groupby(x, sort=False):
        fig.add_box(y=g[y].dropna(), name=str(name), marker_color=T.CATEGORY_COLORS.get(
            {"Environmental": "environmental", "Workforce": "workforce_exposure", "Clinical": "clinical",
             "Absenteeism": "absenteeism", "Incidents": "incidents"}.get(str(name), ""), T.INK_2), boxmean=True)
    fig.update_layout(yaxis_title=ytitle, showlegend=False)
    return fig
