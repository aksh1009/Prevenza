"""Page 3 - Alert investigation (operational: event-matching columns are stripped by the loader)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import charts, components as C, theme as T
from ..data_loader import load_operational, zone_registry
from ..filters import Filters, apply_filters, data_flag_text, measurement_flag_text

LEVEL = {2: "Emerging health risk", 3: "High health risk"}
ACTIONS = [
    "Verify environmental measurements (sensor placement, calibration, and whether the heat-index value is within range).",
    "Review work-rest schedules for the affected zone and shift.",
    "Consider additional hydration and cooling provisions (shade, ventilation, cool-down areas).",
    "Review workforce exposure patterns (shift length, heavy tasks, newly rostered or unacclimatised staff).",
    "Investigate changes in clinical or incident reporting (gaps, delays, coding changes) before drawing conclusions.",
]


def episodes_table(E: pd.DataFrame) -> pd.DataFrame:
    reg = zone_registry().set_index("zone_id")["zone_name"]
    return pd.DataFrame({
        "Episode": E["episode_id"], "Site": E["site_id"], "Zone": E["zone_id"] + " · " + E["zone_id"].map(reg),
        "Start (observation)": E["obs_start"], "End (observation)": E["obs_end"],
        "First detection": E["start_decision"], "Duration h": E["duration_hours"],
        "Alert classification": E["max_level"].map(LEVEL).map(C.class_label),
        "Max risk score": E["peak_risk_score"].round(1), "Alerting decisions": E["n_alerts"],
        "Mean reliability %": E["mean_reliability"].round(0),
    })


def render(f: Filters) -> None:
    C.header("Alert investigation", "Why did the risk-fusion system raise a health-risk alert? Episodes are "
                                    "consecutive Emerging/High health-risk assessments in one zone (both shifts).")
    st.caption("Simulated alerts are not clinical diagnoses. Event-matching outcomes are shown only on the "
               "Model evaluation page.")
    E = load_operational("episodes")
    E = E[(E["model"] == "C") & (E["window"] == "full")]
    E = apply_filters(E.assign(observation_date=E["obs_start"].dt.normalize()), Filters(f.sites, f.zones, [], f.date_range),
                      class_col=None)
    if E.empty:
        C.empty_state("No risk-fusion alert episodes match the current site/zone/date filters.")
        return
    E = E.sort_values("start_decision").reset_index(drop=True)
    tbl = episodes_table(E)
    st.dataframe(tbl, hide_index=True, key="ai_tbl",
                 column_config={"Duration h": st.column_config.NumberColumn(format="%.0f")})
    C.download(tbl, "prevenza_alert_episodes.csv")

    focus = st.session_state.get("focus_zone")
    options = list(E["episode_id"])
    default = next((i for i, z in enumerate(E["zone_id"]) if z == focus), 0)
    ep_id = st.selectbox("Episode to investigate", options, index=default, key="ai_ep",
                         format_func=lambda e: f"{e} · {E.set_index('episode_id').loc[e, 'zone_id']} · "
                                               f"{E.set_index('episode_id').loc[e, 'start_decision']:%Y-%m-%d %H:%M}")
    ep = E.set_index("episode_id").loc[ep_id]
    A = load_operational("assessments")
    Z = A[A["zone_id"] == ep["zone_id"]]
    in_ep = Z[(Z["decision_time"] >= ep["start_decision"]) & (Z["decision_time"] <= ep["end_decision"])]
    first = Z[Z["decision_time"] == ep["start_decision"]].iloc[0]

    if in_ep["alert_relies_on_extrapolated_heat_index"].any():
        n = int(in_ep["alert_relies_on_extrapolated_heat_index"].sum())
        C.caution(f"{n} alerting assessment(s) in this episode use a heat-index value outside the NWS chart range "
                  "(temperature > 43.3 °C or heat index > 58.3 °C). The environmental measurement is an "
                  "extrapolation; verify it before relying on the environmental evidence.")

    k = st.columns(5)
    C.kpi(k[0], "Alert classification", LEVEL[int(ep["max_level"])], f"{int(ep['n_alerts'])} alerting decisions")
    C.kpi(k[1], "Max risk score", f"{ep['peak_risk_score']:.1f}", "0-100 evidence-weighted index")
    C.kpi(k[2], "First detection", C.fmt_ts(ep["start_decision"]), f"zone {ep['zone_id']}")
    C.kpi(k[3], "Duration", f"{ep['duration_hours']:.0f} h", f"{C.fmt_ts(ep['obs_start'])} → {C.fmt_ts(ep['obs_end'])}")
    C.kpi(k[4], "Evidence at detection", f"{first['evidence_level']} · {first['reliability_score']:.0f}%",
          f"{int(first['n_categories_available'])}/5 categories available")
    st.write("")
    st.markdown(f"**Data at first detection:** {data_flag_text(first)} · **Measurement:** {measurement_flag_text(first)}")

    # timeline
    lo, hi = ep["start_decision"] - pd.Timedelta(hours=96), ep["end_decision"] + pd.Timedelta(hours=72)
    W = Z[(Z["decision_time"] >= lo) & (Z["decision_time"] <= hi)]
    fig = charts.risk_trend(W, "Risk score around the episode (shaded = alert episode)")
    fig.add_vrect(x0=ep["start_decision"], x1=ep["end_decision"], fillcolor="#d03b3b", opacity=0.10, line_width=0)
    st.plotly_chart(fig, theme=None, key="ai_tl")
    st.plotly_chart(charts.category_scores(W, "Category scores around the episode"), theme=None, key="ai_cat")

    # explanation at a chosen decision
    st.subheader("Risk explanation")
    choices = list(in_ep.sort_values("decision_time")["decision_time"])
    peak_t = in_ep.loc[in_ep["risk_score"].idxmax(), "decision_time"]
    t = st.select_slider("Decision within the episode", options=choices, value=peak_t, key="ai_t",
                         format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"))
    row = in_ep[in_ep["decision_time"] == t].iloc[0]
    st.markdown(C.class_badge(row["classification"]) + f" &nbsp; risk {row['risk_score']:.1f} · evidence "
                f"{row['evidence_level']} · reliability {row['reliability_score']:.0f}%", unsafe_allow_html=True)
    st.info(row["explanation"])
    X = load_operational("explanations")
    X = X[(X["zone_id"] == row["zone_id"]) & (X["shift"] == row["shift"]) & (X["observation_date"] == row["observation_date"])]
    bar = go.Figure()
    xc = X[X["component"] != "isolation_forest"]
    bar.add_bar(y=[T.CATEGORY_LABELS.get(c, c) for c in xc["component"]], x=xc["contribution_points"], orientation="h",
                marker=dict(color=[T.CATEGORY_COLORS.get(c, T.MUTED) for c in xc["component"]]),
                text=[f"{s} · score {('n/a' if pd.isna(v) else f'{v:.0f}')}" for s, v in zip(xc["status"], xc["signal_score"])],
                textposition="auto", hovertemplate="%{y}: %{x:.1f} points<extra></extra>")
    bar.update_layout(**T.PLOT_LAYOUT)
    bar.update_layout(height=260, title=dict(text="Contribution to risk score (points)", x=0),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, theme=None, key="ai_contrib")
    show = X[["component", "status", "configured_weight", "effective_weight", "signal_score", "contribution_points",
              "observed_value", "baseline_median", "robust_z", "cusum", "staleness_hours", "elevated", "persistent",
              "note"]]
    st.dataframe(show, hide_index=True, key="ai_expl")
    C.download(show, f"prevenza_explanation_{row['zone_id']}_{pd.Timestamp(t):%Y%m%d%H%M}.csv", key="dl_expl")

    st.subheader("Illustrative operational suggestions")
    C.note("Illustrative operational suggestions only — not medical instructions and not a clinical diagnosis. "
           "Any real response must follow site occupational-health procedures and qualified professional judgement.")
    extra = []
    if isinstance(first["missing_sources"], str) and "clinical" in first["missing_sources"]:
        extra.append("Clinical data were unavailable at detection: confirm whether the clinic is reporting.")
    if in_ep["heat_index_validity"].eq("extrapolated").any():
        extra.append("Heat-index readings were extrapolated: cross-check with an independent instrument.")
    for a in extra + ACTIONS:
        st.markdown(f"- {a}")
