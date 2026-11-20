"""Page 3 - Alert investigation (operational: event-matching columns are stripped by the loader)."""
from __future__ import annotations

import html

import pandas as pd
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
        "First detection": E["start_decision"], "Start (observation)": E["obs_start"], "End (observation)": E["obs_end"],
        "Duration (h)": E["duration_hours"], "Peak classification": E["max_level"].map(LEVEL).map(C.class_label),
        "Max risk score": E["peak_risk_score"].round(1), "Alerting decisions": E["n_alerts"],
        "Mean reliability %": E["mean_reliability"].round(0),
    })


def render(f: Filters) -> None:
    C.callout("Simulated episodes are consecutive <b>Emerging</b> or <b>High health risk</b> assessments in one zone "
              "(both shifts). They are not clinical diagnoses. Event-matching outcomes are shown only on Model "
              "evaluation.", html_text=True)
    E = load_operational("episodes")
    E = E[(E["model"] == "C") & (E["window"] == "full")]
    E = apply_filters(E.assign(observation_date=E["obs_start"].dt.normalize()), Filters(f.sites, f.zones, [], f.date_range),
                      class_col=None)
    if E.empty:
        C.empty_state("No episodes match the current filters", "No risk-fusion alert episodes for this site, zone and period.")
        return
    E = E.sort_values("start_decision").reset_index(drop=True)
    tbl = episodes_table(E)
    with C.card("eps", "Alert episodes", f"{len(tbl)} episodes in filter"):
        C.table(tbl, "ai_tbl", height=260, formats={"Duration (h)": st.column_config.NumberColumn(format="%.0f"),
                                                     "Max risk score": st.column_config.NumberColumn(format="%.1f")})
        a, b = st.columns([2, 1], vertical_alignment="bottom")
        focus = st.session_state.get("focus_zone")
        options = list(E["episode_id"])
        default = next((i for i, z in enumerate(E["zone_id"]) if z == focus), 0)
        Ei = E.set_index("episode_id")
        ep_id = a.selectbox("Episode", options, index=default, key="ai_ep",
                            format_func=lambda e: f"{e} · {Ei.loc[e, 'zone_id']} · {Ei.loc[e, 'start_decision']:%Y-%m-%d %H:%M}")
        with b:
            C.download(tbl, "prevenza_alert_episodes.csv")
    ep = Ei.loc[ep_id]
    A = load_operational("assessments")
    Z = A[A["zone_id"] == ep["zone_id"]]
    in_ep = Z[(Z["decision_time"] >= ep["start_decision"]) & (Z["decision_time"] <= ep["end_decision"])]
    first = Z[Z["decision_time"] == ep["start_decision"]].iloc[0]

    C.gap()
    if in_ep["alert_relies_on_extrapolated_heat_index"].any():
        n = int(in_ep["alert_relies_on_extrapolated_heat_index"].sum())
        C.caution(f"{n} alerting assessment(s) in this episode use a heat-index value outside the NWS chart range "
                  "(temperature > 43.3 °C or heat index > 58.3 °C). The environmental measurement is an "
                  "extrapolation; verify it before relying on the environmental evidence.")
    lvl = LEVEL[int(ep["max_level"])]
    k = st.columns(5)
    C.kpi(k[0], "Peak classification", lvl.replace(" health risk", ""), C.badge(lvl), accent=T.CLASS_COLORS[lvl],
          icon_name="shield-alert", small_value=True)
    C.kpi(k[1], "Max risk score", f"{ep['peak_risk_score']:.1f}", '<span class="mono">/ 100 evidence-weighted</span>',
          icon_name="activity")
    C.kpi(k[2], "First detection", f"{ep['start_decision']:%d %b %H:%M}",
          f'<span class="mono">{ep["start_decision"]:%Y-%m-%d} · {ep["zone_id"]}</span>', icon_name="clock")
    C.kpi(k[3], "Episode duration", f"{ep['duration_hours']:.0f} h",
          f'<span class="mono">{int(ep["n_alerts"])} alerting decisions</span>', icon_name="layers")
    C.kpi(k[4], "Evidence at detection", f"{first['reliability_score']:.0f}%",
          C.evidence_bar(int(first["n_categories_available"]), 5, str(first["evidence_level"])), icon_name="database")
    st.markdown(f'<div class="pv-call" style="margin-top:12px">Data at first detection: <b>{html.escape(data_flag_text(first))}</b>'
                f' · Measurement: <b>{html.escape(measurement_flag_text(first))}</b></div>', unsafe_allow_html=True)

    lo, hi = ep["start_decision"] - pd.Timedelta(hours=96), ep["end_decision"] + pd.Timedelta(hours=72)
    W = Z[(Z["decision_time"] >= lo) & (Z["decision_time"] <= hi)]
    peak_row = in_ep.loc[in_ep["risk_score"].idxmax()]
    elevated = {c for c in T.CATEGORIES if bool(peak_row[f"{c}_elevated"])}
    c1, c2 = st.columns(2)
    with c1, C.card("tl", "Risk score around the episode", "shaded = episode window"):
        C.chart(charts.risk_trend(W, shade=(ep["start_decision"], ep["end_decision"])), "ai_tl")
    with c2, C.card("cat", "Category scores around the episode", "elevated streams at peak in full colour"):
        C.chart(charts.category_highlight(W, elevated), "ai_cat")

    C.section("Risk explanation")
    choices = list(in_ep.sort_values("decision_time")["decision_time"])
    t = st.select_slider("Decision within the episode", options=choices, value=peak_row["decision_time"], key="ai_t",
                         format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"))
    row = in_ep[in_ep["decision_time"] == t].iloc[0]
    st.markdown(f'<div class="pv-call" style="background:#F8FAFC">{C.badge(row["classification"])} &nbsp;'
                f'<b>Risk {row["risk_score"]:.1f}</b> · evidence {html.escape(str(row["evidence_level"]))} · '
                f'reliability {row["reliability_score"]:.0f}%<br><span style="display:block;margin-top:6px">'
                f'{html.escape(row["explanation"])}</span></div>', unsafe_allow_html=True)
    X = load_operational("explanations")
    X = X[(X["zone_id"] == row["zone_id"]) & (X["shift"] == row["shift"]) & (X["observation_date"] == row["observation_date"])]
    c3, c4 = st.columns(2)
    with c3, C.card("contrib", "Contribution to risk score", "points · sorted"):
        C.chart(charts.contribution_bars(X), "ai_contrib")
    with c4, C.card("expl", "Component weights & z-scores", "saved explanation output"):
        show = X[["component", "status", "effective_weight", "signal_score", "contribution_points", "observed_value",
                  "baseline_median", "robust_z", "cusum", "staleness_hours"]].copy()
        show["component"] = show["component"].map(lambda c: T.CATEGORY_LABELS.get(c, "Isolation Forest"))
        show["status"] = show["status"].str.replace("_", " ").str.capitalize()
        C.table(C.humanize_frame(show), "ai_expl", height=260)
        C.download(X, f"prevenza_explanation_{row['zone_id']}_{pd.Timestamp(t):%Y%m%d%H%M}.csv", key="dl_expl")

    C.section("Illustrative operational suggestions")
    extra = []
    if isinstance(first["missing_sources"], str) and "clinical" in first["missing_sources"]:
        extra.append("Clinical data were unavailable at detection: confirm whether the clinic is reporting.")
    if in_ep["heat_index_validity"].eq("extrapolated").any():
        extra.append("Heat-index readings were extrapolated: cross-check with an independent instrument.")
    items = "".join(f"<li>{html.escape(a)}</li>" for a in extra + ACTIONS)
    C.callout("<b>Illustrative operational suggestions only — not medical instructions and not a clinical diagnosis.</b> "
              "Any real response must follow site occupational-health procedures and qualified professional judgement."
              f'<ul style="margin:8px 0 0 18px;padding:0">{items}</ul>', "warn", html_text=True)
