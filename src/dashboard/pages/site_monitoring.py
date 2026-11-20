"""Page 2 - Site and zone monitoring (operational: no ground truth)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.features.quality import heat_index_validity

from .. import charts, components as C, theme as T
from ..data_loader import load_operational, zone_registry
from ..filters import Filters, apply_filters, source_state

SOURCES = [("env", "heat_index_celsius", "Environmental"), ("wf", "workforce_count", "Workforce"),
           ("clin", "heat_related_visits", "Clinical"), ("abs", "absence_rate", "Absenteeism"),
           ("inc", "heat_incident_count", "Incidents")]


def zone_context(zone: str, F: pd.DataFrame) -> dict:
    reg = zone_registry().set_index("zone_id").loc[zone]
    feats = F[F["zone_id"] == zone].iloc[0]
    return {"Site": f"{reg['site_id']} · {reg['site_name']}", "Zone": f"{zone} · {reg['zone_name']}",
            "Environment": reg["environment"], "Outdoor / semi-outdoor": "yes" if feats["ctx_outdoor"] else "no",
            "Radiant heat source": "yes" if feats["ctx_radiant_heat"] else "no",
            "Work intensity": reg["work_intensity"], "Shifts": "day 06:00–18:00 · night 18:00–06:00"}


def render(f: Filters) -> None:
    C.header("Site & zone monitoring", "Saved risk assessments and the source measurements behind them. "
                                       "Gaps in charts are unavailable data — never zero.")
    A = apply_filters(load_operational("assessments"), Filters(f.sites, f.zones, f.shifts, f.date_range))
    if A.empty:
        C.empty_state("No observations match the current filters.")
        return
    zones = sorted(A["zone_id"].unique())
    focus = st.session_state.get("focus_zone")
    zone = st.selectbox("Zone", zones, index=zones.index(focus) if focus in zones else 0, key="sm_zone",
                        format_func=lambda z: f"{z} · {zone_registry().set_index('zone_id').loc[z, 'zone_name']}")
    st.session_state["focus_zone"] = zone
    Z = A[A["zone_id"] == zone]
    F = load_operational("features")
    ctx = zone_context(zone, F)
    cols = st.columns([1.3, 1.3, 1, 1, 1, 1, 1.6])
    for c, (k, v) in zip(cols, ctx.items()):
        C.kpi(c, k, v, small=True)
    st.write("")

    st.plotly_chart(charts.risk_trend(Z), theme=None, key="sm_trend")
    C.class_legend()
    st.plotly_chart(charts.classification_timeline(Z), theme=None, key="sm_cls")
    st.plotly_chart(charts.category_scores(Z), theme=None, key="sm_cat")

    U = apply_filters(load_operational("unified"), Filters(f.sites, [zone], f.shifts, f.date_range), class_col=None)
    U = U.copy()
    U["hi_extrapolated"] = heat_index_validity(U["env_temperature_celsius"], U["env_heat_index_celsius"]) == "extrapolated"
    st.subheader("Source measurements (as reported)")
    st.plotly_chart(charts.measurement_lines(
        U, [("env_temperature_celsius", "air temperature °C", T.CATEGORY_COLORS["environmental"]),
            ("env_heat_index_celsius", "heat index °C (NWS approx., not WBGT)", "#0d366b")],
        "Environmental measurements", "°C", flag_col="hi_extrapolated"), theme=None, key="sm_env")
    if U["hi_extrapolated"].any():
        C.caution(f"{int(U['hi_extrapolated'].sum())} heat-index reading(s) in this view are outside the NWS chart "
                  "range (temperature > 43.3 °C or heat index > 58.3 °C). Values are shown unaltered but their "
                  "magnitude is an extrapolation and unreliable.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.measurement_lines(U, [("wf_workforce_count", "workers present", T.CATEGORY_COLORS["workforce_exposure"])],
                                                 "Workforce present", "workers"), theme=None, key="sm_wf1")
        st.plotly_chart(charts.measurement_lines(U, [("clin_heat_related_visits", "heat-related visits", T.CATEGORY_COLORS["clinical"]),
                                                     ("clin_heat_visits_severe", "severe visits", "#7a1414")],
                                                 "Clinical visits (aggregated counts)", "visits"), theme=None, key="sm_clin")
        st.plotly_chart(charts.measurement_lines(U, [("inc_heat_incident_count", "heat incidents", T.CATEGORY_COLORS["incidents"]),
                                                     ("inc_severe_incidents", "severe incidents", "#7a1414")],
                                                 "Incident indicators", "incidents"), theme=None, key="sm_inc")
    with c2:
        st.plotly_chart(charts.measurement_lines(U, [("wf_average_exposure_hours", "avg exposure hours", T.CATEGORY_COLORS["workforce_exposure"])],
                                                 "Workforce exposure hours (per present worker)", "hours"), theme=None, key="sm_wf2")
        st.plotly_chart(charts.measurement_lines(U, [("abs_absence_rate", "absence rate", T.CATEGORY_COLORS["absenteeism"])],
                                                 "Absenteeism (unplanned absence rate)", "fraction of rostered"), theme=None, key="sm_abs")
        states = pd.DataFrame({label: source_state(U, p, v).to_numpy() for p, v, label in SOURCES},
                              index=U["shift_start"] + pd.to_timedelta(0, unit="h"))
        states = states.sort_index()
        st.plotly_chart(charts.availability_strip(states, "Source availability (green reported · amber field "
                                                          "missing · grey not reported)"), theme=None, key="sm_avail")

    st.subheader("Observations")
    tbl = apply_filters(Z, Filters(classes=f.classes))[[
        "observation_date", "shift", "decision_time", "risk_score", "classification", "evidence_level",
        "reliability_score", "n_categories_available", "missing_sources", "stale_sources", "heat_index_validity",
        "environmental_score", "workforce_exposure_score", "clinical_score", "absenteeism_score", "incidents_score"]]
    if f.classes:
        st.caption("Classification filter applied to this table only; charts show the full timeline for context.")
    st.dataframe(tbl.assign(classification=tbl["classification"].map(C.class_label)), hide_index=True, key="sm_tbl")
    C.download(tbl, f"prevenza_{zone}_observations.csv")
