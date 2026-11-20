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
            "Work intensity": reg["work_intensity"], "Shifts": "day 06–18 · night 18–06"}


def render(f: Filters) -> None:
    C.header("Site & zone monitoring", "Saved risk assessments and the source measurements behind them",
             chips=["Gaps = unavailable data, never zero"])
    A = apply_filters(load_operational("assessments"), Filters(f.sites, f.zones, f.shifts, f.date_range))
    if A.empty:
        C.empty_state("No observations match the current filters.")
        return
    zones = sorted(A["zone_id"].unique())
    focus = st.session_state.get("focus_zone")
    reg = zone_registry().set_index("zone_id")
    zone = st.selectbox("Zone", zones, index=zones.index(focus) if focus in zones else 0, key="sm_zone",
                        format_func=lambda z: f"{z} · {reg.loc[z, 'zone_name']}")
    st.session_state["focus_zone"] = zone
    Z = A[A["zone_id"] == zone]
    F = load_operational("features")
    ctx = zone_context(zone, F)
    cols = st.columns([1.4, 1.3, 1, 1, 1, 1, 1.3])
    for c, (k, v) in zip(cols, ctx.items()):
        C.kpi(c, k, v, small=True)
    st.write("")

    U = apply_filters(load_operational("unified"), Filters(f.sites, [zone], f.shifts, f.date_range), class_col=None)
    U = U.copy()
    U["hi_extrapolated"] = heat_index_validity(U["env_temperature_celsius"], U["env_heat_index_celsius"]) == "extrapolated"

    c1, c2 = st.columns([2.4, 1])
    with c1, C.panel("trend", "Risk score", "0-100 evidence-weighted index · day solid, night dotted"):
        st.plotly_chart(charts.risk_trend(Z, ""), theme=None, key="sm_trend")
    with c2, C.panel("gauge", "Latest heat index in view", "measurement · NWS bands"):
        known = U.dropna(subset=["env_heat_index_celsius"]).sort_values("shift_start")
        if known.empty:
            C.empty_state("No heat-index reading available in the current view.")
        else:
            last = known.iloc[-1]
            validity = "extrapolated" if last["hi_extrapolated"] else "within_range"
            st.plotly_chart(charts.heat_index_gauge(last["env_heat_index_celsius"], validity,
                                                    f"{pd.Timestamp(last['shift_start']):%d %b %H:%M} · {last['shift']}"),
                            theme=None, key="sm_gauge")
            band = next(lbl for a, b, _, lbl in charts.HI_BANDS if a <= min(max(last["env_heat_index_celsius"], 20), 69.9) < b)
            st.caption(f"NWS band: **{band}** · air temperature {last['env_temperature_celsius']:.1f} °C · "
                       f"RH {last['env_relative_humidity']:.0f}% · heat index is not WBGT"
                       if pd.notna(last["env_temperature_celsius"]) and pd.notna(last["env_relative_humidity"])
                       else f"NWS band: **{band}**")
            if validity == "extrapolated":
                C.caution("This reading is outside the NWS chart range (white marker at 58.3 °C): extrapolated, "
                          "magnitude unreliable.")

    C.class_legend()
    with C.panel("cls", "Risk classification timeline"):
        st.plotly_chart(charts.classification_timeline(Z, ""), theme=None, key="sm_cls")
    with C.panel("cat", "Category scores", "gaps = unavailable · dashed = elevated threshold 40 (fixed)"):
        st.plotly_chart(charts.category_scores(Z, ""), theme=None, key="sm_cat")

    with C.panel("env", "Environmental measurements (as reported)", "✕ = heat index outside NWS chart range"):
        st.plotly_chart(charts.measurement_lines(
            U, [("env_temperature_celsius", "air temperature °C", T.CATEGORY_COLORS["environmental"]),
                ("env_heat_index_celsius", "heat index °C (NWS approx., not WBGT)", "#fb923c")],
            "", "°C", flag_col="hi_extrapolated"), theme=None, key="sm_env")
        if U["hi_extrapolated"].any():
            C.caution(f"{int(U['hi_extrapolated'].sum())} heat-index reading(s) in this view are outside the NWS chart "
                      "range (temperature > 43.3 °C or heat index > 58.3 °C). Values are shown unaltered but their "
                      "magnitude is an extrapolation and unreliable.")
    c3, c4 = st.columns(2)
    with c3:
        with C.panel("wf1", "Workforce present"):
            st.plotly_chart(charts.measurement_lines(U, [("wf_workforce_count", "workers present", T.CATEGORY_COLORS["workforce_exposure"])],
                                                     "", "workers"), theme=None, key="sm_wf1")
        with C.panel("clin", "Clinical visits (aggregated counts)"):
            st.plotly_chart(charts.measurement_lines(U, [("clin_heat_related_visits", "heat-related visits", T.CATEGORY_COLORS["clinical"]),
                                                         ("clin_heat_visits_severe", "severe visits", "#f87171")],
                                                     "", "visits"), theme=None, key="sm_clin")
        with C.panel("inc", "Incident indicators"):
            st.plotly_chart(charts.measurement_lines(U, [("inc_heat_incident_count", "heat incidents", T.CATEGORY_COLORS["incidents"]),
                                                         ("inc_severe_incidents", "severe incidents", "#f87171")],
                                                     "", "incidents"), theme=None, key="sm_inc")
    with c4:
        with C.panel("wf2", "Workforce exposure hours (per present worker)"):
            st.plotly_chart(charts.measurement_lines(U, [("wf_average_exposure_hours", "avg exposure hours", T.CATEGORY_COLORS["workforce_exposure"])],
                                                     "", "hours"), theme=None, key="sm_wf2")
        with C.panel("abs", "Absenteeism (unplanned absence rate)"):
            st.plotly_chart(charts.measurement_lines(U, [("abs_absence_rate", "absence rate", T.CATEGORY_COLORS["absenteeism"])],
                                                     "", "fraction of rostered"), theme=None, key="sm_abs")
        with C.panel("avail", "Source availability", "green reported · amber field missing · grey not reported"):
            states = pd.DataFrame({label: source_state(U, p, v).to_numpy() for p, v, label in SOURCES},
                                  index=U["shift_start"]).sort_index()
            st.plotly_chart(charts.availability_strip(states, ""), theme=None, key="sm_avail")

    with C.panel("obs", "Observations", "download available"):
        tbl = apply_filters(Z, Filters(classes=f.classes))[[
            "observation_date", "shift", "decision_time", "risk_score", "classification", "evidence_level",
            "reliability_score", "n_categories_available", "missing_sources", "stale_sources", "heat_index_validity",
            "environmental_score", "workforce_exposure_score", "clinical_score", "absenteeism_score", "incidents_score"]]
        if f.classes:
            st.caption("Classification filter applied to this table only; charts show the full timeline for context.")
        st.dataframe(tbl.assign(classification=tbl["classification"].map(C.class_label)), hide_index=True, key="sm_tbl")
        C.download(tbl, f"prevenza_{zone}_observations.csv")
