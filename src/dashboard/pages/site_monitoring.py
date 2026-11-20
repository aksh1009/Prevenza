"""Page 2 - Site and zone monitoring (operational: no ground truth)."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.features.quality import NWS_MAX_HEAT_INDEX_C, heat_index_validity

from .. import charts, components as C, theme as T
from ..data_loader import load_operational, zone_registry
from ..filters import Filters, apply_filters, source_state

SOURCES = [("env", "heat_index_celsius", "Environmental"), ("wf", "workforce_count", "Workforce"),
           ("clin", "heat_related_visits", "Clinical"), ("abs", "absence_rate", "Absenteeism"),
           ("inc", "heat_incident_count", "Incidents")]


def zone_context(zone: str, F: pd.DataFrame) -> dict:
    reg = zone_registry().set_index("zone_id").loc[zone]
    feats = F[F["zone_id"] == zone].iloc[0]
    return {"Site": f"{reg['site_id']} · {reg['site_name']}", "Environment": reg["environment"].capitalize(),
            "Outdoor / semi-outdoor": "Yes" if feats["ctx_outdoor"] else "No",
            "Radiant heat": "Yes" if feats["ctx_radiant_heat"] else "No",
            "Work intensity": reg["work_intensity"].capitalize(), "Shift schedule": "06–18 / 18–06"}


def profile_strip(ctx: dict, active: str) -> None:
    cells = "".join(f'<div><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'
                    for k, v in list(ctx.items())[:5])
    cells += f'<div><div class="k">Active evidence</div><div class="v">{active}</div></div>'
    st.markdown(f'<div class="pv-prof">{cells}</div>', unsafe_allow_html=True)


def thermal_card(last: pd.Series) -> None:
    v = float(last["env_heat_index_celsius"])
    lo, hi = 15.0, 60.0
    segs = "".join(f'<div class="seg" style="width:{(b - a) / (hi - lo) * 100:.2f}%;background:{c}"></div>'
                   for a, b, c, _ in charts.HI_BANDS)
    pos = (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    ticks = "".join(f'<span class="tick" style="left:{(t - lo) / (hi - lo) * 100:.2f}%">{t}°</span>'
                    for t in (27, 32, 39, 51))
    band = charts.nws_band(v)
    extrap = last["hi_extrapolated"]
    pill = C.badge("Invalid measurement", "Outside NWS chart range") if extrap else \
        f'<span class="pv-badge" style="color:#334155;background:#F1F5F9;border-color:#CBD5E1">NWS · {band}</span>'
    t, rh = last["env_temperature_celsius"], last["env_relative_humidity"]
    meta = (f"Air {t:.1f} °C · RH {rh:.0f}%" if pd.notna(t) and pd.notna(rh) else "Air temperature / RH unavailable")
    st.markdown(f'<div class="pv-therm"><div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span class="big">{v:.1f} °C</span>{pill}</div>'
                f'<div class="meta">{meta} · {pd.Timestamp(last["shift_start"]):%Y-%m-%d %H:%M} {last["shift"]}</div>'
                f'<div class="pv-bullet">{segs}<div class="mk" style="left:{pos:.2f}%"><span>{v:.1f}</span></div>{ticks}</div>'
                f'<div class="meta" style="color:{T.MUTED}">Heat index (NWS approximation, not WBGT). Bands: caution 27 · '
                f'extreme caution 32 · danger 39 · extreme danger 51 °C. Chart range ends at {NWS_MAX_HEAT_INDEX_C:.1f} °C.</div>'
                f'</div>', unsafe_allow_html=True)


def render(f: Filters) -> None:
    A = apply_filters(load_operational("assessments"), Filters(f.sites, f.zones, f.shifts, f.date_range))
    if A.empty:
        C.empty_state("No observations match the current filters", "Widen the filters or press Reset.")
        return
    zones = sorted(A["zone_id"].unique())
    focus = st.session_state.get("focus_zone")
    reg = zone_registry().set_index("zone_id")
    c0, _ = st.columns([1, 2])
    zone = c0.selectbox("Zone", zones, index=zones.index(focus) if focus in zones else 0, key="sm_zone",
                        format_func=lambda z: f"{z} · {reg.loc[z, 'zone_name']}")
    st.session_state["focus_zone"] = zone
    Z = A[A["zone_id"] == zone]
    F = load_operational("features")
    lz = Z.sort_values("decision_time").iloc[-1]
    profile_strip(zone_context(zone, F), f"{int(lz['n_categories_available'])}/5 streams")
    C.gap()

    U = apply_filters(load_operational("unified"), Filters(f.sites, [zone], f.shifts, f.date_range), class_col=None).copy()
    U["hi_extrapolated"] = heat_index_validity(U["env_temperature_celsius"], U["env_heat_index_celsius"]) == "extrapolated"

    c1, c2 = st.columns([8, 4])
    with c1, C.card("trend", "Fused risk score trajectory", "0–100 index · day solid · night dotted"):
        C.chart(charts.risk_trend(Z), "sm_trend")
    with c2, C.card("therm", "Latest thermal reading", "last reported heat index in view"):
        known = U.dropna(subset=["env_heat_index_celsius"]).sort_values("shift_start")
        if known.empty:
            C.empty_state("No heat-index reading", "No environmental report in the current view.")
        else:
            thermal_card(known.iloc[-1])

    with C.card("cls", "Classification timeline", "day / night tiles · pale = no assessment"):
        C.class_legend()
        C.chart(charts.classification_tiles(Z), "sm_cls")
    with C.card("cat", "Category scores", "five synchronised streams · gaps = unavailable · dashed = 40"):
        C.chart(charts.category_small_multiples(Z), "sm_cat")
    with C.card("env", "Environmental measurements (as reported)", "⊘ = heat index outside NWS chart range"):
        C.chart(charts.measurement_lines(
            U, [("env_temperature_celsius", "Air temperature °C", T.CATEGORY_COLORS["environmental"]),
                ("env_heat_index_celsius", "Heat index °C (NWS approx., not WBGT)", "#EA580C")],
            "°C", flag_col="hi_extrapolated", height=280), "sm_env")
        if U["hi_extrapolated"].any():
            C.caution(f"{int(U['hi_extrapolated'].sum())} heat-index reading(s) in this view are outside the NWS chart "
                      "range (temperature > 43.3 °C or heat index > 58.3 °C). Values are shown unaltered but their "
                      "magnitude is an extrapolation and unreliable.")
    c3, c4 = st.columns(2)
    with c3:
        with C.card("wf1", "Workforce present", "workers per shift"):
            C.chart(charts.measurement_lines(U, [("wf_workforce_count", "Workers present", T.CATEGORY_COLORS["workforce_exposure"])],
                                             "workers"), "sm_wf1")
        with C.card("clin", "Clinical visits", "aggregated heat-related counts"):
            C.chart(charts.measurement_lines(U, [("clin_heat_related_visits", "Heat-related visits", T.CATEGORY_COLORS["clinical"]),
                                                 ("clin_heat_visits_severe", "Severe visits", "#F87171")], "visits"), "sm_clin")
        with C.card("inc", "Incident indicators", "heat-related incidents"):
            C.chart(charts.measurement_lines(U, [("inc_heat_incident_count", "Heat incidents", T.CATEGORY_COLORS["incidents"]),
                                                 ("inc_severe_incidents", "Severe incidents", "#F87171")], "incidents"), "sm_inc")
    with c4:
        with C.card("wf2", "Workforce exposure hours", "per present worker"):
            C.chart(charts.measurement_lines(U, [("wf_average_exposure_hours", "Average exposure hours", T.CATEGORY_COLORS["workforce_exposure"])],
                                             "hours"), "sm_wf2")
        with C.card("abs", "Absenteeism", "unplanned absence rate"):
            C.chart(charts.measurement_lines(U, [("abs_absence_rate", "Absence rate", T.CATEGORY_COLORS["absenteeism"])],
                                             "fraction of rostered"), "sm_abs")
        with C.card("avail", "Source availability", "green reported · amber field missing · grey not reported"):
            states = pd.DataFrame({label: source_state(U, p, v).to_numpy() for p, v, label in SOURCES},
                                  index=U["shift_start"]).sort_index()
            C.chart(charts.availability_strip(states), "sm_avail")

    with C.card("obs", "Observations", "selected zone · filtered"):
        tbl = apply_filters(Z, Filters(classes=f.classes))[[
            "observation_date", "shift", "decision_time", "risk_score", "classification", "evidence_level",
            "reliability_score", "n_categories_available", "missing_sources", "stale_sources", "heat_index_validity",
            "environmental_score", "workforce_exposure_score", "clinical_score", "absenteeism_score", "incidents_score"]]
        if f.classes:
            st.caption("Classification filter applied to this table only; charts show the full timeline for context.")
        show = C.humanize_frame(tbl.assign(classification=tbl["classification"].map(C.class_label)))
        C.table(show, "sm_tbl", height=320)
        C.download(tbl, f"prevenza_{zone}_observations.csv")
