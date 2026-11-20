"""Page 6 - Data quality and reliability (operational data; one clearly labelled simulation-reference section)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.quality import NWS_MAX_HEAT_INDEX_C, NWS_MAX_TEMPERATURE_C, heat_index_validity

from .. import charts, components as C, theme as T
from ..data_loader import DataUnavailable, load_evaluation, load_operational
from ..filters import Filters, apply_filters, source_state

SOURCES = [("env", "heat_index_celsius", "Environmental"), ("wf", "workforce_count", "Workforce"),
           ("clin", "heat_related_visits", "Clinical"), ("abs", "absence_rate", "Absenteeism"),
           ("inc", "heat_incident_count", "Incidents")]
CAT_STATUS = {"available": "Valid / available", "missing_data": "Missing", "no_baseline": "Insufficient baseline",
              "stale": "Stale"}


def missingness_table(U: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, v, label in SOURCES:
        s = source_state(U, p, v).value_counts()
        rows.append({"source": label, "observations": len(U), "reported": int(s.get("reported", 0)),
                     "field missing (reported row)": int(s.get("field missing", 0)),
                     "not reported (whole period)": int(s.get("not reported", 0)),
                     "missing share": (len(U) - int(s.get("reported", 0))) / len(U) if len(U) else float("nan")})
    return pd.DataFrame(rows)


def render(f: Filters) -> None:
    C.header("Data quality & reliability", "How complete, timely and valid is the evidence behind each assessment?")
    cols = st.columns(len(T.DATA_STATES))
    for c, (name, (color, desc)) in zip(cols, T.DATA_STATES.items()):
        c.markdown(f'<div class="pv-card"><div class="k" style="color:{color}">■ {name}</div>'
                   f'<div class="n">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    C.note("Missing evidence is never treated as normal health status, and missing values are never replaced with "
           "zero. A risk score computed from fewer available categories has lower reliability: the reliability score "
           "is the share of configured evidence weight that was actually available and current.")

    U = apply_filters(load_operational("unified"), Filters(f.sites, f.zones, f.shifts, f.date_range), class_col=None)
    A = apply_filters(load_operational("assessments"), Filters(f.sites, f.zones, f.shifts, f.date_range))
    Fe = apply_filters(load_operational("features"), Filters(f.sites, f.zones, f.shifts, f.date_range), class_col=None)
    if U.empty or A.empty:
        C.empty_state("No observations match the current filters.")
        return

    # availability over time
    daily = pd.DataFrame({label: U[f"{p}_reporting_status"].eq("reported").groupby(U["observation_date"]).mean()
                          for p, _, label in SOURCES})
    fig = go.Figure()
    for label, color in zip(daily.columns, T.CATEGORY_COLORS.values()):
        fig.add_scatter(x=daily.index, y=daily[label], mode="lines", name=label, line=dict(color=color, width=2))
    fig.update_layout(**T.PLOT_LAYOUT)
    fig.update_layout(height=300, title=dict(text="Source availability over time (share of zone-shifts reported, "
                      "eventual reporting)", x=0), yaxis=dict(range=[0, 1.05], gridcolor=T.BORDER))
    st.plotly_chart(fig, theme=None, key="dq_avail")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missingness by source**")
        mt = missingness_table(U)
        st.dataframe(mt.round(3), hide_index=True, key="dq_miss")
        C.download(mt, "prevenza_missingness_by_source.csv", key="dl_miss")
    with c2:
        lat = pd.concat([pd.DataFrame({"source": label, "delay": U[f"{p}_reporting_delay_hours"]})
                         for p, _, label in SOURCES])
        st.plotly_chart(charts.box_by(lat, "source", "delay", "Reporting latency by source (recorded → reported)",
                                      "hours"), theme=None, key="dq_lat")

    c3, c4 = st.columns(2)
    with c3:
        st_ = pd.concat([pd.DataFrame({"source": label, "staleness": Fe[f"{p}_staleness_hours"]})
                         for p, _, label in SOURCES])
        st.plotly_chart(charts.box_by(st_, "source", "staleness", "Staleness at decision time (age of latest known "
                                      "report)", "hours"), theme=None, key="dq_stale")
    with c4:
        status = pd.DataFrame({T.CATEGORY_LABELS[c]: A[f"{c}_status"].map(CAT_STATUS).value_counts()
                               for c in T.CATEGORIES}).fillna(0).astype(int)
        st.markdown("**Fusion category status (assessments)**")
        st.dataframe(status, key="dq_catstatus")

    c5, c6 = st.columns(2)
    with c5:
        cnt = A["n_categories_available"].value_counts().reindex(range(6), fill_value=0)
        fig = go.Figure(go.Bar(x=[str(i) for i in cnt.index], y=cnt.values, marker_color=T.CATEGORY_COLORS["environmental"],
                               hovertemplate="%{x} categories: %{y}<extra></extra>"))
        fig.update_layout(**T.PLOT_LAYOUT, height=280, title=dict(text="Number of available categories (of 5)", x=0),
                          xaxis_title="categories available", yaxis_title="assessments")
        st.plotly_chart(fig, theme=None, key="dq_ncat")
    with c6:
        fig = go.Figure(go.Histogram(x=A["reliability_score"], nbinsx=20, marker_color=T.INK_2))
        fig.update_layout(**T.PLOT_LAYOUT, height=280, title=dict(text="Evidence reliability distribution", x=0),
                          xaxis_title="reliability score (%)", yaxis_title="assessments")
        st.plotly_chart(fig, theme=None, key="dq_rel")

    st.subheader("Insufficient-evidence observations")
    ins = A[A["classification"] == "Insufficient evidence"]
    st.markdown(f"**{len(ins)}** of {len(A)} assessments in the filter are classified *Insufficient evidence* "
                f"(evidence level counts: " + ", ".join(f"{k} {v}" for k, v in A["evidence_level"].value_counts().items()) + ").")
    st.dataframe(ins[["observation_date", "site_id", "zone_id", "shift", "available_weight", "n_categories_available",
                      "missing_sources", "stale_sources", "evidence_level"]], hide_index=True, key="dq_ins")

    st.subheader("Heat-index validity flags")
    st.caption(f"NWS chart range: air temperature ≤ {NWS_MAX_TEMPERATURE_C:.1f} °C and heat index ≤ "
               f"{NWS_MAX_HEAT_INDEX_C:.1f} °C. Values outside are shown unaltered and flagged as invalid "
               "(extrapolated) measurements.")
    raw = U.assign(validity=heat_index_validity(U["env_temperature_celsius"], U["env_heat_index_celsius"]))
    x = raw[raw["validity"] == "extrapolated"][["observation_date", "site_id", "zone_id", "shift",
                                                "env_temperature_celsius", "env_relative_humidity",
                                                "env_heat_index_celsius", "validity"]]
    if x.empty:
        st.success("No extrapolated heat-index readings in the current filter.")
    else:
        C.caution(f"{len(x)} raw heat-index reading(s) outside the NWS chart range in the current filter.")
        st.dataframe(x, hide_index=True, key="dq_hi")
    flagged = A[A["measurement_flags"] != ""]
    st.markdown(f"Assessments carrying a measurement flag: **{len(flagged)}**; health alerts coinciding with an "
                f"extrapolated heat index: **{int(A['alert_relies_on_extrapolated_heat_index'].sum())}**.")

    with st.expander("Simulation reference: injected sensor faults and reporting issues (simulation ground truth)"):
        st.caption("This list comes from the simulator's data-quality log. A real deployment would not know these "
                   "faults in advance; it is shown so reviewers can see which anomalies were deliberately injected.")
        try:
            dq = load_evaluation("dq_log")
            if f.sites:
                dq = dq[dq["site_id"].isin(f.sites)]
            if f.zones:
                dq = dq[dq["zone_id"].isin(f.zones)]
            # health-event identifiers are evaluation information: remove them from this operational page
            dq = dq.assign(details=dq["details"].str.replace(r"scenario EVT-\d+:\s*", "", regex=True))
            st.dataframe(dq.drop(columns=["issue_id"]), hide_index=True, key="dq_log")
        except DataUnavailable as e:
            C.empty_state(str(e))
