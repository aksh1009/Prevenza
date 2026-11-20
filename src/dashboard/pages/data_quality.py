"""Page 6 - Data quality and reliability (operational data; one clearly labelled simulation-reference section)."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.features.quality import NWS_MAX_HEAT_INDEX_C, NWS_MAX_TEMPERATURE_C, heat_index_validity

from .. import charts, components as C, theme as T
from ..data_loader import DataUnavailable, load_evaluation, load_operational
from ..filters import Filters, apply_filters, source_state

SOURCES = [("env", "heat_index_celsius", "Environmental"), ("wf", "workforce_count", "Workforce"),
           ("clin", "heat_related_visits", "Clinical"), ("abs", "absence_rate", "Absenteeism"),
           ("inc", "heat_incident_count", "Incidents")]
CAT_STATUS = {"available": "Valid", "stale": "Stale", "missing_data": "Missing", "no_baseline": "Insufficient baseline"}


def missingness_table(U: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, v, label in SOURCES:
        s = source_state(U, p, v).value_counts()
        rows.append({"source": label, "observations": len(U), "reported": int(s.get("reported", 0)),
                     "field missing (reported row)": int(s.get("field missing", 0)),
                     "not reported (whole period)": int(s.get("not reported", 0)),
                     "missing share": (len(U) - int(s.get("reported", 0))) / len(U) if len(U) else float("nan")})
    return pd.DataFrame(rows)


def taxonomy_strip() -> None:
    cols = st.columns(5)
    for col, (name, (_, desc)) in zip(cols, T.DATA_STATES.items()):
        ink, fill, bd, series, glyph = T.STATES[name]
        dashed = "border-style:dashed;" if name == "Missing" else ""
        col.markdown(f'<div class="pv-kpi" style="border-left:3px solid {series};{dashed}min-height:84px">'
                     f'<div class="top">{C.badge(name)}</div><div class="cap" style="margin-top:8px">{html.escape(desc)}'
                     f'</div></div>', unsafe_allow_html=True)


def render(f: Filters) -> None:
    taxonomy_strip()
    C.gap()
    C.callout("<b>Reliability score</b> = share of configured evidence weight that was available and current at the "
              "decision time (missing, stale and no-baseline streams contribute 0 weight — they are excluded, never "
              "scored as zero risk). Missing evidence is never treated as normal health status.", html_text=True)

    U = apply_filters(load_operational("unified"), Filters(f.sites, f.zones, f.shifts, f.date_range), class_col=None)
    A = apply_filters(load_operational("assessments"), Filters(f.sites, f.zones, f.shifts, f.date_range))
    Fe = apply_filters(load_operational("features"), Filters(f.sites, f.zones, f.shifts, f.date_range), class_col=None)
    if U.empty or A.empty:
        C.empty_state("No observations match the current filters", "Widen the filters or press Reset.")
        return

    daily = pd.DataFrame({label: U[f"{p}_reporting_status"].eq("reported").groupby(U["observation_date"]).mean()
                          for p, _, label in SOURCES})
    with C.card("avail", "Source availability over time", "share of zone-shifts reported (eventual reporting)"):
        C.chart(charts.availability_steps(daily), "dq_avail")

    mt = missingness_table(U)
    c1, c2 = st.columns(2)
    with c1, C.card("miss", "Missingness by source", "reported · field missing · not reported"):
        C.chart(charts.missingness_bars(mt), "dq_missbar")
        C.table(C.humanize_frame(mt), "dq_miss", formats={"Missing share": st.column_config.NumberColumn(format="%.3f")})
        C.download(mt, "prevenza_missingness_by_source.csv", key="dl_miss")
    with c2, C.card("lat", "Reporting latency by source", "hours from recording to report"):
        log = st.toggle("Log scale", value=False, key="dq_log_scale")
        lat = pd.concat([pd.DataFrame({"source": label, "delay": U[f"{p}_reporting_delay_hours"]})
                         for p, _, label in SOURCES])
        C.chart(charts.box_h(lat, "source", "delay", "hours", log=log), "dq_lat")

    c3, c4 = st.columns(2)
    with c3, C.card("stale", "Staleness at decision time", "age of latest known report (h)"):
        st_ = pd.concat([pd.DataFrame({"source": label, "staleness": Fe[f"{p}_staleness_hours"]})
                         for p, _, label in SOURCES])
        C.chart(charts.box_h(st_, "source", "staleness", "hours"), "dq_stale")
    with c4, C.card("status", "Fusion category status", "assessment counts"):
        status = pd.DataFrame({T.CATEGORY_LABELS[c]: A[f"{c}_status"].map(CAT_STATUS).value_counts()
                               for c in T.CATEGORIES}).reindex(list(CAT_STATUS.values())).fillna(0).astype(int)
        C.chart(charts.status_matrix(status), "dq_catstatus")

    c5, c6 = st.columns(2)
    with c5, C.card("ncat", "Available categories", "of 5 per assessment"):
        cnt = A["n_categories_available"].value_counts().reindex(range(6), fill_value=0)
        C.chart(charts.simple_bars([str(i) for i in cnt.index], cnt.values, T.ACCENT, "categories available",
                                   "assessments"), "dq_ncat")
    with c6, C.card("rel", "Evidence reliability distribution", "0–100%"):
        C.chart(charts.histogram(A["reliability_score"], T.BRAND_PRIMARY, "reliability score (%)"), "dq_rel")

    ins = A[A["classification"] == "Insufficient evidence"]
    with C.card("ins", "Insufficient-evidence observations",
                f"{len(ins)} of {len(A)} assessments · " +
                ", ".join(f"{k} {v}" for k, v in A["evidence_level"].value_counts().items())):
        show = ins[["observation_date", "site_id", "zone_id", "shift", "available_weight", "n_categories_available",
                    "missing_sources", "stale_sources", "evidence_level"]]
        if show.empty:
            C.empty_state("No insufficient-evidence assessments in the current filter")
        else:
            C.table(C.humanize_frame(show), "dq_ins", height=280)
            C.download(show, "prevenza_insufficient_evidence.csv", key="dl_ins")

    raw = U.assign(validity=heat_index_validity(U["env_temperature_celsius"], U["env_heat_index_celsius"]))
    x = raw[raw["validity"] == "extrapolated"][["observation_date", "site_id", "zone_id", "shift",
                                                "env_temperature_celsius", "env_relative_humidity",
                                                "env_heat_index_celsius", "validity"]]
    flagged = A[A["measurement_flags"] != ""]
    with C.card("hi", "Heat-index validity flags",
                f"NWS chart range: T ≤ {NWS_MAX_TEMPERATURE_C:.1f} °C · HI ≤ {NWS_MAX_HEAT_INDEX_C:.1f} °C"):
        if x.empty:
            C.empty_state("No extrapolated heat-index readings in the current filter")
        else:
            C.caution(f"{len(x)} raw heat-index reading(s) outside the NWS chart range in the current filter — shown "
                      "unaltered and flagged as invalid (extrapolated) measurements.")
            C.table(C.humanize_frame(x), "dq_hi")
            C.download(x, "prevenza_heat_index_flags.csv", key="dl_hi")
        st.markdown(f'<div class="pv-call">Assessments carrying a measurement flag: <b>{len(flagged)}</b> · health alerts '
                    f'coinciding with an extrapolated heat index: <b>{int(A["alert_relies_on_extrapolated_heat_index"].sum())}'
                    f'</b></div>', unsafe_allow_html=True)

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
            C.table(C.humanize_frame(dq.drop(columns=["issue_id"])), "dq_log")
        except DataUnavailable as e:
            C.empty_state("Data-quality log unavailable", str(e))
