"""Page 1 - Executive overview (operational: no ground truth)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, components as C, theme as T
from ..data_loader import load_operational, zone_registry
from ..filters import Filters, apply_filters, data_flag_text, latest_per_zone, measurement_flag_text


def latest_table(latest: pd.DataFrame) -> pd.DataFrame:
    reg = zone_registry()[["zone_id", "zone_name"]]
    t = latest.merge(reg, on="zone_id", how="left")
    return pd.DataFrame({
        "Site": t["site_id"], "Zone": t["zone_id"] + " · " + t["zone_name"], "Shift": t["shift"],
        "Decision time": t["decision_time"],
        "Risk score": t["risk_score"].round(1),
        "Classification": t["classification"].map(C.class_label),
        "Evidence level": t["evidence_level"], "Reliability %": t["reliability_score"].round(0),
        "Missing / stale data": t.apply(data_flag_text, axis=1),
        "Measurement flags": t.apply(measurement_flag_text, axis=1),
    })


def render(f: Filters) -> None:
    A = load_operational("assessments")
    first, last = A["observation_date"].min(), A["observation_date"].max()
    C.header("Executive overview",
             f"Simulated observation period {first:%d %b %Y} – {last:%d %b %Y} · historical replay of saved "
             "assessments (not live data)")
    times = sorted(A["decision_time"].unique())
    as_of = st.select_slider("Assessment as of (historical simulated decision time)", options=times,
                             value=times[-1], format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"),
                             key="ov_asof")
    scope = apply_filters(A, Filters(sites=f.sites, zones=f.zones))
    latest = latest_per_zone(scope, as_of)
    if latest.empty:
        C.empty_state("No assessments match the current site/zone filters.")
        return

    cls = latest["classification"]
    r1 = st.columns(4)
    C.kpi(r1[0], "Monitored sites", scope["site_id"].nunique(), "synthetic sites")
    C.kpi(r1[1], "Monitored zones", scope["zone_id"].nunique(), "3 per site")
    C.kpi(r1[2], "Latest observation", pd.Timestamp(latest["decision_time"].max()).strftime("%d %b %Y %H:%M"),
          "decision time (historical, simulated)")
    C.kpi(r1[3], "Zones with insufficient evidence", int((cls == "Insufficient evidence").sum()),
          "not the same as normal")
    r2 = st.columns(4)
    C.kpi(r2[0], "■ Emerging health risk", int((cls == "Emerging health risk").sum()), "zones at this time")
    C.kpi(r2[1], "✖ High health risk", int((cls == "High health risk").sum()), "zones at this time")
    C.kpi(r2[2], "▲ Environmental warning", int((cls == "Environmental warning").sum()), "zones at this time")
    C.kpi(r2[3], "◆ Elevated occupational exposure", int((cls == "Elevated occupational exposure").sum()),
          "zones at this time")
    st.write("")
    C.class_legend()

    period = apply_filters(A, f)
    period = period[period["decision_time"] <= as_of]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.class_distribution(period, "Classification distribution (filtered period, up to as-of)"),
                        theme=None, key="ov_dist")
    with c2:
        st.plotly_chart(charts.site_comparison(period, "Site comparison (observations per class)"), theme=None,
                        key="ov_site")

    st.subheader("Latest zone assessments")
    st.caption("One row per zone: its most recent decision at or before the as-of time. Select a row to investigate.")
    tbl = latest_table(latest)
    sel = st.dataframe(tbl, hide_index=True, on_select="rerun", selection_mode="single-row", key="ov_table",
                       column_config={"Risk score": st.column_config.NumberColumn(format="%.1f"),
                                      "Decision time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})
    C.download(tbl, "prevenza_latest_zone_assessments.csv")
    rows = sel.selection.rows if sel and hasattr(sel, "selection") else []
    if rows:
        zone = latest.iloc[rows[0]]["zone_id"]
        st.session_state["focus_zone"] = zone
        st.markdown(f"**Selected:** {zone}")
        b = st.columns(3)
        for col, page in zip(b, ("Site & zone monitoring", "Alert investigation", "24-hour forecasting")):
            col.button(f"Open in {page}", key=f"go_{page}", on_click=_goto, args=(page,))


def _goto(page: str) -> None:
    st.session_state["nav"] = page
