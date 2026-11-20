"""Page 1 - Executive overview (operational: no ground truth)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, components as C, theme as T
from ..data_loader import load_operational, zone_registry
from ..filters import Filters, apply_filters, data_flag_text, latest_per_zone, measurement_flag_text

ALERTING = ["High health risk", "Emerging health risk", "Elevated occupational exposure", "Environmental warning"]


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


def alert_feed_items(period: pd.DataFrame, n: int = 10) -> list[dict]:
    reg = zone_registry().set_index("zone_id")["zone_name"]
    a = period[period["classification"].isin(ALERTING)].sort_values("decision_time", ascending=False).head(n)
    items = []
    for r in a.itertuples():
        extra = []
        if r.missing_sources:
            extra.append("missing " + r.missing_sources.replace(";", ", "))
        if r.stale_sources:
            extra.append("stale " + r.stale_sources.replace(";", ", "))
        if r.heat_index_validity == "extrapolated":
            extra.append("heat index extrapolated")
        items.append({"color": T.CLASS_COLORS[r.classification], "symbol": T.CLASS_ICONS[r.classification],
                      "title": r.classification,
                      "detail": f"{r.zone_id} · {reg.get(r.zone_id, '')} · risk {r.risk_score:.1f} · evidence "
                                f"{r.evidence_level}" + (f" · {'; '.join(extra)}" if extra else ""),
                      "meta": f"{pd.Timestamp(r.decision_time):%d %b %Y %H:%M} · {r.shift} shift"})
    return items


def render(f: Filters) -> None:
    A = load_operational("assessments")
    first, last = A["observation_date"].min(), A["observation_date"].max()
    C.header("Executive overview", "Simulated operational heat-health risk across sites, zones and shifts",
             chips=[f"Simulated period {first:%d %b} – {last:%d %b %Y}", "Historical replay · not live data"])
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
    n = lambda c: int((cls == c).sum())

    r1 = st.columns(4)
    C.kpi(r1[0], "Monitored sites", scope["site_id"].nunique(), "synthetic sites", icon_name="map", color="#3b82f6")
    C.kpi(r1[1], "Monitored zones", scope["zone_id"].nunique(), "3 per site", icon_name="layers", color="#3b82f6")
    C.kpi(r1[2], "Latest decision time", pd.Timestamp(latest["decision_time"].max()).strftime("%d %b %Y %H:%M"),
          "historical, simulated", icon_name="clock", color="#94a3b8")
    C.kpi(r1[3], "Insufficient evidence", n("Insufficient evidence"), "zones · not the same as normal",
          icon_name="help", color=T.CLASS_COLORS["Insufficient evidence"])
    st.write("")
    r2 = st.columns(4)
    C.kpi(r2[0], "✖ High health risk", n("High health risk"), "zones at this time", icon_name="alert",
          color=T.CLASS_COLORS["High health risk"])
    C.kpi(r2[1], "■ Emerging health risk", n("Emerging health risk"), "zones at this time", icon_name="pulse",
          color=T.CLASS_COLORS["Emerging health risk"])
    C.kpi(r2[2], "▲ Environmental warning", n("Environmental warning"), "zones at this time", icon_name="thermo",
          color=T.CLASS_COLORS["Environmental warning"])
    C.kpi(r2[3], "◆ Elevated occupational exposure", n("Elevated occupational exposure"), "zones at this time",
          icon_name="users", color=T.CLASS_COLORS["Elevated occupational exposure"])
    st.write("")

    period = apply_filters(A, f)
    period = period[period["decision_time"] <= as_of]
    c1, c2, c3 = st.columns([1.15, 1.15, 0.9])
    with c1, C.panel("trend", "Risk trend by site", "daily maximum, 0-100 index"):
        st.plotly_chart(charts.site_risk_trend(period, ""), theme=None, key="ov_trend")
    with c2, C.panel("heat", "Risk heatmap (zones × days)", "blank = no score"):
        st.plotly_chart(charts.zone_heatmap(period, ""), theme=None, key="ov_heat")
    with c3, C.panel("feed", "Alerts & notifications", "latest first"):
        items = alert_feed_items(period)
        if items:
            C.feed(items)
        else:
            C.empty_state("No warnings or health-risk assessments in the selected period.")

    c4, c5, c6 = st.columns(3)
    with c4, C.panel("dist", "Classification distribution", "filtered period"):
        st.plotly_chart(charts.class_donut(period, "", f"{len(period)}<br>assessments"), theme=None, key="ov_dist")
    with c5, C.panel("site", "Site comparison", "observations per class"):
        st.plotly_chart(charts.site_comparison(period, ""), theme=None, key="ov_site")
    with c6, C.panel("drivers", "Category drivers (as of)", "mean across zones"):
        means = {c: latest[f"{c}_score"].mean() for c in T.CATEGORIES}
        avail = {c: f"{int((latest[f'{c}_status'] == 'available').sum())}/{len(latest)} zones"
                 for c in T.CATEGORIES}
        st.plotly_chart(charts.category_bars(pd.Series(means), avail, ""), theme=None, key="ov_drivers")
    C.class_legend()

    with C.panel("table", "Latest zone assessments", "select a row to investigate"):
        st.caption("One row per zone: its most recent decision at or before the as-of time.")
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
