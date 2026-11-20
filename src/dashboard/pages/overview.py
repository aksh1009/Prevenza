"""Page 1 - Executive overview (operational: no ground truth).

Hierarchy: status ledger (glance) -> zone status board (the dominant element: where to look) -> alert feed
(what changed) -> trend context -> composition -> full table.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .. import charts, components as C, theme as T
from ..data_loader import load_operational, zone_registry
from ..filters import Filters, apply_filters, data_flag_text, latest_per_zone, measurement_flag_text

ALERTING = ["High health risk", "Emerging health risk", "Elevated occupational exposure", "Environmental warning"]


def evidence_text(row) -> str:
    n = int(row["n_categories_available"])
    return f"{'■' * n}{'□' * (5 - n)}  {str(row['evidence_level']).capitalize()} ({n}/5)"


def latest_table(latest: pd.DataFrame) -> pd.DataFrame:
    reg = zone_registry()[["zone_id", "zone_name"]]
    t = latest.merge(reg, on="zone_id", how="left")
    return pd.DataFrame({
        "Site": t["site_id"], "Zone": t["zone_id"] + " · " + t["zone_name"], "Shift": t["shift"],
        "Decision time": t["decision_time"],
        "Risk score": t["risk_score"].round(1),
        "Classification": t["classification"].map(C.class_label),
        "Evidence": t.apply(evidence_text, axis=1),
        "Reliability": t["reliability_score"].round(0),
        "Data freshness": t.apply(data_flag_text, axis=1),
        "Measurement": t.apply(measurement_flag_text, axis=1),
    })


def zone_board(latest: pd.DataFrame) -> str:
    """3 x 3 control-room board: one row per site, one cell per zone, at the decision time."""
    reg = zone_registry()
    by_zone = latest.set_index("zone_id")
    rows = []
    for site_id, g in reg.groupby("site_id", sort=True):
        rows.append(f'<div class="site"><b>{site_id}</b><span>{html.escape(g["site_name"].iloc[0])}</span></div>')
        for z in g.itertuples():
            if z.zone_id not in by_zone.index:
                rows.append('<div class="pv-cell" style="background:var(--canvas)"></div>')
                continue
            r = by_zone.loc[z.zone_id]
            cls = r["classification"]
            quiet = cls == "Normal"                      # normal cells recede so exceptions carry the eye
            color = "#1F5135" if quiet else T.CLASS_COLORS[cls]
            score = "—" if pd.isna(r["risk_score"]) else f"{r['risk_score']:.0f}"
            flags = []
            if r["stale_sources"]:
                flags.append(f'<span class="flags">◷ stale {html.escape(r["stale_sources"].replace(";", ", "))}</span>')
            if r["missing_sources"]:
                flags.append(f'<span class="flags" style="color:#94A3B8">○ missing {html.escape(r["missing_sources"].replace(";", ", "))}</span>')
            if r["heat_index_validity"] == "extrapolated":
                flags.append('<span class="flags inv">⊘ HI extrapolated</span>')
            rows.append(
                f'<div class="pv-cell{" quiet" if quiet else ""}" style="border-left-color:{color}">'
                f'<div class="top"><span class="zn">{html.escape(z.zone_name)}</span><span class="zid">{z.zone_id} · {r["shift"]}</span></div>'
                f'<div class="score" style="color:{T.MUTED if quiet else (T.INK if cls == "Insufficient evidence" else color)}">{score}<small>/100</small></div>'
                f'<div class="foot">{C.badge(cls)}{C.evidence_bar(int(r["n_categories_available"]), 5, str(r["evidence_level"]))}</div>'
                f'<div class="foot">{"".join(flags)}</div></div>')
    return f'<div class="pv-board" role="table" aria-label="Zone status board">{"".join(rows)}</div>'


def alert_feed_items(period: pd.DataFrame, n: int = 8, max_gap_hours: int = 24) -> list[dict]:
    """Collapse consecutive warnings of the same class in the same zone into one story, newest first."""
    reg = zone_registry().set_index("zone_id")["zone_name"]
    a = period[period["classification"].isin(ALERTING)].sort_values(["zone_id", "decision_time"])
    runs = []
    for zone, g in a.groupby("zone_id"):
        new = (g["classification"] != g["classification"].shift()) |               (g["decision_time"].diff() > pd.Timedelta(hours=max_gap_hours))
        for _, r in g.groupby(new.cumsum()):
            runs.append(r)
    runs.sort(key=lambda r: r["decision_time"].max(), reverse=True)
    items = []
    for r in runs[:n]:
        last = r.iloc[-1]
        flags = []
        if r["missing_sources"].ne("").any():
            flags.append("missing " + ", ".join(sorted({s for v in r["missing_sources"] for s in v.split(";") if s})))
        if r["stale_sources"].ne("").any():
            flags.append("stale " + ", ".join(sorted({s for v in r["stale_sources"] for s in v.split(";") if s})))
        if r["heat_index_validity"].eq("extrapolated").any():
            flags.append("heat index extrapolated")
        span = (f"{r['decision_time'].min():%d %b %H:%M} → {r['decision_time'].max():%d %b %H:%M}"
                if len(r) > 1 else f"{last['decision_time']:%d %b %H:%M}")
        items.append({"zone": f"{reg.get(last['zone_id'], '')} · {last['zone_id']}", "badge": C.badge(last["classification"]),
                      "detail": f"{len(r)} assessment{'s' if len(r) > 1 else ''}, peak risk {r['risk_score'].max():.1f}"
                                + (f" — {'; '.join(flags)}" if flags else ""),
                      "meta": span})
    return items


def drivers_html(latest: pd.DataFrame) -> str:
    """Mean category score across zones as bullet rows; unavailable streams say so instead of drawing zero."""
    rows = []
    for c in T.CATEGORIES:
        v = latest[f"{c}_score"].mean()
        n_av = int((latest[f"{c}_status"] == "available").sum())
        color = T.CATEGORY_COLORS[c]
        fill = "" if pd.isna(v) else f'<i style="width:{max(v, 0.8):.1f}%;background:{color}"></i>'
        val = "n/a" if pd.isna(v) else f"{v:.0f}"
        rows.append(f'<div class="row"><span class="nm">{T.CATEGORY_LABELS[c]}</span><span class="trk">{fill}<b></b></span>'
                    f'<span class="v">{val}</span><span class="sub">{n_av}/{len(latest)} zones reporting</span></div>')
    return f'<div class="pv-drv">{"".join(rows)}</div>'


def render(f: Filters) -> None:
    A = load_operational("assessments")
    times = sorted(A["decision_time"].unique())
    default = times[-1]
    q = st.query_params.get("asof")                      # deep link: ?asof=2025-07-07T19:00
    if q and "ov_asof" not in st.session_state:
        try:
            want = pd.Timestamp(q)
            default = min(times, key=lambda t: abs(pd.Timestamp(t) - want))
        except (ValueError, TypeError):
            pass
    as_of = st.select_slider("Replay a decision time", options=times, value=default,
                             format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"), key="ov_asof")
    scope = apply_filters(A, Filters(sites=f.sites, zones=f.zones))
    latest = latest_per_zone(scope, as_of)
    if latest.empty:
        C.empty_state("No assessments match the current filters", "Widen the site or zone selection, or press Reset.")
        return
    cls = latest["classification"]
    n = lambda c: int((cls == c).sum())
    C.ledger([("High health risk", n("High health risk"), "High health risk"),
              ("Emerging health risk", n("Emerging health risk"), "Emerging health risk"),
              ("Elevated exposure", n("Elevated occupational exposure"), "Elevated occupational exposure"),
              ("Environmental warning", n("Environmental warning"), "Environmental warning"),
              ("Insufficient evidence", n("Insufficient evidence"), "Insufficient evidence"),
              ("Normal", n("Normal"), "Normal")],
             right=f"{scope['site_id'].nunique()} sites · {scope['zone_id'].nunique()} zones<br>"
                   f"decision {pd.Timestamp(latest['decision_time'].max()):%d %b %H:%M}")

    period = apply_filters(A, f)
    period = period[period["decision_time"] <= as_of]
    hero, side = st.columns([8.2, 3.8], gap="medium")
    with hero, st.container(key="hero"):
        C.section("Zone status", "latest decision per zone · left edge = classification · score /100")
        st.markdown(zone_board(latest), unsafe_allow_html=True)
    with side:
        C.section("What changed", "runs of warnings, newest first")
        items = alert_feed_items(period)
        if items:
            C.feed(items)
        else:
            C.empty_state("Quiet period", "No warning or health-risk assessment up to this decision time.")

    C.section("Context over the period")
    c1, c2 = st.columns([5, 7])
    with c1, C.card("trend", "Daily maximum risk by site", "dashed = elevated threshold 40"):
        C.chart(charts.site_risk_trend(period), "ov_trend")
    with c2, C.card("heat", "Zone × day risk", "daily maximum · darkest = no score"):
        C.chart(charts.zone_heatmap(period), "ov_heat")

    c4, c5, c6 = st.columns(3)
    with c4, C.card("dist", "Classification mix", f"{len(period):,} assessments in filter"):
        C.chart(charts.class_donut(period, f"<b>{len(period):,}</b>"), "ov_dist")
    with c5, C.card("site", "Share by site", "% of shift-observations"):
        C.chart(charts.site_comparison(period), "ov_site")
    with c6, C.card("drivers", "What is driving risk now", "mean score across zones · tick = 40"):
        st.markdown(drivers_html(latest), unsafe_allow_html=True)

    C.section("All zones at the decision time")
    tbl = latest_table(latest)
    C.table(tbl, "ov_table", formats={
        "Risk score": st.column_config.NumberColumn(format="%.1f"),
        "Reliability": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
        "Decision time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})
    a, b, c, d, e = st.columns([1.6, 1, 1, 1, 0.8], vertical_alignment="bottom")
    zone = a.selectbox("Open a zone in", list(latest["zone_id"]), key="ov_zone",
                       index=list(latest["zone_id"]).index(st.session_state["focus_zone"])
                       if st.session_state.get("focus_zone") in list(latest["zone_id"]) else 0)
    for col, page, label in ((b, "Site & zone monitoring", "Monitoring"), (c, "Alert investigation", "Alert history"),
                             (d, "24-hour forecasting", "24 h forecast")):
        col.button(label, key=f"go_{page}", on_click=_goto, args=(page, zone), use_container_width=True)
    with e:
        C.download(tbl, "prevenza_latest_zone_assessments.csv")


def _goto(page: str, zone: str) -> None:
    st.session_state["focus_zone"] = zone
    st.session_state["nav"] = page
