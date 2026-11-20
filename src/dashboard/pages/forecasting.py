"""Page 4 - 24-hour forecasting (operational: forecast_results only; no targets)."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .. import charts, components as C, theme as T
from ..data_loader import config_yaml, load_operational, zone_registry
from ..filters import Filters, apply_filters, fmt_ranking_score, persistence_any

LIMITATIONS = [
    "Poor transfer to the foundry scenario (an indoor radiant-heat zone unlike the training event).",
    "No demonstrated early-warning improvement over the simple persistence baseline.",
    "Limited number of independent simulated events (about one per data split).",
    "Requires prospective validation on real, independently labelled data before any operational use.",
]


def forecasts_with_baselines() -> pd.DataFrame:
    """Saved forecasts + 'any warning' persistence flag re-derived from the saved current class using the exact
    rule in config/forecast.yaml (persistence.any_classes)."""
    fc = load_operational("forecasts")
    any_classes = config_yaml("forecast.yaml")["persistence"]["any_classes"]
    fc["persistence_any_flag"] = persistence_any(fc["current_classification"], any_classes)
    return fc


def flag_text(v) -> str:
    return "No forecast" if pd.isna(v) else ("Flag" if bool(v) else "No flag")


def comparison_rows(r: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([
        {"Method": "Logistic regression forecast (24 h)", "Output": flag_text(r["forecast_flag"]),
         "Basis": f"ranking {fmt_ranking_score(r['forecast_score'])} vs fixed threshold {r['threshold']:.3f}"},
        {"Method": "Persistence baseline (any warning)", "Output": "Flag" if r["persistence_any_flag"] else "No flag",
         "Basis": "current class is any non-normal warning"},
        {"Method": "Persistence baseline (health alert)", "Output": "Flag" if r["persistence_health_flag"] else "No flag",
         "Basis": "current class is Emerging/High health risk"},
        {"Method": "Current risk state", "Output": C.class_label(r["current_classification"]),
         "Basis": f"evidence {r['evidence_level']}"},
    ])


def render(f: Filters) -> None:
    g1, g2 = st.columns([1, 1])
    with g1:
        C.callout(f"<b>{html.escape(T.FORECAST_INTERPRETATION)}</b>", "amber", html_text=True)
    with g2:
        with st.expander(f"Known model limitations ({len(LIMITATIONS)})", expanded=False):
            for i, t in enumerate(LIMITATIONS, 1):
                st.markdown(f"{i}. {t}")

    FC = apply_filters(forecasts_with_baselines(), Filters(f.sites, f.zones, f.shifts, f.date_range), class_col=None)
    if FC.empty:
        C.empty_state("No forecasts match the current filters", "Widen the filters or press Reset.")
        return
    reg = zone_registry().set_index("zone_id")["zone_name"]
    zones = sorted(FC["zone_id"].unique())
    focus = st.session_state.get("focus_zone")
    with C.card("insp", "Zone-shift inspection", "saved forecasts only — the model is not refitted here"):
        a, b, c = st.columns([1.2, 0.6, 2.2], vertical_alignment="bottom")
        zone = a.selectbox("Site-zone", zones, index=zones.index(focus) if focus in zones else 0, key="fc_zone",
                           format_func=lambda z: f"{z} · {reg[z]}")
        shifts = sorted(FC.loc[FC["zone_id"] == zone, "shift"].unique())
        shift = b.selectbox("Shift", shifts, key="fc_shift")
        S = FC[(FC["zone_id"] == zone) & (FC["shift"] == shift)].sort_values("decision_time")
        times = list(S["decision_time"])
        latest_t = S.loc[S["is_latest"], "decision_time"]
        default_t = latest_t.iloc[0] if len(latest_t) else times[-1]
        t = c.select_slider("Decision timestamp", options=times, value=default_t, key="fc_t",
                            format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"))
    r = S[S["decision_time"] == t].iloc[0]
    C.gap()

    k = st.columns(4)
    cur = r["current_classification"]
    C.kpi(k[0], "Current risk state", cur.replace(" health risk", "").replace("Elevated occupational ", "Elevated "),
          C.badge(cur), accent=T.CLASS_COLORS.get(cur), icon_name="shield-alert", small_value=True)
    C.kpi(k[1], "24 h forecast ranking", fmt_ranking_score(r["forecast_score"]),
          "uncalibrated ranking — not a probability", accent=T.STATES["Forecast"][3], icon_name="trending-up")
    C.kpi(k[2], "Forecast flag", flag_text(r["forecast_flag"]),
          f'<span class="mono">fixed threshold {r["threshold"]:.3f}</span>', icon_name="alert")
    C.kpi(k[3], "Latest forecast", "Yes" if r["is_latest"] else "No",
          f'<span class="mono">horizon → {C.fmt_ts(r["horizon_end"])}</span>', icon_name="clock")
    C.gap()
    k2 = st.columns(4)
    C.kpi(k2[0], "Eligibility", "Eligible" if r["eligible"] else "Not eligible",
          r["ineligibility_reason"] or "Post warm-up, adequate evidence", icon_name="check")
    C.kpi(k2[1], "Imputed inputs", int(r["n_inputs_missing"]), "training medians + missing indicators", icon_name="database")
    srcs = ", ".join(s for s in [r["missing_sources"], r["stale_sources"]] if isinstance(s, str) and s) or "None"
    C.kpi(k2[2], "Missing / stale sources", srcs, "excluded from fusion, never zero", icon_name="layers", small_value=True)
    C.kpi(k2[3], "Validation split", "In-sample" if r["in_sample"] else "Out-of-sample",
          "model was fitted on this period" if r["in_sample"] else "held-out period", icon_name="help", small_value=True)
    if r["heat_index_validity"] == "extrapolated":
        C.gap()
        C.caution("Heat index input outside the NWS chart range (extrapolated): treat this forecast with extra caution.")

    C.gap()
    c1, c2 = st.columns([5, 7])
    with c1, C.card("cmp", "Forecast vs persistence vs current state", "persistence always shown"):
        C.table(comparison_rows(r), "fc_cmp")
        st.caption(f"Recorded limitations: {r['limitations']}")
    with c2, C.card("tl", f"Ranking score & persistence flags · {zone} {shift}", "dashed = fixed threshold"):
        C.chart(charts.forecast_timeline(S, float(r["threshold"])), "fc_tl")

    with C.card("latest", "Latest forecasts — all zones in filter", "one row per zone-shift"):
        L = FC[FC["is_latest"]].sort_values(["site_id", "zone_id", "shift"])
        tbl = pd.DataFrame({
            "Site": L["site_id"], "Zone": L["zone_id"], "Shift": L["shift"], "Issued": L["decision_time"],
            "Current state": L["current_classification"].map(C.class_label),
            "LR ranking score": L["forecast_score"],
            "LR forecast flag": L["forecast_flag"].map(flag_text),
            "Persistence (any warning) flag": L["persistence_any_flag"].map(lambda v: "Flag" if v else "No flag"),
            "Eligible": L["eligible"].map(lambda v: "Yes" if v else "No"), "Imputed inputs": L["n_inputs_missing"],
            "Missing sources": L["missing_sources"], "Stale sources": L["stale_sources"],
        })
        C.table(tbl, "fc_latest", formats={"LR ranking score": st.column_config.NumberColumn(format="%.3f")})
        C.download(tbl, "prevenza_latest_forecasts.csv")
