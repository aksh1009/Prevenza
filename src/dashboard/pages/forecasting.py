"""Page 4 - 24-hour forecasting (operational: forecast_results only; no targets)."""
from __future__ import annotations

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
    return "no forecast" if pd.isna(v) else ("FLAG" if bool(v) else "no flag")


def comparison_rows(r: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([
        {"Method": "Logistic regression forecast (24 h)", "Output": flag_text(r["forecast_flag"]),
         "Basis": f"ranking score {fmt_ranking_score(r['forecast_score'])} vs fixed threshold {r['threshold']:.3f}"},
        {"Method": "Persistence baseline (any warning)", "Output": "FLAG" if r["persistence_any_flag"] else "no flag",
         "Basis": "current class is any non-normal warning"},
        {"Method": "Persistence baseline (health alert)", "Output": "FLAG" if r["persistence_health_flag"] else "no flag",
         "Basis": "current class is Emerging/High health risk"},
        {"Method": "Current risk state", "Output": C.class_label(r["current_classification"]),
         "Basis": f"evidence {r['evidence_level']}"},
    ])


def render(f: Filters) -> None:
    C.header("24-hour forecasting", "Will a site-zone enter an emerging or escalating heat-risk state within "
                                    "the next 24 hours? Saved forecasts only — the model is not refitted here.")
    C.caution(T.FORECAST_INTERPRETATION)
    C.limitation("Known limitations: " + " ".join(f"({i + 1}) {t}" for i, t in enumerate(LIMITATIONS)))

    FC = apply_filters(forecasts_with_baselines(), Filters(f.sites, f.zones, f.shifts, f.date_range), class_col=None)
    if FC.empty:
        C.empty_state("No forecasts match the current filters.")
        return
    reg = zone_registry().set_index("zone_id")["zone_name"]
    zones = sorted(FC["zone_id"].unique())
    focus = st.session_state.get("focus_zone")
    c1, c2 = st.columns(2)
    zone = c1.selectbox("Site-zone", zones, index=zones.index(focus) if focus in zones else 0, key="fc_zone",
                        format_func=lambda z: f"{z} · {reg[z]}")
    shifts = sorted(FC.loc[FC["zone_id"] == zone, "shift"].unique())
    shift = c2.selectbox("Shift", shifts, key="fc_shift")
    S = FC[(FC["zone_id"] == zone) & (FC["shift"] == shift)].sort_values("decision_time")
    times = list(S["decision_time"])
    latest_t = S.loc[S["is_latest"], "decision_time"]
    default_t = latest_t.iloc[0] if len(latest_t) else times[-1]
    t = st.select_slider("Forecast issued at (historical simulated decision time)", options=times, value=default_t,
                         format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"), key="fc_t")
    r = S[S["decision_time"] == t].iloc[0]

    k = st.columns(4)
    C.kpi(k[0], "Current risk classification", C.class_label(r["current_classification"]), f"evidence {r['evidence_level']}")
    C.kpi(k[1], "24 h forecast score", fmt_ranking_score(r["forecast_score"]), "uncalibrated ranking — not a probability")
    C.kpi(k[2], "Forecast flag", flag_text(r["forecast_flag"]), f"fixed threshold {r['threshold']:.3f}")
    C.kpi(k[3], "Latest forecast", "yes" if r["is_latest"] else "no", f"horizon to {C.fmt_ts(r['horizon_end'])}")
    k2 = st.columns(4)
    C.kpi(k2[0], "Eligibility", "eligible" if r["eligible"] else "not eligible", r["ineligibility_reason"] or "—")
    C.kpi(k2[1], "Inputs filled in (imputed)", int(r["n_inputs_missing"]), "training medians + missing indicators")
    C.kpi(k2[2], "Missing / stale sources",
          ", ".join(s for s in [r["missing_sources"], r["stale_sources"]] if isinstance(s, str) and s) or "none",
          "excluded from fusion, never zero")
    C.kpi(k2[3], "Sample status", "in-sample period" if r["in_sample"] else "out-of-sample period",
          "model was fitted on this period" if r["in_sample"] else "held-out period")
    st.write("")
    if r["heat_index_validity"] == "extrapolated":
        C.caution("Heat index input outside the NWS chart range (extrapolated): treat this forecast with extra caution.")
    st.markdown("**Forecast vs persistence vs current state** (the persistence baseline is always shown)")
    st.dataframe(comparison_rows(r), hide_index=True, key="fc_cmp")
    st.caption(f"Limitations recorded with this forecast: {r['limitations']}")

    st.plotly_chart(charts.forecast_timeline(S, float(r["threshold"]),
                                             f"Forecast ranking score and persistence flags · {zone} {shift} shift"),
                    theme=None, key="fc_tl")

    st.subheader("Latest forecasts — all zones in filter")
    L = FC[FC["is_latest"]].sort_values(["site_id", "zone_id", "shift"])
    tbl = pd.DataFrame({
        "Site": L["site_id"], "Zone": L["zone_id"], "Shift": L["shift"], "Issued": L["decision_time"],
        "Current state": L["current_classification"].map(C.class_label),
        "LR ranking score": L["forecast_score"].map(fmt_ranking_score),
        "LR forecast flag": L["forecast_flag"].map(flag_text),
        "Persistence (any warning) flag": L["persistence_any_flag"].map(lambda v: "FLAG" if v else "no flag"),
        "Eligible": L["eligible"], "Inputs imputed": L["n_inputs_missing"],
        "Missing sources": L["missing_sources"], "Stale sources": L["stale_sources"],
    })
    st.dataframe(tbl, hide_index=True, key="fc_latest")
    C.download(tbl, "prevenza_latest_forecasts.csv")
