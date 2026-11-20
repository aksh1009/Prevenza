"""Prevenza dashboard entry point.

    streamlit run app.py

Read-only: renders saved pipeline outputs. It never regenerates data, refits models or writes files.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import components as C
from src.dashboard import theme as T
from src.dashboard.data_loader import DataUnavailable, LeakageError, clear_cache, load_operational, status
from src.dashboard.filters import Filters
from src.dashboard.pages import PAGES, SLUGS

st.set_page_config(page_title="Prevenza — Synthetic Demonstrator", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
C.inject_css()


def sidebar() -> tuple[str, Filters]:
    with st.sidebar:
        C.brand()
        st.caption("Population & Workforce Health Early Warning · synthetic demonstrator")
        if "nav" not in st.session_state and st.query_params.get("page") in SLUGS:   # deep link: ?page=alerts
            st.session_state["nav"] = SLUGS[st.query_params["page"]]
        page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        st.markdown("**Filters**")
        try:
            A = load_operational("assessments")
        except DataUnavailable:
            return page, Filters()
        sites = st.multiselect("Site", sorted(A["site_id"].unique()), key="f_sites", placeholder="All sites")
        zone_opts = sorted(A.loc[A["site_id"].isin(sites) if sites else A.index, "zone_id"].unique())
        zones = st.multiselect("Zone", zone_opts, key="f_zones", placeholder="All zones")
        shifts = st.multiselect("Shift", ["day", "night"], key="f_shifts", placeholder="Both shifts")
        lo, hi = A["observation_date"].min().date(), A["observation_date"].max().date()
        dr = st.date_input("Observation dates (simulated period)", value=(lo, hi), min_value=lo, max_value=hi,
                           key="f_dates")
        classes = st.multiselect("Risk classification", T.CLASS_ORDER, key="f_classes", placeholder="All classes")
        st.caption("Filters only change what is displayed. There are no controls for model settings or thresholds.")
        st.divider()
        if st.button("Reload saved outputs", help="Re-read Parquet files after the pipeline has been re-run"):
            clear_cache()
            st.rerun()
        with st.expander("Data files"):
            s = status()
            st.dataframe(s[["dataset", "kind", "available"]], hide_index=True)
    date_range = tuple(dr) if isinstance(dr, (tuple, list)) and len(dr) == 2 else None
    return page, Filters(sites, zones, shifts, date_range, classes)


def main() -> None:
    page, filters = sidebar()
    C.disclaimer()
    try:
        PAGES[page].render(filters)
    except DataUnavailable as e:
        st.error(f"Required data is not available. {e}", icon="📂")
    except LeakageError as e:                                  # defensive: should never happen
        st.error(f"Blocked: {e}")


main()
