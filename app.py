"""Prevenza dashboard entry point.

    streamlit run app.py

Read-only: renders saved pipeline outputs. It never regenerates data, refits models or writes files.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.dashboard import components as C
from src.dashboard.data_loader import DataUnavailable, LeakageError, clear_cache, load_operational, status
from src.dashboard.filters import Filters
from src.dashboard.pages import META, PAGES, SLUGS
from src.dashboard.ribbon import filter_ribbon

st.set_page_config(page_title="Prevenza — Synthetic Demonstrator", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
C.inject_css()


def _load_assessments():
    try:
        return load_operational("assessments")
    except DataUnavailable:
        return None


def sidebar(A) -> str:
    with st.sidebar:
        C.brand()
        if "nav" not in st.session_state and st.query_params.get("page") in SLUGS:   # deep link: ?page=alerts
            st.session_state["nav"] = SLUGS[st.query_params["page"]]
        page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed",
                        format_func=lambda p: f":material/{C.NAV_ICONS[p]}: {p}")
        C.rail_divider()
        if A is not None:
            first, last = A["observation_date"].min(), A["observation_date"].max()
            C.snapshot(f"{first:%d %b} – {last:%d %b %Y}", len(A), config.load_simulation()["random_seed"])
        st.write("")
        if st.button("Reload saved outputs", help="Re-read Parquet files after the pipeline has been re-run",
                     use_container_width=True):
            clear_cache()
            st.rerun()
        with st.expander("Data files"):
            for r in status().itertuples():
                st.caption(f"{'●' if r.available else '○'} {r.dataset} · {r.kind}")
    return page


def as_of_text(A, key: str | None) -> str | None:
    if A is None:
        return None
    v = st.session_state.get(key) if key else None
    t = pd.Timestamp(v) if v is not None else A["decision_time"].max()
    return t.strftime("%d %b %Y %H:%M")


def main() -> None:
    A = _load_assessments()
    page = sidebar(A)
    C.disclaimer()
    subtitle, pills, asof_key, filters_on = META[page]
    if subtitle is not None:                              # the front page draws its own hero
        if A is not None:
            first, last = A["observation_date"].min(), A["observation_date"].max()
            pills = [f"Simulated period {first:%d %b} – {last:%d %b %Y}"] + pills
        C.page_header(page, subtitle, pills, as_of_text(A, asof_key))
    filters = filter_ribbon(A, filters_on) if (A is not None and subtitle is not None) else Filters()
    try:
        PAGES[page].render(filters)
    except DataUnavailable as e:
        st.error(f"Required data is not available. {e}", icon="📂")
    except LeakageError as e:                                  # defensive: should never happen
        st.error(f"Blocked: {e}")


main()
