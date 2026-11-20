"""Global filter ribbon (horizontal, below the page header)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as C
from . import theme as T
from .filters import Filters

KEYS = ("f_sites", "f_zones", "f_shifts", "f_dates", "f_classes")


def _reset() -> None:
    for k in KEYS:
        st.session_state.pop(k, None)


def filter_ribbon(A: pd.DataFrame, enabled: bool = True) -> Filters:
    if not enabled:
        C.locked_ribbon()
        return Filters()
    with st.container(key="ribbon"):
        c = st.columns([1.1, 1.1, 0.85, 1.5, 1.5, 0.45], vertical_alignment="bottom")
        sites = c[0].multiselect("Site", sorted(A["site_id"].unique()), key="f_sites", placeholder="All sites")
        zone_opts = sorted(A.loc[A["site_id"].isin(sites) if sites else A.index, "zone_id"].unique())
        zones = c[1].multiselect("Zone", zone_opts, key="f_zones", placeholder="All zones")
        shifts = c[2].multiselect("Shift", ["day", "night"], key="f_shifts", placeholder="Both shifts")
        lo, hi = A["observation_date"].min().date(), A["observation_date"].max().date()
        dr = c[3].date_input("Observation period", value=(lo, hi), min_value=lo, max_value=hi, key="f_dates",
                             format="YYYY-MM-DD")
        classes = c[4].multiselect("Risk classification", T.CLASS_ORDER, key="f_classes", placeholder="All classes")
        c[5].button("Reset", on_click=_reset, key="f_reset", help="Clear all display filters")
    date_range = tuple(dr) if isinstance(dr, (tuple, list)) and len(dr) == 2 else None
    return Filters(sites, zones, shifts, date_range, classes)
