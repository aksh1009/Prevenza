"""Reusable Streamlit UI components (styling, disclaimer, KPI cards, badges, empty states)."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from . import theme as T

CSS = f"""
<style>
[data-testid="stSidebar"] {{ background: {T.NAVY}; }}
[data-testid="stSidebar"] * {{ color: #e8ecf3 !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] * , [data-testid="stSidebar"] input {{ color: {T.INK} !important;
  -webkit-text-fill-color: {T.INK} !important; }}
[data-testid="stSidebar"] [data-baseweb="tag"] * {{ color: #ffffff !important; }}
.block-container {{ padding-top: 3.6rem; max-width: 1400px; }}
[data-testid="stSidebar"] .stButton button {{ background: {T.NAVY_2} !important; border: 1px solid #3a4d6b !important; }}
[data-testid="stSidebar"] .stButton button p {{ color: #ffffff !important; }}
[data-testid="stSidebar"] [data-testid="stDateInput"] input {{ color: {T.INK} !important;
  -webkit-text-fill-color: {T.INK} !important; opacity: 1 !important; }}
.pv-banner {{ background: #fff4e5; border: 1px solid #f0b45a; border-left: 6px solid #d97706; color: #5a3200;
  padding: .7rem 1rem; border-radius: 6px; font-weight: 600; font-size: .92rem; margin-bottom: .8rem; }}
.pv-title {{ font-size: 1.55rem; font-weight: 700; color: {T.INK}; margin: .2rem 0 0 0; }}
.pv-sub {{ color: {T.INK_2}; font-size: .92rem; margin-bottom: .6rem; }}
.pv-card {{ background: {T.PANEL}; border: 1px solid {T.BORDER}; border-radius: 8px; padding: .75rem .9rem; height: 100%; }}
.pv-card .k {{ color: {T.INK_2}; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
.pv-card .v {{ color: {T.INK}; font-size: 1.55rem; font-weight: 700; line-height: 1.25; }}
.pv-card .n {{ color: {T.MUTED}; font-size: .78rem; }}
.pv-card.small .v {{ font-size: 1rem; font-weight: 600; line-height: 1.35; }}
.pv-badge {{ display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
  border: 1px solid; white-space: nowrap; }}
.pv-note {{ background: #eef4fc; border-left: 4px solid #2a78d6; padding: .6rem .9rem; border-radius: 4px;
  color: {T.INK}; font-size: .9rem; margin: .4rem 0 .8rem 0; }}
.pv-caution {{ background: #fdecec; border-left: 6px solid #d03b3b; padding: .7rem 1rem; border-radius: 4px;
  color: #5c1010; font-weight: 600; margin: .4rem 0 .8rem 0; }}
.pv-limit {{ background: #f3f1fb; border-left: 6px solid #6250d6; padding: .7rem 1rem; border-radius: 4px;
  color: #241a5c; font-weight: 600; margin: .4rem 0 .8rem 0; }}
.pv-legend span {{ margin-right: .9rem; font-size: .82rem; color: {T.INK_2}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(f'<div class="pv-banner" role="alert">⚠ {html.escape(T.DISCLAIMER)}</div>', unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="pv-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="pv-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def kpi(col, label: str, value, note: str = "", small: bool = False) -> None:
    col.markdown(f'<div class="pv-card{" small" if small else ""}"><div class="k">{html.escape(label)}</div>'
                 f'<div class="v">{html.escape(str(value))}</div><div class="n">{html.escape(note)}</div></div>',
                 unsafe_allow_html=True)


def class_badge(cls: str) -> str:
    c = T.CLASS_COLORS.get(cls, T.MUTED)
    icon = T.CLASS_ICONS.get(cls, "•")
    return (f'<span class="pv-badge" style="color:{c};border-color:{c};background:{c}14">'
            f'{icon} {html.escape(str(cls))}</span>')


def class_label(cls: str) -> str:
    """Plain-text label with symbol (for tables; colour never carries meaning alone)."""
    return f"{T.CLASS_ICONS.get(cls, '•')} {cls}"


def class_legend() -> None:
    items = "".join(f'<span><b style="color:{T.CLASS_COLORS[c]}">{T.CLASS_ICONS[c]}</b> {c}</span>'
                    for c in T.CLASS_ORDER)
    st.markdown(f'<div class="pv-legend">{items}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="pv-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def caution(text: str) -> None:
    st.markdown(f'<div class="pv-caution" role="alert">⚠ {html.escape(text)}</div>', unsafe_allow_html=True)


def limitation(text: str) -> None:
    st.markdown(f'<div class="pv-limit">{html.escape(text)}</div>', unsafe_allow_html=True)


def empty_state(text: str) -> None:
    st.info(text, icon="ℹ️")


def download(df: pd.DataFrame, filename: str, label: str = "Download filtered CSV", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv",
                       key=key or f"dl_{filename}")


def fmt_ts(ts) -> str:
    return "—" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
