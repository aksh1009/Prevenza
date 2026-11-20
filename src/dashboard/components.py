"""Reusable Streamlit UI components - dark operations-console styling, icon KPI cards, panels, feeds."""
from __future__ import annotations

import html
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from . import theme as T

# Minimal line icons (24x24, stroke = currentColor)
_ICON_PATHS = {
    "shield": '<path d="M12 3l7 3v6c0 4.5-3 7.6-7 9-4-1.4-7-4.5-7-9V6l7-3z"/><path d="M9 12l2 2 4-4"/>',
    "alert": '<path d="M12 3l10 18H2L12 3z"/><path d="M12 10v5"/><path d="M12 18h.01"/>',
    "pulse": '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
    "map": '<path d="M9 4l6 2 6-2v14l-6 2-6-2-6 2V6l6-2z"/><path d="M9 4v14M15 6v14"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
    "thermo": '<path d="M14 14V5a2 2 0 10-4 0v9a4 4 0 104 0z"/><path d="M12 9v6"/>',
    "users": '<circle cx="9" cy="8" r="3"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><path d="M16 4a3 3 0 010 6M21 20c0-2.6-1.6-4.8-4-5.6"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 015 .5c0 1.5-2.5 2-2.5 3.5"/><path d="M12 17h.01"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "forecast": '<path d="M3 17l5-5 4 4 8-8"/><path d="M14 8h6v6"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-5"/>',
}


def icon(name: str, color: str, size: int = 22) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.9" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICON_PATHS[name]}</svg>')


LOGO = ('<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.5l8 3.4v6.4c0 4.9-3.4 8.3-8 9.7-4.6-1.4-8-4.8-8-9.7'
        'V5.9l8-3.4z"/><path d="M6.5 12.5h3l1.5-3 2.5 6 1.5-3h2.5" stroke="#22c55e"/></svg>')

CSS = f"""
<style>
html, body, .stApp {{ background: {T.SURFACE}; }}
.block-container {{ padding-top: 3.4rem; max-width: 1500px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: {T.NAVY}; border-right: 1px solid {T.BORDER}; }}
[data-testid="stSidebar"] * {{ color: #d6deeb; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: .55rem .75rem; border-radius: 8px; margin: 2px 0;
  width: 100%; transition: background .15s; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {T.NAVY_2}; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: #1d4ed8; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] * {{ color: #ffffff !important; font-weight: 600; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none !important; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ width: 100%; gap: 0; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ display: flex; box-sizing: border-box; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:not(:last-child) {{ display: none !important; }}
[data-testid="stSidebar"] .stButton button {{ background: {T.NAVY_2}; border: 1px solid #2b3e60; }}
.pv-brand {{ display: flex; gap: .6rem; align-items: center; margin: .2rem 0 .1rem 0; }}
.pv-brand .name {{ font-size: 1.45rem; font-weight: 800; color: #ffffff; letter-spacing: .01em; }}
.pv-brand .tag {{ font-size: .72rem; color: {T.INK_2}; line-height: 1.25; }}
.pv-banner {{ background: #2a1d07; border: 1px solid #6b4a12; border-left: 5px solid #f59e0b; color: #fde7b0;
  padding: .6rem .95rem; border-radius: 8px; font-weight: 600; font-size: .86rem; margin-bottom: .9rem; }}
.pv-head {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; margin-bottom: .6rem;
  flex-wrap: wrap; }}
.pv-title {{ font-size: 1.6rem; font-weight: 750; color: {T.INK}; margin: 0; }}
.pv-sub {{ color: {T.INK_2}; font-size: .9rem; }}
.pv-chip {{ display: inline-block; background: {T.PANEL}; border: 1px solid {T.BORDER}; color: {T.INK_2};
  border-radius: 8px; padding: .35rem .7rem; font-size: .8rem; margin-left: .4rem; white-space: nowrap; }}
.pv-card {{ background: {T.PANEL}; border: 1px solid {T.BORDER}; border-radius: 12px; padding: .85rem 1rem;
  height: 100%; display: flex; gap: .85rem; align-items: center; }}
.pv-card .ic {{ width: 46px; height: 46px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; flex: 0 0 46px; }}
.pv-card .k {{ color: {T.INK_2}; font-size: .78rem; }}
.pv-card .v {{ color: {T.INK}; font-size: 1.65rem; font-weight: 750; line-height: 1.2; }}
.pv-card .n {{ color: {T.MUTED}; font-size: .74rem; }}
.pv-card.small {{ padding: .6rem .8rem; }}
.pv-card.small .v {{ font-size: 1rem; font-weight: 650; line-height: 1.35; }}
div[class*="st-key-panel"] {{ background: {T.PANEL}; border: 1px solid {T.BORDER}; border-radius: 12px;
  padding: .85rem 1rem .6rem 1rem; }}
.pv-ptitle {{ display: flex; justify-content: space-between; align-items: center; font-size: 1rem; font-weight: 650;
  color: {T.INK}; margin-bottom: .35rem; }}
.pv-ptitle .r {{ font-size: .78rem; color: {T.ACCENT}; font-weight: 500; }}
.pv-badge {{ display: inline-block; padding: .14rem .6rem; border-radius: 999px; font-size: .8rem; font-weight: 650;
  border: 1px solid; white-space: nowrap; }}
.pv-note {{ background: #0f2240; border-left: 4px solid {T.ACCENT}; padding: .6rem .9rem; border-radius: 6px;
  color: {T.INK}; font-size: .88rem; margin: .4rem 0 .8rem 0; }}
.pv-caution {{ background: #2b1015; border: 1px solid #5c1d27; border-left: 5px solid #f43f5e; padding: .65rem 1rem;
  border-radius: 6px; color: #fecdd3; font-weight: 600; margin: .4rem 0 .8rem 0; }}
.pv-limit {{ background: #1b1736; border: 1px solid #332b66; border-left: 5px solid #9085e9; padding: .65rem 1rem;
  border-radius: 6px; color: #dcd8ff; font-weight: 600; margin: .4rem 0 .8rem 0; }}
.pv-legend span {{ margin-right: .9rem; font-size: .8rem; color: {T.INK_2}; }}
.pv-feed {{ display: flex; flex-direction: column; gap: .55rem; max-height: 330px; overflow-y: auto; }}
.pv-item {{ display: flex; gap: .65rem; align-items: flex-start; padding-bottom: .5rem; border-bottom: 1px solid {T.BORDER}; }}
.pv-item .dot {{ width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  flex: 0 0 30px; font-size: .85rem; font-weight: 800; color: #0a1122; }}
.pv-item .t {{ font-weight: 650; font-size: .86rem; }}
.pv-item .d {{ color: {T.INK_2}; font-size: .78rem; line-height: 1.3; }}
.pv-item .m {{ color: {T.MUTED}; font-size: .72rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown(f'<div class="pv-brand">{LOGO}<div><div class="name">Prevenza</div>'
                f'<div class="tag">{html.escape(T.TAGLINE)}</div></div></div>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(f'<div class="pv-banner" role="alert">⚠ {html.escape(T.DISCLAIMER)}</div>', unsafe_allow_html=True)


def header(title: str, subtitle: str = "", chips: list[str] | None = None) -> None:
    chip_html = "".join(f'<span class="pv-chip">{html.escape(c)}</span>' for c in (chips or []))
    st.markdown(f'<div class="pv-head"><div><div class="pv-title">{html.escape(title)}</div>'
                f'<div class="pv-sub">{html.escape(subtitle)}</div></div><div>{chip_html}</div></div>',
                unsafe_allow_html=True)


def kpi(col, label: str, value, note: str = "", small: bool = False, icon_name: str | None = None,
        color: str = T.ACCENT) -> None:
    ic = (f'<div class="ic" style="background:{color}22;border:1px solid {color}55">{icon(icon_name, color)}</div>'
          if icon_name and not small else "")
    cls = "pv-card small" if small else "pv-card"
    col.markdown(f'<div class="{cls}">{ic}<div><div class="k">{html.escape(label)}</div>'
                 f'<div class="v">{html.escape(str(value))}</div>'
                 f'<div class="n">{html.escape(note)}</div></div></div>', unsafe_allow_html=True)


@contextmanager
def panel(key: str, title: str = "", right: str = ""):
    """Dark card container. `key` must be unique per page; it is prefixed so CSS can style it."""
    with st.container(key=f"panel_{key}"):
        if title:
            st.markdown(f'<div class="pv-ptitle"><span>{html.escape(title)}</span>'
                        f'<span class="r">{html.escape(right)}</span></div>', unsafe_allow_html=True)
        yield


def feed(items: list[dict]) -> None:
    """items: dicts with color, symbol, title, detail, meta."""
    rows = "".join(
        f'<div class="pv-item"><div class="dot" style="background:{i["color"]}">{html.escape(i["symbol"])}</div>'
        f'<div><div class="t" style="color:{i["color"]}">{html.escape(i["title"])}</div>'
        f'<div class="d">{html.escape(i["detail"])}</div><div class="m">{html.escape(i["meta"])}</div></div></div>'
        for i in items)
    st.markdown(f'<div class="pv-feed">{rows}</div>', unsafe_allow_html=True)


def class_badge(cls: str) -> str:
    c = T.CLASS_COLORS.get(cls, T.MUTED)
    return (f'<span class="pv-badge" style="color:{c};border-color:{c};background:{c}1f">'
            f'{T.CLASS_ICONS.get(cls, "•")} {html.escape(str(cls))}</span>')


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
