"""Prevenza UI components - control-room console (dark tinted neutrals, one accent, IBM Plex family).

Rules: one dominant element per view; no card-in-card (tables and callouts inside a card lose their own border);
no decorative icons; spacing from a fixed scale (--s1..--s6); visible focus rings; motion none.
"""
from __future__ import annotations

import html
import re
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from . import theme as T

# Icons kept only where they carry meaning (warning, lock, empty state). No decorative KPI icons.
_ICON = {
    "alert": '<path d="M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "inbox": '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.5 5.1L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.5-6.9A2 2 0 0016.8 4H7.2a2 2 0 00-1.7 1.1z"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0110 0v4"/>',
}


def icon(name: str, color: str = T.MUTED, size: int = 14) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICON[name]}</svg>')


LOGO = ('<svg width="24" height="24" viewBox="0 0 24 24" aria-label="Prevenza"><rect x="0.5" y="0.5" width="23" '
        'height="23" rx="5" fill="#141E30" stroke="#33405A"/><path d="M4 16.5h16" stroke="#475569" stroke-width="1.2"/>'
        '<path d="M4 7.5h16" stroke="#EAB308" stroke-width="1.1" stroke-dasharray="2 1.6"/>'
        '<path d="M4 15.5c2.2 0 3-1 4.2-3.2S10.8 9 12.3 9s2.4 2.6 3.6 4.6 2.3 2.4 4.1 2.4" fill="none" '
        'stroke="#67C1D6" stroke-width="1.7" stroke-linecap="round"/></svg>')

NAV_ICONS = {"Home": "home", "Executive overview": "grid_view", "Site & zone monitoring": "monitor_heart",
             "Alert investigation": "gpp_maybe", "24-hour forecasting": "trending_up",
             "Model evaluation": "compare_arrows", "Data quality & reliability": "fact_check"}

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
         'href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family='
         'IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600'
         '&display=swap" rel="stylesheet">')

CSS = f"""
<style>
:root {{
  --canvas: {T.CANVAS}; --surface: {T.SURFACE}; --surface-2: {T.SURFACE_MUTED}; --rail: {T.SIDEBAR};
  --ink: {T.INK}; --ink-2: {T.INK_2}; --muted: {T.MUTED}; --line: {T.BORDER}; --line-2: {T.BORDER_STRONG};
  --accent: {T.ACCENT}; --accent-2: {T.ACCENT_SUBTLE};
  --font-sans: {T.FONT_SANS}; --font-cond: {T.FONT_COND}; --font-mono: {T.FONT_MONO};
  --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 40px;
  --r-card: 6px; --r-ctl: 6px; --r-badge: 3px;
}}
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stToolbar"] {{ display: none !important; }}
.stApp {{ background: var(--canvas); color: var(--ink); font-family: var(--font-sans); }}
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp li {{ font-family: var(--font-sans); }}
.stApp [data-testid="stCaptionContainer"] p {{ font-size: 14px !important; }}
.stApp [data-testid="stMarkdownContainer"] p {{ font-size: 15px; }}
.main .block-container, [data-testid="stMainBlockContainer"] {{ max-width: 1600px; padding: 0 28px var(--s6) 28px; }}
[data-testid="stVerticalBlock"] {{ gap: var(--s4); }}
*:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px !important; border-radius: 4px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}

/* rail */
section[data-testid="stSidebar"] {{ background: var(--rail); width: 248px !important; min-width: 248px !important;
  border-right: 1px solid {T.BORDER_SIDEBAR}; }}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {{ color: {T.ON_DARK_MUTED}; }}
section[data-testid="stSidebar"] .stRadio, section[data-testid="stSidebar"] .stRadio > div,
section[data-testid="stSidebar"] [role="radiogroup"], section[data-testid="stSidebar"] [role="radiogroup"] > label
{{ width: 100% !important; max-width: 100% !important; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {{ min-height: 36px; padding: 0 var(--s3); border-radius: 4px;
  box-sizing: border-box; display: flex; align-items: center; border-left: 2px solid transparent; transition: background .12s ease-out; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: {T.SIDEBAR_HOVER}; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none !important; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size: 14.5px; font-weight: 500; color: #C3CEDD !important; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: {T.SIDEBAR_HOVER};
  border-left: 2px solid var(--accent); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] .stButton button {{ background: transparent !important; border: 1px solid var(--line-2) !important;
  border-radius: var(--r-ctl); min-height: 32px; font-size: 14px; }}
section[data-testid="stSidebar"] .stButton button p {{ color: var(--ink) !important; }}
section[data-testid="stSidebar"] .stButton button:hover {{ border-color: var(--accent) !important; }}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{ border: 1px solid {T.BORDER_SIDEBAR}; background: transparent; }}
.pv-brand {{ display: flex; align-items: center; gap: 10px; }}
.pv-word {{ font-family: var(--font-cond); font-size: 22px; font-weight: 600; letter-spacing: -0.01em; color: #FFFFFF !important; }}
.pv-sim {{ font-family: var(--font-mono); font-size: 10px; color: #FCD34D !important; border: 1px solid #5C4813;
  padding: 1px 5px; border-radius: var(--r-badge); letter-spacing: .06em; }}
.pv-tagline {{ font-size: 13px; line-height: 1.45; color: {T.ON_DARK_MUTED}; margin-top: var(--s2); }}
.pv-rail-div {{ border-top: 1px solid {T.BORDER_SIDEBAR}; margin: var(--s4) 0; }}
.pv-snap {{ font-size: 13px; line-height: 1.6; color: {T.ON_DARK_MUTED}; }}
.pv-snap b {{ color: var(--ink) !important; font-weight: 500; }}
.pv-snap .m {{ font-family: var(--font-mono); font-size: 12.5px; }}

/* governance bar */
.pv-gov {{ position: sticky; top: 0; z-index: 990; display: flex; align-items: center; gap: var(--s2); min-height: 34px;
  margin: 0 -28px var(--s5) -28px; padding: 6px 28px; background: #1E180A; border-bottom: 1px solid #5C4813;
  color: #F5D48A; font-size: 13.5px; font-weight: 500; }}

/* page header */
.pv-head {{ display: flex; flex-direction: column; align-items: center; text-align: center; gap: var(--s2);
  margin: var(--s2) 0 var(--s5) 0; }}
.pv-head > div:first-child {{ display: flex; flex-direction: column; align-items: center; }}
.pv-h1 {{ font-family: var(--font-cond); font-size: 44px; line-height: 50px; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--ink); }}
.pv-sub {{ font-size: 22px; line-height: 26px; color: var(--ink-2); margin-top: var(--s2); max-width: 64ch; }}
.pv-pills {{ margin-top: var(--s2); display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; }}
@media (max-width: 900px) {{ .pv-h1 {{ font-size: 32px; line-height: 38px; }} }}
.pv-pill {{ font-size: 14px; font-weight: 500; color: var(--ink-2); border: 1px solid var(--line-2); border-radius: var(--r-badge); padding: 1px 7px; }}
.pv-asof {{ font-family: var(--font-mono); font-size: 13.5px; color: var(--ink); border: 1px solid var(--line-2);
  border-radius: var(--r-badge); padding: 4px 10px; white-space: nowrap; }}
.pv-asof span {{ color: var(--muted); }}

/* filter ribbon: a rule, not a box */
div[class*="st-key-ribbon"] {{ border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: var(--s2) 0 0 0; margin-bottom: var(--s5); }}
.pv-locked {{ display: flex; align-items: center; gap: var(--s3); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  padding: var(--s3) 0; margin-bottom: var(--s5); color: var(--muted); font-size: 14px; }}
.pv-locked .b {{ color: var(--ink-2); border: 1px solid var(--line-2); border-radius: var(--r-badge); padding: 1px 8px; font-weight: 500; }}

/* native controls */
.stApp label p {{ font-size: 13.5px !important; font-weight: 500 !important; color: var(--ink-2) !important; letter-spacing: .01em; }}
[data-baseweb="select"] > div, [data-testid="stDateInput"] input, [data-baseweb="input"] {{ min-height: 34px;
  border-radius: var(--r-ctl) !important; border-color: var(--line-2) !important; background: var(--surface) !important; font-size: 14.5px; }}
[data-baseweb="tag"] {{ background: var(--accent-2) !important; border-radius: var(--r-badge) !important; }}
[data-baseweb="tag"] span {{ color: var(--accent) !important; font-size: 13.5px; }}
.stApp .stButton button, .stApp .stDownloadButton button {{ border-radius: var(--r-ctl); border: 1px solid var(--line-2);
  background: transparent; color: var(--ink); min-height: 32px; font-size: 14px; font-weight: 500; box-shadow: none;
  transition: border-color .12s ease-out, color .12s ease-out; }}
.stApp .stButton button:hover, .stApp .stDownloadButton button:hover {{ border-color: var(--accent); color: var(--accent); }}
[data-baseweb="tab-list"] {{ gap: var(--s5); border-bottom: 1px solid var(--line); }}
[data-baseweb="tab"] p {{ font-size: 14.5px; font-weight: 500; color: var(--muted); }}
[data-baseweb="tab"][aria-selected="true"] p {{ font-weight: 600; color: var(--ink); }}
[data-baseweb="tab-highlight"] {{ background: var(--accent) !important; height: 2px !important; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); }}
[data-testid="stExpander"] summary p {{ font-size: 14.5px; font-weight: 500; color: var(--ink); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: var(--r-card); }}
[data-testid="stAlert"] {{ border-radius: var(--r-card); }}

/* cards - flat surfaces; anything inside loses its own frame (no card-in-card) */
div[class*="st-key-card"] {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-card);
  padding: var(--s3) var(--s4) var(--s3) var(--s4); gap: var(--s2); }}
div[class*="st-key-card"] [data-testid="stDataFrame"] {{ border: none; }}
div[class*="st-key-card"] .pv-call {{ border-top: none; border-right: none; border-bottom: none; background: transparent; padding-left: var(--s3); }}
div[class*="st-key-hero"] {{ background: transparent; border: none; padding: 0; gap: var(--s3); }}
.pv-chead {{ display: flex; justify-content: space-between; align-items: baseline; gap: var(--s3); }}
.pv-chead .t {{ font-size: 18px; font-weight: 600; color: var(--ink); }}
.pv-chead .c {{ font-size: 13px; color: var(--muted); text-align: right; }}
.pv-h2 {{ font-family: var(--font-cond); font-size: 22px; font-weight: 600; letter-spacing: -0.005em; color: var(--ink);
  margin: var(--s5) 0 var(--s2) 0; }}
.pv-h2 small {{ font-family: var(--font-sans); font-size: 13.5px; font-weight: 400; color: var(--muted); margin-left: var(--s2); }}

/* KPI: number first, label second; no icon */
.pv-kpi {{ border: 1px solid var(--line); border-radius: var(--r-card); padding: var(--s3) var(--s4); min-height: 84px;
  height: 100%; box-sizing: border-box; background: var(--surface); }}
.pv-kpi .eye {{ font-size: 12.5px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--ink-2); }}
.pv-kpi .val {{ font-family: var(--font-cond); font-size: 34px; line-height: 34px; font-weight: 600; color: var(--ink);
  font-variant-numeric: tabular-nums lining-nums; margin-top: var(--s1); }}
.pv-kpi .val.sm {{ font-size: 19px; line-height: 26px; }}
.pv-kpi .cap {{ font-size: 13.5px; color: var(--muted); margin-top: var(--s1); line-height: 16px; }}
.mono {{ font-family: var(--font-mono); }}

/* status ledger: one hairline row instead of eight boxes */
.pv-ledger {{ display: flex; align-items: stretch; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); margin-bottom: var(--s5); }}
.pv-ledger > div {{ padding: var(--s3) var(--s5) var(--s3) 0; margin-right: var(--s5); border-right: 1px solid var(--line); }}
.pv-ledger > div:last-child {{ border-right: none; margin-left: auto; text-align: right; padding-right: 0; margin-right: 0; }}
.pv-ledger .n {{ font-family: var(--font-cond); font-size: 40px; line-height: 36px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.pv-ledger .l {{ font-size: 13.5px; color: var(--ink-2); margin-top: 2px; white-space: nowrap; }}
.pv-ledger .g {{ font-size: 14.5px; margin-right: 6px; }}
.pv-ledger .zero {{ color: var(--muted) !important; }}

/* zone status board - the dominant element */
.pv-board {{ display: grid; grid-template-columns: 128px repeat(3, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: var(--r-card); overflow: hidden; }}
.pv-board .site {{ background: var(--canvas); padding: var(--s3); font-family: var(--font-mono); font-size: 13.5px; color: var(--ink-2);
  display: flex; flex-direction: column; justify-content: center; }}
.pv-board .site b {{ font-family: var(--font-cond); font-size: 22px; color: var(--ink); font-weight: 600; }}
.pv-board .site span {{ font-family: var(--font-sans); font-size: 13.5px; color: var(--muted); line-height: 1.3; margin-top: 2px; }}
.pv-cell.quiet .zn {{ color: var(--ink-2); font-weight: 500; }}
.pv-drv {{ display: flex; flex-direction: column; gap: var(--s3); padding-top: var(--s2); }}
.pv-drv .row {{ display: grid; grid-template-columns: 128px 1fr 64px; align-items: center; gap: var(--s3); }}
.pv-drv .nm {{ font-size: 14.5px; color: var(--ink); }}
.pv-drv .trk {{ position: relative; height: 6px; background: var(--surface-2); border-radius: 2px; }}
.pv-drv .trk i {{ position: absolute; left: 0; top: 0; height: 6px; border-radius: 2px; }}
.pv-drv .trk b {{ position: absolute; left: 40%; top: -4px; width: 1px; height: 14px; background: var(--muted); }}
.pv-drv .v {{ font-family: var(--font-mono); font-size: 13.5px; color: var(--ink); text-align: right; }}
.pv-drv .sub {{ grid-column: 2 / 4; font-family: var(--font-mono); font-size: 12.5px; color: var(--muted); margin-top: -8px; }}
.pv-cell {{ background: var(--surface); padding: var(--s3) var(--s4); border-left: 3px solid transparent; min-height: 116px; }}
.pv-cell .top {{ display: flex; justify-content: space-between; align-items: baseline; gap: var(--s2); }}
.pv-cell .zn {{ font-size: 14.5px; font-weight: 600; color: var(--ink); }}
.pv-cell .zid {{ font-family: var(--font-mono); font-size: 12.5px; color: var(--muted); }}
.pv-cell .score {{ font-family: var(--font-cond); font-size: 40px; line-height: 42px; font-weight: 600; margin-top: var(--s2);
  font-variant-numeric: tabular-nums; }}
.pv-cell .score small {{ font-family: var(--font-mono); font-size: 12.5px; color: var(--muted); font-weight: 400; margin-left: 4px; }}
.pv-cell .foot {{ display: flex; justify-content: space-between; align-items: center; margin-top: var(--s2); gap: var(--s2); flex-wrap: wrap; }}
.pv-cell .flags {{ font-family: var(--font-mono); font-size: 12.5px; color: #FDE047; }}
.pv-cell .flags.inv {{ color: #F0ABFC; }}

/* badges & indicators */
.pv-badge {{ display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 9px; border-radius: var(--r-badge);
  font-size: 13.5px; font-weight: 500; border: 1px solid; white-space: nowrap; line-height: 1; }}
.pv-dot {{ width: 6px; height: 6px; border-radius: 9999px; display: inline-block; }}
.pv-evi {{ display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-mono); font-size: 13px; }}
.pv-evi .blk {{ display: inline-flex; gap: 2px; align-items: center; }}
.pv-evi .blk i {{ width: 9px; height: 4px; border-radius: 1px; display: inline-block; }}
.pv-tag {{ font-family: var(--font-mono); font-size: 13px; padding: 1px 6px; border-radius: var(--r-badge); border: 1px solid; }}

/* callouts */
.pv-call {{ border-left: 2px solid var(--accent); padding: var(--s2) var(--s3); color: var(--ink-2); font-size: 14.5px; line-height: 19px;
  background: var(--surface); border-radius: 0 var(--r-card) var(--r-card) 0; }}
.pv-call.warn {{ border-left-color: #EAB308; }}
.pv-call.amber {{ background: #1E180A; border-left-color: #EAB308; color: #F5D48A; font-weight: 500; }}
.pv-call.danger {{ background: #241029; border-left-color: #D946EF; color: #F5D0FE; font-weight: 500; }}
.pv-call b {{ color: var(--ink); font-weight: 600; }}
.pv-call.amber b, .pv-call.danger b {{ color: inherit; }}

/* empty state */
.pv-empty {{ border: 1px dashed var(--line-2); border-radius: var(--r-card); padding: var(--s6) var(--s5); text-align: center; }}
.pv-empty .t {{ font-size: 15.5px; font-weight: 500; color: var(--ink-2); margin-top: var(--s2); }}
.pv-empty .d {{ font-size: 14px; color: var(--muted); margin-top: var(--s1); }}

/* feed */
.pv-feed {{ display: flex; flex-direction: column; max-height: 520px; overflow-y: auto; }}
.pv-item {{ padding: var(--s3) 0; border-bottom: 1px solid var(--line); }}
.pv-item:first-child {{ padding-top: var(--s1); }}
.pv-item .z {{ font-size: 14.5px; color: var(--ink); font-weight: 500; }}
.pv-item .d {{ font-size: 13.5px; color: var(--ink-2); margin-top: var(--s1); }}
.pv-item .m {{ font-family: var(--font-mono); font-size: 12.5px; color: var(--muted); margin-top: 2px; }}

/* facility profile + thermal */
.pv-prof {{ display: grid; grid-template-columns: repeat(6, 1fr); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
.pv-prof > div {{ padding: var(--s3) var(--s4) var(--s3) 0; }}
.pv-prof .k {{ font-size: 12.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); }}
.pv-prof .v {{ font-size: 16px; font-weight: 500; color: var(--ink); margin-top: var(--s1); }}
.pv-therm .big {{ font-family: var(--font-cond); font-size: 40px; line-height: 44px; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }}
.pv-therm .meta {{ font-family: var(--font-mono); font-size: 13px; color: var(--ink-2); margin-top: var(--s2); line-height: 1.5; }}
.pv-bullet {{ position: relative; height: 8px; border-radius: 2px; margin: 30px 0 26px 0; display: flex; overflow: visible; }}
.pv-bullet .seg {{ height: 8px; }}
.pv-bullet .mk {{ position: absolute; top: -8px; width: 2px; height: 24px; background: var(--ink); }}
.pv-bullet .mk span {{ position: absolute; top: -18px; left: -14px; font-family: var(--font-mono); font-size: 12.5px; color: var(--ink); white-space: nowrap; }}
.pv-bullet .tick {{ position: absolute; top: 14px; font-family: var(--font-mono); font-size: 12px; color: var(--muted); transform: translateX(-50%); white-space: nowrap; }}
/* home / front page */
.hm-eyebrow {{ font-family: var(--font-mono); font-size: 13.5px; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }}
.hm-word {{ font-family: var(--font-cond); font-size: 104px; line-height: 0.95; font-weight: 600; letter-spacing: -0.01em;
  text-transform: uppercase; color: var(--ink); margin: var(--s4) 0 var(--s3) -4px; }}
.hm-word span {{ color: var(--accent); }}
.hm-line {{ font-family: var(--font-cond); font-size: 40px; line-height: 40px; font-weight: 500; color: var(--ink); max-width: 20ch; }}
.hm-lede {{ font-size: 22px; line-height: 28px; color: var(--ink-2); max-width: 54ch; margin-top: var(--s4); }}
.hm-lede b {{ color: var(--ink); font-weight: 500; }}
.hm-spec {{ border: 1px solid var(--line-2); border-radius: var(--r-card); background: var(--surface); overflow: hidden; }}
.hm-spec .hd {{ display: flex; justify-content: space-between; align-items: baseline; padding: var(--s3) var(--s4);
  border-bottom: 1px solid var(--line); font-family: var(--font-mono); font-size: 13px; color: var(--muted); }}
.hm-spec .hd b {{ font-family: var(--font-sans); font-size: 14.5px; color: var(--ink); font-weight: 600; }}
.hm-spec .z {{ display: grid; grid-template-columns: 1fr auto; gap: var(--s1) var(--s3); padding: var(--s3) var(--s4);
  border-bottom: 1px solid var(--line); border-left: 3px solid transparent; }}
.hm-spec .z .n {{ font-size: 16px; color: var(--ink); font-weight: 500; }}
.hm-spec .z .s {{ font-family: var(--font-cond); font-size: 34px; line-height: 30px; font-weight: 600; text-align: right; grid-row: span 2; }}
.hm-spec .why {{ padding: var(--s3) var(--s4); font-size: 14px; line-height: 19px; color: var(--ink-2); }}
.hm-spec .why b {{ color: var(--ink); font-weight: 500; }}
.hm-facts {{ display: grid; grid-template-columns: repeat(5, 1fr); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  margin: var(--s6) 0 var(--s5) 0; }}
.hm-facts > div {{ padding: var(--s4) var(--s4) var(--s4) 0; }}
.hm-facts > div + div {{ border-left: 1px solid var(--line); padding-left: var(--s4); }}
.hm-facts .n {{ font-family: var(--font-cond); font-size: 40px; line-height: 44px; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }}
.hm-facts .l {{ font-size: 14.5px; color: var(--ink-2); margin-top: var(--s1); line-height: 18px; }}
.hm-h {{ font-family: var(--font-cond); font-size: 34px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
  color: var(--ink); margin: var(--s6) 0 var(--s2) 0; }}
.hm-hsub {{ font-size: 16px; color: var(--ink-2); max-width: 70ch; margin-bottom: var(--s5); }}
.hm-steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; border-top: 2px solid var(--line-2); }}
.hm-steps > div {{ padding: var(--s4) var(--s5) var(--s4) 0; position: relative; }}
.hm-steps > div::before {{ content: ""; position: absolute; top: -6px; left: 0; width: 10px; height: 10px; border-radius: 50%;
  background: var(--canvas); border: 2px solid var(--accent); }}
.hm-steps .k {{ font-family: var(--font-mono); font-size: 13.5px; color: var(--accent); margin-top: var(--s3); }}
.hm-steps .t {{ font-family: var(--font-cond); font-size: 22px; font-weight: 600; color: var(--ink); margin-top: var(--s1); }}
.hm-steps .d {{ font-size: 16px; line-height: 21px; color: var(--ink-2); margin-top: var(--s2); }}
.hm-steps .m {{ font-family: var(--font-mono); font-size: 13px; color: var(--muted); margin-top: var(--s2); }}
.hm-quote {{ font-family: var(--font-cond); font-size: 40px; line-height: 44px; font-weight: 600; color: var(--ink); }}
.hm-quote em {{ font-style: normal; color: var(--accent); }}
.hm-qd {{ font-size: 16.5px; line-height: 23px; color: var(--ink-2); margin: var(--s2) 0 var(--s5) 0; max-width: 46ch; }}
.hm-res {{ width: 100%; border-collapse: collapse; font-size: 16px; }}
.hm-res th {{ text-align: left; font-size: 12.5px; letter-spacing: .07em; text-transform: uppercase; color: var(--muted);
  font-weight: 600; padding: 0 var(--s3) var(--s2) 0; border-bottom: 1px solid var(--line-2); }}
.hm-res td {{ padding: var(--s3) var(--s3) var(--s3) 0; border-bottom: 1px solid var(--line); color: var(--ink-2); vertical-align: top; }}
.hm-res td.v {{ font-family: var(--font-mono); color: var(--ink); white-space: nowrap; }}
.hm-res td.v b {{ color: var(--accent); font-weight: 500; }}
.hm-dir {{ border-top: 1px solid var(--line); }}
.hm-dir .r {{ display: grid; grid-template-columns: 48px 260px 1fr; align-items: baseline; padding: var(--s3) 0; border-bottom: 1px solid var(--line); }}
.hm-dir .i {{ font-family: var(--font-mono); font-size: 13.5px; color: var(--muted); }}
.hm-dir .t {{ font-family: var(--font-cond); font-size: 22px; font-weight: 600; color: var(--ink); }}
.hm-dir .d {{ font-size: 16px; color: var(--ink-2); }}
.hm-foot {{ margin-top: var(--s6); padding-top: var(--s4); border-top: 1px solid var(--line); font-size: 14px; color: var(--muted);
  line-height: 19px; display: flex; justify-content: space-between; gap: var(--s5); flex-wrap: wrap; }}
div[class*="st-key-hm_cta"] button {{ min-height: 44px !important; font-size: 16px !important; padding: 0 var(--s5) !important; }}
div[class*="st-key-hm_cta_primary"] button {{ background: var(--accent) !important; border-color: var(--accent) !important; }}
div[class*="st-key-hm_cta_primary"] button p {{ color: #0C1220 !important; font-weight: 600 !important; }}
div[class*="st-key-hm_cta_primary"] button:hover {{ background: #8AD3E3 !important; }}
@media (max-width: 1100px) {{ .hm-word {{ font-size: 72px; }} .hm-facts {{ grid-template-columns: repeat(2, 1fr); }}
  .hm-steps {{ grid-template-columns: repeat(2, 1fr); row-gap: var(--s5); }} .hm-dir .r {{ grid-template-columns: 40px 1fr; }}
  .hm-dir .d {{ grid-column: 2; }} }}
@media (max-width: 1100px) {{ .pv-prof {{ grid-template-columns: repeat(3, 1fr); }} .pv-board {{ grid-template-columns: 88px repeat(3, 1fr); }} }}
</style>
"""


def _compact_css(css: str) -> str:
    """Markdown ends an HTML block at a blank line and reads /* */ as emphasis - strip both before injecting."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return "\n".join(line for line in css.splitlines() if line.strip())


def inject_css() -> None:
    st.markdown(FONTS + _compact_css(CSS), unsafe_allow_html=True)


def gap(size: str = "s4") -> None:
    """Vertical rhythm from the spacing scale (replaces ad-hoc st.write('') spacers)."""
    st.markdown(f'<div style="height:var(--{size})"></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------- frame pieces
def brand() -> None:
    st.markdown(f'<div class="pv-brand">{LOGO}<span class="pv-word">Prevenza</span><span class="pv-sim">SIMULATED</span>'
                f'</div><div class="pv-tagline">{html.escape(T.TAGLINE)}</div><div class="pv-rail-div"></div>',
                unsafe_allow_html=True)


def rail_divider() -> None:
    st.markdown('<div class="pv-rail-div"></div>', unsafe_allow_html=True)


def snapshot(period: str, n_obs: int, seed) -> None:
    st.markdown(f'<div class="pv-snap"><b>Simulated dataset v1.0</b><br><span class="m">{html.escape(period)}</span><br>'
                f'<span class="m">{n_obs:,} zone-shifts · seed {seed}</span><br>Read-only view of saved outputs</div>',
                unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(f'<div class="pv-gov" role="alert">{icon("alert", "#EAB308", 14)}<span>{html.escape(T.DISCLAIMER)}'
                f'</span></div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", pills: list[str] | None = None, as_of: str | None = None) -> None:
    p = "".join(f'<span class="pv-pill">{html.escape(x)}</span>' for x in (pills or []))
    right = f'<div class="pv-asof"><span>as of</span> {html.escape(as_of)}</div>' if as_of else ""
    st.markdown(f'<div class="pv-head"><div><div class="pv-h1">{html.escape(title)}</div>'
                f'<div class="pv-sub">{html.escape(subtitle)}</div><div class="pv-pills">{p}</div></div>{right}</div>',
                unsafe_allow_html=True)


def locked_ribbon(reason: str = "Evaluation-only view · Global filters disabled") -> None:
    st.markdown(f'<div class="pv-locked">{icon("lock", T.MUTED, 14)}<span class="b">{html.escape(reason)}</span>'
                '<span>Results below use fixed evaluation windows, not the display filters.</span></div>',
                unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    small = f"<small>{html.escape(note)}</small>" if note else ""
    st.markdown(f'<div class="pv-h2">{html.escape(title)}{small}</div>', unsafe_allow_html=True)


@contextmanager
def card(key: str, title: str = "", caption: str = ""):
    """Flat surface with a single-row header. Content inside drops its own frame."""
    with st.container(key=f"card_{key}"):
        if title:
            st.markdown(f'<div class="pv-chead"><span class="t">{html.escape(title)}</span>'
                        f'<span class="c">{html.escape(caption)}</span></div>', unsafe_allow_html=True)
        yield


def chart(fig, key: str) -> None:
    st.plotly_chart(fig, theme=None, key=key, config=T.PLOT_CONFIG)


# --------------------------------------------------------------------------------------- cards & badges
def kpi(col, eyebrow: str, value, caption: str = "", accent: str | None = None, icon_name: str | None = None,
        small_value: bool = False) -> None:
    """Metric card: label, figure, one line of context. `icon_name` is accepted but not drawn (no decoration)."""
    border = f"border-left:3px solid {accent};" if accent else ""
    col.markdown(f'<div class="pv-kpi" style="{border}"><div class="eye">{html.escape(eyebrow)}</div>'
                 f'<div class="val{" sm" if small_value else ""}">{html.escape(str(value))}</div>'
                 f'<div class="cap">{caption}</div></div>', unsafe_allow_html=True)


def ledger(items: list[tuple[str, int | str, str]], right: str = "") -> None:
    """Status ledger: (label, count, state) per segment; zero counts recede."""
    segs = []
    for label, n, state in items:
        color = T.STATES[state][3] if state in T.STATES else T.INK
        zero = isinstance(n, int) and n == 0
        glyph = T.STATES[state][4] if state in T.STATES else ""
        segs.append(f'<div><div class="n {"zero" if zero else ""}" style="color:{color}">{n}</div>'
                    f'<div class="l"><span class="g" style="color:{color}">{glyph}</span>{html.escape(label)}</div></div>')
    if right:
        segs.append(f'<div><div class="l mono" style="font-size:12px">{right}</div></div>')
    st.markdown(f'<div class="pv-ledger">{"".join(segs)}</div>', unsafe_allow_html=True)


def badge(state: str, text: str | None = None) -> str:
    ink, fill, bd, series, glyph = T.STATES.get(state, T.STATES["Missing"])
    mark = (f'<span class="pv-dot" style="background:{series}"></span>' if glyph == "●"
            else f'<span style="font-size:11px">{glyph}</span>')
    style = "border-style:dashed;" if state == "Missing" else ""
    return (f'<span class="pv-badge" style="color:{ink};background:{fill};border-color:{bd};{style}">{mark}'
            f'{html.escape(text or state)}</span>')


def evidence_bar(active: int, total: int = 5, level: str = "") -> str:
    blocks = "".join(f'<i style="background:{T.ACCENT}"></i>' if i < active else
                     f'<i style="border:1px dashed {T.BORDER_STRONG};height:2px;width:7px"></i>' for i in range(total))
    color = {"adequate": "#86EFAC", "limited": "#FCD34D", "insufficient": "#CBD5E1"}.get(level, T.INK_2)
    return (f'<span class="pv-evi"><span class="blk">{blocks}</span><span style="color:{color}">'
            f'{html.escape(level.capitalize())} {active}/{total}</span></span>')


def reliability_bar(pct) -> str:
    if pct is None or pd.isna(pct):
        return '<span class="mono">—</span>'
    return f'<span class="mono">{pct:.0f}%</span>'


def data_tag(state: str) -> str:
    ink, fill, bd, _, _ = T.STATES[state]
    style = "border-style:dashed;" if state == "Missing" else ""
    return f'<span class="pv-tag" style="color:{ink};background:{fill};border-color:{bd};{style}">{html.escape(state)}</span>'


def class_label(cls: str) -> str:
    """Plain-text label with glyph (tables: shape + text, never colour alone)."""
    return f"{T.CLASS_ICONS.get(cls, '•')} {cls}"


def class_legend() -> None:
    st.markdown('<div style="display:flex;gap:6px;flex-wrap:wrap">' +
                "".join(badge(c) for c in T.CLASS_ORDER) + "</div>", unsafe_allow_html=True)


def callout(text: str, variant: str = "info", html_text: bool = False) -> None:
    body = text if html_text else html.escape(text)
    st.markdown(f'<div class="pv-call {"" if variant == "info" else variant}">{body}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    callout(text, "info")


def caution(text: str) -> None:
    callout("⊘ " + text, "danger")


def limitation(text: str) -> None:
    callout(text, "warn")


def empty_state(title: str, detail: str = "") -> None:
    st.markdown(f'<div class="pv-empty">{icon("inbox", T.MUTED, 20)}<div class="t">{html.escape(title)}</div>'
                f'<div class="d">{html.escape(detail)}</div></div>', unsafe_allow_html=True)


def feed(items: list[dict]) -> None:
    rows = "".join(f'<div class="pv-item"><div class="z">{html.escape(i["zone"])}</div><div style="margin-top:6px">'
                   f'{i["badge"]}</div><div class="d">{html.escape(i["detail"])}</div>'
                   f'<div class="m">{html.escape(i["meta"])}</div></div>' for i in items)
    st.markdown(f'<div class="pv-feed">{rows}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------- tables
_ACRONYMS = {"ppv": "PPV", "f1": "F1", "pr_auc": "PR-AUC", "cusum": "CUSUM", "id": "ID", "lr": "LR",
             "hi": "HI", "nws": "NWS", "dq": "DQ"}
_HUMAN = {
    "event_zones_included": "Zones included", "event_zones_detected": "Zones detected",
    "event_sensitivity": "Event sensitivity", "mean_detection_coverage": "Detection coverage",
    "median_lead_time_to_escalation_hours": "Median lead time to escalation (h)",
    "median_detection_delay_hours": "Median detection delay (h)", "controls_with_false_alert": "Controls with false alert",
    "n_episodes": "Episodes", "episode_ppv": "Episode PPV", "false_alerts_per_site_week": "False alerts / site / week",
    "median_episode_duration_hours": "Median episode duration (h)", "false_positive_rate": "False-positive rate",
    "available_weight": "Available weight", "n_categories_available": "Categories available",
    "reliability_score": "Reliability %", "risk_score": "Risk score", "observation_date": "Observation date",
    "decision_time": "Decision time", "first_alert_decision": "First alert", "detection_delay_hours": "Detection delay (h)",
    "lead_time_to_escalation_hours": "Lead time to escalation (h)", "n_matched_episodes": "Matched episodes",
    "control_false_alert": "Control false alert", "excluded_reason": "Excluded reason",
    "configured_weight": "Configured weight", "effective_weight": "Effective weight", "signal_score": "Signal score",
    "contribution_points": "Contribution (pts)", "observed_value": "Observed", "baseline_median": "Baseline median",
    "robust_z": "Robust z", "staleness_hours": "Staleness (h)", "heat_index_validity": "Heat-index validity",
    "missing_sources": "Missing sources", "stale_sources": "Stale sources", "evidence_level": "Evidence level",
    "site_id": "Site", "zone_id": "Zone", "env_temperature_celsius": "Air temperature (°C)",
    "env_relative_humidity": "Relative humidity (%)", "env_heat_index_celsius": "Heat index (°C)",
    "source_dataset": "Source", "issue_type": "Issue type", "start_timestamp": "Start", "end_timestamp": "End",
    "pr_auc": "PR-AUC", "n": "Rows", "positives": "Positives", "precision": "Precision", "recall": "Recall", "f1": "F1",
}


def humanize(name: str) -> str:
    if name in _HUMAN:
        return _HUMAN[name]
    if name in _ACRONYMS:
        return _ACRONYMS[name]
    words = [_ACRONYMS.get(w, w) for w in re.split(r"_+", str(name))]
    s = " ".join(words)
    return s[:1].upper() + s[1:]


def humanize_frame(df: pd.DataFrame, values_in: tuple[str, ...] = ()) -> pd.DataFrame:
    out = df.copy()
    for c in values_in:
        if c in out:
            out[c] = out[c].map(lambda v: humanize(v) if isinstance(v, str) else v)
    return out.rename(columns={c: humanize(c) for c in out.columns})


def table(df: pd.DataFrame, key: str, height: int | None = None, formats: dict | None = None) -> None:
    df = df.copy()
    for c in df.columns:                      # no raw checkboxes: booleans read as text
        if pd.api.types.is_bool_dtype(df[c]) or str(df[c].dtype) == "boolean" or \
                (df[c].dtype == object and df[c].notna().any() and df[c].dropna().map(type).isin([bool]).all()):
            df[c] = df[c].map(lambda v: "—" if v is None or (not isinstance(v, bool) and pd.isna(v))
                              else ("Yes" if bool(v) else "No"))
    cfg = {}
    for c in df.columns:
        if (formats or {}).get(c):
            cfg[c] = formats[c]
        elif pd.api.types.is_float_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(format="%.2f")
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            cfg[c] = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")
    kw = {"height": height} if height else {}
    st.dataframe(df, hide_index=True, key=key, column_config=cfg, **kw)


def download(df: pd.DataFrame, filename: str, label: str = "Export CSV", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv",
                       key=key or f"dl_{filename}")


def fmt_ts(ts) -> str:
    return "—" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
