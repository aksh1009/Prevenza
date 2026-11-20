"""Front page - what Prevenza is, how it works, and how it honestly performed (all figures from saved outputs).

Layout: asymmetric hero (wordmark + claim | real product specimen), fact strip, numbered pipeline, principles set as
type, synthetic results table, page directory. No stock imagery, no gradient CTAs, no identical feature cards.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .. import components as C, theme as T
from ..data_loader import DataUnavailable, load, load_operational, zone_registry

DIRECTORY = [
    ("Executive overview", "Zone status board at any replayed decision time, what changed, and what is driving risk."),
    ("Site & zone monitoring", "One zone in depth: risk trajectory, thermal reading, category streams and raw measurements."),
    ("Alert investigation", "Every risk-fusion alert episode, the evidence at first detection and the full explanation."),
    ("24-hour forecasting", "Forecast ranking beside the persistence baseline, with eligibility and data caveats."),
    ("Model evaluation", "Backtests against the simulation's ground truth — including where the models fall short."),
    ("Data quality & reliability", "Missing, stale and invalid evidence, reporting latency and reliability."),
]


def _go(page: str) -> None:
    st.session_state["nav"] = page


def specimen(A: pd.DataFrame) -> str:
    """The highest-risk moment in the simulation, rendered as the product renders it."""
    peak = A.loc[A["risk_score"].idxmax()]
    rows = A[(A["site_id"] == peak["site_id"]) & (A["decision_time"] == peak["decision_time"])].sort_values("zone_id")
    reg = zone_registry().set_index("zone_id")
    zs = []
    for r in rows.itertuples():
        color = T.CLASS_COLORS[r.classification]
        score = "—" if pd.isna(r.risk_score) else f"{r.risk_score:.0f}"
        zs.append(f'<div class="z" style="border-left-color:{color}"><span class="n">{html.escape(reg.loc[r.zone_id, "zone_name"])} '
                  f'<span class="mono" style="color:{T.MUTED};font-size:11px">{r.zone_id}</span></span>'
                  f'<span class="s" style="color:{color}">{score}</span><span>{C.badge(r.classification)}</span></div>')
    elevated = [T.CATEGORY_LABELS[c] for c in T.CATEGORIES if bool(peak[f"{c}_elevated"])]
    why = (f'<b>Why {html.escape(reg.loc[peak["zone_id"], "zone_name"])} scored {peak["risk_score"]:.0f}:</b> '
           f'{html.escape(", ".join(elevated) or "no category")} elevated; {int(peak["n_categories_available"])}/5 evidence '
           f'streams available, reliability {peak["reliability_score"]:.0f}%.'
           + (' Heat index beyond the NWS chart range is flagged, not trusted.' if peak["heat_index_validity"] == "extrapolated" else ""))
    site_name = zone_registry().set_index("site_id")["site_name"].groupby(level=0).first()[peak["site_id"]]
    return (f'<div class="hm-spec"><div class="hd"><b>{peak["site_id"]} · {html.escape(site_name)}</b>'
            f'<span>{pd.Timestamp(peak["decision_time"]):%d %b %Y %H:%M} · {peak["shift"]} shift</span></div>'
            f'{"".join(zs)}<div class="why">{why}</div></div>')


def _metric(bm: pd.DataFrame, model: str, metric: str, window: str = "full"):
    v = bm[(bm["model"] == model) & (bm["window"] == window) & (bm["metric"] == metric)]["value"]
    return float(v.iloc[0]) if len(v) else float("nan")


def results_html() -> str:
    bm = load("backtest_metrics")
    ms = load("model_summary")
    f = ms[ms["phase"] == "4B forecast"]

    def fc(model, subset_prefix, metric, evaluation="test"):
        v = f[(f["model"] == model) & (f["evaluation"] == evaluation) & f["subset"].str.startswith(subset_prefix)
              & (f["metric"] == metric)]["value"]
        return float(v.iloc[0]) if len(v) else float("nan")

    ctrl_a, ctrl_c = _metric(bm, "A", "controls_with_false_alert"), _metric(bm, "C", "controls_with_false_alert")
    n_ctrl = _metric(bm, "C", "controls_included")
    rows = [
        ("False alerts on isolated heat spikes", f"<b>{ctrl_c:.0f} of {n_ctrl:.0f}</b> vs {ctrl_a:.0f} of {n_ctrl:.0f}",
         "Risk fusion ignores a one-shift spike; the simple statistical detector does not."),
        ("False alert episodes per site-week", f"<b>{_metric(bm, 'C', 'false_alerts_per_site_week'):.2f}</b> vs "
                                               f"{_metric(bm, 'A', 'false_alerts_per_site_week'):.2f}",
         "Corroboration and persistence rules cut alert noise sharply."),
        ("Heat events detected", f"<b>{_metric(bm, 'C', 'event_sensitivity'):.0%}</b> vs {_metric(bm, 'A', 'event_sensitivity'):.0%}",
         "Fusion misses the foundry event, where heat is always high."),
        ("Median warning before escalation", f"<b>{_metric(bm, 'C', 'median_lead_time_to_escalation_hours'):+.0f} h</b> vs "
                                             f"{_metric(bm, 'A', 'median_lead_time_to_escalation_hours'):+.0f} h",
         "Fusion warns later: it waits for health evidence, not just heat."),
        ("24 h forecast, unseen event (F1)", f"<b>{fc('logistic regression', 'unseen', 'f1'):.2f}</b> vs "
                                             f"{fc('persistence (any warning)', 'unseen', 'f1'):.2f}",
         "No demonstrated gain over simply carrying today's warning forward."),
    ]
    body = "".join(f'<tr><td style="color:var(--ink)">{a}</td><td class="v">{b}</td><td>{c}</td></tr>' for a, b, c in rows)
    return (f'<table class="hm-res"><thead><tr><th>Measure (synthetic backtest)</th><th>Prevenza vs baseline</th>'
            f'<th>What it means</th></tr></thead><tbody>{body}</tbody></table>')


def render(f) -> None:
    A = load_operational("assessments")
    left, right = st.columns([7, 5], gap="large", vertical_alignment="center")
    with left:
        st.markdown('<div class="hm-eyebrow">Population &amp; workforce health early warning · synthetic demonstrator</div>'
                    '<div class="hm-word">Preven<span>za</span></div>'
                    '<div class="hm-line">See heat-health risk building before it becomes harm.</div>'
                    '<div class="hm-lede">Prevenza fuses five evidence streams — <b>zone heat sensors, workforce exposure, '
                    'clinic visits, absence and incident reports</b> — into one explainable risk call per zone and shift. '
                    'When the evidence is too thin to call, it says so instead of guessing.</div>',
                    unsafe_allow_html=True)
        C.gap("s5")
        b1, b2, _ = st.columns([1.35, 1.15, 0.5])
        b1.button("Open operations overview  →", key="hm_cta_primary", on_click=_go, args=("Executive overview",),
                  use_container_width=True)
        b2.button("How it was evaluated", key="hm_cta_eval", on_click=_go, args=("Model evaluation",),
                  use_container_width=True)
    with right:
        st.markdown('<div class="hm-eyebrow" style="color:var(--muted);margin-bottom:8px">From the replay · '
                    'the simulation\'s highest-risk moment</div>' + specimen(A), unsafe_allow_html=True)

    days = A["observation_date"].nunique()
    zero_scored = int(sum(((A[f"{c}_status"] != "available") & A[f"{c}_score"].notna()).sum() for c in T.CATEGORIES))
    n_missing = int(sum((A[f"{c}_status"] != "available").sum() for c in T.CATEGORIES))
    try:
        P = load("backtest_results")
        cols = [c for c in P.columns if c.startswith("replay_match_")]
        replayed = int(P[cols].all(axis=1).sum()) if cols else 0
        replay_fact = (f'<div><div class="n">{replayed:,}</div><div class="l">of {len(P):,} decisions reproduced exactly '
                       'by a chronological replay</div></div>')
    except DataUnavailable:
        replay_fact = ""
    st.markdown(f'<div class="hm-facts"><div><div class="n">5</div><div class="l">evidence streams fused per zone and shift</div></div>'
                f'<div><div class="n">{A["zone_id"].nunique()}</div><div class="l">zones across {A["site_id"].nunique()} '
                f'simulated sites</div></div><div><div class="n">{len(A):,}</div><div class="l">zone-shift assessments '
                f'over {days} days</div></div><div><div class="n">{zero_scored}</div><div class="l">of {n_missing:,} unavailable '
                f'category readings given a score — missing is never zero</div></div>{replay_fact}</div>', unsafe_allow_html=True)

    st.markdown('<div class="hm-h">How it works</div><div class="hm-hsub">Four steps, each decided only with information '
                'that had actually arrived by the decision time — reporting delays included.</div>'
                '<div class="hm-steps">'
                '<div><div class="k">01 · as-of ingest</div><div class="t">Respect the clock</div><div class="d">Each source is '
                'used only after its report lands. Clinic data arrives a day later; absence two days later.</div>'
                '<div class="m">recorded_at → reporting_timestamp</div></div>'
                '<div><div class="k">02 · per-stream baselines</div><div class="t">Compare a zone with itself</div>'
                '<div class="d">Robust z-scores and CUSUM against each zone\'s own recent history, so a hot foundry is not '
                'judged like a cool packing hall.</div><div class="m">median · MAD · CUSUM</div></div>'
                '<div><div class="k">03 · evidence-aware fusion</div><div class="t">Demand corroboration</div>'
                '<div class="d">Five 0–100 category scores, weights re-spread over what is available. Heat alone never makes '
                'a health alert.</div><div class="m">persistence · independent categories</div></div>'
                '<div><div class="k">04 · explain &amp; look ahead</div><div class="t">Show the working</div>'
                '<div class="d">Every assessment lists contributions, missing and stale sources. A 24 h forecast ranks risk '
                'beside a persistence baseline.</div><div class="m">contribution points · ranking score</div></div>'
                '</div>', unsafe_allow_html=True)

    st.markdown('<div class="hm-h">Principles</div>', unsafe_allow_html=True)
    p1, p2 = st.columns([1, 1], gap="large")
    with p1:
        st.markdown('<div class="hm-quote">Missing is never <em>zero</em>.</div><div class="hm-qd">An unreported clinic shift '
                    'is excluded and named — not counted as a quiet day. Reliability falls as evidence thins.</div>'
                    '<div class="hm-quote">Evidence before <em>alarm</em>.</div><div class="hm-qd">A hot afternoon earns an '
                    'environmental warning. A health alert needs a health signal, corroborated and persistent.</div>',
                    unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="hm-quote">A forecast is a <em>ranking</em>, not a probability.</div><div class="hm-qd">'
                    'Scores order zones by concern. They are uncalibrated and are never shown as a percentage chance.</div>'
                    '<div class="hm-quote">Flag it, never <em>fix</em> it.</div><div class="hm-qd">Heat-index readings beyond '
                    'the NWS chart range stay as recorded, marked as extrapolated, and cannot quietly drive a claim.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="hm-h">How it performed</div><div class="hm-hsub">Measured against the simulation\'s ground '
                'truth with thresholds fixed in advance. A handful of simulated events — read these as behaviour, not '
                'proof.</div>', unsafe_allow_html=True)
    try:
        st.markdown(results_html(), unsafe_allow_html=True)
    except DataUnavailable as e:
        C.empty_state("Evaluation outputs not found", str(e))

    st.markdown('<div class="hm-h">Explore</div>', unsafe_allow_html=True)
    for i, (page, desc) in enumerate(DIRECTORY, start=1):
        c1, c2 = st.columns([6, 1], vertical_alignment="center")
        c1.markdown(f'<div class="hm-dir"><div class="r"><span class="i">0{i}</span><span class="t">{html.escape(page)}</span>'
                    f'<span class="d">{html.escape(desc)}</span></div></div>', unsafe_allow_html=True)
        c2.button("Open →", key=f"hm_dir_{i}", on_click=_go, args=(page,), use_container_width=True)

    st.markdown('<div class="hm-foot"><span>Prevenza is a hackathon demonstrator built on simulated data (3 sites, 9 zones, '
                '90 days). It has not been clinically validated and must not inform real medical, occupational-health or '
                'emergency decisions.</span><span class="mono">Read-only · saved pipeline outputs</span></div>',
                unsafe_allow_html=True)
