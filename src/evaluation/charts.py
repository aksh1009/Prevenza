"""Static report figures (PNG) for docs/backtesting_report.md and docs/model_comparison.md.

Palette: reference data-viz palette. Models use categorical slots 1-3 (validated all-pairs); event phases use the
reserved status colours with text labels; magnitudes use the single-hue blue ramp. Every figure's numbers are also
given as tables in the reports.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
MODEL_NAMES = {"A": "A: MAD + CUSUM", "B": "B: Isolation Forest", "C": "C: Risk fusion"}
STATUS = {"warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
BLUES = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "font.size": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def event_timelines(P: pd.DataFrame, events: pd.DataFrame, path: Path) -> Path:
    evs = events.sort_values(["event_type", "event_id", "zone_id"]).reset_index(drop=True)
    fig, axes = plt.subplots(len(evs), 1, figsize=(10, 2.1 * len(evs)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, ev in zip(axes, evs.itertuples()):
        lo, hi = ev.onset_timestamp - pd.Timedelta(days=7), ev.end_timestamp + pd.Timedelta(days=5)
        z = P[(P["zone_id"] == ev.zone_id) & P["decision_time"].between(lo, hi)].sort_values("decision_time")
        if pd.notna(ev.escalation_timestamp):
            ax.axvspan(ev.onset_timestamp, ev.escalation_timestamp, color=STATUS["warning"], alpha=0.18, lw=0)
            ax.axvspan(ev.escalation_timestamp, ev.end_timestamp, color=STATUS["critical"], alpha=0.12, lw=0)
        else:
            ax.axvspan(ev.onset_timestamp, ev.end_timestamp, color=INK2, alpha=0.12, lw=0)
        ax.plot(z["decision_time"], z["risk_score"], color=INK2, lw=1.5, label="C risk score (0-100)")
        for i, (m, col) in enumerate([("A", "model_a_alert"), ("B", "model_b_alert"), ("C", "model_c_health_alert")]):
            hit = z[z[col].fillna(False).astype(bool)]
            ax.scatter(hit["decision_time"], np.full(len(hit), 108 + 8 * i), marker="|", s=90, lw=2,
                       color=MODEL_COLORS[m], label=f"{MODEL_NAMES[m]} alert")
        ax.set_ylim(0, 128)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(lo, hi)
        kind = "negative control (spike)" if pd.isna(ev.escalation_timestamp) else ev.event_type.replace("_", " ")
        ax.set_title(f"{ev.event_id} - {ev.zone_id} - {kind}", loc="left")
        ax.tick_params(axis="x", labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles += [Patch(color=STATUS["warning"], alpha=0.35), Patch(color=STATUS["critical"], alpha=0.3),
                Patch(color=INK2, alpha=0.25)]
    labels += ["emerging phase (label 1)", "escalating phase (label 2)", "control spike window"]
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Event timelines: risk score and alerts per model (synthetic data)", y=1.05, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def lead_times(R: pd.DataFrame, path: Path) -> Path:
    esc = R[R["role"] == "escalating_event"].copy()
    esc["unit"] = esc["event_id"] + " " + esc["zone_id"]
    units = sorted(esc["unit"].unique())
    models = ["A", "B", "C"]
    fig, ax = plt.subplots(figsize=(8, 0.65 * len(units) + 1.4))
    h = 0.26
    for i, m in enumerate(models):
        d = esc[esc["model"] == m].set_index("unit")
        for j, u in enumerate(units):
            y = j + (i - 1) * h
            if u not in d.index or not d.loc[u, "included"]:
                ax.text(0.2, y, f"{m}: not evaluable (training period)", va="center", fontsize=7, color=INK2)
                continue
            if not d.loc[u, "detected"]:
                ax.text(0.2, y, f"{m}: not detected", va="center", fontsize=7, color=INK2)
                continue
            v = d.loc[u, "lead_time_to_escalation_hours"] / 24
            ax.barh(y, v, height=h * 0.9, color=MODEL_COLORS[m], edgecolor=SURFACE, lw=1)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(units)), units)
    ax.invert_yaxis()
    ax.set_xlabel("days of warning before scheduled escalation (negative = alert after escalation)")
    hs = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in models]
    ax.legend(hs, [MODEL_NAMES[m] for m in models], loc="lower right")
    ax.set_title("Lead time to escalation per event-zone (A, C: full window; B: out-of-sample window)", loc="left")
    return _save(fig, path)


def confusion_matrices(multi: pd.DataFrame, binary: dict[str, dict], path: Path) -> Path:
    fig = plt.figure(figsize=(12, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.2, 1, 1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    v = multi.to_numpy()
    ax.imshow(v, cmap=BLUES, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(v.shape[1]), multi.columns, rotation=20)
    ax.set_yticks(range(v.shape[0]), multi.index)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, int(v[i, j]), ha="center", va="center",
                    color=SURFACE if v[i, j] > v.max() * 0.55 else INK, fontsize=9)
    ax.set_title("C: label x assessment level (full window)", loc="left")
    ax.set_xlabel("model C level"), ax.set_ylabel("ground-truth label")
    for k, (m, d) in enumerate(binary.items()):
        a = fig.add_subplot(gs[k + 1])
        mat = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
        a.imshow(mat, cmap=BLUES, aspect="auto")
        a.grid(False)
        a.set_xticks([0, 1], ["no alert", "alert"])
        a.set_yticks([0, 1], ["negative", "positive"])
        for i in range(2):
            for j in range(2):
                a.text(j, i, mat[i, j], ha="center", va="center", color=SURFACE if mat[i, j] > mat.max() * 0.55 else INK)
        a.set_title(MODEL_NAMES[m], loc="left", fontsize=9)
    fig.suptitle("Observation-level confusion matrices (binary: comparison window, identical rows)", x=0.01,
                 ha="left", y=1.04, fontsize=12)
    return _save(fig, path)


def grouped_bars(table: pd.DataFrame, path: Path, title: str, ylabel: str, ylim=None) -> Path:
    """table: index = groups (x), columns = model keys."""
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(table)), 3.6))
    x = np.arange(len(table))
    w = 0.8 / len(table.columns)
    for i, m in enumerate(table.columns):
        ax.bar(x + (i - (len(table.columns) - 1) / 2) * w, table[m].fillna(0), width=w, color=MODEL_COLORS[m],
               edgecolor=SURFACE, lw=2, label=MODEL_NAMES[m])
        for xi, v in zip(x, table[m]):
            if pd.isna(v):
                ax.text(xi + (i - (len(table.columns) - 1) / 2) * w, 0.01, "n/a", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, table.index, rotation=0)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title(title, loc="left")
    return _save(fig, path)
