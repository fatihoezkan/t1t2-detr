"""Figures for one finished run: prediction stages (A), grouping calibration (B), query
behaviour (C), existence-score distribution (D) and the existence-score metric table (E).

Visual language follows Schlund's diffusion-DETR thesis (his Figures 7, 9, 11, 12, 20 and 23)
with the MD-FA plane replaced by T1-T2 and peak size encoding weight. Entry point:
johannes_figure_set(run_dir, out_dir=...). The run directory is only read.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # cluster nodes have no display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import TargetNormalizer, VoxelDataset
from .device import get_device
from .eval import (
    compute_metrics,
    parameter_recovery_analysis,
    true_compartments,
)
from .model import build_model
from .postprocess import (
    calibrate_grouping,
    existence_score_metrics,
    grouped_predictions,
    query_diagnostics,
    query_table,
    targets_normalized,
    threshold_only_predictions,
)

# ---------------------------------------------------------------------------------------
# Style. Kept local so a figure produced on the cluster and one produced on the laptop look
# the same regardless of which matplotlib stylesheet is installed.
# ---------------------------------------------------------------------------------------

# Colour-vision-safe (Wong/Okabe-Ito). BLUE always means ground truth or the existence score,
# ORANGE always means prediction or the weight; the binding is the same in every figure.
BLUE = "#0072B2"      # ground truth · existence score · label 0
ORANGE = "#D55E00"    # prediction · predicted weight · label 1
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#5A5A5A"
LIGHT = "#BBBBBB"

# Three font sizes by role: body text, annotations, tick labels.
SIZE_BASE, SIZE_ANNOT, SIZE_TICK = 9, 8, 7


def apply_style():
    """Apply the rcParams every figure in this module assumes. Safe to call repeatedly."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": SIZE_BASE, "axes.titlesize": SIZE_BASE, "axes.labelsize": SIZE_BASE,
        "legend.fontsize": SIZE_ANNOT, "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "axes.titlelocation": "left", "axes.titleweight": "normal", "axes.titlepad": 5.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "legend.frameon": False, "lines.linewidth": 1.6,
    })


def panel_letter(ax, letter, dx=-0.14, dy=1.06):
    """Draw the bold panel letter above the top-left corner of the axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=SIZE_BASE + 2,
            fontweight="bold", va="bottom", ha="left")


def _check_no_overlapping_text(fig, ignore_ticks=True):
    """Report visible text boxes that overlap each other.

    Tick labels are skipped by default, since a tick label on its own spine is not a finding.
    Returns a list of description strings; an empty list means the figure is clean.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    if ignore_ticks:
        tickset = {id(t) for ax in fig.axes
                   for t in ax.get_xticklabels() + ax.get_yticklabels()}
        texts = [(t, b) for t, b in texts if id(t) not in tickset]
    findings = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                findings.append(f"{a.get_text()!r} <-> {b.get_text()!r}")
    return findings


def _save(fig, path, verify=True):
    """Save the figure, run the overlap check, and return (path, findings)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    findings = _check_no_overlapping_text(fig) if verify else []
    fig.savefig(path)
    plt.close(fig)
    return str(path), findings


# ---------------------------------------------------------------------------------------
# Figure A: the four postprocessing stages for one voxel (Schlund Figs. 7 and 9)
# ---------------------------------------------------------------------------------------

def _peak_size(weight):
    """Marker area for a compartment of a given weight.

    Affine in the weight rather than proportional, so a 0.05 compartment stays visible while a
    1.0 compartment is clearly dominant. Marker area is therefore an ordinal cue for weight,
    not a quantitative one.
    """
    return 40.0 + 620.0 * np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)


#: Candidate label offsets in points, tried in order: directly above first, then further out
#: and to the sides, so an isolated peak gets the conventional label-above placement.
_LABEL_OFFSETS = [(0, 10), (0, -14), (14, 4), (-14, 4), (14, -10), (-14, -10),
                  (0, 20), (0, -24), (24, 10), (-24, 10), (24, -18), (-24, -18),
                  (34, 2), (-34, 2), (0, 30), (0, -34)]


def _label_queries(ax, params_ms, score, min_leader_pt=16.0):
    """Label every query marker Q1..Qn without letting the labels overlap.

    Must be called after tight_layout: collisions are measured in display coordinates, and a
    later change of axes size moves the data anchors while the point offsets stay fixed.
    Placement is greedy, in descending order of existence score: for each label take the first
    candidate offset that clears every placed label and stays inside the axes. A thin leader
    line is drawn when the chosen offset is more than min_leader_pt from the marker.
    """
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    placed = []
    for q in np.argsort(-np.asarray(score, dtype=float)):
        xy = (params_ms[q, 0], params_ms[q, 1])
        chosen = None
        for dx, dy in _LABEL_OFFSETS:
            txt = ax.annotate(f"Q{q + 1}", xy, textcoords="offset points", xytext=(dx, dy),
                              ha="center", va="center", fontsize=SIZE_TICK, color=GREY,
                              zorder=6)
            box = txt.get_window_extent(r).expanded(1.12, 1.35)
            if not any(box.overlaps(b) for b in placed) and ax.bbox.contains(*box.p0) \
                    and ax.bbox.contains(*box.p1):
                chosen = (txt, box, dx, dy)
                break
            txt.remove()
        if chosen is None:
            # All fixed offsets collided, which happens when many markers pile onto one point.
            # Fall back to a widening ring around the marker: 12 angles at growing radii.
            for radius_pt in (44, 58, 74, 92, 112):
                for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    dx = float(radius_pt * np.cos(ang))
                    dy = float(radius_pt * np.sin(ang))
                    txt = ax.annotate(f"Q{q + 1}", xy, textcoords="offset points",
                                      xytext=(dx, dy), ha="center", va="center",
                                      fontsize=SIZE_TICK, color=GREY, zorder=6)
                    box = txt.get_window_extent(r).expanded(1.12, 1.35)
                    if not any(box.overlaps(b) for b in placed) \
                            and ax.bbox.contains(*box.p0) and ax.bbox.contains(*box.p1):
                        chosen = (txt, box, dx, dy)
                        break
                    txt.remove()
                if chosen is not None:
                    break
        if chosen is None:
            # Out of room. Keep the label anyway and let the overlap check report it.
            dx, dy = _LABEL_OFFSETS[-1]
            txt = ax.annotate(f"Q{q + 1}", xy, textcoords="offset points", xytext=(dx, dy),
                              ha="center", va="center", fontsize=SIZE_TICK, color=GREY,
                              zorder=6)
            chosen = (txt, txt.get_window_extent(r), dx, dy)
        txt, box, dx, dy = chosen
        placed.append(box)
        if (dx ** 2 + dy ** 2) ** 0.5 > min_leader_pt:
            txt.set_arrowprops = None
            ax.annotate("", xy=xy, xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=LIGHT,
                                        shrinkA=1.0, shrinkB=1.0), zorder=1)


def prediction_stages_figure(voxel, query_tbl, true_comps, threshold, radius, normalizer,
                             path, grouping_kwargs=None, t1_lim=(45, 3800), t2_lim=(4.5, 560),
                             label_queries=True, title=None):
    """Four panels for one voxel: ground truth, all queries, thresholded, grouped.

    All panels share the same log axes, passed in as t1_lim and t2_lim rather than derived per
    panel, so a peak can be followed across them. Panel a: true compartments, marker area
    encoding weight. Panel b: all n_queries guesses labelled Q1..Qn, marker opacity encoding
    the existence score. Panel c: queries scoring above the threshold. Panel d: after merging
    peaks within radius. Ground truth is repeated as a hollow outline in c and d. `voxel` is
    the row index into query_tbl and true_comps.
    """
    gk = dict(aggregate="weight", include_weight=False, renormalize=True)
    gk.update(grouping_kwargs or {})

    params_ms = query_tbl["params"][voxel]
    params_nm = query_tbl["params_norm"][voxel]
    score = query_tbl["exist_prob"][voxel]
    keep = score > threshold

    filtered = [(float(params_ms[q, 0]), float(params_ms[q, 1]), float(params_ms[q, 2]))
                for q in np.flatnonzero(keep)]
    # grouped_predictions already returns milliseconds. Grouping happens in normalized space
    # internally and is denormalized on the way out. Do not convert again.
    merged = grouped_predictions(
        {"params_norm": params_nm[None], "exist_prob": score[None]},
        normalizer, threshold, radius, **gk,
    )[0]

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.5), sharex=True, sharey=True)
    stage_titles = ("Ground truth", "Full prediction set",
                    f"Filtered: score > {threshold:.2f}",
                    f"Grouped: radius {radius:.3f}")

    for ax, letter, stitle in zip(axes, "abcd", stage_titles):
        ax.set(xscale="log", yscale="log", xlim=t1_lim, ylim=t2_lim, xlabel="T1 (ms)")
        ax.set_title(f"{letter}) {stitle}")
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
        panel_letter(ax, "", dx=0)          # letters are already in the titles
    axes[0].set_ylabel("T2 (ms)")

    # a) truth
    t = np.asarray(true_comps, dtype=float).reshape(-1, 3)
    axes[0].scatter(t[:, 0], t[:, 1], s=_peak_size(t[:, 2]), facecolor=BLUE,
                    edgecolor="white", lw=0.9, alpha=0.85, zorder=3)

    # b) every query, opacity = existence score
    for q in range(len(score)):
        # Floor the alpha at 0.10 so a query with a near-zero score stays visible.
        a = 0.10 + 0.85 * float(score[q])
        axes[1].scatter(params_ms[q, 0], params_ms[q, 1], s=_peak_size(params_ms[q, 2]),
                        facecolor=ORANGE, edgecolor="white", lw=0.7, alpha=min(a, 0.95),
                        zorder=3)
    # Query labels are placed after tight_layout; see _label_queries.

    # c) and d)
    for ax, peaks in ((axes[2], filtered), (axes[3], merged)):
        p = np.asarray(peaks, dtype=float).reshape(-1, 3)
        if len(p):
            ax.scatter(p[:, 0], p[:, 1], s=_peak_size(p[:, 2]), facecolor=ORANGE,
                       edgecolor="white", lw=0.8, alpha=0.9, zorder=3)
        # Ground truth repeated as a hollow outline so a hit can be told from a near-miss.
        ax.scatter(t[:, 0], t[:, 1], s=_peak_size(t[:, 2]), facecolor="none",
                   edgecolor=BLUE, lw=1.1, zorder=2)

    # One legend for the whole figure, below the panels; panel a contains no predictions.
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=BLUE, mec="white", ms=7,
                   label="true compartment"),
        plt.Line2D([], [], marker="o", ls="", mfc=ORANGE, mec="white", ms=7,
                   label="predicted compartment"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=BLUE, ms=7,
                   label="true compartment, repeated as an outline in c) and d)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=SIZE_ANNOT,
               bbox_to_anchor=(0.5, -0.02))
    if title is None:
        title = (f"Voxel {voxel}: {len(t)} true compartments, "
                 f"{len(filtered)} after threshold, {len(merged)} after grouping "
                 "(marker area = weight, opacity = existence score)")
    fig.suptitle(title, fontsize=SIZE_BASE, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    if label_queries:
        _label_queries(axes[1], params_ms, score)
    return _save(fig, path)


# ---------------------------------------------------------------------------------------
# Figure B: grouping radius, validation sweep and the frozen application to test
# ---------------------------------------------------------------------------------------

def grouping_calibration_figure(calibration, test_stage_table, path, threshold):
    """Validation sensitivity curve on the left, the effect on test on the right.

    Left: validation parameter-set error against radius, one line per aggregation rule, lower
    is better. Radius 0 is the no-grouping control and the dashed line marks the selected
    radius; a rising curve means real pools are being merged away. Right: exact-count accuracy
    per true compartment count on test, threshold only against threshold plus grouping.
    """
    cur = pd.DataFrame(calibration["curve"])
    sel = calibration["selected"]
    # Show the sweep at the selected flag settings so the lines differ only in radius and
    # aggregation rule. Bracket access throughout: `aggregate` is also a DataFrame method, so
    # `cur.aggregate` returns the method rather than the column.
    shown = cur[(cur["include_weight"] == sel["include_weight"])
                & (cur["renormalize"] == sel["renormalize"])]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    for aggregate, colour, marker in zip(("mean", "weight", "confidence"),
                                         (BLUE, ORANGE, GREEN), ("o", "s", "^")):
        d = shown[shown["aggregate"] == aggregate].sort_values("radius")
        if len(d):
            ax.plot(d["radius"], d["parameter_set_error"], color=colour, marker=marker,
                    ms=3.4, label=f"{aggregate} centre")
    base = float(cur[cur["radius"] == 0]["parameter_set_error"].iloc[0])
    ax.axhline(base, color=GREY, ls=":", lw=1.2)
    ax.annotate("no grouping (radius 0)", (ax.get_xlim()[1], base), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=SIZE_ANNOT,
                color=GREY)
    ax.axvline(sel["radius"], color=PURPLE, ls="--", lw=1.4)
    # Anchored at mid height, to the right of the line: the legend takes the top-left and the
    # curves the lower-left.
    ax.annotate(f"selected r = {sel['radius']:.3f}", (sel["radius"], 0.55),
                xycoords=("data", "axes fraction"), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=SIZE_ANNOT, color=PURPLE)
    ax.set(xlabel="grouping radius (normalized log-T1/T2 units)",
           ylabel="validation parameter-set error")
    ax.set_title("a) Validation sensitivity (lower is better)")
    ax.legend(loc="upper left")
    panel_letter(ax, "")

    ax = axes[1]
    labels = list(test_stage_table.index)
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, 100 * test_stage_table["threshold_only"], w, color=BLUE,
                label=f"threshold only (>{threshold:.2f})")
    b2 = ax.bar(x + w / 2, 100 * test_stage_table["grouped"], w, color=ORANGE,
                label=f"+ grouping (r = {sel['radius']:.3f})")
    ax.bar_label(b1, fmt="%.1f", fontsize=SIZE_TICK, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=SIZE_TICK, padding=2)
    ax.set(xticks=x, xticklabels=labels, ylabel="exact-count accuracy (%)", ylim=(0, 128))
    ax.set_title("b) Frozen radius applied once to test (higher is better)")
    # Legend above the bars: they reach about 95 % at K = 1, so a legend inside the axes
    # sits on data.
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    panel_letter(ax, "")

    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------------------
# Figure C: query learning (Schlund Figs. 20 and 23)
# ---------------------------------------------------------------------------------------

def query_histograms_figure(query_tbl, path, threshold=0.5, bins=20, n_cols=5):
    """Per-query histograms of existence score (blue) and predicted weight (orange).

    One panel per query over the whole split, as densities so the two quantities can share the
    [0, 1] axis. Similar score distributions across queries mean balanced activation; a few
    queries with all their mass near 1 and the rest at 0 mean query collapse. Differences in
    the weight distributions are the evidence of specialisation: a query concentrated at low
    weight proposes minor pools. The fraction of voxels in which each query is active is
    printed in its panel.
    """
    S = np.asarray(query_tbl["exist_prob"])
    W = np.asarray(query_tbl["params"])[..., 2]
    n_q = S.shape[1]
    n_rows = int(np.ceil(n_q / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.55 * n_cols, 2.25 * n_rows),
                             sharex=True, squeeze=False)
    edges = np.linspace(0, 1, bins + 1)
    for q in range(n_q):
        ax = axes[q // n_cols][q % n_cols]
        ax.hist(S[:, q], bins=edges, density=True, color=BLUE, alpha=0.85,
                label="existence score")
        ax.hist(W[:, q], bins=edges, density=True, color=ORANGE, alpha=0.65,
                label="predicted weight")
        ax.set_title(f"Query {q + 1}")
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=SIZE_TICK)
        # Fraction of voxels in which this query is active, printed inside the panel.
        ax.text(0.5, 0.96, f"active in {100 * np.mean(S[:, q] > threshold):.0f}% of voxels",
                transform=ax.transAxes, ha="center", va="top", fontsize=SIZE_TICK,
                color=GREY)
    for q in range(n_q, n_rows * n_cols):
        axes[q // n_cols][q % n_cols].axis("off")
    for c in range(n_cols):
        axes[-1][c].set_xlabel("value in [0, 1]")
    for r in range(n_rows):
        axes[r][0].set_ylabel("density")
    axes[0][0].legend(loc="center right", fontsize=SIZE_TICK)
    fig.suptitle("Each query's own output distribution: existence score (blue) versus "
                 f"predicted weight (orange), whole split; \"active\" means score > "
                 f"{threshold:.2f}", fontsize=SIZE_BASE, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _save(fig, path)


def query_scatter_figure(query_tbl, path, n_samples=500, seed=0, n_cols=5,
                         t1_lim=(45, 3800), t2_lim=(4.5, 560), threshold=None):
    """Per-query T1-T2 scatter over a random subsample of voxels, following Schlund's Fig. 23.

    One panel per query on identical log axes; each point is the query's raw output for one
    voxel, before thresholding. A tight cloud means the query has claimed a region of the
    plane (the specialisation Carion et al. described); a cloud filling the panel means a
    general-purpose proposer. With a threshold, points above it are drawn solid and the rest
    faint. The diagonal edge in some panels is the training constraint T2 < T1.
    """
    P = np.asarray(query_tbl["params"])
    S = np.asarray(query_tbl["exist_prob"])
    n_q = P.shape[1]
    rng = np.random.default_rng(seed)
    take = rng.choice(len(P), size=min(n_samples, len(P)), replace=False)

    n_rows = int(np.ceil(n_q / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.55 * n_cols, 2.45 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    for q in range(n_q):
        ax = axes[q // n_cols][q % n_cols]
        t1, t2, sc = P[take, q, 0], P[take, q, 1], S[take, q]
        if threshold is None:
            ax.scatter(t1, t2, s=7, color=BLUE, alpha=0.45, lw=0)
        else:
            on = sc > threshold
            ax.scatter(t1[~on], t2[~on], s=6, color=LIGHT, alpha=0.55, lw=0,
                       label="score below threshold")
            ax.scatter(t1[on], t2[on], s=8, color=BLUE, alpha=0.75, lw=0,
                       label="score above threshold")
        ax.set(xscale="log", yscale="log", xlim=t1_lim, ylim=t2_lim)
        ax.set_title(f"Query {q + 1}")
        ax.grid(True, which="both", ls=":", lw=0.35, alpha=0.45)
    for q in range(n_q, n_rows * n_cols):
        axes[q // n_cols][q % n_cols].axis("off")
    for c in range(n_cols):
        axes[-1][c].set_xlabel("T1 (ms)")
    for r in range(n_rows):
        axes[r][0].set_ylabel("T2 (ms)")
    if threshold is not None:
        axes[0][0].legend(loc="upper left", fontsize=SIZE_TICK, markerscale=1.6)
    fig.suptitle(f"Which part of the T1-T2 plane each query claims "
                 f"({len(take)} random test voxels, unfiltered outputs)",
                 fontsize=SIZE_BASE, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _save(fig, path)


# ---------------------------------------------------------------------------------------
# Figure D: existence-score distribution by Hungarian label (Schlund Figs. 11, 12, 17, 18)
# ---------------------------------------------------------------------------------------

def existence_distribution_figure(diag, exist_prob, n_comp, path, threshold=None, bins=25):
    """Existence scores of matched queries (label 1) against unmatched ones (label 0).

    One panel per true compartment count plus the pooled case. Densities rather than counts:
    with ten queries and one to three compartments, 70 to 90 % of labels are 0, and on a count
    axis the label-0 bars would dwarf everything. The vertical line is the operating threshold;
    label-1 mass to its left is a miss, label-0 mass to its right is a false positive or a
    duplicate. Label-1 mass near 0 means the regression found the compartment but the
    existence head will not report it, a score problem rather than a parameter problem. The
    per-count panels matter because the class balance differs: 10 % of labels are 1 at one
    compartment, 30 % at three.
    """
    label = np.asarray(diag["label"]).astype(bool)
    S = np.asarray(exist_prob)
    n_comp = np.asarray(n_comp)
    groups = [(f"K = {k}", np.flatnonzero(n_comp == k)) for k in sorted(set(n_comp.tolist()))]
    groups.append(("All counts pooled", np.arange(len(n_comp))))

    fig, axes = plt.subplots(1, len(groups), figsize=(3.15 * len(groups), 3.3),
                             sharey=True)
    axes = np.atleast_1d(axes)
    edges = np.linspace(0, 1, bins + 1)
    for ax, letter, (name, idx) in zip(axes, "abcdefgh", groups):
        s0 = S[idx][~label[idx]]
        s1 = S[idx][label[idx]]
        ax.hist(s0, bins=edges, density=True, color=BLUE, alpha=0.75,
                label=f"label 0, unmatched ({len(s0):,})")
        ax.hist(s1, bins=edges, density=True, color=ORANGE, alpha=0.65,
                label=f"label 1, matched ({len(s1):,})")
        if threshold is not None:
            ax.axvline(threshold, color=GREY, ls="--", lw=1.2)
            # Anchored at the bottom of the panel; at the top it collides with the legend.
            ax.annotate(f"threshold {threshold:.2f}", (threshold, 0.02),
                        xycoords=("data", "axes fraction"), xytext=(-4, 0),
                        textcoords="offset points", rotation=90, ha="right", va="bottom",
                        fontsize=SIZE_TICK, color=GREY)
        share = 100 * len(s1) / max(len(s1) + len(s0), 1)
        ax.set_title(f"{letter}) {name}: {share:.0f}% of queries matched")
        ax.set(xlabel="predicted existence score", xlim=(0, 1))
        ax.legend(loc="upper center", fontsize=SIZE_TICK)
    axes[0].set_ylabel("density")
    fig.suptitle("Does the existence head separate matched from unmatched queries? "
                 "(density, because the classes are imbalanced)",
                 fontsize=SIZE_BASE, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


# ---------------------------------------------------------------------------------------
# Table E: existence-score metrics (Schlund Tables 2 and 4)
# ---------------------------------------------------------------------------------------

#: Column order and exported header for the metric table, using Schlund's column names.
TABLE_COLUMNS = [
    ("K", "#comp"), ("n_voxels", "voxels"), ("acc", "Acc"), ("prec", "Prec"),
    ("rec", "Rec"), ("n_peak", "#peak"), ("cost_at_0.5", "costs@0.5"),
    ("cost_at_0.8", "costs@0.8"), ("pacc", "PAcc"), ("scorr", "SCorr"),
]


def existence_metric_table(diag, exist_prob, n_comp, threshold):
    """Schlund's Tables 2 and 4, per compartment count and pooled, as a DataFrame.

    The columns are documented on postprocess.existence_score_metrics. PAcc and SCorr ask
    whether the existence score ranks prediction quality; their no-information values are 0.5
    and 0.0.
    """
    rows = []
    for k in list(sorted(set(np.asarray(n_comp).tolist()))) + ["pooled"]:
        idx = (np.arange(len(n_comp)) if k == "pooled"
               else np.flatnonzero(np.asarray(n_comp) == k))
        sub = {key: np.asarray(v)[idx] for key, v in diag.items()}
        m = existence_score_metrics(sub, np.asarray(exist_prob)[idx], threshold=threshold)
        m["K"] = k
        m["mean_true_count"] = float(np.mean(np.asarray(n_comp)[idx]))
        rows.append(m)
    df = pd.DataFrame(rows)
    ordered = [c for c, _ in TABLE_COLUMNS] + ["mean_true_count", "n_pacc_voxels",
                                               "n_scorr_voxels", "threshold"]
    return df[[c for c in ordered if c in df.columns]]


def existence_metric_markdown(df, threshold, reference=None):
    """Render the metric table as markdown, optionally with a reference row.

    `reference` maps column names to values for a comparison model, such as Schlund's c3 at
    three compartments, rendered as its own row. Different problem and data, so context only.
    """
    head = "| " + " | ".join(h for _, h in TABLE_COLUMNS) + " |"
    rule = "|" + "|".join(["---:"] * len(TABLE_COLUMNS)) + "|"
    lines = [f"Existence-score metrics at threshold {threshold:.2f}", "", head, rule]

    def _fmt(col, v):
        if col in ("K", "n_voxels"):
            return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
        if col == "n_peak":
            return f"{v:.2f}"
        if col.startswith("cost"):
            return f"{v:.4f}"
        return f"{v:.3f}"

    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(c, r[c]) for c, _ in TABLE_COLUMNS) + " |")
    if reference:
        lines.append("| " + " | ".join(
            str(reference.get(c, "-")) for c, _ in TABLE_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
# The one-call entry point
# ---------------------------------------------------------------------------------------

def load_run(run_dir, split="test", device=None, limit=None, checkpoint="best.pt"):
    """Load a finished run's config and checkpoint and build what the figures need.

    Read-only; nothing is written into run_dir. Returns a dict with cfg, model, device,
    normalizer, ds, split, threshold, epoch, query_tbl, trues (millisecond compartment lists),
    targets_norm and n_comp. The threshold is read from the run's own calibration file when it
    exists, so the figures sit at the same operating point as the metrics.
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    dev = get_device(device) if device is not None else torch.device("cpu")

    model = build_model(cfg.model)
    ckpt = torch.load(run_dir / "checkpoints" / checkpoint, map_location="cpu",
                      weights_only=True)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt["state_dict"])
    model.to(dev).eval()

    normalizer = TargetNormalizer.from_config(cfg.data)
    paths = {"train": cfg.data.train_path, "val": cfg.data.val_path,
             "test": cfg.data.test_path}[split]
    ds = VoxelDataset(paths, cfg.data, normalizer, limit=limit)

    threshold = float(cfg.evaluation.fixed_threshold)
    calib = run_dir / "threshold_calibration.json"
    if calib.exists():
        threshold = float(json.loads(calib.read_text())["selected_threshold"])

    targets_n, n_comp = targets_normalized(ds)
    return {
        "cfg": cfg, "model": model, "device": dev, "normalizer": normalizer, "ds": ds,
        "split": split, "threshold": threshold, "epoch": int(ckpt.get("epoch", -1)) + 1,
        "query_tbl": query_table(model, ds, dev, normalizer),
        "trues": true_compartments(ds), "targets_norm": targets_n, "n_comp": n_comp,
    }


def _pick_examples(query_tbl, trues, n_comp, threshold, seed=0):
    """Pick the Figure A voxels: a typical success per compartment count, plus two failures.

    Chosen by rule so the figure regenerates identically and cannot be cherry-picked. A typical
    success has the right predicted count and the median log T1/T2 error among the eligible
    voxels, so it is typical rather than the best available. The failures are a random
    over-count and a random under-count.
    """
    preds = threshold_only_predictions(query_tbl, threshold)
    counts = np.asarray([len(p) for p in preds])
    n_comp = np.asarray(n_comp)
    rng = np.random.default_rng(seed)
    picks = []

    for k in sorted(set(n_comp.tolist())):
        ok = np.flatnonzero((n_comp == k) & (counts == k))
        if not len(ok):
            continue
        err = []
        for i in ok:
            p = np.asarray(preds[i], float).reshape(-1, 3)
            t = np.asarray(trues[i], float).reshape(-1, 3)
            # Order-free score: for each true compartment take the closest prediction in
            # log space. Enough for ranking typicality; the real matching lives in eval.
            d = (np.abs(np.log(p[:, None, 0]) - np.log(t[None, :, 0]))
                 + np.abs(np.log(p[:, None, 1]) - np.log(t[None, :, 1])))
            err.append(float(d.min(axis=0).mean()))
        picks.append((int(ok[np.argsort(err)[len(err) // 2]]), f"typical success, K = {k}"))

    over = np.flatnonzero(counts > n_comp)
    if len(over):
        picks.append((int(rng.choice(over)), "over-count: an extra compartment reported"))
    under = np.flatnonzero(counts < n_comp)
    if len(under):
        picks.append((int(rng.choice(under)), "under-count: a compartment missed"))
    return picks


def johannes_figure_set(run_dir, out_dir=None, split="test", calibration_split="val",
                        radius=None, grouping_kwargs=None, radii=None, n_examples=None,
                        limit=None, seed=0, reference_row=None, log=print):
    """Regenerate the complete figure set and metric table for one run.

    Parameters
    ----------
    run_dir
        A finished run directory containing config.yaml and checkpoints/best.pt. Read only.
    out_dir
        Where the figures are written. Defaults to <run_dir>/figures_johannes, separate from
        the run's own figures/ so nothing there is overwritten.
    calibration_split
        The split the grouping radius is selected on. "val" by default. "test" is accepted for
        demonstrating the leakage and is logged as a warning.
    radius
        Skip calibration and use this radius, for re-applying a previously frozen choice.
    radii, grouping_kwargs
        The sweep grid and the fixed grouping options. See postprocess.calibrate_grouping and
        postprocess.group_peaks.
    reference_row
        Optional extra row for the markdown table, such as the prior thesis's published
        numbers.

    Returns a manifest with the paths written, the settings selected, the metric table, the
    before-and-after test comparison, and any figure legibility findings.
    """
    apply_style()
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "figures_johannes"
    out.mkdir(parents=True, exist_ok=True)

    log(f"[viz] loading {run_dir} split={split}")
    R = load_run(run_dir, split=split, limit=limit)
    thr = R["threshold"]
    log(f"[viz] {len(R['ds'])} voxels, existence threshold {thr:.2f} "
        f"(epoch {R['epoch']}, {R['query_tbl']['exist_prob'].shape[1]} queries)")

    # --- grouping radius, selected on validation ---
    gk = dict(aggregate="weight", include_weight=False, renormalize=True)
    gk.update(grouping_kwargs or {})
    calibration = None
    if radius is None:
        if calibration_split == split:
            log(f"[viz] WARNING: calibrating the grouping radius on the reporting split "
                f"({split!r}). The resulting numbers are NOT out-of-sample.")
        C = R if calibration_split == split else load_run(run_dir, split=calibration_split,
                                                          limit=limit)
        log(f"[viz] sweeping the grouping radius on {calibration_split} "
            f"({len(C['ds'])} voxels)")
        calibration = calibrate_grouping(C["query_tbl"], C["trues"], C["normalizer"],
                                         C["threshold"], radii=radii)
        sel = calibration["selected"]
        radius = sel["radius"]
        gk = {k: sel[k] for k in ("aggregate", "include_weight", "renormalize")}
        log(f"[viz] selected radius {radius:.3f}, {gk} "
            f"({calibration['n_evaluations']} evaluations)")
        (out / "grouping_calibration.json").write_text(json.dumps(calibration, indent=2))
    elif (out / "grouping_calibration.json").exists():
        # An explicit radius skips the sweep; reuse a saved sweep from an earlier call so
        # Figure B is still produced.
        calibration = json.loads((out / "grouping_calibration.json").read_text())
        log(f"[viz] reusing the saved validation sweep for the sensitivity figure "
            f"({calibration['n_evaluations']} evaluations)")

    # --- the two postprocessing stages, scored with the ordinary evaluation code ---
    pred_t = threshold_only_predictions(R["query_tbl"], thr)
    pred_g = grouped_predictions(R["query_tbl"], R["normalizer"], thr, radius, **gk)
    n_q = R["query_tbl"]["exist_prob"].shape[1]
    stages = {}
    for name, preds in (("threshold_only", pred_t), ("grouped", pred_g)):
        m = compute_metrics(preds, R["trues"], n_queries=n_q)
        m["parameter_recovery_summary"] = parameter_recovery_analysis(
            preds, R["trues"])["summary"]
        stages[name] = m
    counts = np.asarray([len(t) for t in R["trues"]])
    stage_tbl = pd.DataFrame(
        {name: [stages[name][f"count_accuracy_n{k}"] for k in sorted(set(counts.tolist()))]
               + [stages[name]["count_accuracy"]]
         for name in stages},
        index=[f"K = {k}" for k in sorted(set(counts.tolist()))] + ["all"],
    )
    changed = int(np.sum(np.asarray([len(p) for p in pred_t])
                         != np.asarray([len(p) for p in pred_g])))
    log(f"[viz] grouping changed the compartment count of {changed} / {len(pred_t)} voxels")

    # --- existence diagnostics + table ---
    diag = query_diagnostics(R["query_tbl"], R["targets_norm"], R["n_comp"], R["cfg"].loss)
    table = existence_metric_table(diag, R["query_tbl"]["exist_prob"], R["n_comp"], thr)
    table.to_csv(out / "existence_score_metrics.csv", index=False)
    (out / "existence_score_metrics.md").write_text(
        existence_metric_markdown(table, thr, reference=reference_row))

    # --- figures ---
    written, findings = [], {}

    def _record(result):
        path, f = result
        written.append(path)
        if f:
            findings[Path(path).name] = f
        log(f"[viz] wrote {path}" + (f"  [{len(f)} text overlaps]" if f else ""))

    examples = _pick_examples(R["query_tbl"], R["trues"], R["n_comp"], thr, seed=seed)
    if n_examples is not None:
        examples = examples[:n_examples]
    for i, (voxel, why) in enumerate(examples):
        _record(prediction_stages_figure(
            voxel, R["query_tbl"], R["trues"][voxel], thr, radius, R["normalizer"],
            out / f"fig_A{i + 1}_prediction_stages_voxel{voxel}.png",
            grouping_kwargs=gk,
            title=(f"{why}, voxel {voxel}: postprocessing turns {n_q} query outputs into "
                   f"{len(pred_g[voxel])} compartments "
                   f"(true {len(R['trues'][voxel])}; marker area = weight, "
                   f"opacity = existence score)"),
        ))
    if calibration is not None:
        _record(grouping_calibration_figure(calibration, stage_tbl,
                                            out / "fig_B_grouping_calibration.png", thr))
    _record(query_histograms_figure(R["query_tbl"], out / "fig_C1_query_histograms.png",
                                    threshold=thr))
    _record(query_scatter_figure(R["query_tbl"], out / "fig_C2_query_t1t2_scatter.png",
                                 seed=seed, threshold=thr))
    _record(existence_distribution_figure(diag, R["query_tbl"]["exist_prob"], R["n_comp"],
                                          out / "fig_D_existence_distribution.png",
                                          threshold=thr))

    manifest = {
        "run_dir": str(run_dir), "out_dir": str(out), "split": split,
        "epoch": R["epoch"], "n_voxels": len(R["ds"]), "n_queries": n_q,
        "existence_threshold": thr,
        "grouping": {"radius": float(radius), "calibration_split": calibration_split, **gk},
        "voxels_changed_by_grouping": changed,
        "examples": [{"voxel": v, "reason": w} for v, w in examples],
        "stage_count_accuracy": stage_tbl.to_dict(),
        "figures": written,
        "legibility_findings": findings,
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    manifest["metric_table"] = table
    manifest["stages"] = stages
    log(f"[viz] done -> {out}")
    return manifest
