"""Grouping near-duplicate query peaks into a compartment set, plus existence-head diagnostics.

Step one is the threshold filter (eval.predictions_from_query_outputs). Step two, added here,
merges queries that land on the same compartment, following the postprocessing in Schlund's
diffusion-DETR thesis (his Figure 7d) moved to (T1, T2, weight). Distances and assignments
are computed in the model's normalised space; see query_table. Read-only analysis of a run.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .eval import parameter_recovery_analysis, predictions_from_query_outputs

# How the T1/T2 centre of a merged group is computed. See group_peaks.
AGGREGATIONS = ("mean", "weight", "confidence")

# Radius grid swept on validation. Starts at 0.0 so the sweep contains an exact no-op control.
# The upper end of 0.40 is past anything sensible on purpose: a curve still rising at the edge
# of the grid would show the grid is too small.
DEFAULT_RADII = np.round(np.arange(0.0, 0.4001, 0.025), 4)


# ---------------------------------------------------------------------------------------
# 1. Model output in both unit systems
# ---------------------------------------------------------------------------------------

def query_table(model, ds, device, normalizer, batch_size=512):
    """Run the model and keep every query, in normalised units as well as physical ones.

    The grouping distance, the Hungarian assignment and the per-query cost all work in the
    model's normalised space: a radius in milliseconds means nothing across pools from 60 ms
    to 3000 ms, and the existence labels reported here must be the ones the loss used. The
    raw sigmoid outputs are kept rather than denormalised and normalised again.

    Returns a dict with:

    params_norm  (N, Q, 3) raw head outputs in [0, 1]: normalised T1, normalised T2, weight.
    params       (N, Q, 3) the same with T1 and T2 in milliseconds.
    exist_prob   (N, Q) existence logit after the sigmoid, a score in [0, 1].
    exist_logit  (N, Q) the raw logit. Kept because it stays informative where the score
                 saturates: 0.9999 and 0.999999 are far apart as logits.
    """
    model.eval()
    norm_chunks, prob_chunks, logit_chunks = [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            X = ds.X[i:i + batch_size].to(device)
            out = model(X)
            out = out["pred"] if isinstance(out, dict) else out
            row = out.detach().cpu().numpy().astype(np.float64)
            norm_chunks.append(row[..., :3])
            logit_chunks.append(row[..., 3])
            # Clip the logit before exp so an unusual checkpoint cannot overflow. At magnitude
            # 80 the sigmoid is already exactly 0 or 1 in double precision.
            prob_chunks.append(1.0 / (1.0 + np.exp(-np.clip(row[..., 3], -80.0, 80.0))))

    n_q = int(getattr(model, "n_queries", 0))
    if not norm_chunks:
        empty3 = np.empty((0, n_q, 3), dtype=np.float64)
        empty1 = np.empty((0, n_q), dtype=np.float64)
        return {"params_norm": empty3, "params": empty3.copy(),
                "exist_prob": empty1, "exist_logit": empty1.copy()}

    params_norm = np.concatenate(norm_chunks, axis=0)
    physical = np.empty_like(params_norm)
    physical[..., 0] = normalizer.denormalize_t1(params_norm[..., 0])
    physical[..., 1] = normalizer.denormalize_t2(params_norm[..., 1])
    physical[..., 2] = params_norm[..., 2]
    return {
        "params_norm": params_norm,
        "params": physical,
        "exist_prob": np.concatenate(prob_chunks, axis=0),
        "exist_logit": np.concatenate(logit_chunks, axis=0),
    }


def targets_normalized(ds):
    """Ground-truth compartments in normalised space, together with the true count.

    Reads ds.y, the tensors built for training, so these are the numbers the loss saw.
    Returns (targets, n_comp) with targets of shape (N, max_comp, 3) and n_comp of shape
    (N,). Slots past n_comp[i] are zeros and every consumer slices them off.
    """
    y = ds.y.numpy().astype(np.float64)
    n = y.shape[1] // 3
    return y.reshape(len(y), n, 3), ds.n_comp.numpy().astype(int)


# ---------------------------------------------------------------------------------------
# 2. Per-query cost and the existence labels the loss used
# ---------------------------------------------------------------------------------------

def _cost_matrices(pred_norm, exist_prob, target_norm, n_comp, loss_cfg):
    """Build both cost matrices for one voxel: with the existence term, and without it.

    The assignment cost must reproduce loss.HungarianLoss exactly, including its existence
    bonus, because the existence labels reported later are defined as the queries the loss
    matched. The reporting cost excludes that term; otherwise every statistic asking whether
    the score predicts the cost (PAcc, SCorr, cost at a score cut) would contain the score by
    construction.

    Shapes: pred_norm (Q, 3), exist_prob (Q,), target_norm (C, 3). Returns two (Q, k) matrices
    with k = n_comp.
    """
    k = int(n_comp)
    t = target_norm[:k]                                     # (k, 3)
    t1_sq = (pred_norm[:, 0:1] - t[None, :, 0]) ** 2         # (Q, k)
    t2_sq = (pred_norm[:, 1:2] - t[None, :, 1]) ** 2
    w_sq = (pred_norm[:, 2:3] - t[None, :, 2]) ** 2

    reg = loss_cfg.t1_weight * t1_sq + loss_cfg.t2_weight * t2_sq
    # uniform switches signal-fraction weighting off; the other modes scale the T1/T2 term by
    # the true weight.
    if getattr(loss_cfg, "t1_t2_weighting", "legacy") != "uniform":
        reg = reg * t[None, :, 2]
    reg = reg + loss_cfg.w_weight * w_sq

    full = reg + loss_cfg.exist_weight * (1.0 - exist_prob)[:, None]
    return full, reg


def query_diagnostics(query_tbl, targets, n_comp, loss_cfg):
    """Per-query existence labels and prediction costs, for every voxel.

    Returns a dict of (N, Q) arrays:

    label        1 if the Hungarian algorithm matched this query to a true compartment, else 0.
                 This is the target the existence head was trained against.
    cost         regression cost minimised over the voxel's true compartments. Defined for
                 every query, including unmatched ones, so a ranking statistic is possible;
                 a duplicate of an already-taken compartment still gets its quality scored.
                 Optimistic by construction. Existence term excluded; see _cost_matrices.
    cost_matched cost against the compartment the query was assigned, NaN for unmatched.
    target_idx   index of the assigned compartment, or -1.
    """
    params = query_tbl["params_norm"]
    probs = query_tbl["exist_prob"]
    n, q = probs.shape
    label = np.zeros((n, q), dtype=np.int8)
    cost = np.full((n, q), np.nan)
    cost_matched = np.full((n, q), np.nan)
    target_idx = np.full((n, q), -1, dtype=np.int16)

    for i in range(n):
        full, reg = _cost_matrices(params[i], probs[i], targets[i], n_comp[i], loss_cfg)
        cost[i] = reg.min(axis=1)
        rows, cols = linear_sum_assignment(full)
        label[i, rows] = 1
        target_idx[i, rows] = cols
        cost_matched[i, rows] = reg[rows, cols]
    return {"label": label, "cost": cost, "cost_matched": cost_matched,
            "target_idx": target_idx}


def existence_score_metrics(diag, exist_prob, threshold=0.5):
    """Existence-head diagnostics, following the columns of Schlund's Tables 2 and 4.

    acc          fraction of (voxel, query) decisions that are correct. With ten queries and
                 one to three compartments, 70 to 90 % of labels are 0, so a model that never
                 activates anything already scores well; do not quote it on its own.
    prec         of the queries the model activates, the fraction the loss matched. Low means
                 redundant or invented peaks.
    rec          of the queries the loss matched, the fraction the model activates. Low means
                 the compartment was found but not admitted.
    n_peak       mean number of activated queries per voxel; compare with the mean true count.
    cost_at_0.5  mean prediction cost among queries scoring above 0.5, and likewise
    cost_at_0.8  above 0.8. Should fall as the cut rises.
    pacc         pairwise accuracy: within a voxel, over every pair of queries, how often the
                 one with the higher cost has the lower existence score. 0.5 is chance, 1.0 a
                 perfect ranking. Pairs tied on either quantity are skipped; averaged over
                 voxels.
    scorr        Spearman rank correlation between negative cost and existence score, per
                 voxel, averaged. Positive is good. Voxels where either quantity is constant
                 are skipped.

    acc, prec, rec and n_peak depend on the threshold; the cost and ranking columns do not,
    apart from the two named score cuts.
    """
    label = diag["label"].astype(bool)
    cost = diag["cost"]
    active = exist_prob > threshold

    tp = int(np.sum(active & label))
    fp = int(np.sum(active & ~label))
    fn = int(np.sum(~active & label))
    tn = int(np.sum(~active & ~label))

    pacc, scorr = [], []
    for i in range(len(cost)):
        c, s = cost[i], exist_prob[i]
        dc = c[:, None] - c[None, :]
        ds = s[:, None] - s[None, :]
        iu = np.triu_indices(len(c), k=1)
        dc, ds = dc[iu], ds[iu]
        usable = (dc != 0) & (ds != 0)
        if usable.any():
            # A concordant pair has the higher cost carrying the lower score, so the two
            # differences have opposite signs.
            pacc.append(float(np.mean((dc[usable] * ds[usable]) < 0)))
        if np.ptp(c) > 0 and np.ptp(s) > 0:
            rho = spearmanr(-c, s).statistic
            if np.isfinite(rho):
                scorr.append(float(rho))

    def _mean_cost_above(cut):
        m = exist_prob > cut
        return float(np.mean(cost[m])) if m.any() else float("nan")

    return {
        "threshold": float(threshold),
        "n_voxels": int(len(cost)),
        "n_queries": int(cost.shape[1]),
        "acc": float((tp + tn) / (tp + tn + fp + fn)) if len(cost) else float("nan"),
        "prec": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "rec": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "n_peak": float(np.mean(active.sum(axis=1))) if len(cost) else float("nan"),
        "cost_at_0.5": _mean_cost_above(0.5),
        "cost_at_0.8": _mean_cost_above(0.8),
        "pacc": float(np.mean(pacc)) if pacc else float("nan"),
        "scorr": float(np.mean(scorr)) if scorr else float("nan"),
        "n_pacc_voxels": int(len(pacc)),
        "n_scorr_voxels": int(len(scorr)),
    }


# ---------------------------------------------------------------------------------------
# 3. Grouping near-duplicate peaks
# ---------------------------------------------------------------------------------------

def peak_distances(points):
    """Condensed pairwise Euclidean distances between peaks, in normalised space.

    The axes are normalised T1 and T2 under log-minmax, each covering the sampled range as
    [0, 1], and optionally the weight. A radius of 0.05 is 5 % of log(3500/50) in T1, a factor
    of about 1.24, and 5 % of log(500/5) in T2, about 1.26, so the radius acts as a ratio
    tolerance. Averaging coordinates in this space is a geometric mean in milliseconds.
    """
    return pdist(np.atleast_2d(points), metric="euclidean")


def group_peaks(params_norm, exist_prob, radius, aggregate="weight",
                include_weight=False, renormalize=True):
    """Merge near-duplicate peaks within one voxel. Input and output are in normalised units.

    params_norm is (k, 3), the queries that passed the existence threshold, and exist_prob is
    their (k,) scores. Follows Schlund: cluster the peaks closer than radius, replace each
    group by the mean of its members, optionally rescale the weights to sum to one. Linkage is
    complete, so the radius bounds the diameter of every group; single linkage would let a
    chain of barely touching peaks collapse into one group and erase real compartments.

    Parameters
    ----------
    radius
        Grouping tolerance in normalised units. Zero or less returns the input unchanged,
        neither merged nor renormalised. Larger values merge more and lower the predicted
        count.
    aggregate
        How the T1/T2 centre of a group is computed. All three are means in normalised space
        (a geometric mean in milliseconds) and differ only in the mixing weights: "mean" is
        unweighted (Schlund's rule), "weight" weights each member by its predicted weight,
        "confidence" by its existence score. The merged weight is always the plain arithmetic
        mean of the member weights: duplicate queries each try to explain the whole
        compartment, so summing would double-count, and weighting a weight-average by weight
        biases it upward.
    include_weight
        Let the weight axis enter the distance. Off by default: an unmatched duplicate query
        gets no regression gradient, so its weight drifts freely and is its least reliable
        output. Swept on validation regardless.
    renormalize
        Rescale the merged weights to sum to one (Schlund's final step). Prior knowledge, not
        leakage: the weights are signal fractions and sum to one in the generator. It improves
        the weight error even when merging did nothing, hence a separate flag.

    Returns a list of (t1_norm, t2_norm, weight) tuples.
    """
    if aggregate not in AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {AGGREGATIONS}; got {aggregate!r}")
    p = np.atleast_2d(np.asarray(params_norm, dtype=np.float64))
    if p.size == 0:
        return []
    if p.shape[1] != 3:
        raise ValueError(f"params_norm must be (k, 3); got {p.shape}")
    conf = np.asarray(exist_prob, dtype=np.float64).reshape(-1)
    if len(conf) != len(p):
        raise ValueError(f"got {len(p)} peaks but {len(conf)} scores")

    # Identity cases, returned before any averaging or renormalisation so that radius 0 is
    # an exact no-op.
    if len(p) <= 1 or radius <= 0:
        return [tuple(float(v) for v in row) for row in p]

    cols = slice(0, 3) if include_weight else slice(0, 2)
    labels = fcluster(
        linkage(peak_distances(p[:, cols]), method="complete"),
        t=float(radius), criterion="distance",
    )

    peaks = []
    for lab in np.unique(labels):
        m = labels == lab
        if aggregate == "mean":
            a = np.ones(int(m.sum()))
        elif aggregate == "weight":
            a = p[m, 2].copy()
        else:
            a = conf[m].copy()
        total = a.sum()
        # An all-zero mixing vector leaves the mean undefined; fall back to the unweighted mean.
        a = a / total if total > 0 else np.full(int(m.sum()), 1.0 / int(m.sum()))
        peaks.append([float(a @ p[m, 0]), float(a @ p[m, 1]), float(np.mean(p[m, 2]))])

    peaks = np.asarray(peaks, dtype=np.float64)
    if renormalize:
        s = peaks[:, 2].sum()
        if s > 0:
            peaks[:, 2] /= s
    return [tuple(float(v) for v in row) for row in peaks]


def grouped_predictions(query_tbl, normalizer, threshold, radius, aggregate="weight",
                        include_weight=False, renormalize=True):
    """Threshold filter and grouping over a whole split, returned in milliseconds.

    Produces the same list of (T1_ms, T2_ms, weight) tuples per voxel that eval.compute_metrics
    and eval.parameter_recovery_analysis consume, so grouped and ungrouped runs are scored by
    the same code. At radius 0 the output is identical to
    eval.predictions_from_query_outputs(query_tbl, threshold); a test asserts it.
    """
    out = []
    for row, score in zip(query_tbl["params_norm"], query_tbl["exist_prob"]):
        keep = score > threshold
        merged = group_peaks(row[keep], score[keep], radius, aggregate=aggregate,
                             include_weight=include_weight, renormalize=renormalize)
        out.append([
            (float(normalizer.denormalize_t1(t1)), float(normalizer.denormalize_t2(t2)), w)
            for t1, t2, w in merged
        ])
    return out


def calibrate_grouping(query_tbl, trues, normalizer, threshold, radii=None,
                       aggregates=AGGREGATIONS, include_weight_options=(False, True),
                       renormalize_options=(False, True),
                       objective="parameter_set_error"):
    """Choose the grouping settings on validation data.

    Sweep on validation, freeze the winner, apply once to test. `objective` names the
    validation quantity minimised. parameter_set_error (default) is the bounded score also used
    to select the existence threshold; it charges a missed compartment at full cost, so a merge
    that erases a real pool is punished while one that removes a duplicate is rewarded.
    count_accuracy, maximised instead, asks only whether the count came out right.

    Returns {"objective", "selected", "curve", "n_evaluations"}; `curve` holds one row per
    (radius, aggregate, include_weight, renormalize) combination.
    """
    if objective not in {"parameter_set_error", "count_accuracy"}:
        raise ValueError("objective must be parameter_set_error|count_accuracy; "
                         f"got {objective!r}")
    radii = DEFAULT_RADII if radii is None else np.asarray(radii, dtype=np.float64)
    true_counts = np.asarray([len(t) for t in trues], dtype=int)

    curve = []
    for aggregate in aggregates:
        for include_weight in include_weight_options:
            for renormalize in renormalize_options:
                for radius in radii:
                    preds = grouped_predictions(
                        query_tbl, normalizer, threshold, float(radius),
                        aggregate=aggregate, include_weight=include_weight,
                        renormalize=renormalize,
                    )
                    s = parameter_recovery_analysis(preds, trues)["summary"]
                    pc = np.asarray([len(p) for p in preds], dtype=int)
                    curve.append({
                        "radius": float(radius),
                        "aggregate": aggregate,
                        "include_weight": bool(include_weight),
                        "renormalize": bool(renormalize),
                        "parameter_set_error": s["parameter_set_error"],
                        "parameter_set_accuracy": s["parameter_set_accuracy"],
                        "count_accuracy": float(np.mean(pc == true_counts)),
                        "recovered_signal_fraction": s["recovered_signal_fraction"],
                        "weight_set_l1_error_mean": s["weight_set_l1_error_mean"],
                        "t1_fraction_weighted_relative_error_matched":
                            s["t1_fraction_weighted_relative_error_matched"],
                        "t2_fraction_weighted_relative_error_matched":
                            s["t2_fraction_weighted_relative_error_matched"],
                        "mean_predicted_count": float(pc.mean()),
                        "n_extra_predictions": s["n_extra_predictions"],
                        "matched_true_compartment_rate":
                            s["matched_true_compartment_rate"],
                    })

    # Explicit tie-breaking so the selection is reproducible: after the objective, more
    # recovered signal, then the smaller radius, then the plain mean, then the fewest flags.
    if objective == "parameter_set_error":
        key = lambda r: (r["parameter_set_error"], -r["recovered_signal_fraction"],
                         r["radius"], AGGREGATIONS.index(r["aggregate"]),
                         r["include_weight"], r["renormalize"])
    else:
        key = lambda r: (-r["count_accuracy"], r["parameter_set_error"],
                         r["radius"], AGGREGATIONS.index(r["aggregate"]),
                         r["include_weight"], r["renormalize"])
    return {"objective": objective, "selected": min(curve, key=key), "curve": curve,
            "n_evaluations": len(curve)}


def threshold_only_predictions(query_tbl, threshold):
    """The threshold filter alone, without grouping, in milliseconds.

    Alias of eval.predictions_from_query_outputs so the two stages are named symmetrically.
    """
    return predictions_from_query_outputs(query_tbl, threshold)
