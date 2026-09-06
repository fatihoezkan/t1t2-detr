"""Normalised-distance (ND) detection metrics for compartments: mAP and precision/recall.

Intersection over union has no meaning for a point in (T1, T2), so the acceptance test is the
Normalised Distance of Wirth (2026, ch. 6): a prediction is accepted for a ground-truth
compartment only if every feature deviates by less than tau, as a fraction of that feature's
global range. Here the features are log T1 and log T2 only; the weight is reported separately.
"""
from __future__ import annotations

import numpy as np

# mAP is reported at tau = 5, 7 and 10 %; 7 % is the base tau for the exact metrics.
TAUS_DEFAULT = (0.05, 0.07, 0.10)
TAU_BASE = 0.07


def log_spans(t1_min, t1_max, t2_min, t2_max):
    """Feature ranges in log space, read from the run's own data config.

    ND is computed in log space, ND_T1 = |log T1_pred - log T1_true| / (log T1_max - log T1_min),
    because T1 and T2 are sampled log-uniformly and normalised by log_minmax; on a linear span a
    35 ms error is negligible at T1 = 2000 ms and a different tissue at T2 = 30 ms. The spans
    are what tau is a fraction of, so they must come from the training config, not from the
    data at hand.
    """
    # log(max) - log(min) for T1 and T2
    return (float(np.log(t1_max) - np.log(t1_min)),
            float(np.log(t2_max) - np.log(t2_min)))


def voxel_records(params, probs, trues, spans, tau, exist_thresh=0.0,
                  include_weight=False):
    """ND assignment for a single voxel.

    Acceptance is per dimension: every ND must be at most tau. Among the accepted candidates
    the assignment goes to the smallest ND sum; the sum only ranks, so one badly wrong
    dimension cannot be averaged away by the others.

    params : (Q, 3) array of T1 ms, T2 ms and weight per query.
    probs  : (Q,) existence probabilities.
    trues  : list of (t1_ms, t2_ms, w) ground-truth compartments.
    spans  : (log span of T1, log span of T2).
    tau    : acceptance threshold as a fraction of each span, so 0.07 is 7 %.
    include_weight : add ND_w = |w_pred - w_true| as a third dimension that must also pass
        tau and enters the ranking sum. The weight span is 1.0 because w lies in [0, 1].
        False by default, as in the original, where weight is excluded from acceptance.

    Returns one record per query above exist_thresh, each a dict with prob, gt (an index into
    trues, or None), dt1_ms, dt2_ms, dw and nd_sum. gt=None means the query passed the
    existence filter but matched no ground-truth compartment.
    """
    # ground truth in log space, (K, 2)
    span1, span2 = spans
    recs = []
    if len(trues):
        T = np.asarray(trues, dtype=np.float64)          # (K, 3)
        logT = np.log(np.maximum(T[:, :2], 1e-9))
    # one record per query above the existence threshold
    for q in range(len(probs)):
        p = float(probs[q])
        if p < exist_thresh:
            continue
        t1, t2, w = (float(params[q, 0]), float(params[q, 1]), float(params[q, 2]))
        # default: matched nothing
        rec = {"prob": p, "gt": None, "dt1_ms": np.nan, "dt2_ms": np.nan,
               "dw": np.nan, "nd_sum": np.inf}
        if len(trues):
            # normalised distance to every true compartment, per dimension
            lp1, lp2 = np.log(max(t1, 1e-9)), np.log(max(t2, 1e-9))
            nd1 = np.abs(lp1 - logT[:, 0]) / span1       # (K,)
            nd2 = np.abs(lp2 - logT[:, 1]) / span2
            ok = (nd1 <= tau) & (nd2 <= tau)             # acceptance is per-dimension
            nd_rank = nd1 + nd2
            if include_weight:
                nd3 = np.abs(w - T[:, 2]) / 1.0          # w span is the unit interval
                ok = ok & (nd3 <= tau)
                nd_rank = nd_rank + nd3
            # among the accepted candidates, take the smallest ND sum
            if ok.any():
                nd_sum = np.where(ok, nd_rank, np.inf)   # sum only ranks candidates
                g = int(np.argmin(nd_sum))
                rec.update(gt=g, nd_sum=float(nd_sum[g]),
                           dt1_ms=abs(t1 - float(T[g, 0])),
                           dt2_ms=abs(t2 - float(T[g, 1])),
                           dw=abs(w - float(T[g, 2])))
        recs.append(rec)
    return recs


def dataset_records(query_outputs, trues, spans, tau, exist_thresh=0.0,
                    include_weight=False):
    """Apply voxel_records across a whole dataset.

    query_outputs is the raw table from eval.detr_query_outputs: params (N, Q, 3) in physical
    units and exist_prob (N, Q). Returns (records_per_voxel, n_gt_per_voxel).
    """
    # per-voxel records and the number of ground-truth compartments per voxel
    params = np.asarray(query_outputs["params"])
    probs = np.asarray(query_outputs["exist_prob"])
    recs = [voxel_records(params[i], probs[i], trues[i], spans, tau, exist_thresh,
                          include_weight=include_weight)
            for i in range(len(trues))]
    n_gt = np.array([len(t) for t in trues], dtype=int)
    return recs, n_gt


def map_101_from_records(recs_per_voxel, n_gt_per_voxel, voxel_ids=None):
    """COCO-style 101-point average precision over the pooled predictions.

    Every prediction enters the ranking with its existence probability as confidence, so on
    records built with exist_thresh=0 this is threshold-free and measures how well the
    existence head ranks its own queries. Walking down the ranked list, the first prediction
    to claim a (voxel, ground truth) pair is a true positive; a later claim on the same pair,
    and any prediction matching nothing, is a false positive. `voxel_ids` scores a subset or
    a bootstrap resample; repeated voxels get distinct ids and claim their pairs separately.

    Returns (mAP, precisions, recalls).
    """
    # pool every prediction with its confidence and a (voxel, gt) key; -1 = matched nothing
    if voxel_ids is None:
        voxel_ids = range(len(recs_per_voxel))
    conf, keys = [], []
    total_gt = 0
    for uid, v in enumerate(voxel_ids):
        total_gt += int(n_gt_per_voxel[v])
        for r in recs_per_voxel[v]:
            conf.append(r["prob"])
            keys.append(-1 if r["gt"] is None else uid * 64 + r["gt"])
    if total_gt == 0 or not conf:
        return 0.0, np.zeros(0), np.zeros(0)
    # rank by confidence, highest first
    conf = np.asarray(conf)
    keys = np.asarray(keys, dtype=np.int64)
    order = np.argsort(-conf, kind="stable")
    keys = keys[order]

    # the first claim on a (voxel, gt) pair is a true positive, everything else a false positive
    tp = np.zeros(len(keys), dtype=np.float64)
    seen = set()
    for i, k in enumerate(keys):
        if k >= 0 and k not in seen:
            tp[i] = 1.0
            seen.add(int(k))
    # running precision and recall down the ranked list
    acc_tp = np.cumsum(tp)
    acc_fp = np.cumsum(1.0 - tp)
    recalls = acc_tp / total_gt
    precisions = acc_tp / np.maximum(acc_tp + acc_fp, np.finfo(float).eps)

    # 101-point interpolation: p_interp(r) = max precision at any recall >= r.
    # Computed via a reverse running maximum instead of a per-point scan.
    p_mono = np.maximum.accumulate(precisions[::-1])[::-1]
    idx = np.searchsorted(recalls, np.linspace(0.0, 1.0, 101), side="left")
    p_at = np.where(idx < len(p_mono), p_mono[np.minimum(idx, len(p_mono) - 1)], 0.0)
    return float(p_at.mean()), precisions, recalls


def exact_metrics_from_records(recs_per_voxel, n_gt_per_voxel):
    """Precision, recall, F1 and true-positive error means at a fixed existence threshold.

    The records must already be filtered at the operating threshold (exist_thresh in
    dataset_records). Per ground-truth compartment the highest-confidence assigned prediction
    is the true positive and further assigned predictions are false positives; predictions
    assigned to nothing are false positives; ground truth with no assigned prediction is a
    false negative. The mean |dT1|, |dT2| (ms) and |dw| over true positives is safe because
    the ND gate bounds every term, but it improves as recall falls, so never read it without
    the recall beside it. Medians are kept as secondary columns.
    """
    # count TP/FP/FN and collect the true-positive errors
    TP = FP = FN = 0
    dt1, dt2, dw = [], [], []
    n_pred = 0
    for recs, n_gt in zip(recs_per_voxel, n_gt_per_voxel):
        # group the voxel's predictions by the ground truth they were assigned to
        by_gt = {}
        for r in recs:
            n_pred += 1
            if r["gt"] is None:
                FP += 1
            else:
                by_gt.setdefault(r["gt"], []).append(r)
        # per ground truth: the best-confidence hit is the TP, extra hits are FPs, no hit is a FN
        for g in range(int(n_gt)):
            hits = by_gt.get(g, [])
            if hits:
                best = max(hits, key=lambda r: r["prob"])
                TP += 1
                FP += len(hits) - 1
                dt1.append(best["dt1_ms"])
                dt2.append(best["dt2_ms"])
                dw.append(best["dw"])
            else:
                FN += 1
    # precision, recall, F1
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_dt1_ms": float(np.mean(dt1)) if dt1 else float("nan"),
        "mean_dt2_ms": float(np.mean(dt2)) if dt2 else float("nan"),
        "mean_dw": float(np.mean(dw)) if dw else float("nan"),
        "median_dt1_ms": float(np.median(dt1)) if dt1 else float("nan"),
        "median_dt2_ms": float(np.median(dt2)) if dt2 else float("nan"),
        "median_dw": float(np.median(dw)) if dw else float("nan"),
        "mean_pred_per_voxel": n_pred / len(recs_per_voxel) if len(recs_per_voxel) else 0.0,
    }


def calibrate_threshold_nd(query_outputs, trues, spans, tau=TAU_BASE,
                           grid=None):
    """Search the existence threshold on the validation split by best F1 at the base tau.

    Default grid 0.25 to 0.75 in steps of 0.05. Returns (best_threshold, table), where table
    maps each threshold to its exact metrics. The original searches on test; this runs on
    validation, so no test data enters the choice, and the result is applied unchanged to test.
    """
    # default grid 0.25 .. 0.75
    if grid is None:
        grid = [round(0.25 + 0.05 * i, 2) for i in range(11)]   # 0.25 .. 0.75
    table = {}
    best_t, best_f1 = grid[0], -1.0
    # score every threshold, keep the best F1
    for t in grid:
        recs, n_gt = dataset_records(query_outputs, trues, spans, tau, exist_thresh=t)
        m = exact_metrics_from_records(recs, n_gt)
        table[t] = m
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return best_t, table


def stratified_map(query_outputs, trues, spans, tau=TAU_BASE, weight_bins=(0.3, 0.6)):
    """mAP restricted to subsets of the ground truth, by weight and by compartment count.

    Ground truth is split by weight into small (below w_small), medium and large (at or above
    w_large), and separately by the compartment count of its voxel. Within a stratum only its
    own ground truth counts toward the total and only predictions assigned to it count as true
    positives. Unassigned predictions are excluded entirely, since no stratum can claim them,
    which makes the absolute values optimistic: compare strata or models with each other, not
    against the global mAP.
    """
    # records at threshold 0, so every prediction is ranked
    w_small, w_large = weight_bins
    recs, n_gt = dataset_records(query_outputs, trues, spans, tau, exist_thresh=0.0)

    def _subset_map(gt_mask_fn):
        """Measure average precision for a chosen group of true compartments."""
        # keep only predictions assigned to ground truth in this stratum
        conf, keys = [], []
        total_gt = 0
        for v, t in enumerate(trues):
            sel = [g for g in range(len(t)) if gt_mask_fn(t[g], len(t))]
            total_gt += len(sel)
            sel_set = set(sel)
            for r in recs[v]:
                if r["gt"] is not None and r["gt"] in sel_set:
                    conf.append(r["prob"])
                    keys.append(v * 64 + r["gt"])
        # same 101-point AP as map_101_from_records
        if total_gt == 0:
            return float("nan"), 0
        conf = np.asarray(conf)
        keys = np.asarray(keys, dtype=np.int64)
        order = np.argsort(-conf, kind="stable")
        tp = np.zeros(len(keys))
        seen = set()
        for i, k in enumerate(keys[order]):
            if k not in seen:
                tp[i] = 1.0
                seen.add(int(k))
        acc_tp, acc_fp = np.cumsum(tp), np.cumsum(1.0 - tp)
        rec_arr = acc_tp / total_gt
        prec = acc_tp / np.maximum(acc_tp + acc_fp, np.finfo(float).eps)
        p_mono = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec_arr, np.linspace(0, 1, 101), side="left")
        p_at = np.where(idx < len(p_mono), p_mono[np.minimum(idx, len(p_mono) - 1)], 0.0)
        return float(p_at.mean()), total_gt

    # strata: three weight bins and each compartment count
    out = {"by_weight": {}, "by_n_comp": {}}
    for label, fn in [
        (f"w<{w_small}", lambda g, n: g[2] < w_small),
        (f"{w_small}<=w<{w_large}", lambda g, n: w_small <= g[2] < w_large),
        (f"w>={w_large}", lambda g, n: g[2] >= w_large),
    ]:
        m, n = _subset_map(fn)
        out["by_weight"][label] = {"map": m, "n_gt": n}
    for k in sorted({len(t) for t in trues}):
        m, n = _subset_map(lambda g, nc, k=k: nc == k)
        out["by_n_comp"][f"n_comp={k}"] = {"map": m, "n_gt": n}
    return out


def bootstrap_map_ci(recs_per_voxel, n_gt_per_voxel, n_boot=500, seed=20260810,
                     recs_other=None, n_gt_other=None):
    """Voxel-level bootstrap confidence interval for mAP.

    Passing a second model's records over the same voxels gives the paired form, where the
    interval is on the difference (this model minus the other). Returns a percentile 95 %
    interval as {"lo", "hi", "n_boot"}.
    """
    # resample voxels with replacement, n_boot times
    rng = np.random.default_rng(seed)
    n = len(recs_per_voxel)
    stats = []
    for _ in range(n_boot):
        ids = rng.integers(0, n, size=n)
        m1, _, _ = map_101_from_records(recs_per_voxel, n_gt_per_voxel, voxel_ids=ids)
        if recs_other is not None:
            m2, _, _ = map_101_from_records(recs_other, n_gt_other, voxel_ids=ids)
            stats.append(m1 - m2)
        else:
            stats.append(m1)
    # percentile interval
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_boot": n_boot}
