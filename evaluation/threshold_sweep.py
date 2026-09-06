"""Sweep the existence threshold for finished runs and score each point under the ND rule.

One inference pass per run on the test split. ND assignment records are built at tau = 7 %,
with and without the weight dimension ("2d" / "3d"), and every threshold on GRID is scored by
filtering those records; nothing is calibrated here. Per threshold: count_acc (right number of
compartments, ignoring where they are), voxel_acc (right number and every compartment inside the
tolerance), precision/recall/f1 (ND detection) and mean_pred (compartments reported per voxel).
Writes results/threshold_sweep/<run>.json. Usage: python3 evaluation/threshold_sweep.py [run ...]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

from t1t2.nd_metrics import dataset_records, TAU_BASE
from t1t2.runs import load_run

GRID = [round(0.05 * i, 2) for i in range(1, 20)]      # 0.05 .. 0.95


def score(recs, ngt, T):
    """All metrics at one existence threshold, from unfiltered records."""
    # filter the records at T and count per voxel
    TP = FP = FN = 0
    n_count_ok = n_voxel_ok = 0
    n_pred = 0
    for r, n in zip(recs, ngt):
        # queries above the threshold
        keep = [x for x in r if x["prob"] >= T]
        n_pred += len(keep)
        if len(keep) == n:
            n_count_ok += 1
        # group the hits by ground truth: first hit TP, extra hits FP, no hit FN
        by = {}
        for x in keep:
            if x["gt"] is None:
                FP += 1
            else:
                by.setdefault(x["gt"], []).append(x)
        for g in range(int(n)):
            hits = by.get(g, [])
            if hits:
                TP += 1
                FP += len(hits) - 1
            else:
                FN += 1
        # every true compartment matched exactly once, and nothing spurious
        if len(keep) == n and len(by) == n and all(len(v) == 1 for v in by.values()):
            n_voxel_ok += 1
    # aggregates
    p = TP / (TP + FP) if TP + FP else 0.0
    r_ = TP / (TP + FN) if TP + FN else 0.0
    nv = len(ngt)
    return {
        "threshold": T, "TP": TP, "FP": FP, "FN": FN,
        "precision": p, "recall": r_,
        "f1": (2 * p * r_ / (p + r_)) if p + r_ else 0.0,
        "count_acc": 100.0 * n_count_ok / nv,
        "voxel_acc": 100.0 * n_voxel_ok / nv,
        "mean_pred": n_pred / nv,
    }


def score_by_k(recs, ngt, T):
    """Strict accuracy split by the true compartment count."""
    # totals, strict hits and count hits per true K
    tot, ok, cnt = {}, {}, {}
    for r, n in zip(recs, ngt):
        n = int(n)
        tot[n] = tot.get(n, 0) + 1
        keep = [x for x in r if x["prob"] >= T]
        by = {}
        for x in keep:
            if x["gt"] is not None:
                by.setdefault(x["gt"], []).append(x)
        if len(keep) == n and len(by) == n and all(len(v) == 1 for v in by.values()):
            ok[n] = ok.get(n, 0) + 1
        if len(keep) == n:
            cnt[n] = cnt.get(n, 0) + 1
    return {"strict": {str(k): 100.0 * ok.get(k, 0) / tot[k] for k in sorted(tot)},
            "count": {str(k): 100.0 * cnt.get(k, 0) / tot[k] for k in sorted(tot)}}


def run_one(run, device="cpu"):
    """Score one model across confidence thresholds in 2D and 3D."""
    # one inference pass, then records with and without the weight dimension
    loaded = load_run(Path("results") / run, device)
    q, trues = loaded.predict("test")
    out = {"run": run, "n_voxels": len(trues), "tau": TAU_BASE}
    for dim, inc_w in (("2d", False), ("3d", True)):
        recs, ngt = dataset_records(q, trues, loaded.spans, TAU_BASE, exist_thresh=0.0,
                                    include_weight=inc_w)
        out[dim] = [score(recs, ngt, T) for T in GRID]
        out[dim + "_by_k"] = {str(T): score_by_k(recs, ngt, T) for T in (0.5, 0.75)}
    return out


def main():
    """Save threshold sweeps and print each run's best test accuracy."""
    # one JSON per run, plus a one-line summary
    runs = sys.argv[1:]
    outdir = Path("results/threshold_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        t0 = time.time()
        res = run_one(run)
        (outdir / f"{run}.json").write_text(json.dumps(res, indent=1))
        best2 = max(res["2d"], key=lambda r: r["voxel_acc"])
        print(f"{run:26s} {time.time()-t0:6.1f}s  best voxel_acc(2D) "
              f"{best2['voxel_acc']:.2f}% at t={best2['threshold']}", flush=True)


if __name__ == "__main__":
    main()
