#!/usr/bin/env python3
"""Write results/review_stats.json: paired count/strict accuracy per K at one threshold, physical
plausibility of the reported set (T1 > T2 share, |sum w - 1|), the minimum pairwise ND between
the true compartments of a voxel, and the weight pass rate given that (T1, T2) already hit.

Reads the test split of baseline_v2_reproduction, results/threshold_val/<run>.json and the
checkpoints of the three four-seed families (cached in results/_review_cache_<run>.npz on first
use). Read-only with respect to everything else.
Usage: PYTHONPATH=.:datagen python3 evaluation/tables/build_review_stats.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datagen"))
os.chdir(ROOT)                      # data paths in config.yaml are relative to the repo root

import torch  # noqa: E402

torch.set_grad_enabled(False)
from t1t2 import nd_metrics as ndm            # noqa: E402
from t1t2.config import load_config            # noqa: E402
from t1t2.data import make_dataloader          # noqa: E402
from t1t2.eval import detr_query_outputs, true_compartments  # noqa: E402
from t1t2.runs import load_run                 # noqa: E402

RES = ROOT / "results"
CONFIGS = ROOT / "configs"
OUT = RES / "review_stats.json"

TAU = ndm.TAU_BASE
FAMILIES = {
    "baseline": ["baseline_v2_reproduction", "baseline_seed20260725",
                 "baseline_seed20260726", "baseline_seed20260727"],
    "loss_uniform": ["loss_uniform", "loss_uniform_seed20260725",
                     "loss_uniform_seed20260726", "loss_uniform_seed20260727"],
    "final_uniform_q6": ["final_uniform_q6_seed20260724", "final_uniform_q6_seed20260725",
                         "final_uniform_q6_seed20260726", "final_uniform_q6_seed20260727"],
}


def cfg_of(run):
    """Load a run's settings from configs or its results folder."""
    p = CONFIGS / f"{run}.yaml"
    return load_config(p if p.exists() else RES / run / "config.yaml")


def query_table(run, test_ds):
    """Raw (unthresholded) query table, cached as float64 exactly as the visuals script does."""
    f = RES / f"_review_cache_{run}.npz"
    if f.exists():
        z = np.load(f)
        return {"params": z["params"], "exist_prob": z["exist_prob"]}
    model = load_run(RES / run).model
    q = detr_query_outputs(model, test_ds, "cpu", test_ds.normalizer, batch_size=2048)
    np.savez_compressed(f, params=q["params"], exist_prob=q["exist_prob"])
    return {"params": q["params"], "exist_prob": q["exist_prob"]}


def calibrated_theta(run):
    """Read the confidence threshold chosen on validation data."""
    return float(json.load(open(RES / "threshold_val" / f"{run}.json"))["val_theta"])


def per_run(run, trues, spans, theta):
    """Count, strict, plausibility and conditional weight pass rate, all at one theta.

    The chapter's headline strict accuracy is at the per-run calibrated theta, so the count
    accuracy quoted beside it has to be measured at the same theta for the difference to be a
    clean "counted right, placed wrong" decomposition. Nothing in the architecture enforces
    T1 > T2 or sum w = 1, so both are measured on the reported set. The share of accepted
    (T1, T2) hits that also pass |dw| <= tau answers the weight question without the confound
    that adding any third acceptance condition lowers AP.
    """
    q = query_table(run, TEST_DS)
    params, probs = q["params"], q["exist_prob"]
    n = len(trues)
    K = np.array([len(t) for t in trues])

    reported = probs >= theta
    n_rep = reported.sum(1)

    # physical plausibility of the reported set
    t1_hat, t2_hat, w_hat = params[..., 0], params[..., 1], params[..., 2]
    ok_order = (t1_hat > t2_hat) & reported
    n_order_ok = int(ok_order.sum())
    n_reported_total = int(reported.sum())
    wsum = (w_hat * reported).sum(1)
    live = n_rep > 0                       # voxels that reported anything at all
    dev = np.abs(wsum[live] - 1.0)

    # acceptance under the ND rule at this theta
    hits = np.zeros(n, dtype=int)
    hits_w = np.zeros(n, dtype=int)        # hits that also pass |dw| <= tau
    for i in range(n):
        recs = ndm.voxel_records(params[i], probs[i], trues[i], spans, TAU,
                                 exist_thresh=theta)
        seen = set()
        for r in recs:
            g = r.get("gt")
            if g is None or g in seen:
                continue
            seen.add(g)
            hits[i] += 1
            if abs(r["dw"]) <= TAU:
                hits_w[i] += 1
    strict = (hits == K) & (n_rep == K)
    out_sep = {}
    for lab, msk in (("below_tau", SEP < TAU), ("above_tau", SEP >= TAU)):
        for k in (2, 3):
            m = msk & (K == k)
            if m.sum():
                out_sep[f"K{k}_{lab}"] = {"n": int(m.sum()),
                                          "strict_acc": 100.0 * float(strict[m].mean())}
    strict_w = (hits_w == K) & (n_rep == K)
    count_ok = n_rep == K

    out = {"theta": theta,
           "count_acc": 100.0 * count_ok.mean(),
           "strict_acc": 100.0 * strict.mean(),
           "strict_acc_with_weight": 100.0 * strict_w.mean(),
           "order_ok_pct": 100.0 * n_order_ok / max(n_reported_total, 1),
           "n_reported": n_reported_total,
           "wsum_dev_median": float(np.median(dev)),
           "wsum_dev_p95": float(np.percentile(dev, 95)),
           "voxels_no_report": int((~live).sum()),
           "strict_by_separation": out_sep,
           "by_k": {}}
    for k in (1, 2, 3):
        m = K == k
        rep_k = reported[m]
        t1k, t2k = t1_hat[m], t2_hat[m]
        ok_k = ((t1k > t2k) & rep_k).sum()
        dev_k = np.abs((w_hat[m] * rep_k).sum(1) - 1.0)[rep_k.sum(1) > 0]
        out["by_k"][str(k)] = {
            "count_acc": 100.0 * count_ok[m].mean(),
            "strict_acc": 100.0 * strict[m].mean(),
            "order_ok_pct": 100.0 * ok_k / max(int(rep_k.sum()), 1),
            "wsum_dev_median": float(np.median(dev_k)),
            "wsum_dev_p95": float(np.percentile(dev_k, 95)),
        }
    # conditional weight pass rate over accepted (T1, T2) hits
    out["hits_total"] = int(hits.sum())
    out["hits_also_weight"] = int(hits_w.sum())
    out["weight_pass_given_placed"] = 100.0 * hits_w.sum() / max(int(hits.sum()), 1)
    return out


def separation_stats(trues, spans):
    """Minimum pairwise ND between the true compartments of a voxel.

    Same normalization and the same max-over-dimensions form as the acceptance rule, so the
    number is directly comparable with tau: a voxel whose minimum pairwise ND is below tau
    holds two compartments that both fit inside one tolerance box. The sampler imposes no
    minimum separation, but the acceptance rule demands a separate prediction for each.
    """
    out = np.full(len(trues), np.inf)
    for i, t in enumerate(trues):
        if len(t) < 2:
            continue
        best = np.inf
        for a in range(len(t)):
            for b in range(a + 1, len(t)):
                d1 = abs(np.log(t[a][0]) - np.log(t[b][0])) / spans[0]
                d2 = abs(np.log(t[a][1]) - np.log(t[b][1])) / spans[1]
                best = min(best, max(d1, d2))
        out[i] = best
    return out


if __name__ == "__main__":
    cfg0 = cfg_of("baseline_v2_reproduction")
    _, TEST_DS = make_dataloader(cfg0.data.test_path, cfg0.data, 512,
                                 shuffle=False, num_workers=0)
    TRUES = true_compartments(TEST_DS)
    SPANS = ndm.log_spans(cfg0.data.t1_min, cfg0.data.t1_max,
                          cfg0.data.t2_min, cfg0.data.t2_max)
    K = np.array([len(t) for t in TRUES])
    print(f"test voxels {len(TRUES):,}   log spans T1 {SPANS[0]:.4f} T2 {SPANS[1]:.4f}")
    print(f"tau = {TAU:.0%}  ->  T1 x/{np.exp(TAU*SPANS[0]):.4f}, "
          f"T2 x/{np.exp(TAU*SPANS[1]):.4f}")

    res = {"tau": TAU, "spans_log": {"t1": SPANS[0], "t2": SPANS[1]},
           "tau_multiplicative": {"t1": float(np.exp(TAU * SPANS[0])),
                                  "t2": float(np.exp(TAU * SPANS[1]))},
           "n_test_voxels": len(TRUES), "runs": {}, "families": {}}

    # ground-truth separation (model-independent)
    sep = separation_stats(TRUES, SPANS)
    globals()["SEP"] = sep
    seprep = {}
    for k in (2, 3):
        s = sep[K == k]
        seprep[str(k)] = {
            "n": int(len(s)),
            "median_min_nd": float(np.median(s)),
            "pct_below_tau": 100.0 * float((s < TAU).mean()),
            "pct_below_2tau": 100.0 * float((s < 2 * TAU).mean()),
        }
    seprep["all_multi"] = {
        "n": int((K >= 2).sum()),
        "pct_below_tau": 100.0 * float((sep[K >= 2] < TAU).mean()),
    }
    res["gt_separation"] = seprep
    print("\n-- ground-truth separation (min pairwise ND) --")
    for k in ("2", "3"):
        d = seprep[k]
        print(f"  K={k}: median {d['median_min_nd']:.3f}, "
              f"{d['pct_below_tau']:.1f}% below tau, "
              f"{d['pct_below_2tau']:.1f}% below 2 tau")

    # per run, at the calibrated and at the declared threshold
    for fam, runs in FAMILIES.items():
        for run in runs:
            th_cal = calibrated_theta(run)
            res["runs"][run] = {"calibrated": per_run(run, TRUES, SPANS, th_cal),
                                "declared_075": per_run(run, TRUES, SPANS, 0.75)}
            print(f"  done {run}")

        def agg(path, sub=None):
            """Collect the same result field across all runs in a family."""
            v = []
            for r in runs:
                d = res["runs"][r][path]
                v.append(d[sub] if sub else d)
            return v

        fam_out = {}
        for proto in ("calibrated", "declared_075"):
            g = {}
            for key in ("count_acc", "strict_acc", "strict_acc_with_weight",
                        "order_ok_pct", "wsum_dev_median", "wsum_dev_p95",
                        "weight_pass_given_placed"):
                v = [res["runs"][r][proto][key] for r in runs]
                g[key] = {"mean": float(np.mean(v)),
                          "range": float(max(v) - min(v))}
            g["by_k"] = {}
            for k in ("1", "2", "3"):
                g["by_k"][k] = {}
                for key in ("count_acc", "strict_acc", "order_ok_pct"):
                    v = [res["runs"][r][proto]["by_k"][k][key] for r in runs]
                    g["by_k"][k] = dict(g["by_k"][k],
                                        **{key: {"mean": float(np.mean(v)),
                                                 "range": float(max(v) - min(v))}})
            fam_out[proto] = g
        res["families"][fam] = fam_out

        for proto in ("calibrated", "declared_075"):
            g = fam_out[proto]
            print(f"\n== {fam} [{proto}] ==")
            print(f"  count {g['count_acc']['mean']:.2f}  "
                  f"strict {g['strict_acc']['mean']:.2f}  "
                  f"strict+w {g['strict_acc_with_weight']['mean']:.2f}")
            for k in ("1", "2", "3"):
                b = g["by_k"][k]
                print(f"   K={k}: count {b['count_acc']['mean']:.2f} "
                      f"(rng {b['count_acc']['range']:.2f})  "
                      f"strict {b['strict_acc']['mean']:.2f} "
                      f"(rng {b['strict_acc']['range']:.2f})")
            print(f"  T1>T2 on reported: {g['order_ok_pct']['mean']:.3f}%   "
                  f"|sum w - 1| median {g['wsum_dev_median']['mean']:.4f}, "
                  f"p95 {g['wsum_dev_p95']['mean']:.4f}")
            print(f"  weight passes given placed: "
                  f"{g['weight_pass_given_placed']['mean']:.2f}%")

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
