"""Paired significance tests between each arm and the reference run.

Both runs are scored on the same test voxels, so the tests are paired: McNemar's exact test with
a Wald interval on the paired difference for strict and count accuracy at both declared
thresholds, and a paired bootstrap (same resample for both runs) for mAP@7, all Holm-Bonferroni
corrected per metric family. Reads the _records_tau7 and _n_gt dumps in
results/nd_evaluation/<arm>.json, so no inference is run; --self-check confirms they reproduce
the stored aggregates. Writes results/paired_tests.json. Usage: paired_tests.py [arm ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

from t1t2.nd_metrics import map_101_from_records

ROOT = Path(__file__).resolve().parents[1]
ND = ROOT / "results" / "nd_evaluation"
OUT = ROOT / "results" / "paired_tests.json"

REFERENCE = "baseline_v2_reproduction"

# Arms the results chapter compares against the reference. Names without an evaluation file
# are skipped.
ARMS = [
    "loss_uniform",
    "data_loguniform",
    "queries_6",
    "queries_4",
    "aux_loss",
    "exist_weight_03",
    "decoder_2",
    "decoder_6",
    "exist_head_shared",
    "physics_clean",
    "physics_noisy",
    "baseline_v3",
    "baseline_v3_no_sqrt",
    "baseline_v3_no_physics",
    "baseline_v4",
]

# The two declared thresholds the chapter uses for between-run comparison.
THETAS = (0.5, 0.75)

N_BOOT = 300          # same as the map7_ci95 bootstrap
BOOT_SEED = 20260826


def load_records(run: str):
    """The per-voxel dump run_nd_evaluation.py wrote for a run, as (records, n_gt, test_paths),
    or (None, None, None) if the run was never evaluated."""
    path = ND / f"{run}.json"
    if not path.exists():
        return None, None, None
    with path.open() as fh:
        blob = json.load(fh)
    recs = blob.get("_records_tau7")
    ngt = blob.get("_n_gt")
    if recs is None or ngt is None:
        return None, None, None
    return recs, np.asarray(ngt, dtype=int), blob.get("test_paths")


def per_voxel_flags(recs, ngt, theta: float):
    """Per-voxel strict and count outcomes at one existence threshold.

    Same rule as threshold_sweep.score(), per voxel instead of aggregated: strictly correct
    means the right number of compartments and a bijection onto the true ones (every true
    compartment hit exactly once, nothing spurious).
    """
    # one flag per voxel
    n = len(ngt)
    strict = np.zeros(n, dtype=bool)
    count = np.zeros(n, dtype=bool)
    for i, (rec, k) in enumerate(zip(recs, ngt)):
        # queries above the threshold, grouped by the ground truth they were assigned to
        keep = [x for x in rec if x["prob"] >= theta]
        count[i] = len(keep) == k
        by: dict = {}
        for x in keep:
            if x["gt"] is not None:
                by.setdefault(x["gt"], []).append(x)
        strict[i] = (
            len(keep) == k
            and len(by) == k
            and all(len(v) == 1 for v in by.values())
        )
    return strict, count


def mcnemar(ref: np.ndarray, arm: np.ndarray):
    """Exact McNemar test plus a Wald interval on the paired difference.

    Returns delta in percentage points (arm minus reference), the two-sided
    exact p value, the 95 % half-width of the paired difference, and the two
    discordant counts.
    """
    # discordant pairs
    n = len(ref)
    b = int(np.sum(~ref & arm))          # arm right, reference wrong
    c = int(np.sum(ref & ~arm))          # reference right, arm wrong
    delta = 100.0 * (arm.mean() - ref.mean())
    # exact binomial test on the discordant pairs
    p = stats.binomtest(c, b + c, 0.5).pvalue if (b + c) else 1.0
    # variance of a paired difference of proportions
    var = (b + c) - (b - c) ** 2 / n
    half = 1.96 * sqrt(max(var, 0.0)) / n * 100.0
    return delta, float(p), half, b, c


def map7(recs, ngt, index=None):
    """mAP@7 through the repository's own scorer, so values match the stored ones.

    map_101_from_records scores a subset or a bootstrap resample through voxel_ids; repeated
    voxels get distinct internal ids so their compartments stay separately creditable.
    """
    value, _, _ = map_101_from_records(recs, ngt, voxel_ids=index)
    return float(value)


def paired_bootstrap_map(ref_recs, ref_ngt, arm_recs, arm_ngt, n_boot=N_BOOT):
    """Bootstrap the mAP@7 difference over shared voxels.

    Both runs are scored on the same resample each iteration, so the run-to-run correlation
    from the shared test set cancels in the difference.
    """
    # point estimate, then n_boot resamples shared by both runs
    rng = np.random.default_rng(BOOT_SEED)
    n = len(ref_ngt)
    base = map7(arm_recs, arm_ngt) - map7(ref_recs, ref_ngt)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = map7(arm_recs, arm_ngt, idx) - map7(ref_recs, ref_ngt, idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(base), float(lo), float(hi)


def holm(pvalues):
    """Holm-Bonferroni adjusted p values, order preserved."""
    # step-down: sort, scale by (m - rank), keep the adjusted values monotone
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def self_check(ref: str = REFERENCE) -> bool:
    """Confirm the per-voxel rule reproduces the stored aggregates exactly."""
    recs, ngt, _ = load_records(ref)
    if recs is None:
        print(f"[self-check] no records for {ref}", file=sys.stderr)
        return False
    sweep_path = ROOT / "results" / "threshold_sweep" / f"{ref}.json"
    with sweep_path.open() as fh:
        sweep = json.load(fh)
    ok = True
    for theta in THETAS:
        strict, count = per_voxel_flags(recs, ngt, theta)
        row = next(r for r in sweep["2d"] if abs(r["threshold"] - theta) < 1e-9)
        for name, mine, stored in (
            ("strict", 100 * strict.mean(), row["voxel_acc"]),
            ("count", 100 * count.mean(), row["count_acc"]),
        ):
            match = abs(mine - stored) < 1e-6
            ok &= match
            print(
                f"[self-check] theta={theta:<5} {name:<7} "
                f"recomputed {mine:.4f}  stored {stored:.4f}  "
                f"{'OK' if match else 'MISMATCH'}"
            )
    nd_map = json.load((ND / f"{ref}.json").open())["map"]["map@7"]
    mine = map7(recs, ngt)
    match = abs(mine - nd_map) < 1e-6
    ok &= match
    print(
        f"[self-check] mAP@7 recomputed {mine:.4f}  stored {nd_map:.4f}  "
        f"{'OK' if match else 'MISMATCH'}"
    )
    return bool(ok)


def main() -> int:
    """Run the paired tests for every arm, or with --self-check confirm that the per-voxel
    rule reproduces the stored aggregates."""
    # command line
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=None)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--no-map", action="store_true",
                    help="skip the mAP bootstrap (much faster)")
    args = ap.parse_args()

    if args.self_check:
        return 0 if self_check() else 1

    # reference flags at both thresholds
    ref_recs, ref_ngt, ref_paths = load_records(REFERENCE)
    if ref_recs is None:
        print(f"no ND records for the reference run {REFERENCE}", file=sys.stderr)
        return 1
    ref_flags = {t: per_voxel_flags(ref_recs, ref_ngt, t) for t in THETAS}

    # one entry per arm
    arms = args.arms or ARMS
    results = {}
    for arm in arms:
        if arm == REFERENCE:
            continue
        recs, ngt, paths = load_records(arm)
        if recs is None:
            print(f"  {arm}: no ND records, skipped")
            continue
        # Pairing needs the same test voxels. Identical n_gt sequences are not enough: every
        # family is balanced 1/3 per compartment count in the same order, so the sequences
        # match even across datasets. The test file paths are the discriminator.
        same_paths = (paths is not None and ref_paths is not None
                      and list(paths) == list(ref_paths))
        same_shape = len(ngt) == len(ref_ngt) and np.array_equal(ngt, ref_ngt)
        if not (same_paths and same_shape):
            reason = ("different test set" if not same_paths
                      else "different voxel count or compartment layout")
            print(f"  {arm}: {reason}, paired test not applicable")
            results[arm] = {
                "paired_applicable": False,
                "reason": reason,
                "test_paths": paths,
            }
            continue

        # McNemar on strict and count accuracy at each threshold
        entry = {"paired_applicable": True, "accuracy": {}}
        for theta in THETAS:
            strict, count = per_voxel_flags(recs, ngt, theta)
            row = {}
            for name, ref_f, arm_f in (
                ("strict", ref_flags[theta][0], strict),
                ("count", ref_flags[theta][1], count),
            ):
                delta, p, half, b, c = mcnemar(ref_f, arm_f)
                row[name] = {
                    "delta_pp": delta,
                    "mcnemar_p": p,
                    "paired_ci95_halfwidth_pp": half,
                    "n_arm_only": b,
                    "n_ref_only": c,
                }
            entry["accuracy"][f"{theta}"] = row

        # paired bootstrap on mAP@7
        if not args.no_map:
            base, lo, hi = paired_bootstrap_map(
                ref_recs, ref_ngt, recs, ngt
            )
            entry["map7_2d"] = {
                "delta": base,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_boot": N_BOOT,
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        results[arm] = entry
        print(f"  {arm}: done")

    # Holm correction within each metric family, across arms and thresholds
    for metric in ("strict", "count"):
        keys = []
        pvals = []
        for arm, entry in results.items():
            if not entry.get("paired_applicable"):
                continue
            for theta, row in entry["accuracy"].items():
                keys.append((arm, theta))
                pvals.append(row[metric]["mcnemar_p"])
        if not pvals:
            continue
        for (arm, theta), adj in zip(keys, holm(pvals)):
            cell = results[arm]["accuracy"][theta][metric]
            cell["holm_p"] = float(adj)
            cell["significant_holm"] = bool(adj < 0.05)

    # write
    payload = {
        "reference": REFERENCE,
        "thetas": list(THETAS),
        "n_test_voxels": int(len(ref_ngt)),
        "multiplicity": {
            "method": "holm-bonferroni",
            "alpha": 0.05,
            "family": "per metric, across arms and both declared thresholds",
        },
        "note": (
            "Strict and count accuracy use McNemar on the shared test voxels; mAP@7 uses a "
            "paired bootstrap over voxels. Derived from the stored ND records, "
            "so no retraining or re-inference is involved."
        ),
        "runs": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(results)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
