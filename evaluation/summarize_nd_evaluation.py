"""Collect results/nd_evaluation/*.json into one comparison table.

Writes, into the same directory: nd_metrics_all_models.csv (one row per model, Wirth's table 3
scheme plus provenance), nd_metrics_table.md (grouped by test set; only rows sharing a test set
are directly comparable) and paired_deltas.json (paired bootstrap 95 % CIs of mAP@7 against the
baseline, same-test-set arms only).

Usage: PYTHONPATH=.:datagen python evaluation/summarize_nd_evaluation.py [results/nd_evaluation]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t1t2.nd_metrics import bootstrap_map_ci  # noqa: E402

BASELINE = "baseline_v2_reproduction"

# Display order and grouping. Only runs scored on the same test set may be compared with each
# other, so the table is grouped rather than sorted into one list. Runs not named here still
# appear, under "Other runs" at the end.
GROUPS = [
    ("Main dataset (t1_3500_t2_500_100k, test 9,999 voxels), directly comparable", [
        "baseline_v2_reproduction", "aux_loss", "loss_uniform", "exist_head_shared",
        "exist_weight_03", "decoder_2", "decoder_6", "queries_4", "queries_6",
        "physics_clean", "physics_noisy", "baseline_v3", "baseline_v3_no_sqrt",
        "baseline_v3_no_physics", "baseline_v4",
        "baseline_seed20260725", "baseline_seed20260726", "baseline_seed20260727",
        "loss_uniform_seed20260725", "loss_uniform_seed20260726", "loss_uniform_seed20260727",
        "final_uniform_q6_seed20260724", "final_uniform_q6_seed20260725",
        "final_uniform_q6_seed20260726", "final_uniform_q6_seed20260727",
    ]),
    ("Log-uniform T1 dataset (different test set, not comparable with above)",
     ["data_loguniform"]),
]


def load_all(res_dir):
    """Load saved model evaluations from a results folder."""
    out = {}
    for p in sorted(Path(res_dir).glob("*.json")):
        if p.name in ("paired_deltas.json",):
            continue
        d = json.loads(p.read_text())
        if "map" in d:
            out[d["name"]] = d
    return out


def row_of(d):
    """Turn one model's evaluation into a summary table row."""
    e = d["exact_at_threshold"]
    return {
        "model": d["name"],
        "test_set": Path(d["test_paths"][0]).parts[-3] if d["test_paths"] else "?",
        "mAP_avg": d["map"]["map_avg"], "mAP@7": d["map"]["map@7"],
        "mAP@5": d["map"]["map@5"], "mAP@10": d["map"]["map@10"],
        "precision": e["precision"], "recall": e["recall"], "f1": e["f1"],
        "mean_dT1_ms": e["mean_dt1_ms"], "mean_dT2_ms": e["mean_dt2_ms"],
        "mean_dw": e["mean_dw"],
        "median_dT1_ms": e["median_dt1_ms"], "median_dT2_ms": e["median_dt2_ms"],
        "exist_threshold(val)": d["existence_threshold"],
        "n_test": d["n_test_voxels"],
    }


def restore_records(d):
    """Restore saved matching records, including infinite unmatched distances."""
    recs = [[{k: (float("inf") if v is None and k == "nd_sum" else v)
              for k, v in r.items()} for r in voxel] for voxel in d["_records_tau7"]]
    return recs, np.asarray(d["_n_gt"])


def main(res_dir="results/nd_evaluation"):
    """Summarize model evaluations and compare their paired results."""
    res_dir = Path(res_dir)
    all_d = load_all(res_dir)
    print(f"loaded {len(all_d)} models: {sorted(all_d)}")

    if not all_d:
        print(f"no evaluated runs found in {res_dir}. Run run_nd_evaluation.py first.")
        return pd.DataFrame(), {}

    # Named runs first, in the order given above, then anything else, so an unlisted run is
    # still reported rather than dropped.
    named = [n for _, names in GROUPS for n in names if n in all_d]
    other = sorted(set(all_d) - set(named))
    groups = GROUPS + ([("Other runs (test sets not checked, compare with care)", other)]
                       if other else [])

    df = pd.DataFrame([row_of(all_d[n]) for n in named + other])
    df.to_csv(res_dir / "nd_metrics_all_models.csv", index=False)

    # markdown, grouped
    lines = ["# ND / mAP evaluation - all models",
             "",
             "Existence threshold calibrated on **val** (grid 0.25-0.75, step 0.05, "
             "best F1 at ND=7%), applied unchanged to **test**. mAP is threshold-free. "
             "Mean errors are over TPs only (bounded by the ND gate); read them next to "
             "recall.", ""]
    cols = ["model", "mAP_avg", "mAP@7", "mAP@5", "mAP@10", "precision", "recall",
            "f1", "mean_dT1_ms", "mean_dT2_ms", "mean_dw", "exist_threshold(val)"]
    for title, names in groups:
        sub = df[df["model"].isin(names)]
        if not len(sub):
            continue
        lines += [f"## {title}", "", "| " + " | ".join(cols) + " |",
                  "|" + "---|" * len(cols)]
        for _, r in sub.iterrows():
            vals = [r["model"]] + [
                f"{r[c]:.4f}" if isinstance(r[c], float) and "dT" not in c and c != "mean_dw"
                else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
                for c in cols[1:]]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    (res_dir / "nd_metrics_table.md").write_text("\n".join(lines))

    # paired bootstrap vs baseline (same test set only)
    deltas = {}
    if BASELINE in all_d:
        rb, nb = restore_records(all_d[BASELINE])
        for name in GROUPS[0][1]:
            if name == BASELINE or name not in all_d:
                continue
            ra, na = restore_records(all_d[name])
            if len(ra) != len(rb):
                continue
            point = (all_d[name]["map"]["map@7"] - all_d[BASELINE]["map"]["map@7"])
            ci = bootstrap_map_ci(ra, na, n_boot=300, recs_other=rb, n_gt_other=nb)
            deltas[name] = {"delta_map7_vs_baseline": point, **ci}
            print(f"{name:32s} dmAP@7 {point:+.4f}  CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
        (res_dir / "paired_deltas.json").write_text(json.dumps(deltas, indent=2))

    return df, deltas


if __name__ == "__main__":
    main(*sys.argv[1:])
