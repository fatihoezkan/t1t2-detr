#!/usr/bin/env python3
"""Write tables/arms/<run>.tex (one per single-change run) and tables/tab_matrix.tex.

Strict accuracy is the share of test voxels with exactly the right number of compartments, every
one accepted under the ND rule at tau = 7%, reported at the declared thresholds 0.50 and 0.75 in
2D and 3D beside the threshold-free mAP@7. Reads results/threshold_sweep/<run>.json,
results/nd_evaluation/<run>.json, results/nd_evaluation/tables_2d_3d.json and
results/<run>/{metrics_detr,parameter_recovery_detr,summary}.json. Nothing is typed by hand.
Usage: python3 evaluation/tables/build_strict_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
TABLES = ROOT / "tables"
ARMS = TABLES / "arms"

REFERENCE = "baseline_v2_reproduction"
FROZEN = "t1_3500_t2_500_weighted_long"   # the v1 baseline, trained on older code; optional

SAMPLING = 0.98       # binomial 95% sampling floor on n=9999 near 50%

# Run-to-run spread, measured 2026-08-27 as the range over the four reference runs that share
# code and data and differ only in train.seed: baseline_v2_reproduction (20260724) plus
# baseline_seed2026072{5,6,7}. The frozen baseline is excluded on purpose: it was trained on
# older code, so pooling it would fold library drift into a seed measurement. These supersede
# the old frozen-vs-repeat differences (0.0036 / 0.0066 / 0.60), which were a same-seed lower
# bound and understated seed variance by up to about 4.7x.
SEED_50 = 1.95        # strict accuracy at theta=0.50, range over 4 runs
SEED_75 = 0.86        # strict accuracy at theta=0.75, range over 4 runs
MAP_SEED = 0.0168     # mAP@7 2D, range over 4 runs (std 0.0081)
MAP_SEED_3D = 0.0199  # mAP@7 3D, range over 4 runs (std 0.0092)

# A difference has to clear both test-set noise and training noise, so the binding ruler is the
# larger of the two. At theta=0.50 training noise dominates; at 0.75 sampling does.
FLOOR_50 = max(SAMPLING, SEED_50)
FLOOR_75 = max(SAMPLING, SEED_75)

MATRIX = [
    ("loss_uniform", r"\texttt{loss\_uniform}", "uniform loss weight"),
    ("data_loguniform", r"\texttt{data\_loguniform}", r"log-uniform $T_1$"),
    ("queries_6", r"\texttt{queries\_6}", "6 queries"),
    ("queries_4", r"\texttt{queries\_4}", "4 queries"),
    ("aux_loss", r"\texttt{aux\_loss}", "aux.\\ decoder losses"),
    ("exist_weight_03", r"\texttt{exist\_weight\_03}", r"exist.\ weight 0.3"),
    ("decoder_2", r"\texttt{decoder\_2}", "2 decoder layers"),
    ("decoder_6", r"\texttt{decoder\_6}", "6 decoder layers"),
    ("exist_head_shared", r"\texttt{exist\_head\_shared}", r"shared exist.\ head"),
]
PHYSICS = [
    ("physics_clean", r"\texttt{physics\_clean}", "consistency, clean target"),
    ("physics_noisy", r"\texttt{physics\_noisy}", "consistency, measured target"),
]

VERDICT = {}   # filled in from the measurements themselves


def _json(path):
    return json.load(open(path))


def _optional(path):
    """A stored table that may be absent; an absent file holds no runs."""
    return _json(path) if Path(path).exists() else {}


def sweep(run, dim, theta, key):
    d = _json(RES / "threshold_sweep" / f"{run}.json")[dim]
    return next(r for r in d if abs(r["threshold"] - theta) < 1e-9)[key]


def map7(run):
    """(2D, 3D) mAP@7. The two stored table files use different layouts."""
    m2 = _json(RES / "nd_evaluation" / f"{run}.json")["map"]["map@7"]
    a = _optional(RES / "nd_evaluation" / "tables_2d_3d.json")
    b = _optional(RES / "nd_evaluation" / "tables_2d_3d_extra.json")
    if run in a:
        assert abs(a[run]["2d"]["map"]["map@7"] - m2) < 5e-4
        return m2, a[run]["3d"]["map"]["map@7"]
    if run in b:
        assert abs(b[run]["2d"]["map7"] - m2) < 5e-4
        return m2, b[run]["3d"]["map7"]
    return m2, None


def load(run):
    m = _json(RES / run / "metrics_detr.json")
    rec = _json(RES / run / "parameter_recovery_detr.json")
    summ = _json(RES / run / "summary.json")
    small = [b for b in rec["bins"] if b["weight_min"] == 0.05][0]
    m2, m3 = map7(run)
    return {
        "a2_50": sweep(run, "2d", 0.5, "voxel_acc"),
        "a2_75": sweep(run, "2d", 0.75, "voxel_acc"),
        "a3_50": sweep(run, "3d", 0.5, "voxel_acc"),
        "a3_75": sweep(run, "3d", 0.75, "voxel_acc"),
        "cacc75": sweep(run, "2d", 0.75, "count_acc"),
        "map2": m2, "map3": m3,
        "theta": summ["threshold_calibration"]["selected_threshold"],
        "t1": m["t1_rel_median"] * 100,
        "small_t1": small["t1_relative_error_median"] * 100,
    }


def verdict_for(x, ref):
    """One phrase, decided by the measurements, not by hand.

    Two independent signals have to agree before a run is called better or worse: strict
    accuracy at both declared thresholds, and the threshold-free mAP@7. A run can beat the
    reference at a fixed threshold only because its scores sit on a more convenient side of
    that threshold; mAP catches this, but only when the mAP drop clears the seed spread. It
    does not for the query-count runs, which is why they are called better here and flat in
    the chapter text: this verdict is measured at the two declared thresholds, whereas
    Chapter 5 reports at each run's own calibrated threshold. Both are stated.
    """
    d75, d50 = x["a2_75"] - ref["a2_75"], x["a2_50"] - ref["a2_50"]
    dmap = x["map2"] - ref["map2"]
    strict_up = d75 > FLOOR_75 and d50 > FLOOR_50
    strict_dn = d75 < -FLOOR_75 and d50 < -FLOOR_50
    strict_flat = abs(d75) <= FLOOR_75 and abs(d50) <= FLOOR_50
    map_up, map_dn = dmap > MAP_SEED, dmap < -MAP_SEED
    if strict_up and not map_dn:
        return "better"
    if strict_dn and not map_up:
        return "worse"
    if strict_flat and not (map_up or map_dn):
        return "no effect"
    return "mixed"


def sgn(v, digits, band=None):
    t = f"{v:+.{digits}f}"
    return r"\textbf{" + t + "}" if band is not None and abs(v) > band else t


def bold_if(val, delta, band, digits=2):
    t = f"{val:.{digits}f}"
    return r"\textbf{" + t + "}" if abs(delta) > band else t


def mini(run, display, ref):
    x = load(run)
    rows = [
        (r"Strict accuracy 2D, $\theta = 0.50$ (\%)", ref["a2_50"], x["a2_50"], 2, FLOOR_50),
        (r"Strict accuracy 2D, $\theta = 0.75$ (\%)", ref["a2_75"], x["a2_75"], 2, FLOOR_75),
        (r"Strict accuracy 3D, $\theta = 0.50$ (\%)", ref["a3_50"], x["a3_50"], 2, FLOOR_50),
        (r"Strict accuracy 3D, $\theta = 0.75$ (\%)", ref["a3_75"], x["a3_75"], 2, FLOOR_75),
        None,
        (r"mAP@7, 2D (no threshold)", ref["map2"], x["map2"], 4, MAP_SEED),
        (r"mAP@7, 3D (no threshold)", ref["map3"], x["map3"], 4, MAP_SEED_3D),
        None,
        (r"Count accuracy, $\theta = 0.75$ (\%)", ref["cacc75"], x["cacc75"], 2, 0.80),
    ]
    out = [r"\renewcommand{\arraystretch}{1.05}", r"\begin{tabular}{lrrr}", r"\toprule",
           r" & Reference & " + display + r" & $\Delta$ \\", r"\midrule"]
    for row in rows:
        if row is None:
            out.append(r"\midrule")
            continue
        label, a, b, dg, band = row
        if b is None:
            out.append(f"{label} & {a:.{dg}f} & --- & --- \\\\")
            continue
        out.append(f"{label} & {a:.{dg}f} & {b:.{dg}f} & {sgn(b - a, dg, band)} \\\\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{4}{l}{\emph{At the run's own fitted $\theta$ "
               r"(not comparable between runs)}} \\")
    out.append(f"\\quad fitted $\\theta$ & {ref['theta']:.2f} & {x['theta']:.2f} & --- \\\\")
    out.append(f"\\quad median rel.\\ $T_1$ error (\\%) & {ref['t1']:.2f} & {x['t1']:.2f} & "
               f"{x['t1'] - ref['t1']:+.2f} \\\\")
    out.append(f"\\quad smallest compartments, rel.\\ $T_1$ (\\%) & {ref['small_t1']:.2f} & "
               f"{x['small_t1']:.2f} & {sgn(x['small_t1'] - ref['small_t1'], 2, 5.23)} \\\\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{4}{l}{\textbf{Verdict at the declared $\theta$:} "
               + VERDICT[run] + r"} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def scorecard(ref):
    out = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
           r"\multirow{2}{*}{Run} & \multirow{2}{*}{What was changed} & "
           r"\multicolumn{2}{c}{strict acc.\ 2D} & \multicolumn{2}{c}{strict acc.\ 3D} & "
           r"\multirow{2}{*}{mAP@7} & \multirow{2}{*}{Verdict$^{\dagger}$} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r" & & $\theta{=}.50$ & $\theta{=}.75$ & $\theta{=}.50$ & $\theta{=}.75$ & & \\",
           r"\midrule",
           "reference & --- & " + f"{ref['a2_50']:.2f} & {ref['a2_75']:.2f} & "
           f"{ref['a3_50']:.2f} & {ref['a3_75']:.2f} & {ref['map2']:.4f}" + r" & --- \\",
           r"\midrule"]
    for run, display, change in MATRIX + PHYSICS:
        x = load(run)
        out.append(" & ".join([
            display, change,
            bold_if(x["a2_50"], x["a2_50"] - ref["a2_50"], FLOOR_50),
            bold_if(x["a2_75"], x["a2_75"] - ref["a2_75"], FLOOR_75),
            bold_if(x["a3_50"], x["a3_50"] - ref["a3_50"], FLOOR_50),
            bold_if(x["a3_75"], x["a3_75"] - ref["a3_75"], FLOOR_75),
            bold_if(x["map2"], x["map2"] - ref["map2"], MAP_SEED, 4),
            VERDICT[run],
        ]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    ARMS.mkdir(parents=True, exist_ok=True)
    ref = load(REFERENCE)
    for run, _d, _c in MATRIX + PHYSICS:
        VERDICT[run] = verdict_for(load(run), ref)

    print(f"run-to-run ruler (range over 4 seeds): strict @0.50 {FLOOR_50:.2f} pp, "
          f"@0.75 {FLOOR_75:.2f} pp, mAP@7 2D {MAP_SEED:.4f}, 3D {MAP_SEED_3D:.4f}")
    if (RES / "threshold_sweep" / f"{FROZEN}.json").exists():
        frozen = load(FROZEN)
        print(f"code/library drift, frozen vs repeat (reported separately): "
              f"{ref['a2_75'] - frozen['a2_75']:+.2f} pp strict 2D @0.75")
    else:
        print(f"frozen baseline {FROZEN} not under results/, drift line skipped")
    for run, _d, _c in MATRIX + PHYSICS:
        (ARMS / f"{run}.tex").write_text(mini(run, _d, ref))
        print(f"  {run:22s} verdict={VERDICT[run]}")
    (TABLES / "tab_matrix.tex").write_text(scorecard(ref))
    print("wrote scorecard + 11 per-run tables")


if __name__ == "__main__":
    main()
