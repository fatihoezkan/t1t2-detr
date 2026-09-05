#!/usr/bin/env python3
"""Write the baseline tables of Section 5.2 from the stored evaluation output.

tab_baseline and tab_baseline_own compare the frozen v1 baseline with the repeat run (declared
thresholds, then each run's own fitted threshold); they are skipped when the frozen run is not
under results/. tab_baseline_perK and tab_seed_spread cover the four reference seeds. Reads
results/threshold_sweep, results/threshold_val, results/nd_evaluation/tables_2d_3d.json,
results/review_stats.json and results/<run>/{metrics_detr,parameter_recovery_detr,summary}.json.
Usage: python3 evaluation/tables/build_baseline_tables.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
TABLES = ROOT / "tables"

SEED1 = "t1_3500_t2_500_weighted_long"
SEED2 = "baseline_v2_reproduction"


def _json(path):
    return json.load(open(path))


def _optional(path):
    """A stored table that may be absent; an absent file holds no runs."""
    return _json(path) if Path(path).exists() else {}


def sweep(run):
    return _json(RES / "threshold_sweep" / f"{run}.json")


def at(run, dim, theta, key):
    return next(r for r in sweep(run)[dim] if abs(r["threshold"] - theta) < 1e-9)[key]


def map7(run):
    m2 = _json(RES / "nd_evaluation" / f"{run}.json")["map"]["map@7"]
    a = _optional(RES / "nd_evaluation" / "tables_2d_3d.json")
    b = _optional(RES / "nd_evaluation" / "tables_2d_3d_extra.json")
    m3 = (a[run]["3d"]["map"]["map@7"] if run in a else b[run]["3d"]["map7"])
    return m2, m3


def headline():
    rows = []
    def add(label, f, digits=2):
        a, b = f(SEED1), f(SEED2)
        rows.append((label, f"{a:.{digits}f}", f"{b:.{digits}f}", f"{b - a:+.{digits}f}"))
    for dim in ("2d", "3d"):
        for th in (0.5, 0.75):
            add(rf"Strict voxel accuracy {dim.upper()}, $\theta = {th:.2f}$ (\%)",
                lambda r, d=dim, t=th: at(r, d, t, "voxel_acc"))
    rows.append(None)
    for dim in ("2d", "3d"):
        for key, name in (("precision", "precision"), ("recall", "recall"), ("f1", r"F$_1$")):
            add(rf"Strict {name} {dim.upper()}, $\theta = 0.75$",
                lambda r, d=dim, k=key: at(r, d, 0.75, k), 4)
    rows.append(None)
    add(r"mAP@7, 2D (no threshold)", lambda r: map7(r)[0], 4)
    add(r"mAP@7, 3D (no threshold)", lambda r: map7(r)[1], 4)
    rows.append(None)
    add(r"Count accuracy, $\theta = 0.75$ (\%)", lambda r: at(r, "2d", 0.75, "count_acc"))
    out = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Measure & Frozen baseline & Repeat run & $\Delta$ \\", r"\midrule"]
    for r in rows:
        out.append(r"\midrule" if r is None else " & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def own():
    def m(run):
        return _json(RES / run / "metrics_detr.json")
    def s(run):
        return _json(RES / run / "summary.json")
    a, b = m(SEED1), m(SEED2)
    ta = s(SEED1)["threshold_calibration"]["selected_threshold"]
    tb = s(SEED2)["threshold_calibration"]["selected_threshold"]
    spec = [
        (r"Fitted threshold $\theta$", "%.2f", ta, tb),
        (r"Count accuracy (\%)", "%.2f", a["count_accuracy"] * 100, b["count_accuracy"] * 100),
        (r"Count MAE (compartments)", "%.4f", a["count_mae"], b["count_mae"]),
        (r"Existence precision", "%.4f", a["existence_precision"], b["existence_precision"]),
        (r"Existence recall", "%.4f", a["existence_recall"], b["existence_recall"]),
        (r"Existence F$_1$", "%.4f", a["existence_f1"], b["existence_f1"]),
        (r"$T_1$ relative error, median (\%)", "%.2f", a["t1_rel_median"] * 100, b["t1_rel_median"] * 100),
        (r"$T_2$ relative error, median (\%)", "%.2f", a["t2_rel_median"] * 100, b["t2_rel_median"] * 100),
        (r"$T_1$ absolute error, median (ms)", "%.2f", a["t1_abs_median_ms"], b["t1_abs_median_ms"]),
        (r"$T_2$ absolute error, median (ms)", "%.2f", a["t2_abs_median_ms"], b["t2_abs_median_ms"]),
        (r"Weight absolute error, median", "%.4f", a["w_abs_median"], b["w_abs_median"]),
        (r"$T_1$ absolute error, mean (ms)", "%.2f", a["t1_abs_mean_ms"], b["t1_abs_mean_ms"]),
        (r"$T_2$ absolute error, mean (ms)", "%.2f", a["t2_abs_mean_ms"], b["t2_abs_mean_ms"]),
        (r"Missed compartments per voxel", "%.3f", a["missed_compartments_per_voxel"], b["missed_compartments_per_voxel"]),
        (r"False positives per voxel", "%.3f", a["false_positive_compartments_per_voxel"], b["false_positive_compartments_per_voxel"]),
    ]
    out = [r"\begin{tabular}{lrr}", r"\toprule",
           r"Measure & Frozen baseline & Repeat run \\", r"\midrule"]
    for label, fmt, x, y in spec:
        out.append(f"{label} & {fmt % x} & {fmt % y} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def by_k(run, dim, key, k, theta="0.75"):
    """Per-compartment-count value, tolerant of the two stored schemas.

    Older sweeps store 2d_by_k[theta][k] (strict only); newer ones store
    2d_by_k[theta]["strict"|"count"][k].
    """
    b = sweep(run)[dim + "_by_k"][theta]
    return b[key][k] if key in b else b[k]


def review_stats():
    return _json(RES / "review_stats.json")


def per_k():
    """Per-K table over the four reference seeds, at each run's calibrated theta.

    Section 5.1 declares the calibrated threshold the primary protocol, so the paired
    count/strict decomposition has to be measured there too: quoting count accuracy at the
    declared theta beside strict accuracy at the calibrated one made their difference
    uninterpretable. The numbers come from build_review_stats.py, which re-runs the four
    checkpoints and reproduces the stored theta = 0.75 evaluation exactly as a check.
    """
    g = review_stats()["families"]["baseline"]["calibrated"]["by_k"]

    # Per-K parameter errors come from the stored metrics_detr.json of the four seed runs,
    # i.e. medians over matched compartments at each run's own fitted threshold (the protocol
    # under which parameter errors are reported throughout the thesis).
    def rel_errs(k, p):
        vals = [_json(RES / r / "metrics_detr.json")[f"n{k}_{p}_rel_median"] * 100
                for r in SEED_RUNS]
        return sum(vals) / len(vals), max(vals) - min(vals)

    out = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
           r"& \multicolumn{2}{c}{Count accuracy} & \multicolumn{2}{c}{Strict 2D}"
           r" & \multicolumn{2}{c}{Rel.\ $T_1$ error (\%)} & \multicolumn{2}{c}{Rel.\ $T_2$ error (\%)} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
           r"& mean & range & mean & range & mean & range & mean & range \\", r"\midrule"]
    for k in ("1", "2", "3"):
        b = g[k]
        t1m, t1r = rel_errs(k, "t1")
        t2m, t2r = rel_errs(k, "t2")
        out.append(f"$K={k}$ & {b['count_acc']['mean']:.2f} & {b['count_acc']['range']:.2f}"
                   f" & {b['strict_acc']['mean']:.2f} & {b['strict_acc']['range']:.2f}"
                   f" & {t1m:.2f} & {t1r:.2f} & {t2m:.2f} & {t2r:.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


SEED_RUNS = ["baseline_v2_reproduction", "baseline_seed20260725",
             "baseline_seed20260726", "baseline_seed20260727"]
SEED_LABELS = ["20260724", "20260725", "20260726", "20260727"]


def seed_spread():
    """The four reference runs that differ only in train.seed, and their spread.

    This is the ruler every comparison in the results chapter is judged against, so it is
    built from the stored output rather than typed. The frozen baseline is not in this table:
    it was trained on older code, so including it would fold library drift into a seed
    measurement. Transposed (quantities down, seeds across) so that every number the text
    quotes as "the seed spread" is visibly in the table, including the calibrated-theta strict
    accuracy that is the ruler itself and the K=3 strict accuracy.
    """
    def val(run):
        cal = _json(RES / "threshold_val" / f"{run}.json")
        smry = _json(RES / run / "summary.json")
        rec = _json(RES / run / "parameter_recovery_detr.json")
        small = [b for b in rec["bins"] if b["weight_min"] == 0.05][0]
        rv = review_stats()["runs"][run]["calibrated"]
        return [smry["best_parameter_val_loss"],
                cal["test_voxel_acc_at_val_theta"],
                rv["by_k"]["1"]["strict_acc"],
                rv["by_k"]["2"]["strict_acc"],
                rv["by_k"]["3"]["strict_acc"],
                rv["count_acc"],
                at(run, "2d", 0.5, "voxel_acc"),
                at(run, "2d", 0.75, "voxel_acc"),
                map7(run)[0], map7(run)[1],
                small["t1_relative_error_median"] * 100,
                small["t2_relative_error_median"] * 100]

    labels = [
        (r"val.\ parameter loss (selection)", 5),
        (r"strict acc.\ 2D, calibrated $\theta$ (\%)", 2),
        (r"\quad of which $K = 1$ (\%)", 2),
        (r"\quad of which $K = 2$ (\%)", 2),
        (r"\quad of which $K = 3$ (\%)", 2),
        (r"count accuracy, calibrated $\theta$ (\%)", 2),
        (r"strict acc.\ 2D, $\theta = 0.50$ (\%)", 2),
        (r"strict acc.\ 2D, $\theta = 0.75$ (\%)", 2),
        (r"mAP@7, 2D (no threshold)", 4),
        (r"mAP@7, 3D (no threshold)", 4),
        (r"smallest-band rel.\ $T_1$ err.\ (\%)", 2),
        (r"smallest-band rel.\ $T_2$ err.\ (\%)", 2),
    ]
    vals = [val(r) for r in SEED_RUNS]          # per run
    rows = list(zip(*vals))                     # per quantity
    out = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"& \multicolumn{4}{c}{seed} & \multirow{2}{*}{\emph{range}} "
           r"& \multirow{2}{*}{\emph{std}} \\",
           r"\cmidrule(lr){2-5}",
           r"Quantity & " + " & ".join(SEED_LABELS) + r" & & \\",
           r"\midrule"]
    for (lab, d), r in zip(labels, rows):
        cells = [f"{x:.{d}f}" for x in r]
        cells.append(rf"\textbf{{{max(r) - min(r):.{d}f}}}")
        cells.append(f"{statistics.stdev(r):.{d}f}")
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    jobs = []
    if (RES / "threshold_sweep" / f"{SEED1}.json").exists() and (RES / SEED1 / "metrics_detr.json").exists():
        jobs += [("tab_baseline", headline), ("tab_baseline_own", own)]
    else:
        print(f"skipping tab_baseline and tab_baseline_own: frozen baseline {SEED1} is not under results/")
    jobs += [("tab_baseline_perK", per_k), ("tab_seed_spread", seed_spread)]
    for name, build in jobs:
        (TABLES / f"{name}.tex").write_text(build())
        print("wrote", name)


if __name__ == "__main__":
    main()
