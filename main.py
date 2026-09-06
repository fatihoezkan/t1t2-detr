#!/usr/bin/env python3
"""Run the thesis pipeline end to end.

    python main.py                                 every stage, in order
    python main.py evaluate figures                only the named stages
    python main.py evaluate --runs loss_uniform    restrict the per-run stages (train, evaluate)
    python main.py --force figures                 redo the named stages even if their outputs exist
    python main.py --dry-run                       print the plan and run nothing

Stages, in order: data, train, evaluate, aggregate, figures, notebook. Every step is
one of the existing scripts, run as a subprocess from the repository root with
PYTHONPATH=.:datagen; this file only orders them and checks their inputs and outputs. A step
is skipped when everything it writes already exists, so on a fresh clone with the release
checkpoints unpacked, `python main.py` regenerates the derived files and touches neither the
data nor the trained runs. A failing step stops the pipeline; rerunning continues from it.

The train stage is the 26 runs of the matrix, about half an hour each on an A100. On a CPU
it is not practical: run `python -m t1t2.experiment --config <yaml>` in your cluster's job
script instead and run the other stages afterwards. The notebook's last cell runs the figure
scripts once more by itself, so the notebook stage repeats the work of the figures stage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'datagen'}")
STAGES = ("data", "train", "evaluate", "aggregate", "figures", "notebook")

REFERENCE = "baseline_v2_reproduction"
FINAL = "loss_uniform"

# The seed families evaluation/snr_ladder.py scores (its DEFAULT_FAMILIES); the list mirrors it.
SNR_RUNS = ["baseline_v2_reproduction", "baseline_seed20260725", "baseline_seed20260726",
            "baseline_seed20260727", "loss_uniform", "loss_uniform_seed20260725",
            "loss_uniform_seed20260726", "loss_uniform_seed20260727"]

# Configs that are not runs of their own: smoke is the development config, and
# final_uniform_q6 is the seed-20260724 run under its family name, carrying the criteria.
DOC_ONLY = {"smoke", "final_uniform_q6"}

# The two dataset families exactly as the README states them; the seeds are the ones the
# shipped manifests record.
FAMILIES = {
    "t1_3500_t2_500_100k": ["--seed", "3500500"],
    "t1_loguniform_100k": ["--seed", "3500501", "--sampling", "t1_log_uniform"],
}
SIZES = ["--n-train", "33333", "--n-val", "3333", "--n-test", "3333", "--n-per-snr", "1667",
         "--t1-min", "50", "--t1-max", "3500", "--t2-min", "5", "--t2-max", "500"]


@dataclass
class Step:
    """One script invocation with the files it writes and the files it cannot do without."""

    name: str
    cmd: list[str]
    outputs: list[Path]                                   # skipped when all exist
    needs: list[tuple[Path, str]] = field(default_factory=list)   # (file, hint when missing)
    done: Callable[[], bool] | None = None                # replaces the outputs check

    def is_done(self) -> bool:
        if self.done is not None:
            return self.done()
        return bool(self.outputs) and all(p.exists() for p in self.outputs)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)


def checkpoint(run: str) -> tuple[Path, str]:
    # The shipped results/<run>/summary.json makes the train step count as done, so a retrain
    # has to be forced.
    return (RESULTS / run / "checkpoints" / "best.pt",
            "no checkpoint; unpack checkpoints_best.zip from the GitHub release (README, Trained "
            f"models) or retrain with `python main.py --force train --runs {run}` on a GPU node")


def configs() -> dict[str, Path]:
    """Run name -> config path for every trained run of the matrix, in config order."""
    out = {}
    for p in sorted(ROOT.glob("configs/**/*.yaml")):
        name = yaml.safe_load(p.read_text())["name"]
        if name not in DOC_ONLY:
            out[name] = p
    return out


CONFIGS = configs()


# ---------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------

def data_stage(force: bool):
    for family, extra in FAMILIES.items():
        for n in (1, 2, 3):
            d = Path("data") / family / f"n{n}"
            cmd = [PY, "datagen/run_generator.py", "--n-comp", str(n), "--out-dir", str(d),
                   *extra, *SIZES] + (["--overwrite"] if force else [])
            yield Step(f"data {d}", cmd,
                       [ROOT / d / f"{split}.parquet" for split in ("train", "val", "test")])


def train_stage(runs: list[str]):
    for name in runs:
        cfg = CONFIGS[name]
        paths = yaml.safe_load(cfg.read_text())["data"]["train_path"]
        paths = [paths] if isinstance(paths, str) else paths
        yield Step(f"train {name}", [PY, "-m", "t1t2.experiment", "--config", rel(cfg)],
                   [RESULTS / name / "summary.json"],
                   needs=[(ROOT / p, "run `python main.py data` first") for p in paths])


def evaluate_stage(runs: list[str]):
    for name in runs:
        ck = [checkpoint(name)]
        yield Step(f"nd-evaluation {name}",
                   [PY, "evaluation/run_nd_evaluation.py", f"results/{name}"],
                   [RESULTS / "nd_evaluation" / f"{name}.json"], ck)
        yield Step(f"threshold-val {name}", [PY, "evaluation/calibrate_threshold.py", name],
                   [RESULTS / "threshold_val" / f"{name}.json"], ck)
        yield Step(f"threshold-sweep {name}", [PY, "evaluation/threshold_sweep.py", name],
                   [RESULTS / "threshold_sweep" / f"{name}.json"], ck)
        if name == REFERENCE:
            yield Step(f"query-analysis {name}", [PY, "evaluation/query_analysis.py", name],
                       [RESULTS / name / "query_analysis.json"], ck)


def aggregate_stage(runs: list[str]):
    nd = RESULTS / "nd_evaluation"
    dumps = [(nd / f"{r}.json", "run `python main.py evaluate` first") for r in runs]
    yield Step("nd-summary", [PY, "evaluation/summarize_nd_evaluation.py"],
               [nd / "nd_metrics_all_models.csv", nd / "nd_metrics_table.md",
                nd / "paired_deltas.json"], dumps)
    yield Step("paired-tests", [PY, "evaluation/paired_tests.py"],
               [RESULTS / "paired_tests.json"], dumps)
    yield Step("compare-experiments",
               [PY, "evaluation/compare_experiments.py", "--all", "--quiet"],
               [RESULTS / "_comparison" / f for f in ("comparison.md", "comparison_metrics.csv",
                                                      "comparison_arms.csv")])
    evaluated = "run `python main.py evaluate` first"
    yield Step("snr-ladder", [PY, "evaluation/snr_ladder.py"],
               [RESULTS / "snr_ladder" / f for f in ("summary.json", "snr_ladder.png",
                                                     "snr_ladder.tex")],
               [checkpoint(r) for r in SNR_RUNS]
               + [(RESULTS / "threshold_val" / f"{r}.json", evaluated) for r in SNR_RUNS])
    yield Step("noise-ratio-table", [PY, "evaluation/figures/make_noise_ratio_table.py"],
               [RESULTS / "compartment_noise_ratio_test.parquet"],
               [(nd / f"{r}.json", evaluated) for r in (REFERENCE, FINAL)])


def figures_stage():
    both = [checkpoint(REFERENCE), checkpoint(FINAL)]
    ratio = [(RESULTS / "compartment_noise_ratio_test.parquet",
              "run `python main.py aggregate` first")]
    # (script and arguments, files written, files needed), in evaluation/figures/README.md order.
    plan = [
        (["make_relaxation_figure.py"], [FIGURES / "00_relaxation.png"], []),
        (["make_query_figure.py"], [FIGURES / "11_queries.png"], [checkpoint(REFERENCE)]),
        (["make_scatter_figure.py"], [FIGURES / "12_pred_true_scatter.png"], both),
        (["make_error_map.py"], [FIGURES / "13_error_map.png"], [checkpoint(FINAL)]),
        (["make_found_scatter.py"], [FIGURES / "14_found_missed.png"], both),
        (["make_found_scatter.py", "--map7"], [FIGURES / "15_found_missed_map7.png"], both),
        (["make_t2_profile.py"], [FIGURES / "16_t2_profile.png"], both),
        (["make_missed_dist.py"], [FIGURES / "17_missed_dist.png"], both),
        (["make_missed_scatter.py"], [FIGURES / "17_missed_scatter.png"], both),
        (["make_error_distribution.py"],
         [FIGURES / "19_error_distribution.png", RESULTS / "error_distribution_summary.json"],
         both),
        (["make_noise_effect_figure.py"],
         [FIGURES / "20_noise_small_compartments.png", RESULTS / "separability_k2_test.parquet"],
         ratio),
        (["plot_threshold_sweep.py"], [FIGURES / "fig_threshold_sweep.png"], []),
    ]
    for args, outputs, needs in plan:
        yield Step(f"figure {' '.join(args)}",
                   [PY, f"evaluation/figures/{args[0]}", *args[1:]], outputs, needs)


def notebook_stage():
    # Executed in place under a real kernel; the notebook's first cell moves to the repository
    # root itself, and no cell has a time limit because two of them run inference.
    yield Step("notebook",
               [PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=-1", "notebooks/thesis.ipynb"],
               [ROOT / "notebooks" / "thesis.ipynb"], [checkpoint(REFERENCE), checkpoint(FINAL)])


# ---------------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------------

def run_steps(steps: list[Step], force: bool, dry: bool) -> None:
    t_all = time.time()
    for s in steps:
        if not force and s.is_done():
            print(f"skip   {s.name}")
            continue
        missing = [(p, hint) for p, hint in s.needs if not p.exists()]
        if missing:
            p, hint = missing[0]
            msg = f"{s.name}: {rel(p)} is missing; {hint}"
            if dry:
                print(f"blocked {msg}")
                continue
            sys.exit(f"stop   {msg}")
        shown = " ".join(["python", *s.cmd[1:]])
        print(f"{'plan' if dry else 'run '}   {s.name}: {shown}", flush=True)
        if dry:
            continue
        t = time.time()
        if subprocess.run(s.cmd, cwd=ROOT, env=ENV).returncode:
            sys.exit(f"stop   {s.name} failed; finished steps are kept, rerun to continue")
        print(f"done   {s.name} ({time.time() - t:.0f} s)", flush=True)
    print(f"finished in {(time.time() - t_all) / 60:.1f} min")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        usage="main.py [STAGE ...] [--runs RUN [RUN ...]] [--force] [--dry-run]",
        description="Run the thesis pipeline end to end, or the named stages of it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"stages, in order: {', '.join(STAGES)}",
    )
    ap.add_argument("stages", nargs="*", metavar="STAGE", help="default: all stages")
    ap.add_argument("--runs", nargs="+", metavar="RUN",
                    help="restrict train and evaluate to these runs (default: the matrix)")
    ap.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    a = ap.parse_args(argv)

    bad = sorted(set(a.stages) - set(STAGES))
    if bad:
        ap.error(f"unknown stage(s) {bad}; choose from {', '.join(STAGES)}")
    if set(a.runs or []) & set(STAGES):
        ap.error("stage names go before --runs, e.g. `python main.py evaluate --runs loss_uniform`")
    if a.force and not a.stages:
        ap.error("--force with no stage would regenerate both dataset families and retrain all "
                 "26 runs; name the stages to redo, e.g. `python main.py --force figures`")
    runs = a.runs or list(CONFIGS)
    bad = sorted(set(runs) - set(CONFIGS))
    if bad:
        ap.error(f"unknown run(s) {bad}; known: {', '.join(CONFIGS)}")

    build = {
        "data": lambda: data_stage(a.force),
        "train": lambda: train_stage(runs),
        "evaluate": lambda: evaluate_stage(runs),
        "aggregate": lambda: aggregate_stage(runs),
        "figures": figures_stage,
        "notebook": notebook_stage,
    }
    wanted = a.stages or list(STAGES)
    steps = [s for stage in STAGES if stage in wanted for s in build[stage]()]
    run_steps(steps, a.force, a.dry_run)


if __name__ == "__main__":
    main()
