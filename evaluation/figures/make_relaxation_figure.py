"""The two relaxation processes and the inversion-recovery experiment (background figure).

(a) Longitudinal recovery 1 - exp(-t/T1) with the 1 - 1/e (63 %) mark. (b) Transverse decay
exp(-t/T2) with the 1/e (37 %) mark. (c) Inversion recovery 1 - 2 exp(-TI/T1) with the zero
crossing at T1 ln 2. Time axes are in units of the respective relaxation time. Needs no
data or checkpoint. Writes figures/00_relaxation.png.
Usage: python3 evaluation/figures/make_relaxation_figure.py
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "00_relaxation.png"

BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "legend.fontsize": SMALL, "xtick.labelsize": TINY, "ytick.labelsize": TINY,
    "axes.titlelocation": "left", "axes.titlepad": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C_GT, C_MATCH, C_FP, GREY = "#1f4e79", "#7b3294", "#d55e00", "#8a8a8a"

# three panels; time in units of the relaxation time
fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.2))

# (a) longitudinal recovery
T1 = 1.0
t = np.linspace(0, 5 * T1, 500)
ax[0].plot(t, 1 - np.exp(-t / T1), color=C_GT, lw=1.8)
ax[0].axhline(1.0, color=GREY, lw=0.8, ls=":")
ax[0].plot([T1, T1], [0, 1 - np.exp(-1)], color=C_FP, lw=0.9, ls="--")
ax[0].plot([0, T1], [1 - np.exp(-1)] * 2, color=C_FP, lw=0.9, ls="--")
ax[0].plot([T1], [1 - np.exp(-1)], "o", color=C_FP, ms=4)
ax[0].annotate(r"$1-1/e \approx 63\,\%$", xy=(T1, 1 - np.exp(-1)),
               xytext=(1.45, 0.44), fontsize=SMALL, color=C_FP,
               arrowprops=dict(arrowstyle="-", color=C_FP, lw=0.7))
ax[0].set_xticks([0, T1, 2 * T1, 3 * T1, 4 * T1, 5 * T1])
ax[0].set_xticklabels(["0", "$T_1$", "$2T_1$", "$3T_1$", "$4T_1$", "$5T_1$"])
ax[0].set_ylim(0, 1.12); ax[0].set_yticks([0, 0.5, 1.0])
ax[0].set_xlabel("time after the pulse")
ax[0].set_ylabel("longitudinal magnetization / $M_0$")
ax[0].set_title("(a) longitudinal relaxation ($T_1$)")

# (b) transverse decay
T2 = 1.0
ax[1].plot(t, np.exp(-t / T2), color=C_MATCH, lw=1.8)
ax[1].plot([T2, T2], [0, np.exp(-1)], color=C_FP, lw=0.9, ls="--")
ax[1].plot([0, T2], [np.exp(-1)] * 2, color=C_FP, lw=0.9, ls="--")
ax[1].plot([T2], [np.exp(-1)], "o", color=C_FP, ms=4)
ax[1].annotate(r"$1/e \approx 37\,\%$", xy=(T2, np.exp(-1)),
               xytext=(1.6, 0.55), fontsize=SMALL, color=C_FP,
               arrowprops=dict(arrowstyle="-", color=C_FP, lw=0.7))
ax[1].set_xticks([0, T2, 2 * T2, 3 * T2, 4 * T2, 5 * T2])
ax[1].set_xticklabels(["0", "$T_2$", "$2T_2$", "$3T_2$", "$4T_2$", "$5T_2$"])
ax[1].set_ylim(0, 1.12); ax[1].set_yticks([0, 0.5, 1.0])
ax[1].set_xlabel("time after the pulse (TE)")
ax[1].set_ylabel("transverse signal / $S_0$")
ax[1].set_title("(b) transverse relaxation ($T_2$)")

# (c) inversion recovery
ti = np.linspace(0, 4 * T1, 500)
ax[2].plot(ti, 1 - 2 * np.exp(-ti / T1), color=C_GT, lw=1.8)
ax[2].axhline(0, color="black", lw=0.8)
zc = T1 * np.log(2)
ax[2].plot([zc], [0], "o", color=C_FP, ms=4)
ax[2].annotate(r"zero crossing at $T_1\ln 2$", xy=(zc, 0),
               xytext=(1.15, -0.55), fontsize=SMALL, color=C_FP,
               arrowprops=dict(arrowstyle="-", color=C_FP, lw=0.7))
ax[2].set_xticks([0, T1, 2 * T1, 3 * T1, 4 * T1])
ax[2].set_xticklabels(["0", "$T_1$", "$2T_1$", "$3T_1$", "$4T_1$"])
ax[2].set_ylim(-1.15, 1.15); ax[2].set_yticks([-1, 0, 1])
ax[2].spines["bottom"].set_position(("data", -1.15))
ax[2].set_xlabel("inversion time TI")
ax[2].set_ylabel("longitudinal magnetization / $M_0$")
ax[2].set_title("(c) inversion recovery, the $T_1$ measurement")

fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
