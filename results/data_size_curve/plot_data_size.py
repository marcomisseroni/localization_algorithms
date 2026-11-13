"""
Test error of the network (y_7m_10db) against the number of training recordings, before and after the random
offset of the training windows. Reads data_size_curve.csv in this folder and writes data_size_curve.png.
    venv/bin/python -m results.data_size_curve.plot_data_size      (or with PYTHONPATH=. from the project root)
aligned: windows starting at multiples of 2400 samples, the network can use the emission time (see
results/leak_check/summary.txt); x0.5 - x3 trained without the learning rate scheduler.
random_offset: stride 4800 with random offset at every epoch, the honest setup.
"""

import csv

from pathlib import Path
from analysis.plot_results import plt, BLUE, INK2, BG

HERE = Path(__file__).parent
ORANGE = "#d1652b"
STYLE = {
    "aligned": {"color": ORANGE, "marker": "s", "ls": "--", "label": "aligned windows (emission-time leak)"},
    "random_offset": {"color": BLUE, "marker": "o", "ls": "-", "label": "random offset (realistic)"},
}

def main() -> None:
    with open(HERE / "data_size_curve.csv") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, key, title in ((axes[0], "test_median_m", "Median error"), (axes[1], "test_p90_m", "P90 error")):
        for setup, st in STYLE.items():
            r = [x for x in rows if x["setup"] == setup]
            n = [int(x["n_recordings"]) for x in r]
            v = [float(x[key]) for x in r]
            ax.plot(n, v, color=st["color"], ls=st["ls"], lw=2, label=st["label"])
            for x, ni, vi in zip(r, n, v):
                hollow = "no scheduler" in x["source"]
                ax.plot(ni, vi, st["marker"], color=st["color"], markersize=8,
                        markerfacecolor=BG if hollow else st["color"], markeredgecolor=st["color"] if hollow else BG,
                        markeredgewidth=2 if hollow else 1.5)
            # direct label at the last point of every series
            ax.annotate(f"{v[-1]:.2f} m", (n[-1], v[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=9)
        ax.set(title=title, xscale="log", xlabel="Training recordings", ylabel="Test error [m]", ylim=(0, None))
        ax.set_xticks([725, 1451, 4353, 14508, 43505, 145116], ["x0.5", "x1", "x3", "x10", "x30", "x100"])
        ax.set_xlim(500, 400000)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Network y_7m_10db: test error vs training set size (hollow markers: trained without lr scheduler)",
                 color=INK2, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "data_size_curve.png", dpi=150)

if __name__ == "__main__":
    main()
