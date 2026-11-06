"""
Plots of the training of the neural network, saved in training/log/.

For every configuration, from training/log/<configuration>/history.csv written by training/train.py:
    loss.png          training and validation loss of every epoch, with the epoch of the saved weights
    val_metrics.png   validation metrics of every epoch: detection of the frames with a step (precision, recall, F1)
                      and localization error of the detected frames (position [m] and direction [deg])
For all the configurations together:
    summary.png       validation loss and median position error of every configuration
    summary.csv       metrics of every configuration at the epoch of the saved weights

train.py calls plot_history() at the end of a training, train.sh calls this script at the end of all the trainings:
    venv/bin/python -m training.plots
"""

import csv
import numpy as np

from pathlib import Path
from matplotlib.ticker import MaxNLocator
from analysis.plot_results import plt, BLUE, INK, INK2, BG

LOG_DIR = Path(__file__).parent / "log"
ORANGE = "#d1652b"
GREEN = "#3a9a5b"
# One colour, marker and line style per configuration, so that lines are not distinguished by colour only
SERIES = [
    {"color": BLUE, "marker": "o", "ls": "-"},
    {"color": ORANGE, "marker": "s", "ls": "--"},
    {"color": GREEN, "marker": "^", "ls": "-."},
    {"color": "#7a4fbf", "marker": "D", "ls": ":"},
    {"color": "#b8a21f", "marker": "v", "ls": "-"},
    {"color": INK2, "marker": "P", "ls": "--"},
]

def load_history(path: Path) -> dict[str, np.ndarray]:
    """
    Columns of a history.csv as arrays.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

def best_index(h: dict[str, np.ndarray]) -> int:
    """
    Index of the epoch with the lowest validation loss, the one of the saved weights.
    """
    return int(np.argmin(h["val_loss"]))

def plot_history(history_path: Path) -> None:
    """
    loss.png and val_metrics.png of one configuration, in the folder of its history.csv.
    """
    h = load_history(history_path)
    out_dir = history_path.parent
    name = out_dir.name
    b = best_index(h)
    ep = h["epoch"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(ep, h["train_loss"], color=BLUE, lw=2, label="training")
    ax.plot(ep, h["val_loss"], color=ORANGE, lw=2, ls="--", label="validation")
    ax.axvline(ep[b], color=INK2, lw=0.8, ls=":")
    ax.plot(ep[b], h["val_loss"][b], "o", color=ORANGE, markersize=8, markeredgecolor=BG)
    ax.text(ep[b], h["val_loss"][b], f"  saved weights (epoch {int(ep[b])})", color=INK2, fontsize=9, va="bottom")
    ax.set(title=f"Loss — {name}", xlabel="Epoch", ylabel="Loss (log scale)", yscale="log")
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    ax = axes[0]
    for key, label, st in (("val_precision", "precision", SERIES[0]), ("val_recall", "recall", SERIES[1]),
                           ("val_f1", "F1", SERIES[2])):
        ax.plot(ep, h[key] * 100, color=st["color"], ls=st["ls"], lw=2, label=label)
    ax.set(title="Detection of the frames with a step", xlabel="Epoch", ylabel="[%]", ylim=(0, 100))
    ax.legend(frameon=False, loc="lower right")
    for ax, key, title, ylabel in ((axes[1], "val_median_error_m", "Position error of the detected frames", "Median error [m]"),
                                   (axes[2], "val_median_angle_deg", "Direction error of the detected frames", "Median error [deg]")):
        ax.plot(ep, h[key], color=BLUE, lw=2)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel, ylim=(0, None))
    for ax in axes:
        ax.axvline(ep[b], color=INK2, lw=0.8, ls=":")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"Validation metrics — {name} (dotted line: epoch of the saved weights)", color=INK, x=0.01, ha="left")
    fig.savefig(out_dir / "val_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Plots of the training written to {out_dir}/{{loss,val_metrics}}.png")

def plot_summary(log_dir: Path) -> None:
    """
    summary.png and summary.csv of all the configurations with a history.csv in log_dir.
    """
    histories = {p.parent.name: load_history(p) for p in sorted(log_dir.glob("*/history.csv"))}
    if(len(histories) == 0):
        print(f"No history.csv in {log_dir}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for k, (name, h) in enumerate(histories.items()):
        st = SERIES[k % len(SERIES)]
        b = best_index(h)
        for ax, key in ((axes[0], "val_loss"), (axes[1], "val_median_error_m")):
            ax.plot(h["epoch"], h[key], color=st["color"], ls=st["ls"], lw=1.8, label=name)
            ax.plot(h["epoch"][b], h[key][b], marker=st["marker"], color=st["color"], markersize=8, markeredgecolor=BG)
    axes[0].set(title="Validation loss", xlabel="Epoch", ylabel="Loss (log scale)", yscale="log")
    axes[1].set(title="Median position error of the detected frames (validation)", xlabel="Epoch",
                ylabel="Median error [m]", ylim=(0, None))
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Training of every configuration (markers: epoch of the saved weights)", color=INK, x=0.01, ha="left")
    fig.savefig(log_dir / "summary.png", dpi=150)
    plt.close(fig)

    cols = ["epoch", "train_loss", "val_loss", "val_precision", "val_recall", "val_f1", "val_median_error_m", "val_median_angle_deg"]
    with open(log_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "epochs_run"] + [f"best_{c}" if c == "epoch" else c for c in cols])
        for name, h in histories.items():
            b = best_index(h)
            writer.writerow([name, len(h["epoch"])] + [f"{h[c][b]:.5g}" for c in cols])
    print(f"Summary of the trainings written to {log_dir}/summary.{{png,csv}}")

def main() -> None:
    for path in sorted(LOG_DIR.glob("*/history.csv")):
        plot_history(path)
    plot_summary(LOG_DIR)

if __name__ == "__main__":
    main()
