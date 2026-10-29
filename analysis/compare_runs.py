"""
Compare the results of all the runs of the experiment campaign.

Reads every results/<dataset>_<localizer>.csv written by evaluate.py, gets the factors of the run from the dataset
name (<layout>_<aperture>m_<position>_snr<snr>, e.g. y_7m_center_snr20 or y_3p5m_center_snr20) and saves in
results/summary/:
    summary.csv                  one row per run: factors and error statistics
    layout_position.png          group A: median and P90 error for every layout x position (7 m, 20 dB)
    aperture.png                 group B: error and estimates per recording vs. aperture (Y, centre, 20 dB)
    snr.png                      group C: error and estimates per recording vs. SNR (Y, 7 m, centre)
    reference_cdf.png            error CDF of every localizer on the reference configuration
    distance_by_layout.png       median error vs. distance from the array centre for every layout (centre, 7 m, 20 dB)

Usage, from the project root (after run_experiments.sh):
    venv/bin/python -m analysis.compare_runs
"""

import argparse
import csv
import json
import re

import matplotlib
import numpy as np

from pathlib import Path
from core.metrics import binned_median, error_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Same style as plot_results.py
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BG = "#fcfcfb"
# One colour, marker and line style per series, so that series are not distinguished by colour only
SERIES = [
    {"color": "#2a78d6", "marker": "o", "ls": "-"},
    {"color": "#d1652b", "marker": "s", "ls": "--"},
    {"color": "#3a9a5b", "marker": "^", "ls": "-."},
]

REFERENCE = {"layout": "y", "aperture": 7.0, "position": "center", "snr": 20}
LAYOUTS = ["y", "square", "l"]
POSITIONS = ["corner", "mid", "center"]
LAYOUT_LABEL = {"y": "Y", "square": "Square", "l": "L"}
NAME_RE = re.compile(r"^(?P<layout>[a-z]+)_(?P<aperture>[0-9p]+)m_(?P<position>[a-z]+)_snr(?P<snr>\d+)_(?P<loc>.+)$")

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the results of all the runs")
    parser.add_argument("--results", type=Path, default=Path("results"), help="folder with the CSVs of evaluate.py")
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder with the datasets")
    return parser.parse_args()


def load_runs(results_dir: Path, data_dir: Path) -> list[dict]:
    """One dict per run with its factors, the errors and the per-estimate distance from the array centre."""
    runs = []
    for path in sorted(results_dir.glob("*.csv")):
        m = NAME_RE.match(path.stem)
        if(m is None):
            continue                                        # not a campaign run (e.g. old test CSVs)
        dataset = path.stem[:-(len(m["loc"]) + 1)]
        with open(data_dir / dataset / "manifest.json") as f:
            centre = np.array(json.load(f)["microphones_xyz_m"])[:, :2].mean(axis=0)
        errors, dists, recordings = [], [], set()
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                recordings.add(r["sample"])
                if(r["error_m"] != ""):
                    errors.append(float(r["error_m"]))
                    dists.append(np.linalg.norm(np.array([float(r["true_x"]), float(r["true_y"])]) - centre))
        err = np.array(errors)
        runs.append({
            "dataset": dataset,
            "loc": m["loc"],
            "layout": m["layout"],
            "aperture": float(m["aperture"].replace("p", ".")),
            "position": m["position"],
            "snr": int(m["snr"]),
            "err": err,
            "dist": np.array(dists),
            "est_per_rec": err.size / max(len(recordings), 1),
            **error_stats(err),
        })
    return runs


def select(runs: list[dict], **factors) -> list[dict]:
    return [r for r in runs if all(r[k] == v for k, v in factors.items())]


def write_summary(runs: list[dict], out: Path) -> None:
    cols = ["dataset", "loc", "layout", "aperture", "position", "snr", "est_per_rec",
            "median", "p90", "p95", "mean", "rmse", "max", "below_10cm", "below_50cm"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            w.writerow([f"{r[c]:.4f}" if isinstance(r[c], float) else r[c] for c in cols])
    print(f"\n{'dataset':24s} {'localizer':12s} {'est/rec':>7s} {'median':>7s} {'P90':>7s} {'<50cm':>6s}")
    for r in runs:
        print(f"{r['dataset']:24s} {r['loc']:12s} {r['est_per_rec']:7.2f} {r['median']:7.3f} {r['p90']:7.2f} {r['below_50cm']:5.0f}%")


def plot_layout_position(runs: list[dict], locs: list[str], out: Path) -> None:
    """Group A: one 3 x 3 grid (layout x position) per localizer and metric, same colour scale everywhere."""
    groupA = [r for r in runs if r["aperture"] == REFERENCE["aperture"] and r["snr"] == REFERENCE["snr"]]
    metrics = [("median", "Median error"), ("p90", "P90 error")]
    vmax = max([r[m] for r in groupA for m, _ in metrics if np.isfinite(r[m])], default=1.0)
    fig, axes = plt.subplots(len(metrics), len(locs), figsize=(4.6 * len(locs), 7.5), constrained_layout=True, squeeze=False)
    for i, (metric, label) in enumerate(metrics):
        for j, loc in enumerate(locs):
            ax = axes[i, j]
            grid = np.full((len(LAYOUTS), len(POSITIONS)), np.nan)
            for a, lay in enumerate(LAYOUTS):
                for b, pos in enumerate(POSITIONS):
                    r = select(groupA, loc=loc, layout=lay, position=pos)
                    if(r):
                        grid[a, b] = r[0][metric]
            mesh = ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax)
            for a in range(len(LAYOUTS)):
                for b in range(len(POSITIONS)):
                    v = grid[a, b]
                    ax.text(b, a, "–" if np.isnan(v) else f"{v:.2f} m", ha="center", va="center", fontsize=10,
                            color=BG if (not np.isnan(v) and v > 0.6 * vmax) else INK)
            ax.set(title=f"{label} — {loc}", xticks=range(len(POSITIONS)), yticks=range(len(LAYOUTS)),
                   xticklabels=POSITIONS, yticklabels=[LAYOUT_LABEL[l] for l in LAYOUTS])
            ax.grid(False)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.8)
    cbar.set_label("Error [m] (same scale in every panel)", color=INK2)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Layout x position (aperture {REFERENCE['aperture']:g} m, SNR {REFERENCE['snr']} dB)", color=INK, x=0.02, ha="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sweep(runs: list[dict], locs: list[str], factor: str, xlabel: str, fixed: dict, out: Path) -> None:
    """Groups B and C: median error, P90 error and estimates per recording vs. one factor, one line per localizer."""
    panels = [("median", "Median error [m]"), ("p90", "P90 error [m]"), ("est_per_rec", "Estimates per recording")]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.2), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        for k, loc in enumerate(locs):
            sel = sorted(select(runs, loc=loc, **fixed), key=lambda r: r[factor])
            if(not sel):
                continue
            st = SERIES[k % len(SERIES)]
            ax.plot([r[factor] for r in sel], [r[metric] for r in sel], color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=2, markersize=8, markeredgecolor=BG, label=loc)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=ylabel.split(" [")[0], ylim=(0, None))
    if(any(len(select(runs, loc=l, **fixed)) for l in locs)):
        axes[0].legend(frameon=False)
    fixed_txt = ", ".join(f"{k} {v}" for k, v in fixed.items())
    fig.suptitle(f"Effect of {factor} ({fixed_txt})", color=INK, x=0.01, ha="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_reference_cdf(runs: list[dict], locs: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for k, loc in enumerate(locs):
        sel = select(runs, loc=loc, **REFERENCE)
        if(not sel or sel[0]["err"].size == 0):
            continue
        e = np.sort(sel[0]["err"])
        st = SERIES[k % len(SERIES)]
        ax.step(e, np.arange(1, e.size + 1) / e.size * 100, where="post", color=st["color"], ls=st["ls"], lw=2,
                label=f"{loc} (median {np.median(e):.2f} m)")
    ax.set(title="Error CDF on the reference configuration", xlabel="Error [m]",
           ylabel="Estimates below the error [%]", xlim=(0, None), ylim=(0, 100))
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_distance_by_layout(runs: list[dict], locs: list[str], out: Path) -> None:
    """Median error vs. distance from the array centre, one line per layout, one panel per localizer."""
    fixed = {"aperture": REFERENCE["aperture"], "position": REFERENCE["position"], "snr": REFERENCE["snr"]}
    fig, axes = plt.subplots(1, len(locs), figsize=(6.5 * len(locs), 4.5), constrained_layout=True, squeeze=False, sharey=True)
    for ax, loc in zip(axes[0], locs):
        for k, lay in enumerate(LAYOUTS):
            sel = select(runs, loc=loc, layout=lay, **fixed)
            if(not sel or sel[0]["err"].size == 0):
                continue
            x, y = binned_median(sel[0]["dist"], sel[0]["err"], bin_width=5.0)
            st = SERIES[k % len(SERIES)]
            ax.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=2, markersize=7,
                    markeredgecolor=BG, label=LAYOUT_LABEL[lay])
        ax.set(title=loc, xlabel="Distance from the array centre [m]", ylim=(0, None))
        ax.legend(frameon=False, loc="upper left")
    axes[0, 0].set_ylabel("Median error per 5 m bin [m]")
    fig.suptitle(f"Error vs. distance by layout (array at the centre, {REFERENCE['aperture']:g} m, {REFERENCE['snr']} dB)",
                 color=INK, x=0.01, ha="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    runs = load_runs(args.results, args.data)
    if(not runs):
        print(f"No campaign CSVs found in {args.results}")
        return
    locs = sorted({r["loc"] for r in runs})
    out_dir = args.results / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_summary(runs, out_dir / "summary.csv")
    plot_layout_position(runs, locs, out_dir / "layout_position.png")
    plot_sweep(runs, locs, "aperture", "Aperture [m]",
               {"layout": REFERENCE["layout"], "position": REFERENCE["position"], "snr": REFERENCE["snr"]},
               out_dir / "aperture.png")
    plot_sweep(runs, locs, "snr", "SNR [dB]",
               {"layout": REFERENCE["layout"], "aperture": REFERENCE["aperture"], "position": REFERENCE["position"]},
               out_dir / "snr.png")
    plot_reference_cdf(runs, locs, out_dir / "reference_cdf.png")
    plot_distance_by_layout(runs, locs, out_dir / "distance_by_layout.png")
    print(f"\nSummary and figures written to {out_dir}")


if(__name__ == "__main__"):
    main()
