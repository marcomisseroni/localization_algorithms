"""
Plot the results written by evaluate.py.

Reads the CSV of the estimates and the manifest of the dataset (for the sensor positions) and saves in results/figures/:
    <csv name>_map.png        median error for every source position, with the sensors
    <csv name>_distance.png   error of every estimate vs. distance from the centre of the array
    <csv name>_cdf.png        cumulative distribution of the error

evaluate.py calls plot_all() automatically; to redo the figures from an existing CSV, from the project root:
    venv/bin/python -m analysis.plot_results results/y_7m_center_snr20_bf_gcc_ls.csv data/y_7m_center_snr20
"""

import argparse
import csv
import json

import matplotlib
import numpy as np

from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2a78d6"        # estimates of complete steps
GREY = "#8f8e89"        # estimates of the first step of each recording
MAP_VMAX = 5.0         # [m] fixed top of the error map colour scale, so maps of different runs are comparable
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BG = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the localization results of evaluate.py")
    parser.add_argument("csv_path", type=Path, help="CSV written by evaluate.py")
    parser.add_argument("dataset_path", type=Path, help="dataset folder (with manifest.json) used for the CSV")
    parser.add_argument("--out", type=Path, default=Path("results"), help="results folder; the PNGs are saved in its figures/ subfolder")
    return parser.parse_args()


def load_results(path: Path) -> list[dict]:
    """
    One dict per CSV row. 'error' is None for samples without estimates.
    'step' is the index of the estimate inside its sample (0 = first step of the recording).
    """
    rows = []
    count = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            has_est = r["error_m"] != ""
            step = count.get(r["sample"], 0)
            if(has_est):
                count[r["sample"]] = step + 1
            rows.append({
                "sample": r["sample"],
                "true": np.array([float(r["true_x"]), float(r["true_y"])]),
                "est": np.array([float(r["est_x"]), float(r["est_y"])]) if has_est else None,
                "error": float(r["error_m"]) if has_est else None,
                "step": step if has_est else None,
            })
    return rows


def load_sensors(dataset_path: Path) -> np.ndarray:
    with open(dataset_path / "manifest.json") as f:
        manifest = json.load(f)
    if("microphones_xyz_m" in manifest):
        return np.array(manifest["microphones_xyz_m"])[:, :2]
    # Datasets exported by the simulator GUI: geometry only in the per-sample JSON
    with open(dataset_path / manifest["samples"][0]["metadata_file"]) as f:
        return np.array(json.load(f)["microphones_xyz_m"])[:, :2]


def plot_map(rows: list[dict], sensors: np.ndarray, out: Path) -> None:
    """Median error of every source position on the grid; positions with no estimate are crossed."""
    per_pos = {}
    for r in rows:
        per_pos.setdefault(tuple(r["true"]), []).append(r["error"])
    xs = np.unique([p[0] for p in per_pos])
    ys = np.unique([p[1] for p in per_pos])
    med = np.full((ys.size, xs.size), np.nan)
    missed = []
    for (x, y), errs in per_pos.items():
        errs = [e for e in errs if e is not None]
        if(errs):
            med[np.searchsorted(ys, y), np.searchsorted(xs, x)] = np.median(errs)
        else:
            missed.append((x, y))

    # Cell edges halfway between the grid nodes
    def edges(v: np.ndarray) -> np.ndarray:
        step = np.diff(v).mean() if v.size > 1 else 1.0
        return np.concatenate(([v[0] - step / 2], v + step / 2))

    vmax = MAP_VMAX
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    mesh = ax.pcolormesh(edges(xs), edges(ys), med, cmap="Blues", vmin=0, vmax=vmax,
                         edgecolors=BG, linewidth=2)
    for (x, y) in missed:
        ax.plot(x, y, marker="x", color=INK2, markersize=8, markeredgewidth=1.5)
    for i in range(ys.size):
        for j in range(xs.size):
            if(not np.isnan(med[i, j])):
                dark = med[i, j] > 0.6 * vmax
                ax.text(xs[j], ys[i], f"{med[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=BG if dark else INK)
    ax.plot(*np.vstack((sensors, sensors[:1])).T, color=INK, lw=1, ls="--")
    ax.plot(sensors[:, 0], sensors[:, 1], "o", color=INK, markersize=9, markeredgecolor=BG, markeredgewidth=2)
    ax.set(title="Median localization error per source position", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, extend="max")    # the arrow marks errors above vmax
    cbar.set_label(f"Median error [m] (fixed scale, darkest = {vmax:.1f} m or more)", color=INK2)
    cbar.outline.set_visible(False)
    ax.text(0, -0.08, "● sensors   × no estimate", transform=ax.transAxes, color=INK2, fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_distance(rows: list[dict], sensors: np.ndarray, out: Path) -> None:
    """Error of every estimate vs. distance of the source from the centre of the array."""
    centre = sensors.mean(axis=0)
    est = [r for r in rows if r["error"] is not None]
    d = np.array([np.linalg.norm(r["true"] - centre) for r in est])
    e = np.array([r["error"] for r in est])
    first = np.array([r["step"] == 0 for r in est])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(d[~first], e[~first], s=22, color=BLUE, alpha=0.75, edgecolors=BG, linewidths=0.6,
               label="other steps")
    ax.scatter(d[first], e[first], s=22, color=GREY, alpha=0.75, marker="s", edgecolors=BG, linewidths=0.6,
               label="first step of the recording")
    # Median of all the estimates in 1 m distance bins
    bins = np.arange(0, np.ceil(d.max()) + 1, 1.0)
    centres, medians = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (d >= lo) & (d < hi)
        if(k.any()):
            centres.append((lo + hi) / 2)
            medians.append(np.median(e[k]))
    ax.plot(centres, medians, color=INK, lw=2, marker="o", markersize=5, label="median per 1 m bin")
    ax.set(title="Localization error vs. distance from the array centre",
           xlabel="Distance from the array centre [m]", ylabel="Error [m]", ylim=(0, None))
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cdf(rows: list[dict], out: Path) -> None:
    """Fraction of the estimates with an error below each value."""
    e = np.sort([r["error"] for r in rows if r["error"] is not None])
    frac = np.arange(1, e.size + 1) / e.size * 100
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.step(e, frac, where="post", color=BLUE, lw=2)
    for p, y in ((50, 12), (90, 4)):        # labels at different heights so they never overlap
        v = np.percentile(e, p)
        ax.axvline(v, color=INK2, lw=0.8, ls="--")
        ax.text(v, y, f" P{p} = {v:.2f} m", color=INK2, fontsize=9)
    ax.set(title="Cumulative distribution of the localization error",
           xlabel="Error [m]", ylabel="Estimates below the error [%]", xlim=(0, None), ylim=(0, 100))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_all(csv_path: Path, dataset_path: Path, out_dir: Path) -> None:
    """
    Make every figure for one CSV and save them in the figures/ subfolder of out_dir.
    Called by evaluate.py after writing the results.
    """
    rows = load_results(csv_path)
    if(all(r["error"] is None for r in rows)):
        print("No estimates: figures not generated")
        return
    sensors = load_sensors(dataset_path)
    out_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    plot_map(rows, sensors, out_dir / f"{stem}_map.png")
    plot_distance(rows, sensors, out_dir / f"{stem}_distance.png")
    plot_cdf(rows, out_dir / f"{stem}_cdf.png")
    print(f"Figures written to {out_dir}/{stem}_{{map,distance,cdf}}.png")


def main() -> None:
    args = parse_arguments()
    plot_all(args.csv_path, args.dataset_path, args.out)


if(__name__ == "__main__"):
    main()
