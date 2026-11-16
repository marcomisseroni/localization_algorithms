"""
Plot the results written by evaluate.py.

Reads the CSV of the estimates and the dataset.json of the dataset (for the sensor positions) and saves in results/figures/:
    <csv name>_map.png        median error for every source position, with the sensors (cells if the positions are
                              a grid, points otherwise)
    <csv name>_map_cells.png  only if the positions are not a grid: the room divided in cells of CELL_SIZE m, median
                              error of all the estimates of the sources inside every cell

evaluate.py calls plot_all() automatically; to redo the figures from an existing CSV, from the project root:
    venv/bin/python -m analysis.plot_results results/y_7m_10db_bf_gcc_ls.csv data/test/y_7m_10db
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
MAP_VMAX = 20.0        # [m] fixed top of the error map colour scale, so maps of different runs are comparable
CELL_SIZE = 7.0        # [m] side of the cells of <csv name>_map_cells.png
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
    parser.add_argument("dataset_path", type=Path, help="dataset folder (with dataset.json) used for the CSV")
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


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, list | None]:
    """
    Sensor positions (N, 2) and room size [x, y, z] (None if the dataset is not simulated) from dataset.json.
    """
    with open(dataset_path / "dataset.json") as f:
        data = json.load(f)
    return np.array(data["sensor_pos"])[:, :2], data.get("simulation", {}).get("room")


def errors_per_position(rows: list[dict]) -> dict:
    """Errors of the estimates (None for a sample without estimates) of every source position."""
    per_pos = {}
    for r in rows:
        per_pos.setdefault(tuple(r["true"]), []).append(r["error"])
    return per_pos


def draw_cells(x_edges: np.ndarray, y_edges: np.ndarray, med: np.ndarray, missed: list, sensors: np.ndarray,
               title: str, out: Path) -> None:
    """
    Map of cells coloured by med (rows along y, columns along x), with the value written in every cell, the sensors
    and a cross on missed (points with sources but no estimate).
    """
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    vmax = MAP_VMAX
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, y_edges, med, cmap="Blues", vmin=0, vmax=vmax, edgecolors=BG, linewidth=2)
    for (x, y) in missed:
        ax.plot(x, y, marker="x", color=INK2, markersize=8, markeredgewidth=1.5)
    for i in range(yc.size):
        for j in range(xc.size):
            if(not np.isnan(med[i, j])):
                dark = med[i, j] > 0.6 * vmax
                ax.text(xc[j], yc[i], f"{med[i, j]:.2f}", ha="center", va="center", fontsize=7 if xc.size <= 12 else 5,
                        color=BG if dark else INK)
    # Only the sensor points: joining them in list order draws a wrong shape for non-polygonal layouts (e.g. Y)
    ax.plot(sensors[:, 0], sensors[:, 1], "o", color=INK, markersize=9, markeredgecolor=BG, markeredgewidth=2)
    ax.set(title=title, xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, extend="max")    # the arrow marks errors above vmax
    cbar.set_label(f"Median error [m] (fixed scale, darkest = {vmax:.1f} m or more)", color=INK2)
    cbar.outline.set_visible(False)
    ax.text(0, -0.08, "● sensors   × no estimate", transform=ax.transAxes, color=INK2, fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def is_grid(per_pos: dict) -> bool:
    """True if the source positions are a regular grid (one source position per grid node)."""
    xs = np.unique([p[0] for p in per_pos])
    ys = np.unique([p[1] for p in per_pos])
    return xs.size * ys.size == len(per_pos)


def plot_map(rows: list[dict], sensors: np.ndarray, out: Path) -> None:
    """
    Median error of every source position; positions with no estimate are crossed.
    Cells with the values if the positions are a regular grid, coloured points otherwise (e.g. random positions).
    """
    per_pos = errors_per_position(rows)
    if(not is_grid(per_pos)):
        plot_map_points(per_pos, sensors, out)
        return
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

    draw_cells(edges(xs), edges(ys), med, missed, sensors, "Median localization error per source position", out)


def plot_map_cells(rows: list[dict], sensors: np.ndarray, room: list | None, out: Path) -> None:
    """
    Room divided in square cells of CELL_SIZE m (from 0 to the room size, or to the largest source coordinate if the
    room is unknown): median error of all the estimates whose source is inside the cell. A cell with sources but no
    estimate is crossed, a cell without sources is empty.
    """
    pos = np.array([r["true"] for r in rows])
    size_x = room[0] if room else pos[:, 0].max()
    size_y = room[1] if room else pos[:, 1].max()
    x_edges = np.linspace(0, size_x, max(int(round(size_x / CELL_SIZE)), 1) + 1)
    y_edges = np.linspace(0, size_y, max(int(round(size_y / CELL_SIZE)), 1) + 1)
    # Cell of every row; sources on the last edge go in the last cell
    col = np.clip(np.searchsorted(x_edges, pos[:, 0], side="right") - 1, 0, x_edges.size - 2)
    row = np.clip(np.searchsorted(y_edges, pos[:, 1], side="right") - 1, 0, y_edges.size - 2)
    errs = {}
    for r, i, j in zip(rows, row, col):
        errs.setdefault((i, j), []).append(r["error"])
    med = np.full((y_edges.size - 1, x_edges.size - 1), np.nan)
    missed = []
    for (i, j), e in errs.items():
        e = [v for v in e if v is not None]
        if(e):
            med[i, j] = np.median(e)
        else:
            missed.append(((x_edges[j] + x_edges[j + 1]) / 2, (y_edges[i] + y_edges[i + 1]) / 2))
    draw_cells(x_edges, y_edges, med, missed, sensors,
               f"Median localization error per {x_edges[1] - x_edges[0]:g} x {y_edges[1] - y_edges[0]:g} m cell", out)


def plot_map_points(per_pos: dict, sensors: np.ndarray, out: Path) -> None:
    """Map of plot_map for source positions that are not a grid: one point per position, coloured by the median error."""
    pos = np.array([p for p, errs in per_pos.items() if any(e is not None for e in errs)])
    med = np.array([np.median([e for e in errs if e is not None]) for errs in per_pos.values()
                    if any(e is not None for e in errs)])
    missed = np.array([p for p, errs in per_pos.items() if all(e is None for e in errs)])

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    if(pos.size):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=med, cmap="Blues", vmin=0, vmax=MAP_VMAX, s=30,
                        edgecolors=INK2, linewidths=0.3)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85, extend="max")
        cbar.set_label(f"Median error [m] (fixed scale, darkest = {MAP_VMAX:.1f} m or more)", color=INK2)
        cbar.outline.set_visible(False)
    if(missed.size):
        ax.plot(missed[:, 0], missed[:, 1], "x", color=INK2, markersize=6, markeredgewidth=1.2)
    ax.plot(sensors[:, 0], sensors[:, 1], "o", color=INK, markersize=9, markeredgecolor=BG, markeredgewidth=2)
    ax.set(title="Median localization error per source position", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax.text(0, -0.08, "● sensors   × no estimate", transform=ax.transAxes, color=INK2, fontsize=9)
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
    sensors, room = load_dataset(dataset_path)
    out_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    written = [out_dir / f"{stem}_map.png"]
    plot_map(rows, sensors, written[0])
    if(not is_grid(errors_per_position(rows))):
        written.append(out_dir / f"{stem}_map_cells.png")
        plot_map_cells(rows, sensors, room, written[1])
    print("Figures written to " + ", ".join(str(p) for p in written))


def main() -> None:
    args = parse_arguments()
    plot_all(args.csv_path, args.dataset_path, args.out)


if(__name__ == "__main__"):
    main()
