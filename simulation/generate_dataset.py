"""
Generate a dataset with the acoustic array simulator (git submodule in ./simulator), without its GUI.

Every experiment is described by a TOML file in simulation/configs/. The output folder has the same
layout as a batch exported by the GUI (one JSON + one WAV per sensor for each source position, plus
manifest.json), so it can be read by core.data.Sample. A copy of the config is saved next to it.

Usage, from the project root:
    venv/bin/python simulation/generate_dataset.py simulation/configs/footstep_square.toml
"""

import argparse
import json
import shutil
import sys
import tomllib

import numpy as np

from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "simulator"))

from selkd_simulator.engine import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    circular_array,
    export_result,
    generate_source_positions,
    linear_array,
    load_mono_wav,
    make_test_signal,
    simulate,
    square_perimeter_array,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simulated dataset from a TOML experiment file")
    parser.add_argument("config", type=Path, help="experiment file, e.g. simulation/configs/footstep_square.toml")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: data/<config name>)")
    parser.add_argument("--overwrite", action="store_true", help="replace the output folder if it exists")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_sim_config(conf: dict) -> SimulationConfig:
    """Map the [simulation] section onto the simulator configuration."""
    sim = conf["simulation"]
    return SimulationConfig(
        sample_rate=sim["fs"],
        speed_of_sound=sim["c"],
        model=sim.get("model", "free_field"),
        room_dimensions=tuple(sim["room"]),
        wall_absorption=sim.get("wall_absorption", 0.35),
        reflection_order=sim.get("reflection_order", 2),
    )


def build_sensors(conf: dict) -> np.ndarray:
    """Sensor positions (N, 3) from the [sensors] section."""
    sen = conf["sensors"]
    layout = sen["layout"]
    if layout == "manual":
        return np.asarray(sen["positions"], dtype=float)
    center = tuple(sen["center"])
    count = sen["count"]
    if layout == "square":
        return square_perimeter_array(center, sen["side"] / 2, count)
    if layout == "circle":
        return circular_array(center, sen["diameter"] / 2, count)
    if layout == "linear":
        return linear_array(center, sen["spacing"], count)
    raise ValueError(f"Unknown sensor layout: {layout}")


def build_sources(conf: dict, room: tuple[float, float, float]) -> np.ndarray:
    """
    Source positions (n_positions, 3) from the [source] section.
    With 'grid_x' / 'grid_y' = [start, stop, step] the positions are the nodes of a regular grid (stop included).
    With 'area' = [[xmin, xmax], [ymin, ymax]] the positions are uniform in that rectangle,
    otherwise in the whole room (minus a margin from the walls).
    """
    src = conf["source"]
    if "grid_x" in src:
        xs = np.arange(src["grid_x"][0], src["grid_x"][1] + src["grid_x"][2] / 2, src["grid_x"][2])
        ys = np.arange(src["grid_y"][0], src["grid_y"][1] + src["grid_y"][2] / 2, src["grid_y"][2])
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, src["z"])))
    n = src["n_positions"]
    seed = src.get("seed", 1)
    if "position" in src:
        return generate_source_positions(np.asarray(src["position"], dtype=float), room, n, randomize_xy=False)
    if "area" in src:
        (x0, x1), (y0, y1) = src["area"]
        rng = np.random.default_rng(seed)
        return np.column_stack((rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), np.full(n, src["z"])))
    base = np.array([room[0] / 2, room[1] / 2, src["z"]])
    return generate_source_positions(base, room, n, randomize_xy=True, seed=seed)


def load_signal(conf: dict, fs: int) -> tuple[np.ndarray, str]:
    """Source waveform resampled to fs, and its name. 'chirp' uses the simulator test chirp."""
    name = conf["source"]["signal"]
    if name == "chirp":
        return make_test_signal(fs), "test_chirp"
    path = Path(name) if Path(name).is_absolute() else ROOT / name
    return load_mono_wav(path, fs), path.stem


def add_noise(result: SimulationResult, snr_db: float, rng: np.random.Generator) -> SimulationResult:
    """
    Add white Gaussian noise, independent on each sensor.
    The SNR is referred to the mean power of the strongest sensor over its active samples
    (|x| above 1% of its peak), so that the silence around the event does not lower it.
    """
    x = result.signals                                      # (T, N)
    powers = []
    for ch in range(x.shape[1]):
        active = np.abs(x[:, ch]) > 0.01 * np.max(np.abs(x[:, ch]))
        powers.append(np.mean(x[active, ch] ** 2))
    noise_std = np.sqrt(max(powers) / 10 ** (snr_db / 10))
    noisy = x + noise_std * rng.standard_normal(x.shape)
    peak = np.max(np.abs(noisy))
    if peak > 0.99:                                         # keep it inside the int16 range
        noisy *= 0.99 / peak
    return replace(result, signals=noisy)


def main() -> None:
    args = parse_arguments()
    conf = load_config(args.config)
    out_dir = args.out or ROOT / "data" / args.config.stem
    if out_dir.exists():
        if not args.overwrite:
            sys.exit(f"{out_dir} already exists, use --overwrite to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    sim_config = build_sim_config(conf)
    sensors = build_sensors(conf)
    sources = build_sources(conf, sim_config.room_dimensions)
    signal, signal_name = load_signal(conf, sim_config.sample_rate)
    snr_db = conf.get("noise", {}).get("snr_db")
    noise_rng = np.random.default_rng(conf["source"].get("seed", 1) + 1)

    manifest = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "experiment": args.config.stem,
        "acoustic_model": sim_config.model,
        "sample_rate_hz": sim_config.sample_rate,
        "speed_of_sound": sim_config.speed_of_sound,
        "input_files": [conf["source"]["signal"]],
        "snr_db": snr_db,
        "microphones_xyz_m": sensors.tolist(),
        "source_positions_xyz_m": sources.tolist(),
        "samples": [],
    }

    for i, position in enumerate(sources):
        result = simulate(signal, sensors, position, sim_config)
        if snr_db is not None:
            result = add_noise(result, snr_db, noise_rng)
        stem = f"rep{i + 1:04d}_file001_{signal_name}"
        wav_paths, json_path = export_result(
            result, sim_config, sensors, position, out_dir, stem=stem, input_audio=conf["source"]["signal"]
        )
        manifest["samples"].append(
            {
                "sample_index": i + 1,
                "source_xyz_m": position.tolist(),
                "wav_files": [p.name for p in wav_paths],
                "metadata_file": json_path.name,
            }
        )
        print(f"[{i + 1}/{len(sources)}] source ({position[0]:.2f}, {position[1]:.2f}) -> {json_path.name}")

    manifest["status"] = "complete"
    manifest["generated_samples"] = len(sources)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy(args.config, out_dir / "config.toml")
    print(f"Dataset written to {out_dir}")


if __name__ == "__main__":
    main()
