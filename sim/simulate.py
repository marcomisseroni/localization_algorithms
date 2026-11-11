import numpy as np
import shutil
import json

from pathlib import Path
from scipy.io import wavfile
from sim.engine import SimulationConfig, load_mono_wav
from sim.engine import simulate as simulate_recording

ROOT = Path(__file__).parent.parent
DEFAULT_SIGNAL = ROOT / "sim" / "test_signals" / "footstep_walk.wav"
FORMAT_VERSION = 2
SIGNAL_RNG_KEY = 1      # second word of the seed of the generator of the signal choice, separate from the noise one

def simulate(out_dir: str | Path, sensor_pos: np.ndarray, source_pos: np.ndarray, fs: int, snr_db: float | np.ndarray | None,
             signal: str | Path = DEFAULT_SIGNAL, signal_pool: str | Path | None = None, c: float = 343.42,
             model: str = "free_field", room: tuple[float, float, float] = (50.0, 50.0, 3.0),
             wall_absorption: float = 0.35, reflection_order: int = 2, seed: int = 0, overwrite: bool = False) -> Path:
    """
    Simulate one recording for every source position and write the dataset in out_dir:
        dataset.json    common information (fs, sensor positions, simulation parameters) and the list of the recordings,
                        each with the source signal it was simulated with
        recXXXX.wav     one float32 WAV per recording, channel i = sensor i of sensor_pos, physical amplitudes
    sensor_pos: (N, 3) [m]
    source_pos: (R, 3) [m], one recording for every row ((3,) for a single recording)
    snr_db: SNR of every recording, a number for all of them, (R,) for one value per recording, None for no noise.
    The white noise is independent on every sensor, the SNR is referred to the power of the strongest sensor
    during the event. seed sets the noise.
    signal: source WAV of every recording, used if signal_pool is None.
    signal_pool: folder of source WAVs; if given, every recording uses one of them chosen at random (uniform, with
    repetition). The choice depends only on seed and on the sorted file names, and uses a generator separate from the
    noise, so without signal_pool the datasets are the same as before its introduction.
    model: "free_field" or "shoebox" (room, wall_absorption and reflection_order are used only by "shoebox",
    except room that is saved in any case).
    Returns out_dir.
    """
    out_dir = Path(out_dir)
    sensor_pos = np.asarray(sensor_pos, dtype=float)
    source_pos = np.atleast_2d(np.asarray(source_pos, dtype=float))
    R = source_pos.shape[0]
    if(sensor_pos.ndim != 2 or sensor_pos.shape[1] != 3):
        raise ValueError(f"sensor_pos must be (N, 3), got {sensor_pos.shape}")
    if(source_pos.shape[1] != 3):
        raise ValueError(f"source_pos must be (R, 3), got {source_pos.shape}")
    if(snr_db is None):
        snr = [None] * R
    else:
        snr = np.broadcast_to(np.asarray(snr_db, dtype=float), (R,))
    if(out_dir.exists() and any(out_dir.iterdir())):
        if(not overwrite):
            raise FileExistsError(f"{out_dir} is not empty, use overwrite=True to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(sample_rate=fs, speed_of_sound=c, model=model, room_dimensions=tuple(room),
                              wall_absorption=wall_absorption, reflection_order=reflection_order)
    rec_signals = choose_signals(signal, signal_pool, R, seed)
    audio_cache = {}        # path -> audio, every signal of the pool is read once
    rng = np.random.default_rng(seed)

    recordings = []
    for r in range(R):
        print(f"Recording {r + 1}/{R}")
        if(rec_signals[r] not in audio_cache):
            audio_cache[rec_signals[r]] = load_mono_wav(rec_signals[r], fs)
        audio = audio_cache[rec_signals[r]]
        x = simulate_recording(audio, sensor_pos, source_pos[r], config).signals       # (L, N)
        if(snr[r] is not None):
            x = add_noise(x, snr[r], rng)
        file_name = f"rec{r:04d}.wav"
        wavfile.write(out_dir / file_name, fs, x.astype(np.float32))
        recordings.append({
            "file": file_name,
            "source_pos": source_pos[r].tolist(),
            "snr_db": None if snr[r] is None else float(snr[r]),
            "n_samples": x.shape[0],
            "signal": signal_name(rec_signals[r]),
        })

    # Written at the end, so that a folder without dataset.json is an interrupted simulation
    dataset = {
        "format_version": FORMAT_VERSION,
        "origin": "simulated",
        "fs": fs,
        "sensor_pos": sensor_pos.tolist(),
        "simulation": {
            "c": c,
            "model": model,
            "room": list(room),
            "wall_absorption": wall_absorption,
            "reflection_order": reflection_order,
            "signal_pool": None if signal_pool is None else signal_name(signal_pool),
            "seed": seed,
        },
        "recordings": recordings,
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)
    return out_dir

def choose_signals(signal: str | Path, signal_pool: str | Path | None, R: int, seed: int) -> list[Path]:
    """
    Source signal of every recording: signal for all of them if signal_pool is None, otherwise one WAV of the folder
    signal_pool per recording, chosen uniformly with repetition. The files are sorted by name, so that the choice
    with the same seed does not depend on the order of the filesystem.
    """
    if(signal_pool is None):
        return [Path(signal)] * R
    pool = sorted(Path(signal_pool).glob("*.wav"))
    if(len(pool) == 0):
        raise FileNotFoundError(f"no WAV file in the signal pool {signal_pool}")
    rng = np.random.default_rng([seed, SIGNAL_RNG_KEY])
    idx = rng.integers(len(pool), size=R)       # (R,)
    return [pool[i] for i in idx]

def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add white Gaussian noise, independent on every sensor.
    The SNR is referred to the mean power of the strongest sensor over its active samples
    (|x| above 1% of its peak), so that the silence around the event does not lower it.
    """
    # x: (L, N)
    powers = []
    for ch in range(x.shape[1]):
        active = np.abs(x[:, ch]) > 0.01 * np.max(np.abs(x[:, ch]))
        powers.append(np.mean(x[active, ch]**2))
    noise_std = np.sqrt(max(powers) / 10**(snr_db / 10))
    return x + noise_std * rng.standard_normal(x.shape)

def signal_name(signal: str | Path) -> str:
    """
    Path of the source signal relative to the project root if it is inside it, absolute otherwise.
    """
    path = Path(signal).resolve()
    if(path.is_relative_to(ROOT)):
        return str(path.relative_to(ROOT))
    return str(path)
