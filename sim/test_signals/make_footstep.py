"""
Generate synthetic footstep-like vibration signals to be used as input of the simulator.
Each step is a heel impact followed by a toe impact, both modelled as band-limited noise
with an exponential decay. The result is impulsive and broadband in the low frequencies.
Besides the two fixed signals (footstep_single.wav, footstep_walk.wav) it generates a pool of random walks in pool/,
used by sim.simulate through signal_pool: different number of steps, step times and step shapes.
"""

import json
import numpy as np

from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

FS = 16000              # Simulator sample rate [Hz]
BAND = (20, 400)        # Band of the impacts [Hz]
OUT_DIR = Path(__file__).parent
STEP_LEN = 0.35         # [s] buffer of one step: heel + toe, the toe impact ends within it

# Pool of random walks
POOL_DIR = OUT_DIR / "pool"
POOL_SIZE = 100
POOL_SEED = 1
N_STEPS = (1, 6)                # number of steps, both included
STEP_INTERVAL = (0.4, 0.8)      # [s] time between the onsets of two consecutive steps
FIRST_STEP = (0.3, 1.0)         # [s] onset of the first step: noise before it for the detector
TAIL = (0.3, 1.0)               # [s] silence after the end of the last step
MIN_DURATION = 2.0              # [s] at least one window of the network (1.5 s) in every recording
STEP_AMP = (0.7, 1.0)           # amplitude of every step
HEEL_DECAY = (0.020, 0.040)     # [s] decay constant of the heel impact
TOE_DECAY = (0.015, 0.035)      # [s] decay constant of the toe impact
TOE_DELAY = (0.06, 0.10)        # [s] delay of the toe impact after the heel
TOE_AMP = (0.4, 0.8)            # amplitude of the toe impact relative to the heel
BAND_LOW = (15, 30)             # [Hz] lower edge of the band of the impacts
BAND_HIGH = (300, 450)          # [Hz] upper edge of the band of the impacts


def impact(rng: np.random.Generator, duration: float, decay: float, attack: float = 0.003,
           band: tuple[float, float] = BAND) -> np.ndarray:
    """Band-limited noise burst with a short attack and exponential decay."""
    n = int(duration * FS)
    t = np.arange(n) / FS
    sos = butter(4, band, btype="bandpass", fs=FS, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n))
    # A smooth attack avoids a step at the onset, which would add energy far above BAND
    return x * (1 - np.exp(-t / attack)) * np.exp(-t / decay)


def footstep(rng: np.random.Generator, heel_decay: float = 0.030, toe_decay: float = 0.025, toe_delay: float = 0.08,
             toe_amp: float = 0.6, band: tuple[float, float] = BAND) -> np.ndarray:
    """Heel strike and, toe_delay later, a weaker toe strike. The defaults give the fixed signals."""
    heel = impact(rng, 0.25, heel_decay, band=band)
    toe = toe_amp * impact(rng, 0.25, toe_decay, band=band)
    step = np.zeros(int(STEP_LEN * FS))
    step[:heel.size] += heel
    k = int(toe_delay * FS)
    step[k:k + toe.size] += toe
    return step


def random_walk(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    One walk of the pool: random number of steps, step times and shape of every step (see the constants).
    Returns the signal and its parameters (onsets and shape of every step), written in pool/pool.json.
    """
    n_steps = int(rng.integers(N_STEPS[0], N_STEPS[1] + 1))
    onsets = [rng.uniform(*FIRST_STEP)]
    for _ in range(n_steps - 1):
        onsets.append(onsets[-1] + rng.uniform(*STEP_INTERVAL))
    duration = max(onsets[-1] + STEP_LEN + rng.uniform(*TAIL), MIN_DURATION)
    x = np.zeros(int(duration * FS))
    steps = []
    for onset in onsets:
        params = {
            "onset_s": onset,
            "amp": rng.uniform(*STEP_AMP),
            "heel_decay": rng.uniform(*HEEL_DECAY),
            "toe_decay": rng.uniform(*TOE_DECAY),
            "toe_delay": rng.uniform(*TOE_DELAY),
            "toe_amp": rng.uniform(*TOE_AMP),
            "band": [rng.uniform(*BAND_LOW), rng.uniform(*BAND_HIGH)],
        }
        s = params["amp"] * footstep(rng, params["heel_decay"], params["toe_decay"], params["toe_delay"],
                                     params["toe_amp"], tuple(params["band"]))
        k = int(onset * FS)
        x[k:k + s.size] += s
        steps.append(params)
    return x, {"n_steps": n_steps, "duration_s": x.size / FS, "steps": steps}


def save(path: Path, x: np.ndarray, verbose: bool = True) -> None:
    x = x / np.max(np.abs(x)) * 10 ** (-1 / 20)     # peak at -1 dBFS
    wavfile.write(path, FS, (x * 32767).astype(np.int16))
    if(verbose):
        print(f"{path.name}: {x.size / FS:.2f} s")


def generate_pool() -> None:
    """
    POOL_SIZE random walks in POOL_DIR (step_000.wav, ...) and their parameters in POOL_DIR/pool.json.
    The pool is replaced at every run; with the same POOL_SEED it is the same.
    """
    POOL_DIR.mkdir(exist_ok=True)
    for old in POOL_DIR.glob("*.wav"):
        old.unlink()
    rng = np.random.default_rng(POOL_SEED)
    info = []
    for i in range(POOL_SIZE):
        x, params = random_walk(rng)
        name = f"step_{i:03d}.wav"
        save(POOL_DIR / name, x, verbose=False)
        info.append({"file": name, **params})
    with open(POOL_DIR / "pool.json", "w") as f:
        json.dump({"fs": FS, "seed": POOL_SEED, "signals": info}, f, indent=4)
    n = [s["n_steps"] for s in info]
    d = [s["duration_s"] for s in info]
    print(f"pool: {POOL_SIZE} signals in {POOL_DIR}, steps {min(n)}-{max(n)}, duration {min(d):.2f}-{max(d):.2f} s")


def main() -> None:
    rng = np.random.default_rng(0)

    # Single step with 0.5 s of silence before and 1 s after
    single = np.zeros(int(1.85 * FS))
    s = footstep(rng)
    k = int(0.5 * FS)
    single[k:k + s.size] = s
    save(OUT_DIR / "footstep_single.wav", single)

    # Walk: 4 steps every 0.55 s with slightly different amplitudes
    walk = np.zeros(int(3.5 * FS))
    for i in range(4):
        s = footstep(rng) * rng.uniform(0.7, 1.0)
        k = int((0.5 + 0.55 * i) * FS)
        walk[k:k + s.size] += s
    save(OUT_DIR / "footstep_walk.wav", walk)

    generate_pool()


if(__name__ == "__main__"):
    main()
