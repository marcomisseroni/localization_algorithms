"""
Generate synthetic footstep-like vibration signals to be used as input of the simulator.
Each step is a heel impact followed by a toe impact, both modelled as band-limited noise
with an exponential decay. The result is impulsive and broadband in the low frequencies.
"""

import numpy as np

from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

FS = 16000              # Simulator sample rate [Hz]
BAND = (20, 400)        # Band of the impacts [Hz]
OUT_DIR = Path(__file__).parent


def impact(rng: np.random.Generator, duration: float, decay: float, attack: float = 0.003) -> np.ndarray:
    """Band-limited noise burst with a short attack and exponential decay."""
    n = int(duration * FS)
    t = np.arange(n) / FS
    sos = butter(4, BAND, btype="bandpass", fs=FS, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n))
    # A smooth attack avoids a step at the onset, which would add energy far above BAND
    return x * (1 - np.exp(-t / attack)) * np.exp(-t / decay)


def footstep(rng: np.random.Generator) -> np.ndarray:
    """Heel strike and, ~80 ms later, a weaker toe strike."""
    heel = impact(rng, 0.25, 0.030)
    toe = 0.6 * impact(rng, 0.25, 0.025)
    step = np.zeros(int(0.35 * FS))
    step[:heel.size] += heel
    k = int(0.08 * FS)
    step[k:k + toe.size] += toe
    return step


def save(name: str, x: np.ndarray) -> None:
    x = x / np.max(np.abs(x)) * 10 ** (-1 / 20)     # peak at -1 dBFS
    wavfile.write(OUT_DIR / name, FS, (x * 32767).astype(np.int16))
    print(f"{name}: {x.size / FS:.2f} s")


def main() -> None:
    rng = np.random.default_rng(0)

    # Single step with 0.5 s of silence before and 1 s after
    single = np.zeros(int(1.85 * FS))
    s = footstep(rng)
    k = int(0.5 * FS)
    single[k:k + s.size] = s
    save("footstep_single.wav", single)

    # Walk: 4 steps every 0.55 s with slightly different amplitudes
    walk = np.zeros(int(3.5 * FS))
    for i in range(4):
        s = footstep(rng) * rng.uniform(0.7, 1.0)
        k = int((0.5 + 0.55 * i) * FS)
        walk[k:k + s.size] += s
    save("footstep_walk.wav", walk)


if(__name__ == "__main__"):
    main()
