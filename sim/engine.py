"""
Propagation engine of the simulator.

Taken from selkd_simulator/engine.py of https://github.com/jjavierk/acustic-array-simulator.git
(commit 58592714ea34aa0ed022552dbe317572307c487c). Changes w.r.t. the original:
- the signals are not normalized: the amplitudes are the physical ones (1 / distance for every path),
  so they can be compared between recordings and datasets;
- removed what is not used by sim/simulate.py: array layouts, random source positions, test chirp,
  export to WAV/JSON (now in sim/simulate.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, gcd
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly


Model = Literal["free_field", "shoebox"]


@dataclass(frozen=True)
class SimulationConfig:
    sample_rate: int = 48_000
    speed_of_sound: float = 343.42
    model: Model = "free_field"
    room_dimensions: tuple[float, float, float] = (4.0, 3.0, 2.5)
    wall_absorption: float = 0.35
    reflection_order: int = 2
    fractional_filter_half_length: int = 24

    def validate(self) -> None:
        if self.sample_rate < 8_000:
            raise ValueError("Sample rate must be at least 8000 Hz")
        if self.speed_of_sound <= 0:
            raise ValueError("Speed of sound must be positive")
        if self.model not in ("free_field", "shoebox"):
            raise ValueError(f"Unknown acoustic model: {self.model}")
        if any(value <= 0 for value in self.room_dimensions):
            raise ValueError("All room dimensions must be positive")
        if not 0.0 <= self.wall_absorption <= 1.0:
            raise ValueError("Wall absorption must lie between 0 and 1")
        if not 0 <= self.reflection_order <= 8:
            raise ValueError("Reflection order must lie between 0 and 8")
        if self.fractional_filter_half_length < 4:
            raise ValueError("Fractional-delay half length must be at least 4")


@dataclass
class SimulationResult:
    signals: np.ndarray                 # (L, N) physical amplitudes
    impulse_responses: np.ndarray       # (N, rir_length)
    direct_distances_m: np.ndarray      # (N,)
    direct_arrival_seconds: np.ndarray  # (N,)
    relative_arrival_seconds: np.ndarray
    common_filter_latency_samples: int
    sample_rate: int


def load_mono_wav(path: str | Path, target_sample_rate: int) -> np.ndarray:
    source_rate, audio = wavfile.read(str(path))
    original_dtype = audio.dtype
    if audio.ndim == 2:
        audio = np.mean(_as_float(audio, original_dtype), axis=1)
    else:
        audio = _as_float(audio, original_dtype)
    if source_rate != target_sample_rate:
        divisor = gcd(int(source_rate), int(target_sample_rate))
        audio = resample_poly(audio, target_sample_rate // divisor, source_rate // divisor)
    return np.ascontiguousarray(audio, dtype=np.float64)


def _as_float(audio: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return audio.astype(np.float64)
    if dtype == np.uint8:
        return (audio.astype(np.float64) - 128.0) / 128.0
    info = np.iinfo(dtype)
    scale = float(max(abs(info.min), info.max))
    return audio.astype(np.float64) / scale


def _axis_images(source: float, length: float, max_order: int) -> list[tuple[float, int]]:
    extent = ceil(max_order / 2) + 1
    images: dict[tuple[int, int], tuple[float, int]] = {}
    for n in range(-extent, extent + 1):
        candidates = (
            (2.0 * n * length + source, 2 * abs(n)),
            (2.0 * n * length - source, abs(2 * n - 1)),
        )
        for family, (position, order) in enumerate(candidates):
            if order <= max_order:
                images[(n, family)] = (position, order)
    return list(images.values())


def _image_sources(
    source_xyz: np.ndarray, room_dimensions: tuple[float, float, float], max_order: int
) -> list[tuple[np.ndarray, int]]:
    axes = [
        _axis_images(float(source_xyz[axis]), room_dimensions[axis], max_order)
        for axis in range(3)
    ]
    images: list[tuple[np.ndarray, int]] = []
    for x_image, y_image, z_image in product(*axes):
        order = x_image[1] + y_image[1] + z_image[1]
        if order <= max_order:
            images.append(
                (np.array((x_image[0], y_image[0], z_image[0]), dtype=np.float64), order)
            )
    return images


def _fractional_kernel(offsets: np.ndarray, half_length: int) -> np.ndarray:
    normalized = offsets / half_length
    window = np.where(np.abs(normalized) < 1.0, 0.5 * (1.0 + np.cos(np.pi * normalized)), 0.0)
    kernel = np.sinc(offsets) * window
    total = np.sum(kernel)
    if abs(total) > 1e-12:
        kernel /= total
    return kernel


def _build_rirs(
    microphone_xyz: np.ndarray,
    source_xyz: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct_distances = np.linalg.norm(microphone_xyz - source_xyz, axis=1)
    if np.any(direct_distances < 0.01):
        raise ValueError("The source must be at least 1 cm from every microphone")

    if config.model == "free_field":
        paths = [(source_xyz, 0)]
    else:
        paths = _image_sources(source_xyz, config.room_dimensions, config.reflection_order)

    reflection_coefficient = np.sqrt(1.0 - config.wall_absorption)
    half = config.fractional_filter_half_length
    path_data: list[list[tuple[float, float]]] = []
    maximum_delay = 0.0
    for microphone in microphone_xyz:
        microphone_paths: list[tuple[float, float]] = []
        for image_position, order in paths:
            distance = float(np.linalg.norm(microphone - image_position))
            if distance < 0.01:
                continue
            delay_samples = distance / config.speed_of_sound * config.sample_rate + half
            amplitude = (reflection_coefficient**order) / distance
            microphone_paths.append((delay_samples, amplitude))
            maximum_delay = max(maximum_delay, delay_samples)
        path_data.append(microphone_paths)

    rir_length = int(np.ceil(maximum_delay)) + half + 2
    rirs = np.zeros((len(microphone_xyz), rir_length), dtype=np.float64)
    for mic_index, microphone_paths in enumerate(path_data):
        for delay_samples, amplitude in microphone_paths:
            left = int(np.floor(delay_samples)) - half + 1
            indices = np.arange(left, left + 2 * half, dtype=np.int64)
            valid = (indices >= 0) & (indices < rir_length)
            offsets = indices[valid].astype(np.float64) - delay_samples
            rirs[mic_index, indices[valid]] += amplitude * _fractional_kernel(offsets, half)

    direct_arrivals = direct_distances / config.speed_of_sound
    return rirs, direct_distances, direct_arrivals


def simulate(
    source_signal: np.ndarray,
    microphone_xyz: np.ndarray,
    source_xyz: np.ndarray,
    config: SimulationConfig,
) -> SimulationResult:
    config.validate()
    microphones = np.asarray(microphone_xyz, dtype=np.float64)
    source = np.asarray(source_xyz, dtype=np.float64)
    audio = np.asarray(source_signal, dtype=np.float64)
    if microphones.ndim != 2 or microphones.shape[1] != 3 or microphones.shape[0] < 1:
        raise ValueError("At least one microphone XYZ position is required")
    if source.shape != (3,):
        raise ValueError("Source position must contain X, Y, and Z")
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Source signal must be a non-empty mono array")
    if not np.all(np.isfinite(microphones)) or not np.all(np.isfinite(source)):
        raise ValueError("Geometry contains a non-finite value")
    if config.model == "shoebox":
        room = np.asarray(config.room_dimensions)
        if np.any(source <= 0) or np.any(source >= room):
            raise ValueError("The source must lie strictly inside the room")
        if np.any(microphones <= 0) or np.any(microphones >= room):
            raise ValueError("Every microphone must lie strictly inside the room")

    rirs, distances, arrivals = _build_rirs(microphones, source, config)
    # No normalization: physical amplitudes
    recordings = np.vstack([fftconvolve(audio, rir, mode="full") for rir in rirs]).T
    relative = arrivals - np.min(arrivals)
    return SimulationResult(
        signals=recordings,
        impulse_responses=rirs,
        direct_distances_m=distances,
        direct_arrival_seconds=arrivals,
        relative_arrival_seconds=relative,
        common_filter_latency_samples=config.fractional_filter_half_length,
        sample_rate=config.sample_rate,
    )
