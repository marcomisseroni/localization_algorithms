# Footstep detection and localization

Detection and localization algorithms for an array of sensors, developed for a master thesis.
The goal is to localize people walking from the floor vibrations produced by their footsteps.
During development, the algorithms are tested on data from the acoustic array simulator
(`../acustic-array-simulator`), fed to them block by block as if it were arriving in real time.

## Project structure

```
.
├── configs/
│   └── default.toml            # parameters of every algorithm + stream settings
├── core/                       # code shared by detection and localization
│   ├── data.py                 # Sample: loads a simulator recording, streams it in blocks
│   └── metrics.py              # (empty for now) shared evaluation metrics
├── detection/
│   ├── base.py                 # Detector: abstract base class of every detector
│   └── amp_thr.py              # AmpThr: amplitude-threshold detector (work in progress)
├── localization/
│   ├── base.py                 # Localizer: abstract base class of every localizer
│   └── beamforming_gcc_ls.py   # BeamformingGccLs: beamforming + cross-correlation + least squares
├── test_signals/               # synthetic footstep WAVs to use as simulator input (git-ignored)
│   ├── make_footstep.py        # generates the WAVs below
│   ├── footstep_single.wav
│   ├── footstep_walk.wav
│   └── footstep_signals.png    # plot of the two signals
├── evaluate.py                 # entry point: runs a detector + localizer on a recording
└── requirements.txt
```

Dependencies only go one way: `evaluate.py` uses `core`, `detection` and `localization`;
`detection` and `localization` may use `core`; `core` uses neither.

## Components

### `core/data.py` - `Sample`

One recording produced by the simulator: one source, N sensors.

- Built from the per-recording JSON written by the simulator (e.g. `rep0001_file001_footstep_single.json`),
  which lists the WAV files and holds the geometry and ground truth.
- Attributes: `signals` `(N, T)` as float in [-1, 1], `fs`, `N`, `sensors_pos` `(N, 3)`,
  `source_pos` `(3,)`, `c`.
- `read(hop)` returns the next `(N, hop)` block in chronological order, or `None` when the recording
  is over. Blocks do not overlap, like the buffers delivered by a real ADC/sound card.
- `rewind()` restarts from the beginning.

### `localization/base.py` - `Localizer`

Interface of every localization algorithm.

- `__init__(sensor_pos, fs, c)` stores what every algorithm needs.
- `push(block)` receives a `(N, hop)` block and returns the estimated position, or `None`
  if no estimate is available yet (e.g. the internal buffer is not full).
- `reset()` clears the internal state before a new recording.

### `localization/beamforming_gcc_ls.py` - `BeamformingGccLs`

1. **Buffer** `M` `(N, Nw)` with the last `Nw` samples of every sensor; the most recent sample is in column 0.
2. **Beamforming**: correlation matrix `C = D Dᵀ / K` of the zero-mean sub-windows of length `L`;
   the eigenvector of the largest eigenvalue gives the weights and the output `y = wᵀ X(t)`.
   The eigenvector sign is fixed so that its dominant tap is positive.
3. **Cross-correlation** of `y` with every sensor, lags `0 … L-1`; the TDOA of sensor n with respect
   to sensor 1 is `-(τn - τ1) / fs` (the minus sign comes from the reversed time order of `M`).
4. **Least squares** (2D, linear): unknowns `[px, py, r1]`, one row per sensor `n = 2 … N`,
   solved with `np.linalg.lstsq`. Requires at least 4 sensors.

Constraints checked in the constructor: `L - 1 >= tau_max` (maximum delay between two sensors, in samples),
`L <= Nw`, `N >= 4`. In practice `K = Nw - L + 1` should be at least ~10 × `N·L` for a reliable `C`.

### `detection/base.py` - `Detector`

Interface of every detector. The detector sees every block and, when a footstep is over, returns the
whole event segment `(N, length)`, so the localizer works on a complete event.

- `__init__(pre_margin, post_margin, fs, N, max_duration)`: margins and maximum duration in samples.
  `post_margin` must be at least `tau_max`, so that the event has reached every sensor.
- `detect(block)` returns the event segment when an event has ended, otherwise `None`.
- `reset()` clears the internal state.

### `evaluate.py`

```
Sample ──read(hop)──► block ──► Localizer.push ──► estimate ──► error vs. ground truth ──► report
```

1. Reads the command line: recording JSON, localizer name (`--loc`) and detector name (`--det`).
2. Loads `configs/default.toml`.
3. Builds the `Sample`, then the localizer and detector chosen by name through the
   `LOCALIZERS` / `DETECTORS` dictionaries, passing the common arguments plus the parameters of
   their config section.
4. Streams the recording block by block, collects the estimates, computes the 2D position error
   and prints the statistics (mean, median, RMSE, percentiles, % below thresholds).

## Configuration

`configs/default.toml` holds the parameters of **all** the algorithms; the command line only picks
which ones to use. Every section key must have the same name as a constructor argument.
All lengths are **in samples**, so the values depend on `fs` (one config file per sampling rate).

```toml
[stream]
hop = 512                 # samples per block

[localization.bf_gcc_ls]  # section name = key in LOCALIZERS
L = 96
Nw = 4096

[detection.amp_thr]       # section name = key in DETECTORS
pre_margin = 320
post_margin = 320
max_duration = 3000
thr = 0.5
```

## Usage

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt

# run from the project root
venv/bin/python evaluate.py <recording.json> --loc bf_gcc_ls --det amp_thr
venv/bin/python evaluate.py --help      # lists the available algorithms
```

`<recording.json>` is the per-recording JSON inside a simulator batch folder,
e.g. `../acustic-array-simulator/outputs/selkd_batch_<timestamp>/rep0001_file001_footstep_single.json`.

Single modules can be run as `venv/bin/python -m localization.beamforming_gcc_ls` (from the root,
so that the `core`, `detection` and `localization` packages are importable).

## Adding an algorithm

1. Create a file in `localization/` (or `detection/`) with a class derived from `Localizer` (or `Detector`).
2. Add it to `LOCALIZERS` (or `DETECTORS`) in `evaluate.py` with a short name.
3. Add a section `[localization.<name>]` (or `[detection.<name>]`) to the config with its parameters.

`evaluate.py` needs no other change.

## Test signals

The simulator's default chirp is narrowband inside every analysis window, so the cross-correlation
has many equal peaks and the TDOAs become ambiguous at high frequencies. Footsteps are impulsive and
broadband, so `test_signals/make_footstep.py` generates footstep-like inputs: a heel strike followed
~80 ms later by a weaker toe strike, each a 20–400 Hz noise burst with a 3 ms attack and an
exponential decay, at 16 kHz. Load them in the simulator as input WAV files, with the source in the
same plane as the sensors (the least squares is 2D).

## Conventions

- Blocks entering `push` / `detect` are chronological (last column = most recent sample).
- Inside `BeamformingGccLs`, `M` is reversed: column 0 = most recent sample.
- Sensor 1 (index 0) is the TDOA reference.
- Positions are in metres; the error is computed on x and y only.

## Status and limitations

- `c` is currently taken from the simulator. For floor vibrations it is unknown and frequency dependent
  (dispersive waves); it will have to be estimated or removed from the model.
- TDOAs are integer lags; sub-sample interpolation of the peak is still to be added.
- The physical consistency checks on the least-squares result (`r1 > 0`, `r1 ≈ ‖p - s1‖`) are not implemented yet.
- `AmpThr` is a stub and `evaluate.py` does not use the detector yet: every block goes to the localizer,
  so windows with no footstep also produce (meaningless) estimates.
- Only one recording per run; looping over the whole batch (`manifest.json`) is still to be added.
