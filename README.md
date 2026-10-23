# Footstep detection and localization

Detection and localization algorithms for an array of sensors, developed for a master thesis.
The goal is to localize people walking from the floor vibrations produced by their footsteps.
During development, the algorithms are tested on data from an acoustic array simulator
(included as a git submodule), fed to them block by block as if it were arriving in real time.

## Project structure

```
.
├── analysis/
│   └── plot_results.py         # figures from a results CSV (called by evaluate.py)
├── configs/
│   └── default.toml            # parameters of every algorithm + stream settings
├── core/                       # code shared by detection and localization
│   ├── data.py                 # Sample: loads a simulated recording, streams it in blocks
│   └── metrics.py              # (empty for now) shared evaluation metrics
├── detection/
│   ├── base.py                 # Detector: abstract base class of every detector
│   └── amp_thr.py              # AmpThr: moving-average amplitude-threshold detector
├── localization/
│   ├── base.py                 # Localizer: abstract base class of every localizer
│   └── beamforming_gcc_ls.py   # BeamformingGccLs: beamforming + cross-correlation + least squares
├── simulation/
│   ├── configs/                # one TOML per simulated experiment
│   │   └── corner_square_grid.toml
│   └── generate_dataset.py     # runs the simulator engine without its GUI
├── simulator/                  # git submodule: acustic-array-simulator
├── test_signals/               # synthetic footstep WAVs used as simulator input (git-ignored)
│   └── make_footstep.py        # generates footstep_single.wav and footstep_walk.wav
├── tests/                      # unit tests (to be written)
├── data/                       # generated datasets (git-ignored)
├── results/                    # CSVs and figures written by evaluate.py
├── evaluate.py                 # entry point: runs a detector + localizer on a whole dataset
└── requirements.txt
```

Dependencies only go one way: `evaluate.py` uses `core`, `detection`, `localization` and `analysis`;
`detection` and `localization` may use `core`; `core` uses neither.

## Workflow

```
simulation/configs/<exp>.toml ──generate_dataset.py──► data/<exp>/ ──evaluate.py──► results/<name>.csv + figures
```

```bash
# once
git submodule update --init                 # if the repo was cloned without --recursive
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python test_signals/make_footstep.py

# for every experiment (from the project root)
venv/bin/python simulation/generate_dataset.py simulation/configs/corner_square_grid.toml [--overwrite]
venv/bin/python evaluate.py data/corner_square_grid --loc bf_gcc_ls --det amp_thr --out results/corner_square_grid.csv

# redo the figures of an existing CSV without running the evaluation again
venv/bin/python -m analysis.plot_results results/corner_square_grid.csv data/corner_square_grid
```

`evaluate.py --help` lists the available algorithms.

## Simulation

`simulation/generate_dataset.py` calls the engine of the simulator (`simulator/selkd_simulator/engine.py`,
which only needs numpy and scipy) and writes a folder with the same layout as a batch exported by the
simulator GUI: for every source position one JSON (geometry and ground truth) and one WAV per sensor,
plus `manifest.json` (list of the samples, geometry, speed, SNR) and a copy of the experiment config.

Options: `--out <folder>` (default `data/<config name>`), `--overwrite` to replace an existing dataset.

The experiment TOML has four sections:

| Section | Keys |
|---|---|
| `[simulation]` | `fs`, `c` (propagation speed), `model` (`free_field` / `shoebox`), `room`, `wall_absorption`, `reflection_order` |
| `[sensors]` | `layout` = `square` (`side`), `circle` (`diameter`), `linear` (`spacing`) with `center` and `count`, or `manual` (`positions`) |
| `[source]` | `signal` (WAV path from the project root, or `"chirp"`), `z`, and the positions: `grid_x` / `grid_y` = `[start, stop, step]`, or `n_positions` + `seed` + optional `area = [[xmin, xmax], [ymin, ymax]]`, or a fixed `position` |
| `[noise]` | optional `snr_db`: white noise on every sensor, SNR referred to the power of the strongest sensor during the event |

Keep sources and sensors at the same height: the least squares is 2D.
Datasets are not versioned; the config and the seed regenerate them identically.

## Components

### `core/data.py` - `Sample`

One simulated recording: one source, N sensors.

- Built from the per-recording JSON (e.g. `data/<exp>/rep0001_file001_footstep_walk.json`).
- Attributes: `signals` `(N, T)` as float in [-1, 1], `fs`, `N`, `sensors_pos` `(N, 3)`, `source_pos` `(3,)`, `c`.
- `read(hop)` returns a copy of the next `(N, hop)` block in chronological order, or `None` at the end.
  Blocks do not overlap, like the buffers delivered by a real ADC.
- `rewind()` restarts from the beginning.

### `localization/base.py` - `Localizer`

- `__init__(sensor_pos, fs, c)`.
- `push(block)`: streaming use, adds a block to the internal buffer and estimates when it is full.
- `load_block_from_det(segment)`: loads a whole event returned by a detector and returns the estimate.
- `estimate()`: position `(x, y)` from the current buffer, or `None` if not available or not reliable.
- `reset()`: clears the internal state.

### `localization/beamforming_gcc_ls.py` - `BeamformingGccLs`

1. **Buffer** `M` `(N, Nw)`, most recent sample in column 0. With `load_block_from_det`, `Nw` becomes the segment length.
2. **Beamforming**: correlation matrix `C = D Dᵀ / K` of the zero-mean sub-windows of length `L`;
   the eigenvector of the largest eigenvalue gives the weights and the output `y = wᵀ X(t)`.
   The eigenvector sign is fixed so that its dominant tap is positive.
3. **Cross-correlation** of `y` with every sensor, lags `0 … L-1`; the TDOA of sensor n with respect to sensor 1
   is `-(τn - τ1) / fs` (the minus sign comes from the reversed time order of `M`).
4. **Least squares** (2D, linear): unknowns `[px, py, r1]`, one row per sensor `n = 2 … N`, solved with `lstsq`.
5. **Consistency checks**: the linearization treats `r1` as a free unknown, so the estimate is rejected if
   `r1 < 0` (this is also the only symptom of TDOAs with the wrong sign) or if `|r1 - ‖p - s1‖| > 10 % r1`.

Constraints checked in the constructor: `L - 1 >= tau_max` (maximum delay between two sensors, in samples),
`L <= Nw`, `N >= 4`. For a reliable `C`, `K = Nw - L + 1` should be at least ~10 × `N·L`.

### `detection/base.py` - `Detector`

- `__init__(pre_margin, post_margin, fs, N, max_duration)`, lengths in samples.
- `detect(block)`: returns the segment `(N, length)` containing an event, or `None`.
- `reset()`: clears the internal state.

### `detection/amp_thr.py` - `AmpThr`

Moving average of `|x|` over `window` samples on every sensor; an event is where at least one sensor is above `thr`.
In the current block it returns the longest event, extended by `pre_margin` (taken from the previous block if needed)
and `post_margin` (cut at the end of the block). It works one block at a time, so an event crossing a block boundary
is returned as two fragments (see *Status and limitations*).

### `evaluate.py`

1. Reads the dataset folder, `--loc`, `--det` and `--out` from the command line, and `configs/default.toml`.
2. For every sample of `manifest.json`: builds the `Sample`, a new localizer and a new detector (chosen by name
   through the `LOCALIZERS` / `DETECTORS` dictionaries, with the parameters of their config section),
   streams the recording block by block, passes every detected segment to the localizer and collects the estimates.
3. Writes the CSV (`sample, true_x, true_y, est_x, est_y, error_m`; one row per estimate, an empty row for
   samples with no estimate), prints a line per sample (with the distance from the array centre) and the overall
   statistics, then calls `analysis.plot_results.plot_all`.

### `analysis/plot_results.py`

Saves next to the CSV: `<name>_map.png` (median error per source position, with the sensors),
`<name>_distance.png` (error vs. distance from the array centre, first step of each recording highlighted)
and `<name>_cdf.png` (cumulative distribution of the error).

## Configuration

`configs/default.toml` holds the parameters of **all** the algorithms; the command line only picks which ones to use.
Every key must have the same name as a constructor argument. All lengths are **in samples**, so the values depend
on `fs` and on the array size (`L - 1` must exceed the maximum delay between two sensors).

```toml
[stream]
hop = 8192                # samples per block

[localization.bf_gcc_ls]  # section name = key in LOCALIZERS
L = 144
Nw = 4096                 # only used with push(); overwritten by the segment length with a detector

[detection.amp_thr]       # section name = key in DETECTORS
pre_margin = 320
post_margin = 320
max_duration = 7000
thr = 0.05
window = 30
```

## Adding an algorithm

1. Create a file in `localization/` (or `detection/`) with a class derived from `Localizer` (or `Detector`).
2. Add it to `LOCALIZERS` (or `DETECTORS`) in `evaluate.py` with a short name.
3. Add a section `[localization.<name>]` (or `[detection.<name>]`) to the config with its parameters.

## Test signals

The simulator's default chirp is narrowband inside every analysis window, so the cross-correlation has many
equal peaks and the TDOAs become ambiguous at high frequencies. Footsteps are impulsive and broadband, so
`test_signals/make_footstep.py` generates footstep-like inputs at 16 kHz: a heel strike followed ~80 ms later by
a weaker toe strike, each a 20–400 Hz noise burst with a 3 ms attack and an exponential decay.
`footstep_single.wav` has one step at 0.5 s, `footstep_walk.wav` four steps every 0.55 s.

## Conventions

- Blocks entering `push` / `detect` are chronological (last column = most recent sample).
- Inside `BeamformingGccLs`, `M` is reversed: column 0 = most recent sample.
- Sensor 1 (index 0) is the TDOA reference.
- Positions are in metres; the error is computed on x and y only.

## Status and limitations

- `AmpThr` has no memory between blocks: a step crossing a block boundary is split into fragments, which can give
  wrong TDOAs. With a fixed step time in the input WAV this happens systematically (e.g. the step at 0.5 s and
  `hop = 8192`). A detector that keeps the event open across blocks is still to be written.
- `c` is currently taken from the simulator. For floor vibrations it is unknown and frequency dependent
  (dispersive waves); it will have to be estimated or removed from the model.
- TDOAs are integer lags; sub-sample interpolation of the peak is still to be added. Far from the array the
  radial error grows quickly (one sample of TDOA error moves the estimate by metres).
- No unit tests yet; `core/metrics.py` is empty (the error and the statistics are computed in `evaluate.py`).
- The config path is fixed (`configs/default.toml`).
