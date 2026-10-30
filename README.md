# Footstep detection and localization

Detection and localization algorithms for an array of sensors, developed for a master thesis.
The goal is to localize people walking from the floor vibrations produced by their footsteps.
During development the algorithms are tested on data from an acoustic array simulator (git submodule), fed to
them block by block as if it were arriving in real time, and evaluated over campaigns of simulated datasets.

## Project structure

```
.
├── analysis/
│   ├── plot_results.py         # figures of one run (error map, error vs. distance, CDF)
│   └── compare_runs.py         # summary table and comparison figures of a whole campaign
├── configs/
│   └── default.toml            # parameters of the algorithms, the stream, the event and the frequency band
├── core/
│   ├── data.py                 # Sample: loads a simulated recording, band-pass filters it, streams it in blocks
│   └── metrics.py              # error statistics shared by evaluate.py and the analysis scripts
├── detection/
│   ├── base.py                 # Detector: abstract base class
│   └── amp_thr.py              # AmpThr: envelope detector with adaptive threshold
├── localization/
│   ├── base.py                 # Localizer: abstract base class
│   ├── beamforming_gcc_ls.py   # BeamformingGccLs: beamforming + (generalized) cross-correlation + least squares
│   └── wavelet_tdoa.py         # WaveletTdoaMult: wavelet arrival times + adaptive multilateration (unknown speed)
├── simulation/
│   ├── configs/                # one TOML per simulated dataset (the experiment campaign)
│   └── generate_dataset.py     # runs the simulator engine without its GUI
├── simulator/                  # git submodule: acustic-array-simulator
├── test_signals/               # synthetic footstep WAVs used as simulator input (git-ignored)
│   └── make_footstep.py
├── tests/                      # unit tests (to be written)
├── data/                       # generated datasets (git-ignored)
├── results/                    # CSVs, logs/, figures/ and summary/ (CSVs and figures git-ignored)
├── evaluate.py                 # runs a detector + localizer on one dataset
├── run_experiments.sh          # generates the missing datasets, evaluates every dataset in parallel, compares
└── requirements.txt
```

Dependencies only go one way: `evaluate.py` uses `core`, `detection`, `localization` and `analysis`;
`detection` and `localization` may use `core`; `core` uses neither.

## Workflow

```
simulation/configs/*.toml ─generate_dataset.py─► data/<exp>/ ─evaluate.py─► results/<exp>_<loc>.csv ─compare_runs.py─► results/summary/
```

```bash
# once
git submodule update --init                 # if the repo was cloned without --recursive
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python test_signals/make_footstep.py

# whole campaign: missing datasets, every dataset x localizer in parallel, comparison
./run_experiments.sh

# single steps, from the project root
venv/bin/python simulation/generate_dataset.py simulation/configs/y_7m_center_snr20.toml [--overwrite]
venv/bin/python evaluate.py data/y_7m_center_snr20 --loc bf_gcc_ls [--det amp_thr] [--out results/x.csv] [--config configs/default.toml]
venv/bin/python -m analysis.plot_results results/y_7m_center_snr20_bf_gcc_ls.csv data/y_7m_center_snr20
venv/bin/python -m analysis.compare_runs
```

`evaluate.py --help` lists the available algorithms. Without `--out` the CSV is `results/<dataset>_<loc>.csv`.

### `run_experiments.sh`

1. Creates the venv if missing, then generates every dataset of `simulation/configs/` that has no `manifest.json`.
2. Runs `evaluate.py` for every dataset and localizer, up to `nproc --all` processes at a time, with numpy limited to
   one thread per process (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`). Each run writes its CSV in `results/` and
   its output in `results/logs/<dataset>_<loc>.log`; the script prints `start` / `done` / `FAILED` for every run.
3. Prints how many CSVs are present and runs `analysis.compare_runs`.

The whole campaign (13 datasets x 2 localizers) takes about 15 minutes on 8 cores.

## Simulation

`simulation/generate_dataset.py` calls the engine of the simulator (`simulator/selkd_simulator/engine.py`, only numpy
and scipy) and writes, for every source position, one JSON (geometry and ground truth) and one WAV per sensor, plus
`manifest.json` and a copy of the config. Options: `--out <folder>` (default `data/<config name>`), `--overwrite`.

| Section | Keys |
|---|---|
| `[simulation]` | `fs`, `c` (propagation speed), `model` (`free_field` / `shoebox`), `room`, `wall_absorption`, `reflection_order` |
| `[sensors]` | `layout` = `square` (`side`), `circle` (`diameter`), `linear` (`spacing`) with `center` and `count`, or `manual` (`positions`) |
| `[source]` | `signal` (WAV from the project root, or `"chirp"`), `z`, and the positions: `grid_x` / `grid_y` = `[start, stop, step]`, or `n_positions` + `seed` + optional `area`, or a fixed `position` |
| `[noise]` | optional `snr_db`: white noise on every sensor, SNR referred to the power of the strongest sensor during the event |

Keep sources and sensors at the same height: the localization is 2D.

### Current campaign

50 x 50 m room, free field, 16 kHz, `footstep_walk.wav` (4 steps), sources on a 5 m grid (100 positions),
4 sensors in every layout, arrays rotated by 10 deg so that no sensor is closer than 0.6 m to a grid node.
The aperture is the maximum distance between two sensors. Dataset names: `<layout>_<aperture>m_<position>_snr<dB>`.

| Group | Varied | Fixed | Datasets |
|---|---|---|---|
| A | layout (Y, square, L) x position (corner (5,5), mid (15,15), centre (25,25)) | 7 m, 20 dB | 9 |
| B | aperture 3.5 / 10 m | Y, centre, 20 dB | 2 |
| C | SNR 30 / 10 dB | Y, 7 m, centre | 2 |

Reference configuration: **Y, 7 m, centre, 20 dB** (`y_7m_center_snr20`).

## Components

### `core/data.py` - `Sample`

One simulated recording. Built from the per-recording JSON and from the band of the config
(`Sample(path, low_cutoff, high_cutoff, filter_order)`).

- Attributes: `signals` `(N, T)` in [-1, 1], `fs`, `N`, `sensors_pos` `(N, 3)`, `source_pos` `(3,)`, `c`, `room_dim`.
- `read(hop)` returns the next `(N, hop)` block, band-pass filtered (Butterworth, second-order sections, `sosfilt`
  with the filter state kept between blocks, so block-wise filtering equals filtering the whole signal),
  or `None` at the end. The same filter on every sensor shifts all the signals equally, so the TDoAs are unchanged.
- `rewind()` restarts from the beginning and resets the filter state.

### `detection/base.py` - `Detector`, `detection/amp_thr.py` - `AmpThr`

`detect(block)` returns the list of the events that ended in the block, each one a segment `(N, length)` with
`pre_margin` samples of noise before it; `reset()` clears the state. Events are kept open across blocks.

`AmpThr`: envelope = moving average of `|x|` over `window` samples on every sensor; an event is where at least one
sensor is above its threshold. The threshold is adaptive and per sensor: `max(k * median(envelope), min_thr)`,
initialized on the first block and updated on every block with an exponential average (`learning_rate`).
The median of the envelope is not moved by a step shorter than half a block, so the update needs no event logic.

### `localization/base.py` - `Localizer`

`__init__(sensor_pos, fs, c, pre_margin, room_dim, low_cutoff, high_cutoff)`: common information, passed to every
localizer even if it does not use it. `load_block_from_det(segment)` loads a detected event and returns the estimated
position `(x, y)` or `None`; `estimate()` computes it from the loaded segment; `reset()` clears the state.

### `localization/beamforming_gcc_ls.py` - `BeamformingGccLs`

1. **Beamforming**: correlation matrix `C = D Dᵀ / K` of the zero-mean sub-windows of length `L` of every sensor;
   the eigenvector of the largest eigenvalue gives the weights and the output `y = wᵀ X(t)`.
   `L = "auto"` sets `L = ceil(1.1 tau_max) + 1` from the sensor positions (`tau_max` = max delay between two sensors).
2. **Cross-correlation** of `y` with every sensor over all the lags, plain (`cc_type = "cc"`) or generalized with the
   `roth`, `scot` or `phat` weight (with a regularization `epsilon`); TDoA of sensor n w.r.t. sensor 1 = `-(τn - τ1) / fs`.
3. **Least squares** (2D, linear, speed `c` known): unknowns `[px, py, r1]`, solved with `lstsq`, at least 4 sensors.
4. **Checks**: the estimate is rejected if `r1 < 0`, if `|r1 - ‖p - s1‖| > 10 % r1` (the linearization treats `r1` as
   a free unknown) or if it is outside the room.

### `localization/wavelet_tdoa.py` - `WaveletTdoaMult`

Approach of Mirshekari et al., *Occupant localization using footstep-induced structural vibration*, MSSP 2018:
the speed is not needed.

1. **Wavelet**: CWT with the Mexican hat on `n_scales` scales covering the band; `edge_margin` (5 x the largest scale)
   samples are excluded at both ends because of the edge effects, so `pre_margin` must exceed `edge_margin`.
2. **Arrival times**: for every scale and sensor, first sample above `max(3σ_noise, k · peak)`; the relative term makes
   the threshold crossing happen at the same relative point of the waveform on every sensor.
3. **Sensor selection**: the 4 sensors reached first; the first one is the anchor.
4. **Adaptive multilateration**: for every scale and every speed in `[v_min, v_max]` (`v_count` values) one hyperbola
   per anchor-sensor pair (`h_range`, `h_steps` points), intersections computed with shapely and kept only inside the
   room. A pair of hyperbolas sharing a focus can intersect twice: among the combinations with one candidate per pair,
   the one with minimum cost (sum of the distances between the 3 points) is kept. The best speed of every scale gives a
   position (mean of the intersections); the final estimate is the mean of the 8 scales with lowest cost.

### `evaluate.py`

For every sample of `manifest.json`: builds the `Sample`, a new localizer and a new detector (chosen by name through
`LOCALIZERS` / `DETECTORS`, with the parameters of their config section plus the common ones), streams the recording,
passes every detected event to the localizer and collects the estimates. Writes the CSV
(`sample, true_x, true_y, est_x, est_y, error_m`; one row per estimate, an empty row for samples with no estimate),
prints a line per sample and the statistics of `core/metrics.py`, then calls `analysis.plot_results.plot_all`.

### `analysis/plot_results.py` and `analysis/compare_runs.py`

- `plot_results`: in `results/figures/`, `<name>_map.png` (median error per source position, fixed colour scale up to
  `MAP_VMAX` = 5 m so that maps of different runs are comparable), `<name>_distance.png`, `<name>_cdf.png`.
- `compare_runs`: reads every campaign CSV, gets the factors from the dataset name and writes in `results/summary/`
  `summary.csv` and the figures of the three groups (`layout_position.png`, `aperture.png`, `snr.png`),
  `reference_cdf.png` and `distance_by_layout.png`. "Estimates per recording" is 4 when every step is localized.

## Configuration (`configs/default.toml`)

All lengths are **in samples**, so they depend on `fs`. Every key must match a constructor argument.

| Section | Keys |
|---|---|
| `[stream]` | `hop`: samples per block |
| `[localization.bf_gcc_ls]` | `L` (`"auto"` or integer), `Nw`, `cc_type` (`cc`, `roth`, `scot`, `phat`) |
| `[localization.wv_tdoa_mlt]` | `n_scales`, `v_min`, `v_max`, `v_count`, `h_range`, `h_steps`, `k` |
| `[detection.amp_thr]` | `post_margin`, `window`, `learning_rate`, `k`, `min_thr` |
| `[event]` | `pre_margin`: noise samples before every event, shared by detector and localizers |
| `[band]` | `low_cutoff`, `high_cutoff` [Hz], `filter_order`: band-pass filter and band of the localizers |

## Adding an algorithm

1. Create a file in `localization/` (or `detection/`) with a class derived from `Localizer` (or `Detector`).
2. Add it to `LOCALIZERS` (or `DETECTORS`) in `evaluate.py` with a short name.
3. Add a section `[localization.<name>]` (or `[detection.<name>]`) to the config with its parameters.
4. To include it in the campaign, add the name to `locs` in `run_experiments.sh`.

## Test signals

`test_signals/make_footstep.py` generates footstep-like inputs at 16 kHz: a heel strike followed ~80 ms later by a
weaker toe strike, each a 20–400 Hz noise burst with a 3 ms attack and an exponential decay.
`footstep_single.wav` has one step at 0.5 s, `footstep_walk.wav` four steps every 0.55 s.
The default chirp of the simulator is narrowband inside every analysis window and gives ambiguous TDoAs.

## Conventions

- Blocks entering `detect` and segments entering `load_block_from_det` are chronological.
- Inside `BeamformingGccLs` the buffer `M` is reversed (column 0 = most recent sample).
- Sensor 1 (index 0) is the TDoA reference of `BeamformingGccLs`; `WaveletTdoaMult` uses the earliest sensor.
- Positions are in metres, with the origin in a corner of the room; the error is computed on x and y only.

## Status and limitations

- `BeamformingGccLs` needs the speed `c`; for floor vibrations it is unknown and frequency dependent (dispersion).
- The GCC weights whiten the spectrum over all the frequencies: limiting them to the band is still to be done.
- TDoAs and arrival times are integer samples; sub-sample interpolation is still to be added.
- In `WaveletTdoaMult` the noise σ is computed on `[0, pre_margin]`, edge region included, the threshold mask uses
  `3 * thr` and the speed range of the config is narrow around the true speed of the simulator.
- The noise of the simulations is white; reflections (`shoebox`) and real data are still to be tested.
- No unit tests yet.
