# Footstep detection and localization

Detection and localization algorithms for an array of sensors, developed for a master thesis.
The goal is to localize people walking from the floor vibrations produced by their footsteps.
During development the algorithms are tested on simulated datasets, fed to them block by block as if the data were
arriving in real time. Three localizers are compared: two classic ones, based on the time differences of arrival,
and a neural network.

## Quick start

All the commands are run from the project root.

```bash
# once
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python sim/test_signals/make_footstep.py     # footstep WAVs used as source signal

# 1. datasets: data/training/, data/validation/ and data/test/, one folder per sensor configuration
venv/bin/python -m executables.generate_dataset

# 2. one network per configuration: weights in checkpoints/, plots of the training in training/log/
./executables/train.sh

# 3. every localizer on every test dataset: CSVs and figures in results/, comparison in results/summary/
./executables/eval.sh
```

The same steps one by one:

```bash
# training of the network of one configuration (a subfolder of data/training/ and data/validation/)
venv/bin/python -m training.train y_7m_10db [--train_conf configs/train.toml] [--def_conf configs/default.toml]

# one localizer on one dataset: CSV in results/, figures in results/figures/
venv/bin/python -m executables.evaluate data/test/y_7m_10db --loc bf_gcc_ls --det amp_thr [--out results/x.csv] [--config configs/default.toml]
venv/bin/python -m executables.evaluate data/test/y_7m_10db --loc cnn_attention --det raw

# figures of an existing CSV again
venv/bin/python -m analysis.plot_results results/y_7m_10db_bf_gcc_ls.csv data/test/y_7m_10db

# comparison of all the CSVs in results/
venv/bin/python -m analysis.compare_runs

# plots of all the trainings in training/log/
venv/bin/python -m training.plots

# check of the targets built for the network (shapes, fraction of active frames)
venv/bin/python -m training.loader
```

The classic localizers are used with the `amp_thr` detector, the network with `raw` (the network detects the steps
itself). `executables.evaluate --help` lists the localizers and detectors; without `--out` the CSV is
`results/<dataset>_<loc>.csv`.

The scripts import `core`, `detection`, `localization` and `sim`, so they are run as modules from the project root
(`python -m executables.evaluate`, with dots and without `.py`). Run as `python executables/evaluate.py`, Python looks
for the imports in `executables/` and fails with `ModuleNotFoundError`. In VS Code use a `launch.json` configuration
with `"module": "executables.evaluate"` and `"cwd": "${workspaceFolder}"`.

## Project structure

```
.
├── sim/                        # simulation, used as a black box through sim.simulate()
│   ├── simulate.py             # API: sensors + source positions + fs + SNR -> dataset folder
│   ├── engine.py               # propagation engine (from acustic-array-simulator, without normalization)
│   └── test_signals/           # make_footstep.py, the footstep WAVs and the pool/ of random walks (git-ignored)
├── core/
│   ├── data.py                 # Sample: one recording of a dataset, band-pass filtered and streamed in blocks
│   └── metrics.py              # error statistics shared by evaluate and the analysis scripts
├── detection/
│   ├── base.py                 # Detector: abstract base class
│   ├── amp_thr.py              # AmpThr: envelope detector with adaptive threshold
│   └── raw.py                  # Raw: fixed-length overlapping blocks, input of the neural network
├── localization/
│   ├── base.py                 # Localizer: abstract base class
│   ├── beamforming_gcc_ls.py   # BeamformingGccLs: beamforming + (generalized) cross-correlation + least squares
│   ├── wavelet_tdoa.py         # WaveletTdoaMult: wavelet arrival times + adaptive multilateration (unknown speed)
│   └── cnn_attention.py        # CNNAttention: neural network (CNN + self-attention), targets, input normalization
├── training/
│   ├── loader.py               # FootstepDataset: windows and per-frame targets read from a dataset folder
│   ├── train.py                # training of the network of one sensor configuration
│   ├── plots.py                # plots of the trainings
│   └── log/                    # history and plots of every training (git-ignored)
├── analysis/
│   ├── plot_results.py         # figures of one run (error map, error vs. distance, CDF)
│   └── compare_runs.py         # summary table and comparison figures of all the runs
├── executables/
│   ├── generate_dataset.py     # sensor configurations and datasets of the campaign
│   ├── evaluate.py             # runs a detector + localizer on one dataset
│   ├── train.sh                # trains one network per configuration of data/training/
│   └── eval.sh                 # evaluates every localizer on every dataset of data/test/, compares
├── configs/
│   ├── default.toml            # parameters of the algorithms, the stream, the event and the frequency band
│   └── train.toml              # parameters of the training of the network
├── data/                       # datasets: training/, validation/, test/ (git-ignored)
├── checkpoints/                # weights of the networks, one per configuration (git-ignored)
├── results/                    # CSVs, logs/, figures/ and summary/ (git-ignored)
└── requirements.txt
```

Dependencies only go one way: `sim`, `core`, `detection` and `localization` use no other folder of the project;
`analysis` uses `core`; `training` uses `core`, `detection`, `localization` and the plot style of `analysis`;
`executables` use everything.

## Workflow

```
executables/generate_dataset.py ─sim.simulate─► data/{training,validation,test}/<conf>/
data/{training,validation}/<conf>/ ─training.train─► checkpoints/<conf>.pt + training/log/<conf>/
data/test/<conf>/ ─executables.evaluate (every localizer)─► results/<conf>_<loc>.csv ─compare_runs─► results/summary/
```

## Simulation (`sim/`)

```python
from sim import simulate
simulate(out_dir, sensor_pos, source_pos, fs, snr_db,
         signal=..., signal_pool=None, c=343.42, model="free_field", room=(50.0, 50.0, 3.0),
         wall_absorption=0.35, reflection_order=2, seed=0, overwrite=False)
```

- `sensor_pos` `(N, 3)`, `source_pos` `(R, 3)` [m]: one recording for every source position.
- `snr_db`: one value for all the recordings, `(R,)` for one value per recording, or `None` for no noise. White noise,
  independent on every sensor, with the SNR referred to the power of the strongest sensor during the event; `seed`
  sets it.
- `model`: `free_field` (direct path only) or `shoebox` (image-source reflections, `wall_absorption`,
  `reflection_order`). `signal`: source WAV of every recording, default `sim/test_signals/footstep_walk.wav`.
- `signal_pool`: folder of source WAVs; if given (instead of `signal`), every recording uses one of them chosen at
  random, uniform with repetition. The choice depends only on `seed` and on the sorted file names, and uses a
  generator separate from the noise, so without `signal_pool` the datasets are the same as before its introduction.
- If `out_dir` is not empty it raises an error, unless `overwrite=True`.

`sim/engine.py` is the propagation engine of
[acustic-array-simulator](https://github.com/jjavierk/acustic-array-simulator) (commit `5859271`), with fractional
delays and 1 / distance attenuation. Unlike the original, the signals are **not normalized**: they keep their physical
amplitude, so it can be compared between recordings and datasets, as with real measurements.

### Dataset format

```
<dataset>/
├── dataset.json     common information and list of the recordings, written last
├── rec0000.wav      float32, N channels (channel i = sensor i of sensor_pos), physical amplitude
└── ...
```

```json
{
  "format_version": 2,
  "origin": "simulated",
  "fs": 16000,
  "sensor_pos": [[35.0, 35.0, 1.0], ...],
  "simulation": {"c": 343.42, "model": "free_field", "room": [70, 70, 3], "wall_absorption": 0.35,
                 "reflection_order": 2, "signal_pool": null, "seed": 3},
  "recordings": [
    {"file": "rec0000.wav", "source_pos": [1.0, 1.0, 1.0], "snr_db": 10.0, "n_samples": 58519,
     "signal": "sim/test_signals/footstep_walk.wav"},
    ...
  ]
}
```

A folder without `dataset.json` is an interrupted simulation. The recordings can have different lengths (`n_samples`).
In version 1 (the datasets generated before `signal_pool`) the source WAV is `simulation.signal`, common to all the
recordings, and the recordings have no `signal`; no code of the project reads these fields.
Real measurements can be stored in the same format (`"origin": "measured"`, without `simulation`).

### Campaign (`executables/generate_dataset.py`)

70 x 70 x 3 m room, free field, 16 kHz, SNR 10 dB, array at the centre of the room (35, 35). Every recording uses a
random signal of the pool `sim/test_signals/pool/` (`SIGNAL_POOL`, see Test signals), chosen with the seed of its
group: the same position of the same group has the same signal in every configuration. Six sensor configurations of 4 sensors: Y and square layouts, apertures 3.5, 7 and 10 m (maximum distance
between two sensors). Names: `<layout>_<aperture>m_<SNR>db`, with `p` for the decimal point (`y_3p5m_10db`).

- Y: one sensor in the centre and three at 120 deg (100, 220, 340 deg) at distance aperture / √3.
- Square: vertices at 235, 325, 55, 145 deg at distance aperture / 2.

| Group | Source positions | Seed of the noise |
|---|---|---|
| `data/test/<conf>` | 16 x 16 grid from 1 to 69 m (256) | 3 |
| `data/validation/<conf>` | 16 x 16 grid of cell centres (256), no node in common with the test grid | 1 |
| `data/training/<conf>` | random, uniform in [1, 69] m, `(1 - VAL_SPLIT) / VAL_SPLIT` x the validation size (1451) | 2 |

Both grid sides are even: with an odd side a node is in the centre of the room, on the central sensor of the Y. Every
grid node is at least 0.65 m from the sensors. Existing datasets (with `dataset.json`) are skipped.

## Components

### `core/data.py` - `Sample`

One recording of a dataset: `Sample(path of dataset.json, index of the recording, low_cutoff, high_cutoff,
filter_order)`.

- Attributes: `signals` `(N, T)`, `fs`, `N`, `sensors_pos` `(N, 3)`, `source_pos` `(3,)`, `c`, `room_dim`.
  It raises an error if the WAV has a different number of channels than the sensors of the JSON.
- `read(hop)` returns the next `(N, hop)` block, band-pass filtered (Butterworth, second-order sections, `sosfilt`
  with the filter state kept between blocks, so block-wise filtering equals filtering the whole signal),
  or `None` at the end. The same filter on every sensor shifts all the signals equally, so the TDoAs are unchanged.
- `rewind()` restarts from the beginning and resets the filter state.

### `detection/` - `Detector`, `AmpThr`, `Raw`

`detect(block)` returns the list of the segments `(N, length)` ready for the localizer; `reset()` clears the state.

`AmpThr`: envelope = moving average of `|x|` over `window` samples on every sensor; an event is where at least one
sensor is above its threshold (`compute_mask(block)` returns this mask, used also to label the training data).
The threshold is adaptive and per sensor: `max(k * median(envelope), min_thr)`, initialized on the first block and
updated on every block with an exponential average (`learning_rate`). Since the amplitudes are physical, `min_thr` is
only a floor for signals without noise. Every event is returned with `pre_margin` samples of noise before it, and
events are kept open across blocks.

`Raw`: no detection, it returns every block of `input_size` samples, one every `stride` samples (overlapping if
`stride < input_size`). It is the input of the neural network.

### `localization/base.py` - `Localizer`

`__init__(sensor_pos, fs, c, pre_margin, room_dim, low_cutoff, high_cutoff)`: common information, passed to every
localizer even if it does not use it. `load_block_from_det(segment)` loads a segment and returns the estimated
position `(x, y)`, several positions `(K, 2)` or `None`; `estimate()` computes it from the loaded segment; `reset()`
clears the state.

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
   per anchor-sensor pair, sampled with `h_steps` points of the parameter `h` in `[-h_max, h_max]`,
   `h_max = asinh(D / b)` (`D` = diagonal of the room, `b` = semi-minor axis), so that the points cover the room;
   intersections computed with shapely and kept only inside the room. A pair of hyperbolas sharing a focus can intersect twice: among the combinations with one candidate per pair,
   the one with minimum cost (sum of the distances between the 3 points) is kept. The best speed of every scale gives a
   position (mean of the intersections); the final estimate is the mean of the 8 scales with lowest cost.

### `localization/cnn_attention.py` - `CNNAttention`

Network `SELDNet`, from the SELD branch of *Multi-Task Learning for Sound Source Localization and Keyword Spotting on
Nano-Drones* (ETH): 4 convolutional blocks (Conv1d, batch norm, ReLU, max pooling, dropout, squeeze-and-excitation),
4 self-attention blocks (8 heads, residual + layer norm), 2 linear layers. Input `(B, N, L)`, output `(B, T, 3)` with
one frame every ~2400 samples (~150 ms at 16 kHz): `(dx, dy)` with tanh and `d` with ReLU.

- **Targets** (ACCDOA-like, one class, one source): `encode_pos` gives `(dx, dy, d / scale)`, with `(dx, dy)` the unit
  vector from the centroid of the sensors to the source and `d` their distance, multiplied by the activity of the
  frame, so an inactive frame is `(0, 0, 0)`; `decode_pos` is the inverse.
- **Detection**: a frame is a step if `‖(dx, dy)‖ > det_thr`; the estimate of a window is the median of the positions
  decoded from its active frames (one estimate per window, `None` if no frame is active).
- **Window alignment**: the training windows start at multiples of `window_stride` (2400 samples) from the beginning
  of the recording, and the network works only with windows aligned in the same way (a shift of 1200 samples gives
  errors of ~20 m).
- **Input normalization**: `normalize_input` divides every window by its maximum absolute value over all the sensors,
  in training and in inference. The physical amplitude changes by orders of magnitude with the distance, and without
  it the validation is unstable. It uses only the current window, so it works in real time.
- **Weights**: the network does not receive the sensor positions, so there is one checkpoint per configuration in
  `weights_dir`. `find_checkpoint` chooses the one trained with the same `sensor_pos` of the data (`.tmp.pt` files and
  checkpoints without `sensor_pos` are ignored). The constructor stops if the number of sensors, `fs`, the band or the
  input normalization differ from the training, and takes `input_size` and `dist_scale` from the checkpoint;
  `load_block_from_det` accepts only blocks of `input_size` samples.
- **Initialization**: `init_weights` sets Conv1d and Linear layers to the PyTorch default (uniform in
  `±1 / √fan_in`). With Xavier the ReLU of the distance output got stuck at 0 (median error 27.7 m).

### `executables/evaluate.py`

For every recording of `dataset.json`: builds the `Sample`, a new localizer and a new detector (chosen by name through
`LOCALIZERS` / `DETECTORS`, with the parameters of their config section plus the common ones), streams the recording,
passes every segment of the detector to the localizer and collects the estimates. Writes the CSV
(`sample, true_x, true_y, est_x, est_y, error_m`; one row per estimate, an empty row for recordings with no estimate),
prints a line per recording and the statistics of `core/metrics.py`, then calls `analysis.plot_results.plot_all`.

### `analysis/plot_results.py` and `analysis/compare_runs.py`

- `plot_results`: in `results/figures/`, `<name>_map.png` (median error per source position: cells for a grid, points
  otherwise; fixed colour scale up to `MAP_VMAX` = 20 m so that maps of different runs are comparable),
  `<name>_distance.png`, `<name>_cdf.png`.
- `compare_runs`: reads every `results/<dataset>_<loc>.csv`, gets layout, aperture and SNR from the dataset name and
  writes in `results/summary/` `summary.csv` and the figures `layout_aperture.png` (median and P90),
  `layout_aperture_stats.png` (mean, std, min, max, one colour scale per statistic), `aperture_<layout>.png`,
  `snr.png` (only with more than one SNR), `reference_cdf.png` and `distance_by_layout.png` (one y scale per
  localizer). The reference configuration is `REFERENCE` (Y, 7 m, 10 dB).

### `training/`

- `loader.py` - `FootstepDataset(data_dir, input_size, window_stride, frame_thr, dist_scale, T, def_toml)`: `data_dir`
  is the folder of one dataset. Every recording is read with `Sample` (same filter as `evaluate`) and labeled sample by
  sample with `AmpThr.compute_mask`, block by block. Filtered signal (float32) and mask are computed once and saved in
  `<data_dir>/cache/` (`recXXXX_signal.npy`, `recXXXX_mask.npy`, about 0.7 MB per recording, and `cache.json` with the
  parameters and the lengths, written last); every window is read from disk with a memory map, so the RAM does not grow
  with the dataset. The cache is built again if `dataset.json` (SHA-256) or band, hop, `pre_margin` or `amp_thr`
  of `default.toml` change. Element `k`: `x` `(N, input_size)`, the normalized window, and `y`
  `(T, 3)`: the activity mask of the window is split in `T` frames, a frame is active if more than `frame_thr` of its
  samples are active, and its target is `encode_pos(source, active, centroid, dist_scale)`.
  `output_frames(N, input_size)` gives `T` with a forward pass of the network.
- `train.py <conf>`: trains on `data/training/<conf>` and validates on `data/validation/<conf>`. Loss: MSE on
  `(dx, dy)` of all the frames + MSE on `d` of the active frames. Adam, early stopping after `patience` epochs without
  improvement of the validation loss. The best weights are written in `checkpoints/<conf>.tmp.pt`, renamed to
  `<conf>.pt` at the end, with `N`, `fs`, `input_size`, `T`, `frame_thr`, `dist_scale`, `input_norm`, `sensor_pos` and
  `data_params` (band, hop, `amp_thr`). Every epoch it also computes on the validation set the detection precision,
  recall and F1 of the frames and the median position and direction error of the detected frames, written in
  `training/log/<conf>/history.csv`.
- `plots.py`: `training/log/<conf>/loss.png` and `val_metrics.png` (called by `train.py`), and
  `training/log/summary.png` and `summary.csv` for all the configurations (called by `train.sh`).

### `executables/train.sh` and `executables/eval.sh`

- `train.sh`: runs `training.train` for every folder of `data/training/` with a `dataset.json`, then
  `training.plots`. Every run replaces the checkpoint of its configuration.
- `eval.sh`: runs `executables.evaluate` for every dataset of `data/test/` and every localizer (`bf_gcc_ls` and
  `wv_tdoa_mlt` with `amp_thr`, `cnn_attention` with `raw`), up to `nproc --all` processes at a time with numpy limited
  to one thread per process. Each run writes its CSV in `results/` and its output in `results/logs/`; the script
  prints `start` / `done` / `FAILED` for every run and at the end runs `analysis.compare_runs`.

## Configuration

`configs/default.toml`: all lengths are **in samples**, so they depend on `fs`. Every key must match a constructor
argument.

| Section | Keys |
|---|---|
| `[stream]` | `hop`: samples per block |
| `[localization.bf_gcc_ls]` | `L` (`"auto"` or integer), `Nw`, `cc_type` (`cc`, `roth`, `scot`, `phat`) |
| `[localization.wv_tdoa_mlt]` | `n_scales`, `v_min`, `v_max`, `v_count`, `h_range`, `h_steps`, `k` |
| `[localization.cnn_attention]` | `det_thr` (threshold on `‖(dx, dy)‖`), `weights_dir` (folder of the checkpoints) |
| `[detection.amp_thr]` | `post_margin`, `window`, `learning_rate`, `k`, `min_thr` |
| `[detection.raw]` | `input_size` (must be the `input_size` of the training), `stride` |
| `[event]` | `pre_margin`: noise samples before every event, shared by detector and localizers |
| `[band]` | `low_cutoff`, `high_cutoff` [Hz], `filter_order`: band-pass filter and band of the localizers |

`configs/train.toml`:

| Section | Keys |
|---|---|
| `[data]` | `train_dir`, `val_dir`: folders with one subfolder per configuration |
| `[window]` | `input_size` (multiple of 2400), `window_stride`, `frame_thr`, `dist_scale` [m] |
| `[training]` | `batch_size`, `lr` (Adam), `max_epochs`, `patience` (early stopping), `seed` |
| `[output]` | `checkpoint_dir`: weights in `<checkpoint_dir>/<conf>.pt` |

## Adding an algorithm

1. Create a file in `localization/` (or `detection/`) with a class derived from `Localizer` (or `Detector`).
2. Add it to `LOCALIZERS` (or `DETECTORS`) in `executables/evaluate.py` with a short name.
3. Add a section `[localization.<name>]` (or `[detection.<name>]`) to `configs/default.toml` with its parameters.
4. To include it in the comparison, add the name to `locs` in `executables/eval.sh` (and its detector).

## Test signals

`sim/test_signals/make_footstep.py` generates footstep-like inputs at 16 kHz: a heel strike followed ~80 ms later by a
weaker toe strike, each a 20–400 Hz noise burst with a 3 ms attack and an exponential decay.
`footstep_single.wav` has one step at 0.5 s, `footstep_walk.wav` four steps every 0.55 s.

It also writes a pool of `POOL_SIZE` (100) random walks in `sim/test_signals/pool/` (`step_000.wav`, ..., seed
`POOL_SEED`), the input of `simulate(..., signal_pool=...)`:

- 1 to 6 steps, first step at 0.3-1.0 s, 0.4-0.8 s between two steps, 0.3-1.0 s of silence after the last one,
  at least 2 s in total (one window of the network is 1.5 s);
- every step with its own amplitude (0.7-1.0), heel and toe decay, heel-toe delay (60-100 ms), toe amplitude
  (0.4-0.8 of the heel) and band (15-30 Hz to 300-450 Hz).

`pool/pool.json` lists the parameters of every signal (number of steps, onset and shape of every step). The pool is
replaced at every run and is the same with the same seed; the two fixed WAVs do not change.

## Conventions

- Blocks entering `detect` and segments entering `load_block_from_det` are chronological.
- Inside `BeamformingGccLs` the buffer `M` is reversed (column 0 = most recent sample).
- Sensor 1 (index 0) is the TDoA reference of `BeamformingGccLs`; `WaveletTdoaMult` uses the earliest sensor.
- Positions are in metres, with the origin in a corner of the room; the error is computed on x and y only.
- Sensors and sources are at the same height: the localization is 2D.

## Status and limitations

- The simulations are free field, with white noise, one propagation speed and the synthetic signals of the pool
  (the same pool for training, validation and test); reflections (`shoebox`), other speeds and real data are still
  to be tested.
- `BeamformingGccLs` needs the speed `c`; for floor vibrations it is unknown and frequency dependent (dispersion).
  It works only within about ±10 % of the true speed.
  The speed range of `WaveletTdoaMult` in the config is narrow around the true speed of the simulator.
- `WaveletTdoaMult`: the hyperbola sampling is not the limit (errors of millimetres); the limit is the integer-sample
  arrival times, so its error grows quickly with the distance of the source.
- TDoAs and arrival times are integer samples; sub-sample interpolation (arrival times of `WaveletTdoaMult`, then
  GCC peak of `BeamformingGccLs`) is the next step.
- The GCC weights whiten the spectrum over all the frequencies: limiting them to the band is still to be done.
- The network needs one training per sensor configuration and is trained only in the conditions of the test sets;
  its error grows near the corners of the room, which are little covered by the training positions.
  It works only with windows aligned as in training (see `CNNAttention`); planned fix: datasets simulated with the
  signal pool (steps at different times and with different shapes in every recording) and a new training.
- The CSVs of the campaign in `results/*.csv` and `results/summary/` predate `h_max`, the retraining of the networks
  and the median per window: run `./executables/eval.sh` again before using them. The newer comparisons are in
  `results/wv_k08_hmax/` and `results/cnn_median/`.
- No unit tests yet.
