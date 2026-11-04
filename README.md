# Footstep detection and localization

Detection and localization algorithms for an array of sensors, developed for a master thesis.
The goal is to localize people walking from the floor vibrations produced by their footsteps.
During development the algorithms are tested on data from an acoustic array simulator (git submodule), fed to
them block by block as if it were arriving in real time, and evaluated over campaigns of simulated datasets.

## Quick start

All the commands are run from the project root.

### 1. Setup (once)

```bash
git submodule update --init                 # simulator, if the repo was cloned without --recursive
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python test_signals/make_footstep.py   # footstep WAVs used as simulator input
```

### 2. Test campaign of the classic localizers

Everything at once: generates the missing datasets of `simulation/configs/general/`, evaluates every dataset with
every localizer in parallel and writes the comparison in `results/summary/`.

```bash
./eval_classic_alg.sh
```

The same steps one by one:

```bash
# one dataset: simulation/configs/general/<name>.toml -> data/general/<name>/
venv/bin/python simulation/generate_dataset.py simulation/configs/general/y_7m_center_snr20.toml [--overwrite]

# one localizer on one dataset: CSV in results/, figures in results/figures/
venv/bin/python evaluate.py data/general/y_7m_center_snr20 --loc bf_gcc_ls [--det amp_thr] [--out results/x.csv] [--config configs/default.toml]

# figures of an existing CSV again
venv/bin/python -m analysis.plot_results results/y_7m_center_snr20_bf_gcc_ls.csv data/general/y_7m_center_snr20

# summary table and comparison figures of all the CSVs of the campaign
venv/bin/python -m analysis.compare_runs
```

`evaluate.py --help` lists the available localizers (`--loc`) and detectors (`--det`).

### 3. Neural network localizer

Everything at once: for every array of `simulation/arrays/` with a test config in `simulation/configs/general/`,
generates its training data, trains one network (skipped if `checkpoints/cnn_attention_<array>.pt` exists) and tests
it on the datasets of the campaign with the same array, then writes the comparison with the classic localizers.

```bash
./eval_nn.sh
```

The same steps one by one, for one array:

```bash
# configs of the training and validation datasets: simulation/configs/{training,validation}/<array>_*.toml
venv/bin/python simulation/generate_nn_configs.py [--array y_7m_center]

# datasets: data/training/<name>/ and data/validation/<name>/
for f in simulation/configs/training/*.toml simulation/configs/validation/*.toml; do
    venv/bin/python simulation/generate_dataset.py "$f"
done

# .npz files for the network: training/data/training/ and training/data/validation/
venv/bin/python -m training.dataset --data data/training --out training/data/training
venv/bin/python -m training.dataset --data data/validation --out training/data/validation

# check of the targets built for the network (shapes, fraction of active frames)
venv/bin/python -m training.loader

# training: weights and loss of every epoch in the checkpoint of configs/train.toml
venv/bin/python -m training.train [--conf configs/train.toml]

# test on a dataset of the campaign with the same array (weights_path of configs/default.toml)
venv/bin/python evaluate.py data/general/y_7m_center_snr20 --loc cnn_attention --det raw
```

## Project structure

```
.
├── analysis/
│   ├── plot_results.py         # figures of one run (error map, error vs. distance, CDF)
│   └── compare_runs.py         # summary table and comparison figures of a whole campaign
├── configs/
│   ├── default.toml            # parameters of the algorithms, the stream, the event and the frequency band
│   └── train.toml              # parameters of the training of the neural network
├── core/
│   ├── data.py                 # Sample: loads a simulated recording, band-pass filters it, streams it in blocks
│   └── metrics.py              # error statistics shared by evaluate.py and the analysis scripts
├── detection/
│   ├── base.py                 # Detector: abstract base class
│   ├── amp_thr.py              # AmpThr: envelope detector with adaptive threshold
│   └── raw.py                  # Raw: fixed-length overlapping blocks, input of the neural network
├── localization/
│   ├── base.py                 # Localizer: abstract base class
│   ├── beamforming_gcc_ls.py   # BeamformingGccLs: beamforming + (generalized) cross-correlation + least squares
│   ├── wavelet_tdoa.py         # WaveletTdoaMult: wavelet arrival times + adaptive multilateration (unknown speed)
│   └── cnn_attention.py        # CNNAttention: neural network (CNN + self-attention) and encoding of the targets
├── simulation/
│   ├── arrays/                 # one TOML per array geometry ([sensors] only, no dataset is generated from them)
│   ├── configs/
│   │   ├── general/            # one TOML per test dataset (the experiment campaign, comparison of the methods)
│   │   ├── training/           # datasets used to train the neural network
│   │   └── validation/         # datasets used to validate the neural network
│   ├── generate_dataset.py     # runs the simulator engine without its GUI
│   └── generate_nn_configs.py  # writes the training/ and validation/ configs of the neural network
├── simulator/                  # git submodule: acustic-array-simulator
├── test_signals/               # synthetic footstep WAVs used as simulator input (git-ignored)
│   └── make_footstep.py
├── tests/                      # unit tests (to be written)
├── training/
│   ├── dataset.py              # converts the simulated datasets into .npz files for the neural network
│   ├── loader.py               # FootstepDataset: windows and per-frame targets for the training
│   ├── train.py                # training of the neural network
│   └── data/                   # generated training data, training/ and validation/ (git-ignored)
├── checkpoints/                # weights of the networks, loss of every epoch, configs/ of eval_nn.sh (git-ignored)
├── data/                       # generated datasets, general/, training/ and validation/ (git-ignored)
├── results/                    # CSVs, logs/, figures/ and summary/ (CSVs and figures git-ignored)
├── evaluate.py                 # runs a detector + localizer on one dataset
├── eval_classic_alg.sh         # generates the missing datasets, evaluates every dataset in parallel, compares
├── eval_nn.sh                  # for every array: training data, training and test of the network, compares
└── requirements.txt
```

Dependencies only go one way: `evaluate.py` uses `core`, `detection`, `localization` and `analysis`;
`detection` and `localization` may use `core`; `core` uses neither. `training` uses `core`, `detection` and
`localization/cnn_attention.py`.

## Workflow

```
simulation/configs/general/*.toml ─generate_dataset.py─► data/general/<exp>/ ─evaluate.py─► results/<exp>_<loc>.csv ─compare_runs.py─► results/summary/
```

Neural network:

```
simulation/arrays/<array>.toml ─generate_nn_configs.py─► simulation/configs/{training,validation}/*.toml
    ─generate_dataset.py─► data/{training,validation}/<name>/ ─training.dataset─► training/data/{training,validation}/<name>/*.npz
    ─training.train─► checkpoints/cnn_attention_<array>.pt ─evaluate.py --loc cnn_attention --det raw─► results/
```

The commands of every step are in [Quick start](#quick-start).

Without `--out` the CSV of `evaluate.py` is `results/<dataset>_<loc>.csv`.

Scripts inside a subfolder (`analysis/`, `training/`) import `core`, `detection` and `localization`, so they are run
as modules from the project root: `python -m training.dataset`, with dots and without `.py`. Run as
`python training/dataset.py` (or with the VS Code run button), Python looks for the imports in `training/` and fails
with `ModuleNotFoundError: No module named 'core'`. In VS Code use a `launch.json` configuration with
`"module": "training.dataset"` and `"cwd": "${workspaceFolder}"`.

### `eval_classic_alg.sh`

1. Creates the venv if missing, then generates every dataset of `simulation/configs/general/` that has no `manifest.json`.
2. Runs `evaluate.py` for every dataset and localizer, up to `nproc --all` processes at a time, with numpy limited to
   one thread per process (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`). Each run writes its CSV in `results/` and
   its output in `results/logs/<dataset>_<loc>.log`; the script prints `start` / `done` / `FAILED` for every run.
3. Prints how many CSVs are present and runs `analysis.compare_runs`.

The whole campaign (13 datasets x 2 localizers) takes about 15 minutes on 8 cores.

### `eval_nn.sh`

For every array of `simulation/arrays/` with at least one test config `simulation/configs/general/<array>_snr*.toml`:

1. Generates the missing test datasets of the array.
2. If `checkpoints/cnn_attention_<array>.pt` does not exist: empties the training folders
   (`simulation/configs/{training,validation}`, `data/{training,validation}`, `training/data/{training,validation}`,
   about 4 GB per array), runs `generate_nn_configs.py --array <array>`, `generate_dataset.py` on every config,
   `training.dataset` and `training.train` with a copy of `train.toml` (`checkpoints/configs/train_<array>.toml`).
   The weights are written in a `.tmp.pt` file renamed at the end, so an interrupted training is redone at the next
   call. Output in `results/logs/train_<array>.log`.
3. Runs `evaluate.py --loc cnn_attention --det raw` on every test dataset of the array, with a copy of `default.toml`
   whose `weights_path` is the checkpoint of the array (`checkpoints/configs/default_<array>.toml`); CSV in
   `results/<dataset>_cnn_attention.csv`, output in `results/logs/`.

A failed training or test prints `FAILED` and goes on with the next array. At the end the training folders are
emptied, the number of checkpoints and CSVs is printed and `analysis.compare_runs` compares all the localizers.
11 networks (one per array) are trained and tested on the 13 datasets of the campaign: `y_7m_center` is tested on
SNR 10, 20 and 30 dB. One training takes 5-25 minutes on a GPU.

## Simulation

`simulation/generate_dataset.py` calls the engine of the simulator (`simulator/selkd_simulator/engine.py`, only numpy
and scipy) and writes, for every source position, one JSON (geometry and ground truth) and one WAV per sensor, plus
`manifest.json` and a copy of the config. Options: `--out <folder>` (default `data/<config folder>/<config name>`, e.g. `data/general/y_7m_center_snr20`),
`--overwrite`. The configs are grouped by use: `general/` (test datasets of the campaign, used by
`eval_classic_alg.sh` and `eval_nn.sh`), `training/` and `validation/` (data of the neural network, written by
`generate_nn_configs.py`, not part of the campaign).

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

### Arrays and neural network datasets

`simulation/arrays/<array>.toml` contains only the `[sensors]` of one geometry of the campaign (11 arrays, the 3 SNR
of `y_7m_center` share one), copied from the configs of `general/`. No dataset is generated from them.

`simulation/generate_nn_configs.py [--array <array>]` (default `y_7m_center`) reads the sensors of the array and
writes, for every SNR of `SNRS` (10, 20, 30 dB):

- `training/<array>_random_snr<SNR>.toml`: `TRAINING_POSITIONS` (1000) random source positions in `AREA`
  (`[1, 49] x [1, 49]` m);
- `validation/<array>_grid_snr<SNR>.toml`: grid `VALIDATION_GRID` (1.25 to 46.25 m, step 5 m, 100 nodes), shifted
  w.r.t. the 5 m test grid so that no validation position is a test position.

Every file has its own seed (from `SEED_START`), so positions and noise are different in every dataset; with a grid
the seed only sets the noise. `fs`, `c`, `model`, `room`, `signal` and `z` are constants of the script, equal to the
test configs. The script stops if the sensors differ from the test configs `general/<array>_snr*.toml` or if a
validation node is closer than `MIN_SENSOR_DIST` (0.6 m) to a sensor.

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

### `detection/raw.py` - `Raw`

Accumulates the stream and returns every block of `input_size` samples, one every `stride` samples (overlapping if
`stride < input_size`), without any detection: the input of the neural network, which detects the steps itself.

### `localization/cnn_attention.py` - `CNNAttention`

Network `SELDNet`, from the SELD branch of *Multi-Task Learning for Sound Source Localization and Keyword Spotting on
Nano-Drones* (ETH): 4 convolutional blocks (Conv1d, batch norm, ReLU, max pooling, dropout, squeeze-and-excitation),
4 self-attention blocks (8 heads, residual + layer norm), 2 linear layers. Input `(B, N, L)`, output `(B, T, 3)` with
one frame every ~2400 samples (~150 ms at 16 kHz): `(dx, dy)` with tanh and `d` with ReLU.

- **Targets** (ACCDOA-like, one class, one source): `encode_pos` gives `(dx, dy, d / scale)`, with `(dx, dy)` the unit
  vector from the centroid of the sensors to the source and `d` their distance, multiplied by the activity of the
  frame, so an inactive frame is `(0, 0, 0)`; `decode_pos` is the inverse. A source in the centroid is `(1, 0, 0)`.
- **Detection**: a frame is an event if `‖(dx, dy)‖ > det_thr`; every such frame gives one estimate, so one block
  can give several estimates.
- **Weights**: `weights_path` (from the project root) is a checkpoint of `training/train.py`. The constructor stops
  if its number of sensors, `fs` or band differ from the current ones, and takes `input_size` and `dist_scale` from
  it; `load_block_from_det` accepts only blocks of `input_size` samples, so `[detection.raw] input_size` must be the
  same.

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

### `training/dataset.py`

Prepares the training data of the neural network. For every dataset of `--data` with a `manifest.json` it streams
every recording with `Sample.read(hop)` (same band-pass filter as `evaluate.py`) and labels every sample as active or
not with `AmpThr.compute_mask`, block by block as in streaming. In `<out>/<dataset>/` (default
`training/data/training/<dataset>/`) it writes:

- `<recording>.npz`: `signal` `(N, L)` float32, filtered; `active` `(L,)` bool, True where at least one sensor is
  above the threshold (no pre/post margin); `source_pos` `(3,)`; `sensor_pos` `(N, 3)`; `fs`.
  `L` is a multiple of `hop`: the last incomplete block is dropped.
- `index.csv`: one row per recording, `file, length, active_ratio, source_x, source_y, dist`
  (`dist` from the centroid of the sensors).
- `params.json`: band, `hop`, `pre_margin` and `amp_thr` parameters used, so that the data can be matched with the
  configuration used at inference.

Windows and per-frame targets are built at training time, so they can change without regenerating the files.

### `training/loader.py` - `FootstepDataset`

PyTorch `Dataset` on the folders written by `training/dataset.py` (every subfolder with an `index.csv`). All the
recordings are loaded in memory (about 2.6 GB for 3300 recordings) and cut in windows of `input_size` samples, one
every `window_stride` samples. Element `k`: `x` `(N, input_size)`, the filtered signal, and `y` `(T, 3)`: the
`active` mask of the window is split in `T` frames of `input_size // T` samples, a frame is active if more than
`frame_thr` of its samples are active, and its target is `encode_pos(source, active, centroid, dist_scale)`.
`output_frames(N, input_size)` gives `T` with a forward pass of the network. `python -m training.loader` checks the
validation set (shapes, one target, fraction of active frames).

### `training/train.py`

`--conf` (default `configs/train.toml`). Builds the training and validation `FootstepDataset`s (`shuffle` only for
training), `SELDNet` and Adam, and for every epoch trains and computes the validation loss without gradients.

- **Loss**: MSE on `(dx, dy)` of all the frames (direction and detection) + MSE on `d` of the active frames only.
- **Early stopping**: the checkpoint is written when the validation loss improves; the training stops after
  `patience` epochs without improvement.
- **Checkpoint**: `model_state` and the conditions of the training, used by `CNNAttention`: `N`, `fs`, `input_size`,
  `T`, `frame_thr`, `dist_scale`, `data_params` (the `params.json` of the data: band, `hop`, `amp_thr`), the data
  folders, epoch and validation loss. The loss of every epoch is written next to it (`.csv`).

## Configuration (`configs/default.toml`)

All lengths are **in samples**, so they depend on `fs`. Every key must match a constructor argument.

| Section | Keys |
|---|---|
| `[stream]` | `hop`: samples per block |
| `[localization.bf_gcc_ls]` | `L` (`"auto"` or integer), `Nw`, `cc_type` (`cc`, `roth`, `scot`, `phat`) |
| `[localization.wv_tdoa_mlt]` | `n_scales`, `v_min`, `v_max`, `v_count`, `h_range`, `h_steps`, `k` |
| `[localization.cnn_attention]` | `det_thr` (threshold on `‖(dx, dy)‖`), `weights_path` (checkpoint of `training/train.py`) |
| `[detection.amp_thr]` | `post_margin`, `window`, `learning_rate`, `k`, `min_thr` |
| `[detection.raw]` | `input_size` (must be the `input_size` of the training), `stride` |
| `[event]` | `pre_margin`: noise samples before every event, shared by detector and localizers |
| `[band]` | `low_cutoff`, `high_cutoff` [Hz], `filter_order`: band-pass filter and band of the localizers |

### Training (`configs/train.toml`)

| Section | Keys |
|---|---|
| `[data]` | `train_dir`, `val_dir`: folders written by `training/dataset.py` |
| `[window]` | `input_size` (multiple of 2400), `window_stride`, `frame_thr`, `dist_scale` [m] |
| `[training]` | `batch_size`, `lr` (Adam), `max_epochs`, `patience` (early stopping), `seed` |
| `[output]` | `checkpoint`: weights and metadata, the loss of every epoch is written with the `.csv` extension |

## Adding an algorithm

1. Create a file in `localization/` (or `detection/`) with a class derived from `Localizer` (or `Detector`).
2. Add it to `LOCALIZERS` (or `DETECTORS`) in `evaluate.py` with a short name.
3. Add a section `[localization.<name>]` (or `[detection.<name>]`) to the config with its parameters.
4. To include it in the campaign, add the name to `locs` in `eval_classic_alg.sh`.

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
- The neural network needs one training per array, since it does not receive the sensor positions; it is trained
  only on simulated data, with one footstep signal and one propagation speed.
- No unit tests yet.
