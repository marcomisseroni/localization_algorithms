Run x100, y_7m_10db, started 12/11/2026 16:42 (PID 2031323, tmux session train_x100)
Data: 145116 training recordings (VAL_SPLIT = 0.001761), validation and test the same as x10 / x30
Training config at launch (configs/train.toml was edited afterwards, so it is not copied here):
  input_size 24000, window_stride 4800 with random offset per recording in [0, 4800) drawn at every epoch,
  val_stride 2400 (fixed), frame_thr 0.5, dist_scale 70
  batch_size 128, lr 1e-3, max_epochs 200, patience 60, seed 0, ReduceLROnPlateau factor 0.1, patience_lr 24, min_lr 0
  NUM_WORKERS_TRAIN 12 (not persistent), NUM_WORKERS_VAL 2
Files copied when the run ended: checkpoints/y_7m_10db.pt, log/, eval/ (test and validation)
