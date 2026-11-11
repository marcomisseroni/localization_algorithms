Campaign with the signal pool (11/11), training 1x (1451 positions), checkpoints and training logs of all 6 configs.
Data-size study on y_7m_10db (same validation and test, one seed per point):
  0.5x: results/train_x0p5/   1x: results/diag_train_error/ (CSVs of these checkpoints)   3x: results/train_x3/
  data_size_curve_y7m.txt: median / P90 / radial / angle errors of the three points on validation and test.
Validation (seed 1) and test (seed 3) are regenerated identically by generate_dataset.py with the same pool and seeds.
