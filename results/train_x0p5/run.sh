#!/bin/bash
# Study of the amount of training data (y_7m_10db, 0.5x positions); no file of the project is modified
set -e
cd /home/m4rch0/marco/master/master_thesis/localization_algorithms
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=$(pwd)
echo "[$(date +%H:%M)] training"
cp -r training/log/y_7m_10db results/train_x0p5/log_campaign_backup
venv/bin/python -m training.train y_7m_10db --train_conf results/train_x0p5/train_x0p5.toml > results/train_x0p5/train.log 2>&1 || { rm -rf training/log/y_7m_10db; cp -r results/train_x0p5/log_campaign_backup training/log/y_7m_10db; exit 1; }
mv training/log/y_7m_10db results/train_x0p5/log
cp -r results/train_x0p5/log_campaign_backup training/log/y_7m_10db
echo "[$(date +%H:%M)] evaluation"
for s in validation test; do
  venv/bin/python -m executables.evaluate data/$s/y_7m_10db --loc cnn_attention --det raw --config results/train_x0p5/eval_x0p5.toml --out results/train_x0p5/x0p5_$s.csv > results/train_x0p5/eval_$s.log 2>&1 &
done
wait
echo "[$(date +%H:%M)] done"
