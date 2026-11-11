#!/bin/bash
# Study of the amount of training data (y_7m_10db, 3x positions); no file of the project is modified
set -e
cd /home/m4rch0/marco/master/master_thesis/localization_algorithms
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=$(pwd)
echo "[$(date +%H:%M)] generation"
venv/bin/python /tmp/claude-1000/-home-m4rch0-marco-master-master-thesis-localization-algorithms/ccfac90d-f5bb-4fe9-b9b2-f2efa4a71040/scratchpad/gen_x3.py > results/train_x3/gen.log 2>&1
echo "[$(date +%H:%M)] training"
cp -r training/log/y_7m_10db results/train_x3/log_campaign_backup
venv/bin/python -m training.train y_7m_10db --train_conf results/train_x3/train_x3.toml > results/train_x3/train.log 2>&1 || { rm -rf training/log/y_7m_10db; cp -r results/train_x3/log_campaign_backup training/log/y_7m_10db; exit 1; }
mv training/log/y_7m_10db results/train_x3/log
cp -r results/train_x3/log_campaign_backup training/log/y_7m_10db
echo "[$(date +%H:%M)] evaluation"
cp results/diag_train_error/y_7m_10db_*.csv results/train_x3/ 2>/dev/null || true
for s in validation test; do
  venv/bin/python -m executables.evaluate data/$s/y_7m_10db --loc cnn_attention --det raw --config results/train_x3/eval_x3.toml --out results/train_x3/x3_$s.csv > results/train_x3/eval_$s.log 2>&1 &
done
wait
echo "[$(date +%H:%M)] done"
