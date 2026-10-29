#!/bin/bash

set -e

if [ "$(basename "$PWD")" != "localization_algorithms" ]; then
    echo "Error: call this script from the project root."
    exit 1
fi

mkdir -p results/logs

if [ ! -d venv ]; then
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
fi
source venv/bin/activate

configs=(simulation/configs/*.toml)

for file in "${configs[@]}"; do
    name=$(basename "$file" .toml)
    if [ ! -f "data/$name/manifest.json" ]; then
        echo "[$(date +%H:%M:%S)] generating dataset $name"
        python3 simulation/generate_dataset.py "$file" --overwrite
    fi
done

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

locs=(bf_gcc_ls wv_tdoa_mlt)
total=$(( ${#configs[@]} * ${#locs[@]} ))
i=0

echo "[$(date +%H:%M:%S)] $total evaluations, up to $(nproc --all) in parallel"

for file in "${configs[@]}"; do
    name=$(basename "$file" .toml)
    for loc in "${locs[@]}"; do
        i=$((i + 1))
        csv=results/${name}_${loc}.csv
        log=results/logs/${name}_${loc}.log
        echo "[$(date +%H:%M:%S)] start  $i/$total: $name $loc"
        (
            if python3 evaluate.py "data/$name" --loc "$loc" --out "$csv" > "$log" 2>&1; then
                echo "[$(date +%H:%M:%S)] done   $i/$total: $name $loc"
            else
                echo "[$(date +%H:%M:%S)] FAILED $i/$total: $name $loc (see $log)"
            fi
        ) &
        if (( $(jobs -rp | wc -l) >= $(nproc --all) )); then
            wait -n || true
        fi
    done
done

wait

n_csv=0
for file in "${configs[@]}"; do
    name=$(basename "$file" .toml)
    for loc in "${locs[@]}"; do
        if [ -f "results/${name}_${loc}.csv" ]; then
            n_csv=$((n_csv + 1))
        fi
    done
done
echo "[$(date +%H:%M:%S)] finished: $n_csv/$total result CSVs present"

python3 -m analysis.compare_runs