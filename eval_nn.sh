#!/bin/bash

set -e

if [ "$(basename "$PWD")" != "localization_algorithms" ]; then
    echo "Error: call this script from the project root."
    exit 1
fi

if [ ! -d venv ]; then
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
fi
source venv/bin/activate

mkdir -p checkpoints/configs
mkdir -p results/logs

# Training data of one array, emptied before every training
nn_dirs=(
    simulation/configs/training
    simulation/configs/validation
    data/training
    data/validation
    training/data/training
    training/data/validation
)

clean_nn_dirs() {
    for dir in "${nn_dirs[@]}"; do
        rm -rf "$dir"
        mkdir -p "$dir"
    done
}

# Data and training of the network of one array, the weights are written in $2.
# Inside a function called by an if, set -e is disabled: every step returns on error.
train_array() {
    local array=$1
    local ckpt=$2
    clean_nn_dirs
    python3 simulation/generate_nn_configs.py --array "$array" || return 1
    for file in simulation/configs/training/*.toml simulation/configs/validation/*.toml; do
        python3 simulation/generate_dataset.py "$file" || return 1
    done
    python3 -m training.dataset --data data/training --out training/data/training || return 1
    python3 -m training.dataset --data data/validation --out training/data/validation || return 1
    sed "s|^checkpoint = .*|checkpoint = \"$ckpt\"|" configs/train.toml > "checkpoints/configs/train_${array}.toml"
    python3 -m training.train --conf "checkpoints/configs/train_${array}.toml" || return 1
}

# Arrays of simulation/arrays/ with at least one test config in simulation/configs/general/
arrays=()
for file in simulation/arrays/*.toml; do
    array=$(basename "$file" .toml)
    if [ -n "$(ls simulation/configs/general/${array}_snr*.toml 2> /dev/null)" ]; then
        arrays+=("$array")
    else
        echo "[$(date +%H:%M:%S)] skip $array: no test config in simulation/configs/general/"
    fi
done

total=${#arrays[@]}
i=0
echo "[$(date +%H:%M:%S)] $total arrays"

for array in "${arrays[@]}"; do
    i=$((i + 1))
    ckpt=checkpoints/cnn_attention_${array}.pt
    log=results/logs/train_${array}.log

    # Test datasets of the array
    for file in simulation/configs/general/${array}_snr*.toml; do
        name=$(basename "$file" .toml)
        if [ ! -f "data/general/$name/manifest.json" ]; then
            echo "[$(date +%H:%M:%S)] generating dataset $name"
            python3 simulation/generate_dataset.py "$file" --overwrite
        fi
    done

    # Training, only if the array has no weights. The weights are written in a temporary file and renamed
    # at the end, so that an interrupted training is not taken as finished at the next call
    if [ -f "$ckpt" ]; then
        echo "[$(date +%H:%M:%S)] skip training $i/$total: $array ($ckpt present)"
    else
        echo "[$(date +%H:%M:%S)] start  training $i/$total: $array"
        tmp_ckpt=checkpoints/cnn_attention_${array}.tmp.pt
        if train_array "$array" "$tmp_ckpt" > "$log" 2>&1; then
            mv "$tmp_ckpt" "$ckpt"
            mv "checkpoints/cnn_attention_${array}.tmp.csv" "checkpoints/cnn_attention_${array}.csv"
            echo "[$(date +%H:%M:%S)] done   training $i/$total: $array"
        else
            echo "[$(date +%H:%M:%S)] FAILED training $i/$total: $array (see $log)"
            continue
        fi
    fi

    # Test on every dataset of the array, with a copy of the default config pointing to its weights
    eval_conf=checkpoints/configs/default_${array}.toml
    sed "s|^weights_path = .*|weights_path = \"$ckpt\"|" configs/default.toml > "$eval_conf"
    for file in simulation/configs/general/${array}_snr*.toml; do
        name=$(basename "$file" .toml)
        csv=results/${name}_cnn_attention.csv
        eval_log=results/logs/${name}_cnn_attention.log
        if python3 evaluate.py "data/general/$name" --loc cnn_attention --det raw --config "$eval_conf" --out "$csv" > "$eval_log" 2>&1; then
            echo "[$(date +%H:%M:%S)] done   test: $name"
        else
            echo "[$(date +%H:%M:%S)] FAILED test: $name (see $eval_log)"
        fi
    done
done

# The training data of the last array is not needed anymore
clean_nn_dirs

n_ckpt=0
n_csv=0
for array in "${arrays[@]}"; do
    if [ -f "checkpoints/cnn_attention_${array}.pt" ]; then
        n_ckpt=$((n_ckpt + 1))
    fi
done
n_tests=$(ls simulation/configs/general/*.toml | wc -l)
for file in simulation/configs/general/*.toml; do
    if [ -f "results/$(basename "$file" .toml)_cnn_attention.csv" ]; then
        n_csv=$((n_csv + 1))
    fi
done
echo "[$(date +%H:%M:%S)] finished: $n_ckpt/$total checkpoints, $n_csv/$n_tests result CSVs present"

python3 -m analysis.compare_runs
