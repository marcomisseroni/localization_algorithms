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

for file in data/training/*/dataset.json; do
    name=$(basename "$(dirname "$file")")
    python3 -m training.train "$name"
done

# Plots of every training and comparison of the configurations in training/log/
python3 -m training.plots
