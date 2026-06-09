#!/bin/bash
#set -e

#source /home/fxwu/anaconda3/bin/activate base

scripts=(
  "00_gen_data.py"
  "01_fwi_ms.py"
  "02_fwi_gau_ml.py"
  "03_fwi_lap_ml.py"
  "11_fwi_l1.py"
  "12_fwi_gau_loss.py"
  "13_fwi_lap_loss.py"
)

for script in "${scripts[@]}"; do
    echo "========================================"
    echo "Running $script at $(date)"
    echo "========================================"

    python "$script"

    echo "Sleeping to release GPU..."
    sleep 30
    nvidia-smi
done
