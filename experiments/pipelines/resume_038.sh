#!/usr/bin/env bash
set -euo pipefail
cd /home/marchezi/python_projects/solar_filaments_detection
B=.venv/bin/filament
L=work/logs/rerun038_0907_1012
echo "[$(date +%H:%M:%S)] 3/4 train-unet (YOLO ja treinado: mAP50=0.702)"
$B train-unet --epochs 40 > "$L/unet.log" 2>&1
grep "best val dice" "$L/unet.log"
echo "[$(date +%H:%M:%S)] 4/4 avaliacao local"
for c in 0.25 0.35 0.45; do
  echo "--- conf=$c ---"
  $B evaluate --conf-thres $c --mask-thresh 0.8 --tta --full 2>&1 | grep -E "PQ=" || true
done | tee "$L/evaluate.log"
echo "[$(date +%H:%M:%S)] COMPLETO"
