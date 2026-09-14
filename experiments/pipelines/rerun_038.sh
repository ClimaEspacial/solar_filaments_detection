#!/usr/bin/env bash
# Reconstroi a configuracao que marcou 0.38: box@1024 + U-Net, alvo de anotador unico.
# Mesma semente e val-frac do modelo seg, entao o split e identico e os dois sao comparaveis.
set -euo pipefail
cd /home/marchezi/python_projects/solar_filaments_detection
B=.venv/bin/filament
L=work/logs/rerun038_$(date +%m%d_%H%M); mkdir -p "$L"; echo "logs em $L"

echo "[$(date +%H:%M:%S)] 1/4 prepare (alvo single, dataset de caixas)"
$B prepare --val-frac 0.05 --annotator-merge single > "$L/prepare.log" 2>&1

echo "[$(date +%H:%M:%S)] 2/4 train-yolo box@1024"
$B train-yolo --epochs 60 > "$L/yolo.log" 2>&1

echo "[$(date +%H:%M:%S)] 3/4 train-unet"
$B train-unet --epochs 40 > "$L/unet.log" 2>&1

echo "[$(date +%H:%M:%S)] 4/4 avaliacao local (conf 0.35 = a do 0.38, mais vizinhos)"
for c in 0.25 0.35 0.45; do
  echo "--- conf=$c ---"
  $B evaluate --conf-thres $c --mask-thresh 0.8 --tta --full 2>&1 | grep -E "PQ=" || true
done | tee "$L/evaluate.log"
echo "[$(date +%H:%M:%S)] COMPLETO"
