#!/usr/bin/env bash
# Cadeia da noite de 2026-09-06, alvo `union`. Cada etapa so roda se a anterior
# terminou com sucesso, e tudo fica em log.
set -euo pipefail
cd /home/marchezi/python_projects/solar_filaments_detection
B=.venv/bin/filament
L=work/logs/night_$(date +%m%d_%H%M)
mkdir -p "$L"
echo "logs em $L"

echo "[$(date +%H:%M:%S)] 1/4 train-yolo seg@2048 union"
$B train-yolo --seg --imgsz 2048 --weights yolo11l-seg.pt --epochs 60 --batch 0 > "$L/yolo.log" 2>&1
echo "[$(date +%H:%M:%S)] yolo ok"

echo "[$(date +%H:%M:%S)] 2/4 train-unet 400 epocas"
$B train-unet --epochs 400 > "$L/unet.log" 2>&1
echo "[$(date +%H:%M:%S)] unet ok"

echo "[$(date +%H:%M:%S)] 3/4 train-fulldisk (segmentador semantico, all-data union)"
$B train-fulldisk --epochs 80 --annotator-merge union > "$L/fulldisk.log" 2>&1
echo "[$(date +%H:%M:%S)] fulldisk ok"

echo "[$(date +%H:%M:%S)] 4/4 avaliacao local do modelo union"
for c in 0.15 0.20 0.30; do
  echo "--- conf=$c ---"
  $B evaluate --detector yolo-seg --tta --imgsz 2048 --conf-thres $c \
     --mask-thresh 0.8 --mask-nms-thresh 0.15 --full 2>&1 | grep -E "PQ=" || true
done | tee "$L/evaluate.log"
echo "[$(date +%H:%M:%S)] CADEIA COMPLETA"
