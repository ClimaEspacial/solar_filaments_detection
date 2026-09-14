#!/usr/bin/env bash
set -euo pipefail
cd /home/marchezi/python_projects/solar_filaments_detection
echo "[$(date +%H:%M)] 1/3 pseudo-rotulos por invariancia a espelhamento"
.venv/bin/python experiments/semisupervised/pseudo_flipconsistent.py > work/logs/pseudo_v3.log 2>&1
tail -3 work/logs/pseudo_v3.log
echo "[$(date +%H:%M)] 2/3 dataset"
.venv/bin/python experiments/semisupervised/build_semisup_dataset.py --pseudo /data/marchezi/gong_pseudo_flip \
  --out work/yolo_seg_semisup_v3 --ratio 99 2>&1 | tail -3
echo "[$(date +%H:%M)] 3/3 treino"
.venv/bin/filament train-yolo --seg --imgsz 2048 --weights yolo11l-seg.pt --epochs 60 --batch 0 \
  --data work/yolo_seg_semisup_v3/data.yaml --name yolo11s_filament_detect_seg_semisup_v3 \
  > work/logs/yolo_semisup_v3.log 2>&1
echo "[$(date +%H:%M)] COMPLETO"
