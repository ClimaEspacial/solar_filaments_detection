#!/usr/bin/env bash
# Semi-supervisionado v2: filtro de COMPLETUDE por imagem.
# v1 falhou porque rotulos faltantes viraram fundo (PQ 0.457 -> 0.424, GT perdido 22% -> 40%).
# Aqui so entram imagens cuja fracao de instancias confirmadas e alta -- proxy para
# "encontramos quase tudo nesta imagem" -- em vez de filtrar instancia a instancia.
set -euo pipefail
cd /home/marchezi/python_projects/solar_filaments_detection
B=.venv/bin/python

echo "[$(date +%H:%M)] 1/4 re-rotulando com manifesto por imagem"
$B experiments/semisupervised/pseudo_label.py --out /data/marchezi/gong_pseudo_v2 --min-instances 1 \
   --manifest work/pseudo_manifest.csv > work/logs/pseudo_v2.log 2>&1
tail -4 work/logs/pseudo_v2.log

echo "[$(date +%H:%M)] 2/4 filtrando por completude"
$B - <<'PY'
import csv, shutil
from pathlib import Path
rows = list(csv.DictReader(open("work/pseudo_manifest.csv")))
src, dst = Path("/data/marchezi/gong_pseudo_v2"), Path("/data/marchezi/gong_pseudo_complete")
for k in ("images", "labels"):
    d = dst / k
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
import numpy as np
r = [(x["stem"], int(x["n_pred"]), int(x["n_agreed"])) for x in rows]
ratio = np.array([a / p for _, p, a in r if p > 0])
print(f"razao de concordancia por imagem: mediana {np.median(ratio):.2f}, p75 {np.percentile(ratio,75):.2f}")
THR, MIN = 0.80, 4
kept = 0
for stem, p, a in r:
    if p == 0 or a / p < THR or a < MIN: continue
    lab = src / "labels" / f"{stem}.txt"
    img = src / "images" / f"{stem}.jpg"
    if not lab.exists() or not img.exists(): continue
    (dst / "labels" / lab.name).symlink_to(lab.resolve())
    (dst / "images" / img.name).symlink_to(img.resolve())
    kept += 1
print(f"criterio: concordancia >= {THR:.0%} E pelo menos {MIN} instancias -> {kept} imagens de {len(r)}")
PY

echo "[$(date +%H:%M)] 3/4 montando dataset"
$B experiments/semisupervised/build_semisup_dataset.py --pseudo /data/marchezi/gong_pseudo_complete \
   --out work/yolo_seg_semisup_v2 --ratio 99 2>&1 | tail -3

echo "[$(date +%H:%M)] 4/4 treinando"
.venv/bin/filament train-yolo --seg --imgsz 2048 --weights yolo11l-seg.pt --epochs 60 --batch 0 \
  --data work/yolo_seg_semisup_v2/data.yaml --name yolo11s_filament_detect_seg_semisup_v2 \
  > work/logs/yolo_semisup_v2.log 2>&1
echo "[$(date +%H:%M)] COMPLETO"
tr -d '\r' < work/logs/yolo_semisup_v2.log | grep -iE "epochs completed|EarlyStopping" | tail -2
