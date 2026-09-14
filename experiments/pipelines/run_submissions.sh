#!/usr/bin/env bash
# Gera as submissoes do modelo all-data (YOLO11l-seg @2048 + refinador U-Net + TTA).
# Um arquivo por threshold, nomes distintos - nada sobrescreve nada.
set -euo pipefail

for c in 0.15 0.20 0.30; do
  out="work/sub_alldata_seg2048_conf${c#0.}.csv"
  echo "=== conf=$c -> $out ==="
  filament predict --detector yolo-seg --tta --imgsz 2048 --conf-thres "$c" --mask-thresh 0.8 --mask-nms-thresh 0.15 --out "$out"
done

echo
echo "=== resumo ==="
for f in work/sub_alldata_seg2048_conf*.csv; do
  echo "$f: $(($(wc -l < "$f") - 1)) linhas"
done
