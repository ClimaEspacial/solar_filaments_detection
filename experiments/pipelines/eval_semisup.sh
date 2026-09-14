#!/usr/bin/env bash
set -euo pipefail
cd /home/marchezi/python_projects/solar_filaments_detection
while pgrep -f "yolo11s_filament_detect_seg_semisup" > /dev/null 2>&1; do sleep 60; done
echo "=== treino terminou $(date +%H:%M) ==="
tr -d '\r' < work/logs/yolo_semisup.log | grep -iE "epochs completed|EarlyStopping" | tail -2
tail -1 work/runs/yolo11s_filament_detect_seg_semisup/results.csv | cut -d, -f1,12-15
echo "=== avaliacao no split real retido ==="
.venv/bin/python - <<'PY'
import pandas as pd, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.eval.self_eval import format_summary, get_overlap_df, summarize_overlap
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner
from solar_filaments.rle import encode_rle_counts
paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path); _, val_ids = load_split(paths.split_path)
unet = load_refiner(paths.ckpt_dir / "unet_best.pt", dev, use_edge_attention=UNetConfig().use_edge_attention)
ccfg = CropConfig(); COLS = ["filament_id", "segmentation_rle"]
gt = []
for img_id in val_ids:
    info = d.images_by_id[img_id]
    for k, a in enumerate(d.anns_by_image.get(img_id, []), 1):
        m = polygon_to_mask(a["segmentation"], info["height"], info["width"]).astype(bool)
        if m.any(): gt.append({"filament_id": f"{img_id}_{k}", "segmentation_rle": encode_rle_counts(m)})
gt_df = pd.DataFrame(gt, columns=COLS)
seen, uniq = set(), []
for i in val_ids:
    fn = d.images_by_id[i]["file_name"]
    if fn not in seen: seen.add(fn); uniq.append(fn)
imgs = {fn: load_image(paths.train_img_dir / fn) for fn in uniq}
for tag, name in (("baseline ", f"{YoloConfig().name}_seg"),
                  ("semi-sup ", f"{YoloConfig().name}_seg_semisup")):
    W = str(paths.runs_dir / name / "weights" / "best.pt")
    for conf in (0.20, 0.25, 0.30):
        det = YoloSegDetector(YOLO(W), conf_thres=conf, iou_thres=0.5, imgsz=2048)
        rows = []
        for fn in uniq:
            lm, _ = refine_candidates(imgs[fn], det.detect(imgs[fn]), unet, dev, crop_size=ccfg.crop_size,
                pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side, mask_thresh=0.5,
                min_instance_area=25, mask_nms_thresh=0.15, use_tta=True)
            stem = fn.rsplit(".", 1)[0]
            rows += [{"filament_id": f"{stem}_{i}", "segmentation_rle": encode_rle_counts(lm == i)}
                     for i in range(1, int(lm.max()) + 1) if (lm == i).any()]
        s = summarize_overlap(get_overlap_df(gt_df, pd.DataFrame(rows, columns=COLS)))
        print(f"{tag} conf={conf:.2f}  {format_summary(s)}", flush=True)
PY
echo "=== FIM ==="
