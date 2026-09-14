"""Encoder base vs ResNet18: o ganho de dice atravessa o corte de casamento do PQ?

Val dice mede pixels dentro de recortes que o detector ja acertou. PQ so paga acima de
IoU 0.5. O teste real e a faixa 0.40-0.50: 23 filamentos estavam ali, valendo +0.021 se
convertessem. Se o dice sobe e a faixa nao anda, o ganho foi em mascaras que ja casavam.
"""
import numpy as np, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, merge_annotator_records, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.eval.self_eval import format_summary, get_overlap_df, summarize_overlap
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner
from solar_filaments.rle import encode_rle_counts
import pandas as pd

paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path)
_, val_ids = load_split(paths.split_path)
icfg = InferenceConfig(conf_thres=0.20, imgsz=2048, mask_thresh=0.8, mask_nms_thresh=0.15)
ccfg = CropConfig()
yolo = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                       conf_thres=0.20, iou_thres=icfg.iou_thres, imgsz=2048)
RK = dict(crop_size=ccfg.crop_size, pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
          mask_thresh=icfg.mask_thresh, min_instance_area=icfg.min_instance_area,
          mask_nms_thresh=icfg.mask_nms_thresh, use_tta=True)

def gt_rows(img_id):
    info = d.images_by_id[img_id]
    out = []
    for k, ann in enumerate(d.anns_by_image.get(img_id, []), start=1):
        m = polygon_to_mask(ann["segmentation"], info["height"], info["width"]).astype(bool)
        if m.any():
            out.append({"filament_id": f"{img_id}_{k}", "segmentation_rle": encode_rle_counts(m)})
    return out

for name in ("unet_best.pt", "unet_r18_best.pt"):
    unet = load_refiner(paths.ckpt_dir / name, dev, use_edge_attention=UNetConfig().use_edge_attention)
    gr, pr, best = [], {}, []
    for img_id in val_ids:
        info = d.images_by_id[img_id]
        stem = info["file_name"].rsplit(".", 1)[0]
        gr.extend(gt_rows(img_id))
        if stem in pr:
            continue
        img = load_image(paths.train_img_dir / info["file_name"])
        lm, _ = refine_candidates(img, yolo.detect(img), unet, dev, **RK)
        masks = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]
        pr[stem] = [{"filament_id": f"{stem}_{i+1}", "segmentation_rle": encode_rle_counts(m)}
                    for i, m in enumerate(masks)]
        for c in merge_annotator_records(d, info["file_name"], "union"):
            g = polygon_to_mask(c["segmentation"], img.shape[0], img.shape[1]).astype(bool)
            best.append(max(((g & m).sum() / max((g | m).sum(), 1) for m in masks), default=0.0))
    cols = ["filament_id", "segmentation_rle"]
    s = summarize_overlap(get_overlap_df(pd.DataFrame(gr, columns=cols),
                                          pd.DataFrame([r for v in pr.values() for r in v], columns=cols)))
    b = np.array(best)
    print(f"\n=== {name} ===")
    print(format_summary(s))
    print(f"  casados (IoU>0.5): {(b > 0.5).mean():.1%} de {len(b)} filamentos")
    for lo, hi, tag in [(0.0, 0.01, "sem sobreposicao"), (0.01, 0.40, "longe"),
                        (0.40, 0.50, "QUASE LA"), (0.50, 1.01, "casados")]:
        k = ((b >= lo) & (b < hi)).sum()
        print(f"  IoU {lo:.2f}-{hi:.2f} {tag:18s}: {k:4d}")
