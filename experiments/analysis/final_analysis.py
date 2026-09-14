"""(1) Quanto do offset local->placar vem da composicao da validacao?
   (2) Recall do ensemble por tamanho - a decisao operacional."""
import collections
import numpy as np, pandas as pd, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import (Dataset, load_image, load_split, merge_annotator_records,
                                        polygon_to_mask)
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.detectors.fulldisk import FullDiskSegmenter, load_flat_unet
from solar_filaments.eval.self_eval import get_overlap_df, get_pq_score, summarize_overlap
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner
from solar_filaments.rle import encode_rle_counts

paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path); _, val_ids = load_split(paths.split_path)
unet = load_refiner(paths.ckpt_dir / "unet_best.pt", dev, use_edge_attention=UNetConfig().use_edge_attention)
icfg = InferenceConfig(conf_thres=0.20, imgsz=2048, mask_thresh=0.8, mask_nms_thresh=0.15); ccfg = CropConfig()
yolo = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                       conf_thres=0.20, iou_thres=0.5, imgsz=2048)
smod, si = load_flat_unet(str(paths.ckpt_dir / "flat_unet_best.pt"), dev)
sem = FullDiskSegmenter(smod, dev, imgsz=si)
RK = dict(crop_size=ccfg.crop_size, pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
          mask_thresh=icfg.mask_thresh, min_instance_area=icfg.min_instance_area,
          mask_nms_thresh=icfg.mask_nms_thresh, use_tta=True)

files = sorted({d.images_by_id[i]["file_name"] for i in val_ids if i in d.images_by_id})
preds, sizes = {}, {"yolo": [], "yolo+sem": []}
for fn in files:
    img = load_image(paths.train_img_dir / fn)
    c_y = yolo.detect(img)
    for tag, cands in (("yolo", c_y), ("yolo+sem", c_y + sem.detect(img))):
        lm, _ = refine_candidates(img, cands, unet, dev, **RK)
        ms = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]
        preds.setdefault(tag, {})[fn.rsplit(".", 1)[0]] = ms
        for c in merge_annotator_records(d, fn, "union"):
            g = polygon_to_mask(c["segmentation"], img.shape[0], img.shape[1]).astype(bool)
            best = max(((g & m).sum()/max((g | m).sum(),1) for m in ms), default=0.0)
            sizes[tag].append((g.sum(), best))

cols = ["filament_id", "segmentation_rle"]
def frames(tag, keep):
    gr, pr = [], []
    for img_id in val_ids:
        info = d.images_by_id[img_id]; fn = info["file_name"]
        if len(d.file_name_to_image_ids[fn]) not in keep:
            continue
        for k, ann in enumerate(d.anns_by_image.get(img_id, []), 1):
            m = polygon_to_mask(ann["segmentation"], info["height"], info["width"]).astype(bool)
            if m.any():
                gr.append({"filament_id": f"{img_id}_{k}", "segmentation_rle": encode_rle_counts(m)})
    seen = set()
    for img_id in val_ids:
        fn = d.images_by_id[img_id]["file_name"]; stem = fn.rsplit(".", 1)[0]
        if len(d.file_name_to_image_ids[fn]) not in keep or stem in seen:
            continue
        seen.add(stem)
        pr += [{"filament_id": f"{stem}_{i+1}", "segmentation_rle": encode_rle_counts(m)}
               for i, m in enumerate(preds[tag][stem])]
    return pd.DataFrame(gr, columns=cols), pd.DataFrame(pr, columns=cols)

print("=== 1. PQ por numero de anotadores da imagem (config yolo, conf 0.20) ===")
pq = {}
for name, keep in [("1 anotador", {1}), ("2+ anotadores", {2, 3}), ("todas", {1, 2, 3})]:
    g, p_ = frames("yolo", keep)
    pq[name] = get_pq_score(get_overlap_df(g, p_))
    print(f"  {name:16s} PQ={pq[name]:.4f}   ({len(g)} registros GT, {len(p_)} predicoes)")
w = 0.581  # fracao de imagens 1-anotador no dataset completo
print(f"\n  nossa validacao tem 68.6% de imagens 1-anotador; o dataset tem 58.1%")
print(f"  PQ reponderado para a composicao do dataset: "
      f"{w*pq['1 anotador'] + (1-w)*pq['2+ anotadores']:.4f}  (medido: {pq['todas']:.4f})")

print("\n=== 2. Recall por tamanho: YOLO sozinho vs + ramo semantico ===")
a_y = np.array(sizes["yolo"]); a_s = np.array(sizes["yolo+sem"])
q = np.percentile(a_y[:, 0], [20, 40, 60, 80])
lims = [(0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], 1e9)]
print(f"{'area (px)':22s} {'n':>4s} {'YOLO':>8s} {'+sem':>8s} {'ganho':>8s}")
for lo, hi in lims:
    sy = a_y[(a_y[:,0] >= lo) & (a_y[:,0] < hi)]; ss = a_s[(a_s[:,0] >= lo) & (a_s[:,0] < hi)]
    ry, rs = (sy[:,1] > 0.5).mean(), (ss[:,1] > 0.5).mean()
    print(f"{int(lo):>8d} - {int(min(hi, a_y[:,0].max())):<10d} {len(sy):4d} {ry:7.1%} {rs:7.1%} {rs-ry:+7.1%}")
print(f"\n{'TOTAL':22s} {len(a_y):4d} {(a_y[:,1]>0.5).mean():7.1%} {(a_s[:,1]>0.5).mean():7.1%} "
      f"{(a_s[:,1]>0.5).mean()-(a_y[:,1]>0.5).mean():+7.1%}")
for tag, arr in (("yolo", a_y), ("yolo+sem", a_s)):
    g, p_ = frames(tag, {1,2,3})
    s = summarize_overlap(get_overlap_df(g, p_))
    print(f"  {tag:9s} PQ={s['PQ']:.4f}  FP={s['FP']}  n_pred={len(p_)}")
