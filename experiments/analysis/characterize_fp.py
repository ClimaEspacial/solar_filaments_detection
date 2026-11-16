"""O que sao os falsos positivos? Filamentos reais nao anotados, ou lixo?

Decide se a remocao de FP (a alavanca de maior retorno no PQ, +0.077 se total)
e atacavel por modelagem ou se e o teto da anotacao. Compara, para predicoes
SEM nenhuma sobreposicao com o GT, as mesmas estatisticas de forma e contraste
usadas nos filamentos verdadeiros.
"""
import sys, json, numpy as np, cv2, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, merge_annotator_records, polygon_to_mask
from solar_filaments.detectors.yolo import YoloSegDetector
from solar_filaments.inference import refine_candidates
from solar_filaments.models.unet import EdgeAttUNet
sys.path.insert(0, "experiments/multiwavelength")
from geom import fit_limb

N = int(sys.argv[1]) if len(sys.argv) > 1 else 35
paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path); _, val = load_split(paths.split_path)
files = sorted({d.images_by_id[i]["file_name"] for i in val if i in d.images_by_id})[:N]
unet = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(dev)
unet.load_state_dict(torch.load("weights/refiner_unet_edgeatt.pt", map_location=dev)); unet.eval()
icfg = InferenceConfig(conf_thres=0.25, imgsz=2048, mask_thresh=0.5, mask_nms_thresh=0.15)
ccfg = CropConfig()
yolo = YoloSegDetector(YOLO("weights/detector_yolo11seg_2048.pt"), conf_thres=0.25,
                       iou_thres=icfg.iou_thres, imgsz=2048)

def stats(img, m, disk):
    k = cv2.dilate(m.astype(np.uint8), np.ones((31,31), np.uint8))
    ring = (k>0) & (~m) & disk
    if ring.sum() < 50: return None
    width = 2*cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5).max()
    return dict(area=int(m.sum()), width=float(width), length=float(m.sum()/max(width,1)),
                rank=float((img[ring] < np.median(img[m])).mean()))

rows, crops = [], []
for n, fn in enumerate(files, 1):
    img = load_image(paths.train_img_dir / fn)
    cx, cy, r = fit_limb(img)
    yy, xx = np.mgrid[0:2048, 0:2048]
    rad = np.hypot(xx-cx, yy-cy); disk = rad < r*0.97
    gt = [polygon_to_mask(c["segmentation"],2048,2048).astype(bool)
          for c in merge_annotator_records(d, fn, "union")]
    gt_any = np.zeros((2048,2048), bool)
    for g in gt: gt_any |= g
    lm, sc = refine_candidates(img, yolo.detect(img), unet, dev, crop_size=ccfg.crop_size,
        pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side, mask_thresh=icfg.mask_thresh,
        min_instance_area=icfg.min_instance_area, mask_nms_thresh=icfg.mask_nms_thresh, use_tta=True)
    for i in range(1, int(lm.max())+1):
        p = lm == i
        if not p.any(): continue
        best = max(((g&p).sum()/max((g|p).sum(),1) for g in gt), default=0.0)
        kind = "TP" if best > 0.5 else ("FP0" if not (p & gt_any).any() else "FPpartial")
        s = stats(img, p, disk)
        if s is None: continue
        s.update(kind=kind, file=fn, score=float(sc[i-1]) if i-1 < len(sc) else np.nan,
                 r_frac=float(rad[p].mean()/r))
        rows.append(s)
        if kind == "FP0" and len(crops) < 24:
            ys, xs = np.where(p); y0,y1,x0,x1 = ys.min(),ys.max(),xs.min(),xs.max()
            pad = 40
            crops.append(img[max(0,y0-pad):y1+pad, max(0,x0-pad):x1+pad].copy())
    print(f"  {n}/{len(files)}", flush=True)
json.dump(rows, open("experiments/data/fp_stats.json","w"))
np.save("experiments/data/fp_crops.npy", np.array(crops, dtype=object), allow_pickle=True)
print(f"\n{len(rows)} instancias preditas -> experiments/data/fp_stats.json")
