"""Premissa: os filamentos que o modelo PERDE em H-alfa sao visiveis em AIA 304?

Para cada filamento GT da validacao mede-se (a) o melhor IoU com as predicoes
e (b) o contraste contra um anel local de fundo, nos dois canais. Se os
perdidos tem contraste ~0 no GONG mas contraste real no 304, um segundo canal
acrescenta informacao que a imagem H-alfa nao contem.
"""
import os, sys, json, numpy as np, cv2, torch
S = os.path.dirname(os.path.abspath(__file__))   # este diretorio
OUT = os.path.join("experiments", "data")
sys.path.insert(0, S)
from geom import fit_limb, ts_of, aia_matched
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, merge_annotator_records, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.inference import refine_candidates
from solar_filaments.models.unet import EdgeAttUNet

N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 15
paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path)
_, val_ids = load_split(paths.split_path)
files = sorted({d.images_by_id[i]["file_name"] for i in val_ids if i in d.images_by_id})
files = files[:: max(1, len(files) // N_FRAMES)][:N_FRAMES]

unet = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(dev)
unet.load_state_dict(torch.load("weights/refiner_unet_edgeatt.pt", map_location=dev)); unet.eval()
icfg = InferenceConfig(conf_thres=0.25, imgsz=2048, mask_thresh=0.5, mask_nms_thresh=0.15)
ccfg = CropConfig()
yolo = YoloSegDetector(YOLO("weights/detector_yolo11seg_2048.pt"),
                       conf_thres=0.25, iou_thres=icfg.iou_thres, imgsz=2048)

def contrast(img, m, disk):
    """(fundo - filamento) / sigma_local, usando um anel ao redor da estrutura."""
    k = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8))
    ring = (k > 0) & (~m) & disk
    if ring.sum() < 50 or m.sum() < 10:
        return np.nan
    bg = img[ring].astype(float)
    return (bg.mean() - img[m].astype(float).mean()) / max(bg.std(), 1e-6)


def darkrank(img, m, disk):
    """Fracao dos pixels do anel de fundo que sao MAIS ESCUROS que a mediana do
    filamento. 0.5 = indistinguivel; perto de 0 = filamento claramente escuro.
    Invariante a qualquer transformacao monotonica (paleta de cores, gamma)."""
    k = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8))
    ring = (k > 0) & (~m) & disk
    if ring.sum() < 50 or m.sum() < 10:
        return np.nan
    return float((img[ring] < np.median(img[m])).mean())

rows = []
for n, fn in enumerate(files, 1):
    img = load_image(paths.train_img_dir / fn)
    try:
        cx, cy, r = fit_limb(img if img.ndim == 2 else img[..., 0])
        aia, meta = aia_matched(ts_of(fn), cx, cy, r, 2048)
        if aia is None: raise RuntimeError("sem AIA")
    except Exception as e:
        print(f"  {fn}: AIA falhou ({e})", flush=True); continue
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    yy, xx = np.mgrid[0:2048, 0:2048]; disk = np.hypot(xx - cx, yy - cy) < r * 0.97

    lm, _ = refine_candidates(img, yolo.detect(img), unet, dev, crop_size=ccfg.crop_size,
                              pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
                              mask_thresh=icfg.mask_thresh, min_instance_area=icfg.min_instance_area,
                              mask_nms_thresh=icfg.mask_nms_thresh, use_tta=True)
    pm = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]

    for c in merge_annotator_records(d, fn, "union"):
        g = polygon_to_mask(c["segmentation"], 2048, 2048).astype(bool)
        iou = max(((g & p).sum() / max((g | p).sum(), 1) for p in pm), default=0.0)
        rows.append(dict(file=fn, area=int(g.sum()), iou=float(iou),
                         c_ha=contrast(gray, g, disk), c_304=contrast(aia, g, disk),
                         r_ha=darkrank(gray, g, disk), r_304=darkrank(aia, g, disk)))
    print(f"  {n}/{len(files)} {fn} ({meta['date']})", flush=True)

os.makedirs(OUT, exist_ok=True)
json.dump(rows, open(os.path.join(OUT, "premise2.json"), "w"))
print(f"\n{len(rows)} filamentos medidos -> {OUT}/premise2.json")
