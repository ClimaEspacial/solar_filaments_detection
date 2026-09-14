"""Pseudo-rotulos por CONCORDANCIA entre duas fontes independentes.

Pseudo-rotulagem ingenua reforca o proprio vies: o modelo rotula o que ja acerta e nunca
os filamentos pequenos e fracos que perde -- exatamente o nosso gargalo. Aqui uma instancia
so vira rotulo se o detector E o segmentador semantico de disco inteiro concordarem
(IoU > 0.5). As duas fontes concordam em 46.9% dos casos na validacao, entao a intersecao
e de alta precisao, e o sinal e de co-treinamento, nao de auto-confirmacao.
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import _polygon_of_mask, load_image
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.detectors.fulldisk import FullDiskSegmenter, load_flat_unet
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/data/marchezi/gong_unlabeled")
ap.add_argument("--out", default="/data/marchezi/gong_pseudo")
ap.add_argument("--conf-thres", type=float, default=0.25)
ap.add_argument("--mask-thresh", type=float, default=0.5)
ap.add_argument("--agree-iou", type=float, default=0.5)
ap.add_argument("--min-instances", type=int, default=2,
                help="descarta imagens com menos que isto: sem filamento confirmado, o rotulo "
                     "vazio ensinaria o detector a nao detectar nada")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--manifest", default=None, help="CSV com n_pred e n_agreed por imagem")
a = ap.parse_args()

paths, dev = Paths(), torch.device("cuda")
out_img, out_lab = Path(a.out) / "images", Path(a.out) / "labels"
out_img.mkdir(parents=True, exist_ok=True); out_lab.mkdir(parents=True, exist_ok=True)

unet = load_refiner(paths.ckpt_dir / "unet_best.pt", dev, use_edge_attention=UNetConfig().use_edge_attention)
ccfg = CropConfig()
det = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                      conf_thres=a.conf_thres, iou_thres=0.5, imgsz=2048)
smod, si = load_flat_unet(str(paths.ckpt_dir / "flat_unet_best.pt"), dev)
sem = FullDiskSegmenter(smod, dev, imgsz=si)

files = sorted(Path(a.src).glob("*.jpg"))
if a.limit: files = files[:a.limit]
print(f"{len(files)} imagens nao rotuladas", flush=True)

kept_img, kept_inst, seen_inst = 0, 0, 0
manifest = []
for n, f in enumerate(files, 1):
    try:
        img = load_image(f)
    except Exception:
        continue
    if img.shape != (2048, 2048):
        continue
    lm, _ = refine_candidates(img, det.detect(img), unet, dev, crop_size=ccfg.crop_size,
        pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side, mask_thresh=a.mask_thresh,
        min_instance_area=25, mask_nms_thresh=0.15, use_tta=True)
    preds = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]
    sems = [c.coarse_mask for c in sem.detect(img) if c.coarse_mask is not None and c.coarse_mask.any()]
    seen_inst += len(preds)

    lines = []
    for m in preds:
        agree = any((m & s).sum() / max((m | s).sum(), 1) > a.agree_iou for s in sems)
        if not agree:
            continue
        poly = _polygon_of_mask(m)
        if len(poly) < 6:
            continue
        p = np.clip(np.asarray(poly, np.float32).reshape(-1, 2) / 2048.0, 0.0, 1.0)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in p.flatten()))
    manifest.append((f.stem, len(preds), len(lines)))
    if len(lines) >= a.min_instances:
        (out_lab / f"{f.stem}.txt").write_text("\n".join(lines))
        dst = out_img / f"{f.stem}.jpg"
        if not dst.exists():
            dst.symlink_to(f.resolve())
        kept_img += 1; kept_inst += len(lines)
    if n % 100 == 0:
        print(f"  {n}/{len(files)}  imagens mantidas={kept_img}  instancias={kept_inst}"
              f"  (concordancia {kept_inst/max(seen_inst,1):.1%})", flush=True)

print(f"\n{kept_img} imagens com pseudo-rotulo, {kept_inst} instancias")
print(f"taxa de concordancia detector-semantico: {kept_inst/max(seen_inst,1):.1%}")
print(f"media {kept_inst/max(kept_img,1):.2f} inst/img  (GT real union = 9.18)")
if a.manifest:
    import csv
    with open(a.manifest, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stem", "n_pred", "n_agreed"]); w.writerows(manifest)
    print(f"manifesto: {a.manifest}")
