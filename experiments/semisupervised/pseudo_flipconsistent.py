"""Pseudo-rotulos por INVARIANCIA A ESPELHAMENTO, nao por concordancia entre modelos.

A v2 selecionou imagens onde o detector e o segmentador semantico concordam -- que sao,
por construcao, as imagens FACEIS. O SQ subiu para 0.690 (o mais alto do projeto) e o
reconhecimento caiu, assinatura de deslocamento de distribuicao.

Aqui o criterio e outro: uma instancia so entra se o MESMO detector a produz na imagem
original e na espelhada horizontalmente. O disco solar nao tem razao fisica para quebrar
essa simetria, entao instabilidade sob espelhamento indica predicao nao confiavel -- sem
selecionar por facilidade.
"""
import argparse
from pathlib import Path
import numpy as np, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import _polygon_of_mask, load_image
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.inference import refine_candidates
from solar_filaments.models.refiner import load_refiner

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/data/marchezi/gong_unlabeled")
ap.add_argument("--out", default="/data/marchezi/gong_pseudo_flip")
ap.add_argument("--conf-thres", type=float, default=0.20)
ap.add_argument("--consist-iou", type=float, default=0.5)
ap.add_argument("--min-instances", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

paths, dev = Paths(), torch.device("cuda")
oi, ol = Path(a.out) / "images", Path(a.out) / "labels"
oi.mkdir(parents=True, exist_ok=True); ol.mkdir(parents=True, exist_ok=True)
unet = load_refiner(paths.ckpt_dir / "unet_best.pt", dev, use_edge_attention=UNetConfig().use_edge_attention)
ccfg = CropConfig()
det = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                      conf_thres=a.conf_thres, iou_thres=0.5, imgsz=2048)
RK = dict(crop_size=ccfg.crop_size, pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
          mask_thresh=0.5, min_instance_area=25, mask_nms_thresh=0.15, use_tta=True)

def instances(img):
    lm, _ = refine_candidates(img, det.detect(img), unet, dev, **RK)
    return [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]

files = sorted(Path(a.src).glob("*.jpg"))
if a.limit: files = files[:a.limit]
print(f"{len(files)} imagens", flush=True)
kept_img = kept_inst = seen = 0
for n, f in enumerate(files, 1):
    try: img = load_image(f)
    except Exception: continue
    if img.shape != (2048, 2048): continue
    direct = instances(img)
    mirrored = [m[:, ::-1] for m in instances(img[:, ::-1].copy())]   # desfaz o espelho
    seen += len(direct)
    lines = []
    for m in direct:
        if not any((m & q).sum() / max((m | q).sum(), 1) > a.consist_iou for q in mirrored):
            continue
        poly = _polygon_of_mask(m)
        if len(poly) < 6: continue
        p = np.clip(np.asarray(poly, np.float32).reshape(-1, 2) / 2048.0, 0.0, 1.0)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in p.flatten()))
    if len(lines) >= a.min_instances:
        (ol / f"{f.stem}.txt").write_text("\n".join(lines))
        d = oi / f"{f.stem}.jpg"
        if not d.exists(): d.symlink_to(f.resolve())
        kept_img += 1; kept_inst += len(lines)
    if n % 200 == 0:
        print(f"  {n}/{len(files)} imgs={kept_img} inst={kept_inst} "
              f"estabilidade={kept_inst/max(seen,1):.1%}", flush=True)
print(f"\n{kept_img} imagens, {kept_inst} instancias, {kept_inst/max(kept_img,1):.2f} inst/img")
print(f"estabilidade sob espelhamento: {kept_inst/max(seen,1):.1%}  (GT real = 9.18 inst/img)")
