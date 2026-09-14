"""Fase 3: as propostas semanticas encontram o que o detector YOLO-seg perde?

O gargalo medido nao e a qualidade das mascaras (SQ fixo em 0.67) nem o alvo de treino
sozinho - e que 31% dos filamentos UNANIMES nunca sao propostos. Se um modelo semantico
denso (Flat U-Net, componentes conexos) acerta uma fracao desses, a uniao das duas fontes
de proposta ataca o unico eixo com folga real. Se acertar quase nada, as duas fontes
falham juntas e o problema e a imagem, nao o detector.
"""
import argparse, collections
import numpy as np, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split, merge_annotator_records, polygon_to_mask
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.detectors.fulldisk import FullDiskSegmenter, load_flat_unet
from solar_filaments.inference import refine_candidates
from solar_filaments.models.unet import EdgeAttUNet

ap = argparse.ArgumentParser()
ap.add_argument("--conf-thres", type=float, default=0.20)
ap.add_argument("--sem-weights", default=None)
ap.add_argument("--prob-thres", type=float, default=0.5)
a = ap.parse_args()

paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path)
_, val_ids = load_split(paths.split_path)
files = sorted({d.images_by_id[i]["file_name"] for i in val_ids if i in d.images_by_id})

unet = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(dev)
unet.load_state_dict(torch.load(paths.ckpt_dir / "unet_best.pt", map_location=dev)); unet.eval()
icfg = InferenceConfig(conf_thres=a.conf_thres, imgsz=2048, mask_thresh=0.8, mask_nms_thresh=0.15)
ccfg = CropConfig()
yolo = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                       conf_thres=a.conf_thres, iou_thres=icfg.iou_thres, imgsz=2048)
sm, si = load_flat_unet(a.sem_weights or str(paths.ckpt_dir / "flat_unet_best.pt"), dev)
sem = FullDiskSegmenter(sm, dev, imgsz=si, prob_thres=a.prob_thres)

def iou_best(g, masks):
    return max(((g & m).sum() / max((g | m).sum(), 1) for m in masks), default=0.0)

st = collections.defaultdict(collections.Counter)
for n, fn in enumerate(files, 1):
    img = load_image(paths.train_img_dir / fn)
    lm, _ = refine_candidates(img, yolo.detect(img), unet, dev, crop_size=ccfg.crop_size,
                              pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
                              mask_thresh=icfg.mask_thresh, min_instance_area=icfg.min_instance_area,
                              mask_nms_thresh=icfg.mask_nms_thresh, use_tta=True)
    ym = [m for m in (lm == i for i in range(1, int(lm.max()) + 1)) if m.any()]
    sm_ = [c.coarse_mask for c in sem.detect(img) if c.coarse_mask is not None and c.coarse_mask.any()]

    for c in merge_annotator_records(d, fn, "union"):
        g = polygon_to_mask(c["segmentation"], img.shape[0], img.shape[1]).astype(bool)
        k = "unanime" if c["support"] >= 0.99 else ("alto" if c["support"] >= 0.5 else "baixo")
        y, s = iou_best(g, ym) > 0.5, iou_best(g, sm_) > 0.5
        st[k]["n"] += 1
        st[k]["yolo"] += y; st[k]["sem"] += s
        st[k]["ambos"] += (y and s); st[k]["so_sem"] += (s and not y); st[k]["nenhum"] += (not y and not s)
    if n % 10 == 0: print(f"  {n}/{len(files)}", flush=True)

print(f"\nsemantico: {a.sem_weights or 'flat_unet_best.pt'}  prob_thres={a.prob_thres}  conf={a.conf_thres}")
print(f"{'grupo':10s} {'n':>5s} {'YOLO':>7s} {'SEM':>7s} {'ambos':>7s} {'SO SEM':>8s} {'nenhum':>7s}")
tot = collections.Counter()
for k in ("unanime", "alto", "baixo"):
    c = st[k]; tot.update(c)
    if c["n"]:
        print(f"{k:10s} {c['n']:5d} {c['yolo']/c['n']:6.1%} {c['sem']/c['n']:6.1%} "
              f"{c['ambos']/c['n']:6.1%} {c['so_sem']/c['n']:7.1%} {c['nenhum']/c['n']:6.1%}")
print(f"{'TOTAL':10s} {tot['n']:5d} {tot['yolo']/tot['n']:6.1%} {tot['sem']/tot['n']:6.1%} "
      f"{tot['ambos']/tot['n']:6.1%} {tot['so_sem']/tot['n']:7.1%} {tot['nenhum']/tot['n']:6.1%}")
print(f"\nrecall da UNIAO das duas fontes: {(tot['yolo']+tot['so_sem'])/tot['n']:.1%} "
      f"(so YOLO: {tot['yolo']/tot['n']:.1%})")
