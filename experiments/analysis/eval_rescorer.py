"""O rescorer paga em PQ? Varredura do limiar nos dois pools de candidatos.

Pool A (so YOLO-seg): a referencia e PQ=0.4436. O rescorer tem que superar isso.
Pool B (YOLO-seg + semantico): sem filtro da PQ=0.3637 com recall muito maior
(GT perdido 9.7% vs 20.6%). O teto com filtro perfeito e ~0.460. Se o rescorer
chegar perto disso, a fonte semantica volta ao jogo; se nao, esta rota fecha.
"""
import argparse
import numpy as np, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.detectors.fulldisk import FullDiskSegmenter, load_flat_unet
from solar_filaments.eval.self_eval import (build_val_gt_pred_dataframes, format_summary,
                                             get_overlap_df, summarize_overlap)
from solar_filaments.inference import refine_candidates
from solar_filaments.models.unet import EdgeAttUNet
from solar_filaments.rescore import Rescorer

ap = argparse.ArgumentParser()
ap.add_argument("--conf-thres", type=float, default=0.15)
ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.20, 0.32, 0.45])
a = ap.parse_args()

paths, dev = Paths(), torch.device("cuda")
d = Dataset(paths.train_ann_path)
_, val_ids = load_split(paths.split_path)
unet = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(dev)
unet.load_state_dict(torch.load(paths.ckpt_dir / "unet_best.pt", map_location=dev)); unet.eval()
icfg = InferenceConfig(conf_thres=a.conf_thres, imgsz=2048, mask_thresh=0.8, mask_nms_thresh=0.15)
ccfg = CropConfig()
yolo = YoloSegDetector(YOLO(str(paths.runs_dir / f"{YoloConfig().name}_seg" / "weights" / "best.pt")),
                       conf_thres=a.conf_thres, iou_thres=icfg.iou_thres, imgsz=2048)
smod, si = load_flat_unet(str(paths.ckpt_dir / "flat_unet_best.pt"), dev)
sem = FullDiskSegmenter(smod, dev, imgsz=si)

RK = dict(crop_size=ccfg.crop_size, pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
          mask_thresh=icfg.mask_thresh, min_instance_area=icfg.min_instance_area,
          mask_nms_thresh=icfg.mask_nms_thresh, use_tta=True)

print(f"conf={a.conf_thres}  (limiar 0.0 = rescorer desligado)\n")
for pool in ("yolo", "yolo+sem"):
    for t in a.thresholds:
        rs = Rescorer.load(paths.ckpt_dir / "rescorer.pkl", threshold=t) if t > 0 else None
        def pf(path, rs=rs, pool=pool):
            img = load_image(path)
            prob = sem.probability_map(img)
            c = yolo.detect(img) + (sem.detect(img) if pool == "yolo+sem" else [])
            return refine_candidates(img, c, unet, dev, rescorer=rs, semantic_prob=prob, **RK)
        gt, pred = build_val_gt_pred_dataframes(d, paths.train_img_dir, val_ids, None, None, dev,
                                                 ccfg, icfg, predict_fn=pf)
        s = summarize_overlap(get_overlap_df(gt, pred))
        print(f"{pool:9s} limiar={t:<5} n_pred={len(pred):5d}  {format_summary(s)}", flush=True)
