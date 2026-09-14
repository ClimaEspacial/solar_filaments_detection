"""PQ da uniao de propostas YOLO-seg + semantico denso.

A complementaridade medida e grande (+10.5 pontos de recall), mas o retreino `union`
mostrou que recall a mais nao vale nada se os FP crescerem junto. Isto mede o PQ real.
O peso do semantico reescala a confianca dessa fonte antes do ranqueamento: peso baixo
faz ela so preencher buracos, peso alto a deixa competir de igual para igual.
"""
import argparse
import numpy as np, torch
from ultralytics import YOLO
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_split
from solar_filaments.detectors import YoloSegDetector
from solar_filaments.detectors.fulldisk import FullDiskSegmenter, load_flat_unet
from solar_filaments.eval.self_eval import (build_val_gt_pred_dataframes_ensemble, format_summary,
                                             get_overlap_df, summarize_overlap)
from solar_filaments.inference import refine_candidates
from solar_filaments.models.unet import EdgeAttUNet

ap = argparse.ArgumentParser()
ap.add_argument("--conf-thres", type=float, default=0.20)
ap.add_argument("--prob-thres", type=float, default=0.5)
ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.8])
ap.add_argument("--mask-nms-thresh", type=float, default=0.15)
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
sm, si = load_flat_unet(str(paths.ckpt_dir / "flat_unet_best.pt"), dev)
sem = FullDiskSegmenter(sm, dev, imgsz=si, prob_thres=a.prob_thres)

RK = dict(crop_size=ccfg.crop_size, pad_factor=ccfg.pad_factor, min_crop_side=ccfg.min_crop_side,
          mask_thresh=icfg.mask_thresh, min_instance_area=icfg.min_instance_area,
          mask_nms_thresh=icfg.mask_nms_thresh, use_tta=True)

print(f"conf={a.conf_thres} prob_thres={a.prob_thres}  (peso 0.0 = so YOLO, a referencia)")
for w in a.weights:
    from solar_filaments.data.coco import load_image
    def make_sources(path, w=w):
        img = load_image(path)
        src = [(lambda i=img: refine_candidates(i, yolo.detect(i), unet, dev, **RK), 1.0)]
        if w > 0:
            src.append((lambda i=img: refine_candidates(i, sem.detect(i), unet, dev, **RK), w))
        return src
    gt, pred = build_val_gt_pred_dataframes_ensemble(
        d, paths.train_img_dir, val_ids, make_sources,
        {"mask_nms_thresh": a.mask_nms_thresh, "min_instance_area": icfg.min_instance_area})
    s = summarize_overlap(get_overlap_df(gt, pred))
    print(f"peso_sem={w:<4} n_pred={len(pred):5d}  {format_summary(s)}", flush=True)
