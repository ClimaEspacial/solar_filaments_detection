"""Shared CLI plumbing for the detector backends (`--detector yolo-seg|maskrcnn`),
used by evaluate and predict so the two never drift apart."""
import argparse
from typing import Callable, Optional

import torch
from ultralytics import YOLO

from solar_filaments.config import MaskRCNNConfig, Paths, YoloConfig
from solar_filaments.detectors import MaskRCNNDetector, YoloSegDetector, build_maskrcnn
from solar_filaments.inference import predict_instances_from_candidates

DETECTOR_CHOICES = ("yolo", "yolo-seg", "maskrcnn")


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", choices=DETECTOR_CHOICES, default="yolo",
                         help="detector backend feeding the U-Net refiner: 'yolo' = the original axis-aligned "
                              "box detector (default, unchanged code path); 'yolo-seg' = YOLO11 instance "
                              "segmentation (from `train-yolo --seg`); 'maskrcnn' = torchvision Mask R-CNN "
                              "(from `train-maskrcnn`)")
    parser.add_argument("--maskrcnn-weights", default=None,
                         help="default: work_dir/checkpoints/maskrcnn[_fold<N>]_best.pt")
    parser.add_argument("--score-thres", type=float, default=MaskRCNNConfig().score_thres,
                         help="maskrcnn: min detection score (the analogue of --conf-thres)")
    parser.add_argument("--fallback-coarse", action="store_true",
                         help="yolo-seg/maskrcnn: if the refiner returns an empty mask for a candidate, claim the "
                              "detector's own coarse mask instead (protects recall on faint filaments)")
    parser.add_argument("--no-refine", action="store_true",
                         help="yolo-seg/maskrcnn: skip the U-Net entirely and score the detector's coarse masks "
                              "(ablation: how much does the refiner add?)")


def maskrcnn_weights_path(paths: Paths, fold: Optional[int]) -> str:
    name = f"maskrcnn_fold{fold}_best.pt" if fold is not None else "maskrcnn_best.pt"
    return str(paths.ckpt_dir / name)


def load_maskrcnn(weights: str, device) -> torch.nn.Module:
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = build_maskrcnn(pretrained=False, imgsz=ckpt["imgsz"], anchor_sizes=ckpt["anchor_sizes"],
                           aspect_ratios=ckpt["aspect_ratios"])
    model.load_state_dict(ckpt["model"])
    return model


def build_detector(args: argparse.Namespace, paths: Paths, device, fold: Optional[int], inference_cfg):
    if args.detector == "yolo-seg":
        base = f"{YoloConfig().name}_seg"
        name = f"{base}_fold{fold}" if fold is not None else base
        weights = args.yolo_weights or str(paths.runs_dir / name / "weights" / "best.pt")
        return YoloSegDetector(YOLO(weights), conf_thres=inference_cfg.conf_thres,
                               iou_thres=inference_cfg.iou_thres, imgsz=inference_cfg.imgsz)
    if args.detector == "maskrcnn":
        weights = args.maskrcnn_weights or maskrcnn_weights_path(paths, fold)
        ckpt_imgsz = torch.load(weights, map_location="cpu", weights_only=False)["imgsz"]
        return MaskRCNNDetector(load_maskrcnn(weights, device), device, score_thres=args.score_thres,
                                imgsz=ckpt_imgsz)
    raise ValueError(f"no backend object for --detector {args.detector}")


def make_predict_fn(args: argparse.Namespace, detector, unet_model, device, crop_cfg, inference_cfg,
                    preprocess: bool = False, use_tta: bool = False) -> Callable:
    """image_path -> (label_map, scores), for eval/self_eval.build_val_gt_pred_dataframes(predict_fn=...)."""
    def predict_fn(image_path):
        return predict_instances_from_candidates(
            image_path, detector, unet_model, device, preprocess=preprocess, refine=not args.no_refine,
            crop_size=crop_cfg.crop_size, pad_factor=crop_cfg.pad_factor, min_crop_side=crop_cfg.min_crop_side,
            mask_thresh=inference_cfg.mask_thresh, min_instance_area=inference_cfg.min_instance_area,
            mask_nms_thresh=inference_cfg.mask_nms_thresh, mask_conf_thres=inference_cfg.mask_conf_thres,
            use_tta=use_tta, fallback_coarse=args.fallback_coarse)
    return predict_fn
