import argparse

import pandas as pd
import torch
from ultralytics import YOLO

from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.inference import predict_instances
from solar_filaments.models.unet import EdgeAttUNet
from solar_filaments.rle import encode_rle_counts


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yolo-weights", default=None, help="default: work_dir/runs/<name>/weights/best.pt")
    parser.add_argument("--unet-weights", default=None, help="default: work_dir/checkpoints/unet_best.pt")
    parser.add_argument("--out", default=None, help="default: work_dir/submission.csv")
    parser.add_argument("--conf-thres", type=float, default=InferenceConfig().conf_thres)
    parser.add_argument("--iou-thres", type=float, default=InferenceConfig().iou_thres)
    parser.add_argument("--imgsz", type=int, default=InferenceConfig().imgsz)
    parser.add_argument("--mask-nms-thresh", type=float, default=InferenceConfig().mask_nms_thresh,
                         help="suppress a candidate mask this fraction covered by a higher-confidence instance")


def run(args: argparse.Namespace) -> None:
    paths = Paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_weights = args.yolo_weights or str(paths.runs_dir / YoloConfig().name / "weights" / "best.pt")
    unet_weights = args.unet_weights or str(paths.ckpt_dir / "unet_best.pt")
    out_path = args.out or str(paths.work_dir / "submission.csv")

    yolo_model = YOLO(yolo_weights)
    unet_model = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(device)
    unet_model.load_state_dict(torch.load(unet_weights, map_location=device))
    unet_model.eval()

    crop_cfg = CropConfig()
    inference_cfg = InferenceConfig(conf_thres=args.conf_thres, iou_thres=args.iou_thres, imgsz=args.imgsz,
                                     mask_nms_thresh=args.mask_nms_thresh)

    test_image_paths = sorted(paths.test_img_dir.glob("*.jpeg"))
    submission_rows = []

    for path in test_image_paths:
        image_id = path.stem
        label_map, _ = predict_instances(
            path, yolo_model, unet_model, device,
            conf_thres=inference_cfg.conf_thres, iou_thres=inference_cfg.iou_thres,
            imgsz=inference_cfg.imgsz, crop_size=crop_cfg.crop_size,
            pad_factor=crop_cfg.pad_factor, min_crop_side=crop_cfg.min_crop_side,
            mask_thresh=inference_cfg.mask_thresh, min_instance_area=inference_cfg.min_instance_area,
            mask_nms_thresh=inference_cfg.mask_nms_thresh,
        )
        for inst_id in range(1, int(label_map.max()) + 1):
            mask = label_map == inst_id
            if not mask.any():
                continue
            submission_rows.append({
                "filament_id": f"{image_id}_{inst_id}",
                "segmentation_rle": encode_rle_counts(mask),
            })

    submission_df = pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"])
    submission_df.to_csv(out_path, index=False)
    print(f"wrote {len(submission_df)} rows to {out_path}")
