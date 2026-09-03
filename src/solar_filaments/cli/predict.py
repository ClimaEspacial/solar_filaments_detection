import argparse

import pandas as pd
import torch
from ultralytics import YOLO

from solar_filaments.cli._backends import add_backend_args, build_detector, make_predict_fn
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.inference import predict_instances, predict_instances_obb
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
    parser.add_argument("--mask-conf-thres", type=float, default=InferenceConfig().mask_conf_thres,
                         help="min mean U-Net probability inside a candidate's own mask; 0 disables")
    parser.add_argument("--spine-orientation", action="store_true",
                         help="use a spine-oriented crop/U-Net pair instead of axis-aligned "
                              "(default unet-weights becomes unet_spine_best.pt)")
    parser.add_argument("--min-elongation", type=float, default=InferenceConfig().min_elongation,
                         help="spine-orientation only: min PCA elongation of a rough mask to trust its angle")
    parser.add_argument("--obb", action="store_true",
                         help="use an OBB-trained detector + matching U-Net (single-pass rotated crop, "
                              "no bootstrap) instead of axis-aligned - default weights become "
                              "<yolo-name>_obb / unet_obb_best.pt")
    parser.add_argument("--instance-norm", action="store_true",
                         help="load a U-Net trained with --instance-norm (default unet-weights becomes "
                              "*_in_best.pt) and construct it with matching InstanceNorm layers")
    parser.add_argument("--normalize", action="store_true",
                         help="use limb-darkening-flattened + CLAHE-preprocessed images at inference and load "
                              "*_norm-suffixed weights - not combinable with --obb this round")
    parser.add_argument("--tta", action="store_true",
                         help="average the U-Net's prediction over 4 flips (identity/h/v/hv), only trusting the "
                              "average when it doesn't change the single-view mask's topology (area/Dice/"
                              "component-count guards) - inference-only, no retraining, no separate weights")
    add_backend_args(parser)
    parser.add_argument("--fold", type=int, default=None,
                         help="use this fold's weights (from `filament prepare --k-folds`); "
                              "default: no fold suffix (the single-split model)")


def run(args: argparse.Namespace) -> None:
    if args.normalize and args.obb:
        raise ValueError("--normalize and --obb can't be combined this round")

    paths = Paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_name = f"{YoloConfig().name}_obb" if args.obb else (f"{YoloConfig().name}_norm" if args.normalize else YoloConfig().name)
    yolo_name = f"{base_name}_fold{args.fold}" if args.fold is not None else base_name
    unet_stem = "unet_obb" if args.obb else ("unet_spine" if args.spine_orientation else "unet")
    if args.normalize:
        unet_stem += "_norm"
    if args.instance_norm:
        unet_stem += "_in"
    default_unet_name = f"{unet_stem}_fold{args.fold}_best.pt" if args.fold is not None else f"{unet_stem}_best.pt"
    yolo_weights = args.yolo_weights or str(paths.runs_dir / yolo_name / "weights" / "best.pt")
    unet_weights = args.unet_weights or str(paths.ckpt_dir / default_unet_name)
    out_path = args.out or str(paths.work_dir / "submission.csv")

    norm_layer = "instance" if args.instance_norm else "batch"
    unet_model = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention, norm_layer=norm_layer).to(device)
    unet_model.load_state_dict(torch.load(unet_weights, map_location=device))
    unet_model.eval()

    crop_cfg = CropConfig(use_spine_orientation=args.spine_orientation, use_obb_orientation=args.obb)
    inference_cfg = InferenceConfig(conf_thres=args.conf_thres, iou_thres=args.iou_thres, imgsz=args.imgsz,
                                     mask_nms_thresh=args.mask_nms_thresh, mask_conf_thres=args.mask_conf_thres,
                                     min_elongation=args.min_elongation)

    if args.detector == "yolo":
        yolo_model = YOLO(yolo_weights)
        backend_predict = None
    else:
        detector = build_detector(args, paths, device, args.fold, inference_cfg)
        backend_predict = make_predict_fn(args, detector, unet_model, device, crop_cfg, inference_cfg,
                                          preprocess=args.normalize, use_tta=args.tta)

    test_image_paths = sorted(paths.test_img_dir.glob("*.jpeg"))
    submission_rows = []

    for path in test_image_paths:
        image_id = path.stem
        if backend_predict is not None:
            label_map, _ = backend_predict(path)
        elif args.obb:
            label_map, _ = predict_instances_obb(
                path, yolo_model, unet_model, device,
                conf_thres=inference_cfg.conf_thres, iou_thres=inference_cfg.iou_thres,
                imgsz=inference_cfg.imgsz, crop_size=crop_cfg.crop_size,
                pad_factor=crop_cfg.pad_factor, min_crop_side=crop_cfg.min_crop_side,
                mask_thresh=inference_cfg.mask_thresh, min_instance_area=inference_cfg.min_instance_area,
                mask_nms_thresh=inference_cfg.mask_nms_thresh, mask_conf_thres=inference_cfg.mask_conf_thres,
                use_tta=args.tta,
            )
        else:
            label_map, _ = predict_instances(
                path, yolo_model, unet_model, device,
                conf_thres=inference_cfg.conf_thres, iou_thres=inference_cfg.iou_thres,
                imgsz=inference_cfg.imgsz, crop_size=crop_cfg.crop_size,
                pad_factor=crop_cfg.pad_factor, min_crop_side=crop_cfg.min_crop_side,
                mask_thresh=inference_cfg.mask_thresh, min_instance_area=inference_cfg.min_instance_area,
                mask_nms_thresh=inference_cfg.mask_nms_thresh, mask_conf_thres=inference_cfg.mask_conf_thres,
                use_spine_orientation=crop_cfg.use_spine_orientation, min_elongation=inference_cfg.min_elongation,
                preprocess=args.normalize, use_tta=args.tta,
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
