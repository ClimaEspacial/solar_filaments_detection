import argparse
import random

import matplotlib
matplotlib.use("Agg")  # headless: only ever save plots to disk, never open a window
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_split
from solar_filaments.eval.self_eval import build_val_gt_pred_dataframes, get_overlap_df, get_pq_score, \
    plot_distribution, plot_m2n_counts
from solar_filaments.models.unet import EdgeAttUNet


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yolo-weights", default=None, help="default: work_dir/runs/<name>/weights/best.pt")
    parser.add_argument("--unet-weights", default=None, help="default: work_dir/checkpoints/unet_best.pt")
    parser.add_argument("--quick", type=int, default=8, help="quick-check sample size; 0 to skip")
    parser.add_argument("--full", action="store_true", help="run the full validation-set self-eval")
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


def run(args: argparse.Namespace) -> None:
    paths = Paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_unet_name = "unet_spine_best.pt" if args.spine_orientation else "unet_best.pt"
    yolo_weights = args.yolo_weights or str(paths.runs_dir / YoloConfig().name / "weights" / "best.pt")
    unet_weights = args.unet_weights or str(paths.ckpt_dir / default_unet_name)

    yolo_model = YOLO(yolo_weights)
    unet_model = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(device)
    unet_model.load_state_dict(torch.load(unet_weights, map_location=device))
    unet_model.eval()

    dataset = Dataset(paths.train_ann_path)
    _, val_ids = load_split(paths.split_path)

    crop_cfg = CropConfig(use_spine_orientation=args.spine_orientation)
    inference_cfg = InferenceConfig(conf_thres=args.conf_thres, iou_thres=args.iou_thres, imgsz=args.imgsz,
                                     mask_nms_thresh=args.mask_nms_thresh, mask_conf_thres=args.mask_conf_thres)

    def run_self_eval(image_ids, label):
        gt_df, pred_df = build_val_gt_pred_dataframes(
            dataset, paths.train_img_dir, image_ids, yolo_model, unet_model, device, crop_cfg, inference_cfg)
        print(f"[{label}] GT filaments: {len(gt_df)}   predicted filaments: {len(pred_df)}")
        overlap_df = get_overlap_df(gt_df, pred_df)
        pq_score = get_pq_score(overlap_df)
        print(f"[{label}] PQ score: {pq_score:.4f}")
        return overlap_df, pq_score

    if args.quick > 0:
        quick_ids = random.sample(sorted(val_ids), k=min(args.quick, len(val_ids)))
        run_self_eval(quick_ids, f"quick check ({len(quick_ids)} imgs)")

    if args.full:
        overlap_df, pq_score = run_self_eval(sorted(val_ids), "full val")

        out_dir = paths.work_dir / "eval_plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 4))
        plot_distribution(overlap_df["iou_matrix"], "IoU Distribution (val)", "IoU")
        plt.savefig(out_dir / "iou_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        plot_distribution(overlap_df["dice_matrix"], "Dice Distribution (val)", "Dice")
        plt.savefig(out_dir / "dice_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 5))
        plot_m2n_counts(overlap_df, "Many-to-One / One-to-Many Distribution (val)")
        plt.savefig(out_dir / "m2n_counts.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("plots saved to", out_dir)
