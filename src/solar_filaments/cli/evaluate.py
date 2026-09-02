import argparse
import json
import random

import matplotlib
matplotlib.use("Agg")  # headless: only ever save plots to disk, never open a window
import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO

from solar_filaments.classical import detect_filaments_classical
from solar_filaments.config import CropConfig, InferenceConfig, Paths, UNetConfig, YoloConfig
from solar_filaments.data.coco import Dataset, load_image, load_split
from solar_filaments.eval.self_eval import build_val_gt_pred_dataframes, build_val_gt_pred_dataframes_classical, \
    build_val_gt_pred_dataframes_ensemble, get_overlap_df, get_pq_score, plot_distribution, plot_m2n_counts
from solar_filaments.inference import predict_instances, predict_instances_obb
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
    parser.add_argument("--fold", default=None,
                         help="evaluate this fold (int, from `filament prepare --k-folds`), or 'all' to pool "
                              "out-of-fold PQ across every fold's own held-out val set")
    parser.add_argument("--classical", action="store_true",
                         help="use the classical (non-learned) Shih & Kowalski-style detector instead of "
                              "YOLO+U-Net - no weights/GPU needed. --fold picks which images to evaluate on "
                              "(no model to load per fold); --fold all means every image in the dataset")
    parser.add_argument("--median-size", type=int, default=20, help="classical: local-median window (px, working res)")
    parser.add_argument("--cutoff-m", type=float, default=0.96, help="classical: asymmetric threshold multiplier")
    parser.add_argument("--directional-size", type=int, default=11,
                         help="classical: directional closing footprint size (px, working res)")
    parser.add_argument("--min-area", type=int, default=100, help="classical: min instance area (px, working res)")
    parser.add_argument("--min-eccentricity", type=float, default=0.9,
                         help="classical: min region eccentricity (rounder = more likely an active region)")
    parser.add_argument("--ensemble", action="store_true",
                         help="merge the axis-aligned and OBB pipelines (each at its own already-tuned "
                              "conf-thres), plus optionally the classical detector, via mask-level NMS across "
                              "all their instances. No --fold support (single-split weights only)")
    parser.add_argument("--axis-yolo-weights", default=None, help="ensemble: default work_dir/runs/<name>/weights/best.pt")
    parser.add_argument("--axis-unet-weights", default=None, help="ensemble: default work_dir/checkpoints/unet_best.pt")
    parser.add_argument("--obb-yolo-weights", default=None, help="ensemble: default work_dir/runs/<name>_obb/weights/best.pt")
    parser.add_argument("--obb-unet-weights", default=None, help="ensemble: default work_dir/checkpoints/unet_obb_best.pt")
    parser.add_argument("--axis-conf-thres", type=float, default=0.35, help="ensemble: axis-aligned source's own conf_thres")
    parser.add_argument("--obb-conf-thres", type=float, default=0.25, help="ensemble: OBB source's own conf_thres")
    parser.add_argument("--axis-weight", type=float, default=1.0, help="ensemble: axis-aligned source confidence multiplier")
    parser.add_argument("--obb-weight", type=float, default=1.0, help="ensemble: OBB source confidence multiplier")
    parser.add_argument("--classical-weight", type=float, default=0.0,
                         help="ensemble: classical source confidence multiplier; 0 = excluded (default)")
    parser.add_argument("--ensemble-mask-nms-thresh", type=float, default=0.3,
                         help="ensemble: cross-source mask overlap threshold to suppress a duplicate instance")


def _default_weights(paths: Paths, args: argparse.Namespace, fold):
    base_name = f"{YoloConfig().name}_obb" if args.obb else (f"{YoloConfig().name}_norm" if args.normalize else YoloConfig().name)
    name = f"{base_name}_fold{fold}" if fold is not None else base_name
    unet_stem = "unet_obb" if args.obb else ("unet_spine" if args.spine_orientation else "unet")
    if args.normalize:
        unet_stem += "_norm"
    if args.instance_norm:
        unet_stem += "_in"
    unet_name = f"{unet_stem}_fold{fold}_best.pt" if fold is not None else f"{unet_stem}_best.pt"
    yolo_weights = args.yolo_weights or str(paths.runs_dir / name / "weights" / "best.pt")
    unet_weights = args.unet_weights or str(paths.ckpt_dir / unet_name)
    return yolo_weights, unet_weights


def _load_models(paths: Paths, args: argparse.Namespace, device, fold):
    yolo_weights, unet_weights = _default_weights(paths, args, fold)
    yolo_model = YOLO(yolo_weights)
    norm_layer = "instance" if args.instance_norm else "batch"
    unet_model = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention, norm_layer=norm_layer).to(device)
    unet_model.load_state_dict(torch.load(unet_weights, map_location=device))
    unet_model.eval()
    return yolo_model, unet_model


def _save_plots(paths: Paths, overlap_df: pd.DataFrame) -> None:
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


def _run_all_folds(paths: Paths, args: argparse.Namespace, dataset, device, crop_cfg, inference_cfg) -> None:
    if not paths.kfold_meta_path.exists():
        raise FileNotFoundError(f"{paths.kfold_meta_path} not found - run `filament prepare --k-folds N` first")
    with open(paths.kfold_meta_path) as f:
        k = json.load(f)["k"]

    all_gt, all_pred = [], []
    for fold in range(k):
        yolo_model, unet_model = _load_models(paths, args, device, fold)
        _, val_ids = load_split(paths.split_path_for(fold))
        gt_df, pred_df = build_val_gt_pred_dataframes(
            dataset, paths.train_img_dir, sorted(val_ids), yolo_model, unet_model, device, crop_cfg, inference_cfg,
            obb=args.obb, preprocess=args.normalize, use_tta=args.tta)
        print(f"[fold {fold}] GT filaments: {len(gt_df)}   predicted filaments: {len(pred_df)}")
        all_gt.append(gt_df)
        all_pred.append(pred_df)

    gt_df = pd.concat(all_gt, ignore_index=True)
    pred_df = pd.concat(all_pred, ignore_index=True)
    print(f"[all {k} folds, pooled] GT filaments: {len(gt_df)}   predicted filaments: {len(pred_df)}")
    overlap_df = get_overlap_df(gt_df, pred_df)
    pq_score = get_pq_score(overlap_df)
    print(f"[all {k} folds, pooled] out-of-fold PQ score: {pq_score:.4f}")

    if args.full:
        _save_plots(paths, overlap_df)


def _run_classical(paths: Paths, args: argparse.Namespace, dataset: Dataset) -> None:
    classical_kwargs = dict(median_size=args.median_size, cutoff_m=args.cutoff_m,
                             directional_size=args.directional_size, min_area=args.min_area,
                             min_eccentricity=args.min_eccentricity)

    if args.fold == "all":
        image_ids = sorted(dataset.images_by_id.keys())
    else:
        fold = int(args.fold) if args.fold is not None else None
        _, val_ids = load_split(paths.split_path_for(fold))
        image_ids = sorted(val_ids)

    def run_self_eval(ids, label):
        gt_df, pred_df = build_val_gt_pred_dataframes_classical(dataset, paths.train_img_dir, ids, classical_kwargs)
        print(f"[{label}] GT filaments: {len(gt_df)}   predicted filaments: {len(pred_df)}")
        overlap_df = get_overlap_df(gt_df, pred_df)
        pq_score = get_pq_score(overlap_df)
        print(f"[{label}] PQ score: {pq_score:.4f}")
        return overlap_df, pq_score

    if args.quick > 0:
        quick_ids = random.sample(image_ids, k=min(args.quick, len(image_ids)))
        run_self_eval(quick_ids, f"quick check ({len(quick_ids)} imgs, classical)")

    if args.full:
        overlap_df, pq_score = run_self_eval(image_ids, "full (classical)")
        _save_plots(paths, overlap_df)


def _run_ensemble(paths: Paths, args: argparse.Namespace, dataset: Dataset, device) -> None:
    axis_yolo_weights = args.axis_yolo_weights or str(paths.runs_dir / YoloConfig().name / "weights" / "best.pt")
    axis_unet_weights = args.axis_unet_weights or str(paths.ckpt_dir / "unet_best.pt")
    obb_yolo_weights = args.obb_yolo_weights or str(paths.runs_dir / f"{YoloConfig().name}_obb" / "weights" / "best.pt")
    obb_unet_weights = args.obb_unet_weights or str(paths.ckpt_dir / "unet_obb_best.pt")

    axis_yolo = YOLO(axis_yolo_weights)
    axis_unet = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(device)
    axis_unet.load_state_dict(torch.load(axis_unet_weights, map_location=device))
    axis_unet.eval()

    obb_yolo = YOLO(obb_yolo_weights)
    obb_unet = EdgeAttUNet(use_edge_attention=UNetConfig().use_edge_attention).to(device)
    obb_unet.load_state_dict(torch.load(obb_unet_weights, map_location=device))
    obb_unet.eval()

    crop_cfg = CropConfig()
    inference_cfg = InferenceConfig()
    classical_kwargs = dict(median_size=args.median_size, cutoff_m=args.cutoff_m,
                             directional_size=args.directional_size, min_area=args.min_area,
                             min_eccentricity=args.min_eccentricity)

    def make_sources(image_path):
        sources = [
            (lambda: predict_instances(
                image_path, axis_yolo, axis_unet, device, conf_thres=args.axis_conf_thres,
                iou_thres=inference_cfg.iou_thres, imgsz=inference_cfg.imgsz, crop_size=crop_cfg.crop_size,
                pad_factor=crop_cfg.pad_factor, min_crop_side=crop_cfg.min_crop_side,
                mask_thresh=inference_cfg.mask_thresh, min_instance_area=inference_cfg.min_instance_area,
                mask_nms_thresh=inference_cfg.mask_nms_thresh), args.axis_weight),
            (lambda: predict_instances_obb(
                image_path, obb_yolo, obb_unet, device, conf_thres=args.obb_conf_thres,
                iou_thres=inference_cfg.iou_thres, imgsz=inference_cfg.imgsz, crop_size=crop_cfg.crop_size,
                pad_factor=crop_cfg.pad_factor, min_crop_side=crop_cfg.min_crop_side,
                mask_thresh=inference_cfg.mask_thresh, min_instance_area=inference_cfg.min_instance_area,
                mask_nms_thresh=inference_cfg.mask_nms_thresh), args.obb_weight),
        ]
        if args.classical_weight > 0:
            sources.append((lambda: detect_filaments_classical(load_image(image_path), **classical_kwargs),
                             args.classical_weight))
        return sources

    ensemble_kwargs = dict(mask_nms_thresh=args.ensemble_mask_nms_thresh,
                            min_instance_area=inference_cfg.min_instance_area)

    _, val_ids = load_split(paths.split_path)
    val_ids = sorted(val_ids)

    def run_self_eval(ids, label):
        gt_df, pred_df = build_val_gt_pred_dataframes_ensemble(
            dataset, paths.train_img_dir, ids, make_sources, ensemble_kwargs)
        print(f"[{label}] GT filaments: {len(gt_df)}   predicted filaments: {len(pred_df)}")
        overlap_df = get_overlap_df(gt_df, pred_df)
        pq_score = get_pq_score(overlap_df)
        print(f"[{label}] PQ score: {pq_score:.4f}")
        return overlap_df, pq_score

    if args.quick > 0:
        quick_ids = random.sample(val_ids, k=min(args.quick, len(val_ids)))
        run_self_eval(quick_ids, f"quick check ({len(quick_ids)} imgs, ensemble)")

    if args.full:
        overlap_df, pq_score = run_self_eval(val_ids, "full (ensemble)")
        _save_plots(paths, overlap_df)


def run(args: argparse.Namespace) -> None:
    if args.normalize and args.obb:
        raise ValueError("--normalize and --obb can't be combined this round")

    paths = Paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(paths.train_ann_path)

    if args.classical:
        _run_classical(paths, args, dataset)
        return

    if args.ensemble:
        _run_ensemble(paths, args, dataset, device)
        return

    crop_cfg = CropConfig(use_spine_orientation=args.spine_orientation, use_obb_orientation=args.obb)
    inference_cfg = InferenceConfig(conf_thres=args.conf_thres, iou_thres=args.iou_thres, imgsz=args.imgsz,
                                     mask_nms_thresh=args.mask_nms_thresh, mask_conf_thres=args.mask_conf_thres,
                                     min_elongation=args.min_elongation)

    if args.fold == "all":
        _run_all_folds(paths, args, dataset, device, crop_cfg, inference_cfg)
        return

    fold = int(args.fold) if args.fold is not None else None
    yolo_model, unet_model = _load_models(paths, args, device, fold)
    _, val_ids = load_split(paths.split_path_for(fold))

    def run_self_eval(image_ids, label):
        gt_df, pred_df = build_val_gt_pred_dataframes(
            dataset, paths.train_img_dir, image_ids, yolo_model, unet_model, device, crop_cfg, inference_cfg,
            obb=args.obb, preprocess=args.normalize, use_tta=args.tta)
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
        _save_plots(paths, overlap_df)
