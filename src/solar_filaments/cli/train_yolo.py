import argparse

import torch
from ultralytics import YOLO

from solar_filaments.config import Paths, YoloConfig


def add_args(parser: argparse.ArgumentParser) -> None:
    cfg = YoloConfig()
    parser.add_argument("--weights", default=None,
                         help=f"pretrained checkpoint, or a run's last.pt/best.pt to resume from; "
                              f"default: {cfg.base_weights}, or yolo11s-obb.pt with --obb")
    parser.add_argument("--imgsz", type=int, default=cfg.imgsz)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch", type=int, default=cfg.batch,
                         help="images per step; 0 = ultralytics AutoBatch (sizes the batch to ~60%% of the GPU "
                              "memory free when the run starts)")
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--name", default=None, help=f"default: {cfg.name}, or {cfg.name}_fold<N> with --fold")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted run (weights must be its last.pt)")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto-detect")
    parser.add_argument("--fold", type=int, default=None,
                         help="train on this fold's data (from `filament prepare --k-folds`); "
                              "default run name becomes <name>_fold<N>")
    parser.add_argument("--obb", action="store_true",
                         help="train an oriented-bounding-box detector on the OBB dataset "
                              "(from `filament prepare --obb`) instead of axis-aligned boxes")
    parser.add_argument("--normalize", action="store_true",
                         help="train on the limb-darkening-flattened + CLAHE-preprocessed dataset "
                              "(from `filament prepare --normalize`) instead of raw images - "
                              "not combinable with --obb this round")
    parser.add_argument("--seg", action="store_true",
                         help="train a YOLO11 instance-segmentation model (yolo11s-seg.pt, downloaded by "
                              "ultralytics on first use) on the polygon dataset from `filament prepare --seg`; "
                              "run name <name>_seg. Pair with `--detector yolo-seg` at evaluate/predict time")


def run(args: argparse.Namespace) -> None:
    paths = Paths()
    paths.ensure()
    cfg = YoloConfig()

    if sum([args.obb, args.normalize, args.seg]) > 1:
        raise ValueError("--obb, --normalize and --seg are mutually exclusive")

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    variant = "_obb" if args.obb else ("_norm" if args.normalize else ("_seg" if args.seg else ""))
    base_name = f"{cfg.name}{variant}"
    name = args.name or (f"{base_name}_fold{args.fold}" if args.fold is not None else base_name)
    weights = args.weights or ("yolo11s-obb.pt" if args.obb else ("yolo11s-seg.pt" if args.seg else cfg.base_weights))

    if args.obb:
        yolo_dir = paths.yolo_obb_dir_for(args.fold)
    elif args.normalize:
        yolo_dir = paths.yolo_norm_dir_for(args.fold)
    elif args.seg:
        yolo_dir = paths.yolo_seg_dir_for(args.fold)
    else:
        yolo_dir = paths.yolo_dir_for(args.fold)
    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.exists():
        hint = " --obb" if args.obb else (" --normalize" if args.normalize else (" --seg" if args.seg else ""))
        fold_hint = f" --k-folds ..." if args.fold is not None else ""
        raise FileNotFoundError(f"{data_yaml} not found - run `filament prepare{hint}{fold_hint}` first")

    model = YOLO(weights)

    if args.resume:
        model.train(resume=True)
        return

    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch if args.batch > 0 else -1,  # -1 = AutoBatch
        device=device,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        patience=args.patience,
        mosaic=0.0,
        degrees=10, translate=0.05, scale=0.2, shear=0.0, perspective=0.0,
        flipud=0.5, fliplr=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
        project=str(paths.runs_dir),
        name=name,
    )
    weights_path = paths.runs_dir / name / "weights" / "best.pt"
    print("best weights:", weights_path)
