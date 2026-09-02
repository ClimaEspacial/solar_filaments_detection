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
    parser.add_argument("--batch", type=int, default=cfg.batch)
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


def run(args: argparse.Namespace) -> None:
    paths = Paths()
    paths.ensure()
    cfg = YoloConfig()

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    base_name = f"{cfg.name}_obb" if args.obb else cfg.name
    name = args.name or (f"{base_name}_fold{args.fold}" if args.fold is not None else base_name)
    weights = args.weights or ("yolo11s-obb.pt" if args.obb else cfg.base_weights)

    yolo_dir = paths.yolo_obb_dir_for(args.fold) if args.obb else paths.yolo_dir_for(args.fold)
    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.exists():
        obb_hint = " --obb" if args.obb else ""
        fold_hint = f" --k-folds ..." if args.fold is not None else ""
        raise FileNotFoundError(f"{data_yaml} not found - run `filament prepare{obb_hint}{fold_hint}` first")

    model = YOLO(weights)

    if args.resume:
        model.train(resume=True)
        return

    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
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
