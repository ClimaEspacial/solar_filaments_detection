import argparse

import torch
from ultralytics import YOLO

from solar_filaments.config import Paths, YoloConfig


def add_args(parser: argparse.ArgumentParser) -> None:
    cfg = YoloConfig()
    parser.add_argument("--weights", default=cfg.base_weights,
                         help="pretrained checkpoint, or a run's last.pt/best.pt to resume from")
    parser.add_argument("--imgsz", type=int, default=cfg.imgsz)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch", type=int, default=cfg.batch)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--name", default=cfg.name)
    parser.add_argument("--resume", action="store_true", help="resume an interrupted run (weights must be its last.pt)")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto-detect")


def run(args: argparse.Namespace) -> None:
    paths = Paths()
    paths.ensure()

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    yolo_dir = paths.yolo_dir
    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found - run `filament prepare` first")

    model = YOLO(args.weights)

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
        name=args.name,
    )
    weights_path = paths.runs_dir / args.name / "weights" / "best.pt"
    print("best weights:", weights_path)
