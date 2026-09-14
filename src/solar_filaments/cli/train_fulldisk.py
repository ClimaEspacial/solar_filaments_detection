"""Train the Flat U-Net full-disk semantic segmenter (models/flat_unet.py).

This is not a standalone submission model - it produces one filaments-vs-
background probability map per image, which inference uses as an *independent*
per-candidate confidence signal (see `--semantic-thres`). Its own selection
metric here is therefore plain pixel Dice on the validation split.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from solar_filaments.config import FullDiskConfig, Paths
from solar_filaments.data.coco import ANNOTATOR_MERGE_MODES, Dataset, load_split
from solar_filaments.data.fulldisk_dataset import FullDiskDataset
from solar_filaments.losses import bce_dice_loss, dice_score
from solar_filaments.models.flat_unet import FlatUNet


def add_args(parser: argparse.ArgumentParser) -> None:
    cfg = FullDiskConfig()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--imgsz", type=int, default=cfg.imgsz,
                         help="working resolution (the paper uses 512; filaments are thin, so 1024 is worth a try)")
    parser.add_argument("--width", type=int, default=cfg.width,
                         help="constant channel width; 32 gives ~0.26M params")
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--fold", type=int, default=None,
                         help="train on this fold's split; checkpoint flat_unet_fold<N>_best.pt")
    parser.add_argument("--annotator-merge", choices=ANNOTATOR_MERGE_MODES, default="union",
                         help="how to build the semantic target from 1-3 annotator records: 'union' = every "
                              "filament anyone marked (max recall, but 31%% of them are marked by a minority "
                              "and are the same low-contrast cases the detector misses); 'majority' = only "
                              "filaments >=50%% of annotators agree on (cleaner target); 'single' = first record")
    parser.add_argument("--bce-weight", type=float, default=cfg.bce_weight,
                         help="weight of BCE vs Dice in the loss (filaments cover ~1%% of pixels, so pure "
                              "BCE under-segments; the default mixes in Dice)")


def run(args: argparse.Namespace) -> None:
    paths = Paths()
    paths.ensure()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = Dataset(paths.train_ann_path)
    train_ids, val_ids = load_split(paths.split_path_for(args.fold))
    synthetic_img_dir = paths.synthetic_img_dir_for(args.fold)

    train_ds = FullDiskDataset(dataset, paths.train_img_dir, train_ids, imgsz=args.imgsz, augment=True,
                               synthetic_img_dir=synthetic_img_dir, annotator_merge=args.annotator_merge)
    # the validation target always uses the same rule, so val dice stays comparable across runs
    val_ds = FullDiskDataset(dataset, paths.train_img_dir, val_ids, imgsz=args.imgsz, augment=False,
                             synthetic_img_dir=synthetic_img_dir, annotator_merge=args.annotator_merge)
    has_val = len(val_ds) > 0
    print(f"train images: {len(train_ds)}  val images: {len(val_ds)}  @ {args.imgsz}x{args.imgsz}")
    if not has_val:
        print("no validation images (val-frac 0) - saving every epoch instead of selecting on val dice")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = FlatUNet(width=args.width).to(device)
    print(f"Flat U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    stem = "flat_unet" if args.annotator_merge == "union" else f"flat_unet_{args.annotator_merge}"
    name = f"{stem}_fold{args.fold}_best.pt" if args.fold is not None else f"{stem}_best.pt"
    ckpt_path = paths.ckpt_dir / name
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = bce_dice_loss(logits, masks, bce_weight=args.bce_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * imgs.size(0)
        sched.step()
        train_loss = running / max(len(train_loader.dataset), 1)

        if has_val:
            model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    total += dice_score(model(imgs), masks) * imgs.size(0)
                    n += imgs.size(0)
            val_dice = total / max(n, 1)
            print(f"epoch {epoch + 1:03d}/{args.epochs}  train_loss={train_loss:.4f}  val_dice={val_dice:.4f}")
            if val_dice > best:
                best = val_dice
                torch.save({"model": model.state_dict(), "imgsz": args.imgsz, "width": args.width}, ckpt_path)
        else:
            print(f"epoch {epoch + 1:03d}/{args.epochs}  train_loss={train_loss:.4f}")
            torch.save({"model": model.state_dict(), "imgsz": args.imgsz, "width": args.width}, ckpt_path)

    print(("best val dice: %.4f -> saved to %s" % (best, ckpt_path)) if has_val
          else f"final epoch weights (no val set) -> saved to {ckpt_path}")
