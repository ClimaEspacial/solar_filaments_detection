import argparse

import torch
from torch.utils.data import DataLoader

from solar_filaments.config import CropConfig, Paths, UNetConfig
from solar_filaments.data.coco import Dataset, load_split
from solar_filaments.data.crop_dataset import FilamentCropDataset
from solar_filaments.losses import bce_dice_loss, dice_score
from solar_filaments.models.unet import EdgeAttUNet


def add_args(parser: argparse.ArgumentParser) -> None:
    cfg = UNetConfig()
    crop_cfg = CropConfig()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--base-channels", type=int, default=cfg.base_channels)
    parser.add_argument("--no-edge-attention", action="store_true")
    parser.add_argument("--crop-size", type=int, default=crop_cfg.crop_size)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--spine-orientation", action="store_true",
                         help="rotate crops so the GT spine lies horizontal, instead of axis-aligned bbox crops "
                              "(saves to a separate unet_spine_best.pt - pair with --spine-orientation at "
                              "evaluate/predict time too)")
    parser.add_argument("--obb", action="store_true",
                         help="rotate crops to match the minAreaRect of the GT polygon (same geometry an OBB "
                              "detector predicts), instead of axis-aligned bbox crops (saves to unet_obb_best.pt - "
                              "pair with --obb at evaluate/predict time too)")
    parser.add_argument("--fold", type=int, default=None,
                         help="train on this fold's data (from `filament prepare --k-folds`); "
                              "default checkpoint name becomes unet_fold<N>_best.pt")


def run(args: argparse.Namespace) -> None:
    paths = Paths()
    paths.ensure()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = Dataset(paths.train_ann_path)
    train_ids, val_ids = load_split(paths.split_path_for(args.fold))
    synthetic_img_dir = paths.synthetic_img_dir_for(args.fold)

    crop_cfg = CropConfig(crop_size=args.crop_size, use_spine_orientation=args.spine_orientation,
                           use_obb_orientation=args.obb)
    train_crop_ds = FilamentCropDataset(dataset, paths.train_img_dir, train_ids,
                                         crop_cfg.crop_size, crop_cfg.pad_factor, crop_cfg.min_crop_side,
                                         augment=True, use_spine_orientation=crop_cfg.use_spine_orientation,
                                         use_obb_orientation=crop_cfg.use_obb_orientation,
                                         synthetic_img_dir=synthetic_img_dir)
    val_crop_ds = FilamentCropDataset(dataset, paths.train_img_dir, val_ids,
                                       crop_cfg.crop_size, crop_cfg.pad_factor, crop_cfg.min_crop_side,
                                       augment=False, use_spine_orientation=crop_cfg.use_spine_orientation,
                                       use_obb_orientation=crop_cfg.use_obb_orientation,
                                       synthetic_img_dir=synthetic_img_dir)
    print(f"train crops: {len(train_crop_ds)}  val crops: {len(val_crop_ds)}")

    train_loader = DataLoader(train_crop_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_crop_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = EdgeAttUNet(base=args.base_channels, use_edge_attention=not args.no_edge_attention).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    ckpt_name = "unet_obb_best.pt" if args.obb else ("unet_spine_best.pt" if args.spine_orientation else "unet_best.pt")
    if args.fold is not None:
        ckpt_name = ckpt_name.replace("_best.pt", f"_fold{args.fold}_best.pt")
    ckpt_path = paths.ckpt_dir / ckpt_name
    best_val_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for crops, masks in train_loader:
            crops, masks = crops.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(crops)
                loss = bce_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * crops.size(0)
        sched.step()

        model.eval()
        val_dice, n = 0.0, 0
        with torch.no_grad():
            for crops, masks in val_loader:
                crops, masks = crops.to(device), masks.to(device)
                logits = model(crops)
                val_dice += dice_score(logits, masks) * crops.size(0)
                n += crops.size(0)
        val_dice /= max(n, 1)

        print(f"epoch {epoch+1:03d}/{args.epochs}  "
              f"train_loss={running_loss/len(train_loader.dataset):.4f}  val_dice={val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)

    print("best val dice:", best_val_dice, "-> saved to", ckpt_path)
