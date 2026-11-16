import argparse

import torch
from torch.utils.data import DataLoader

from solar_filaments.config import CropConfig, Paths, UNetConfig
from solar_filaments.data.coco import Dataset, load_split
from solar_filaments.data.crop_dataset import FilamentCropDataset
from solar_filaments.losses import bce_dice_loss, dice_score
from solar_filaments.models.refiner import ENCODERS, build_refiner


def add_args(parser: argparse.ArgumentParser) -> None:
    cfg = UNetConfig()
    crop_cfg = CropConfig()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=None,
                         help=f"default {cfg.lr} for the scratch encoder, 2e-4 for --encoder resnet18 "
                              "(fine-tuning ImageNet weights at 1e-3 diverges to nan within 3 epochs)")
    parser.add_argument("--base-channels", type=int, default=cfg.base_channels)
    parser.add_argument("--no-edge-attention", action="store_true")
    parser.add_argument("--spine-loss", type=float, default=0.0,
                         help="weight of an auxiliary head predicting the filament's medial axis, "
                              "rasterized from the MAGFiLO spine polyline. 0 disables it. A region "
                              "loss on a few-pixel-wide structure supplies almost no gradient; the "
                              "spine says where the filament *runs*. Saves to a _spine-suffixed "
                              "checkpoint; inference reads the mask channel and ignores the head")
    parser.add_argument("--spine-thickness", type=int, default=5)
    parser.add_argument("--encoder", choices=ENCODERS, default="base",
                         help="'base' = the scratch EdgeAttUNet encoder (8.8M params, default); "
                              "'resnet18' = ImageNet-pretrained ResNet18 encoder (15.4M), aimed at the "
                              "IoU 0.40-0.50 near-miss band that mask quality decides. Saves to a "
                              "_r18-suffixed checkpoint; no flag needed at inference (the encoder is "
                              "read back from the checkpoint)")
    parser.add_argument("--crop-size", type=int, default=crop_cfg.crop_size)
    parser.add_argument("--pad-factor", type=float, default=crop_cfg.pad_factor,
                         help="context around each filament's box in the training crops. The refiner "
                              "memorises this scale: a model trained at one value and run at another "
                              "loses 0.08 PQ per 0.25 of mismatch, so whatever is set here must be "
                              "passed to evaluate/predict as well. Checkpoint gets a _padNNN suffix")
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
    parser.add_argument("--instance-norm", action="store_true",
                         help="use InstanceNorm instead of BatchNorm in the U-Net (arXiv:2607.24525) - "
                              "saves to a _in-suffixed checkpoint; pair with --instance-norm at "
                              "evaluate/predict time too")
    parser.add_argument("--normalize", action="store_true",
                         help="train on limb-darkening-flattened + CLAHE-preprocessed crops instead of raw - "
                              "saves to a _norm-suffixed checkpoint; pair with --normalize at evaluate/predict "
                              "time too. Not combinable with --obb this round")


def run(args: argparse.Namespace) -> None:
    if args.normalize and args.obb:
        raise ValueError("--normalize and --obb can't be combined this round")

    paths = Paths()
    paths.ensure()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = Dataset(paths.train_ann_path)
    train_ids, val_ids = load_split(paths.split_path_for(args.fold))
    synthetic_img_dir = paths.synthetic_img_dir_for(args.fold)

    crop_cfg = CropConfig(crop_size=args.crop_size, pad_factor=args.pad_factor,
                           use_spine_orientation=args.spine_orientation,
                           use_obb_orientation=args.obb)
    train_crop_ds = FilamentCropDataset(dataset, paths.train_img_dir, train_ids,
                                         crop_cfg.crop_size, crop_cfg.pad_factor, crop_cfg.min_crop_side,
                                         augment=True, with_spine_target=args.spine_loss > 0,
                                         spine_thickness=args.spine_thickness,
                                         use_spine_orientation=crop_cfg.use_spine_orientation,
                                         use_obb_orientation=crop_cfg.use_obb_orientation,
                                         synthetic_img_dir=synthetic_img_dir, preprocess=args.normalize)
    val_crop_ds = FilamentCropDataset(dataset, paths.train_img_dir, val_ids,
                                       crop_cfg.crop_size, crop_cfg.pad_factor, crop_cfg.min_crop_side,
                                       augment=False, use_spine_orientation=crop_cfg.use_spine_orientation,
                                       use_obb_orientation=crop_cfg.use_obb_orientation,
                                       synthetic_img_dir=synthetic_img_dir, preprocess=args.normalize)
    print(f"train crops: {len(train_crop_ds)}  val crops: {len(val_crop_ds)}")

    train_loader = DataLoader(train_crop_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_crop_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    norm_layer = "instance" if args.instance_norm else "batch"
    model = build_refiner(args.encoder, use_edge_attention=not args.no_edge_attention,
                           norm_layer=norm_layer, base_channels=args.base_channels,
                           out_ch=2 if args.spine_loss > 0 else 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params / 1e6:.2f}M")

    lr = args.lr if args.lr is not None else (2e-4 if args.encoder != "base" else UNetConfig().lr)
    print(f"encoder={args.encoder}  lr={lr}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    ckpt_stem = "unet_obb" if args.obb else ("unet_spine" if args.spine_orientation else "unet")
    if args.normalize:
        ckpt_stem += "_norm"
    if args.instance_norm:
        ckpt_stem += "_in"
    if args.encoder != "base":
        ckpt_stem += f"_{args.encoder.replace('resnet', 'r')}"
    if args.spine_loss > 0:
        ckpt_stem += "_spine"
    if abs(args.pad_factor - CropConfig().pad_factor) > 1e-6:
        ckpt_stem += f"_pad{int(round(args.pad_factor * 100))}"
    ckpt_name = f"{ckpt_stem}_best.pt"
    if args.fold is not None:
        ckpt_name = ckpt_name.replace("_best.pt", f"_fold{args.fold}_best.pt")
    ckpt_path = paths.ckpt_dir / ckpt_name
    has_val = len(val_crop_ds) > 0
    if not has_val:
        print("no validation crops (val-frac 0 / final all-data training) - "
              "saving every epoch instead of selecting on val_dice")
    best_val_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            crops, masks = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            spines = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(crops)
                if spines is None:
                    loss = bce_dice_loss(logits, masks)
                else:
                    # channel 0 is the mask (the only one used at inference); channel 1 is the
                    # auxiliary medial-axis head, present only to shape the shared encoder
                    loss = (bce_dice_loss(logits[:, :1], masks)
                            + args.spine_loss * bce_dice_loss(logits[:, 1:2], spines))
            scaler.scale(loss).backward()
            # Unscale before clipping so the norm is measured on real gradients. Without
            # this a single bad batch under fp16 sends the loss to nan and it never
            # recovers - observed with the pretrained ResNet18 encoder at lr 1e-3.
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * crops.size(0)
        sched.step()

        if has_val:
            model.eval()
            val_dice, n = 0.0, 0
            with torch.no_grad():
                for crops, masks in val_loader:
                    crops, masks = crops.to(device), masks.to(device)
                    logits = model(crops)[:, :1]   # canal 0 = mascara, o unico usado na inferencia
                    val_dice += dice_score(logits, masks) * crops.size(0)
                    n += crops.size(0)
            val_dice /= max(n, 1)
            print(f"epoch {epoch+1:03d}/{args.epochs}  "
                  f"train_loss={running_loss/len(train_loader.dataset):.4f}  val_dice={val_dice:.4f}")
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), ckpt_path)
        else:
            # nothing to select "best" on - just keep the latest epoch's weights
            print(f"epoch {epoch+1:03d}/{args.epochs}  train_loss={running_loss/len(train_loader.dataset):.4f}")
            torch.save(model.state_dict(), ckpt_path)

    if has_val:
        print("best val dice:", best_val_dice, "-> saved to", ckpt_path)
    else:
        print("final epoch weights (no val set) -> saved to", ckpt_path)
