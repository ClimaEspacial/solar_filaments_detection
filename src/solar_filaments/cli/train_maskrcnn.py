"""Train the Mask R-CNN detector backend on annotator-consensus instance targets.

Checkpoint selection uses the metric we actually care about - PQ of the
detector's own coarse masks (no refiner) on a fixed validation subset - rather
than a box-mAP proxy, so "best epoch" means best instance matching."""
import argparse
import random

import torch
from torch.utils.data import DataLoader

from solar_filaments.cli._backends import maskrcnn_weights_path
from solar_filaments.config import ANNOTATOR_MERGE_DEFAULT, MaskRCNNConfig, Paths
from solar_filaments.data.coco import ANNOTATOR_MERGE_MODES, Dataset, load_split
from solar_filaments.data.instance_dataset import FilamentInstanceDataset, collate
from solar_filaments.detectors import MaskRCNNDetector, build_maskrcnn
from solar_filaments.eval.self_eval import build_val_gt_pred_dataframes, format_summary, get_overlap_df, \
    summarize_overlap
from solar_filaments.inference import predict_instances_from_candidates


def add_args(parser: argparse.ArgumentParser) -> None:
    cfg = MaskRCNNConfig()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch", type=int, default=cfg.batch,
                         help="images per step; 0 = auto (probe 8/6/4/2/1 with a real training step and keep the "
                              "largest that fits the GPU memory free *right now*)")
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--imgsz", type=int, default=cfg.imgsz)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--val-subset", type=int, default=cfg.val_subset,
                         help="images scored each epoch for checkpoint selection (0 = whole val split)")
    parser.add_argument("--aspect-ratios", default=",".join(str(r) for r in cfg.aspect_ratios),
                         help="RPN anchor aspect ratios; wide set by default since filaments are elongated")
    parser.add_argument("--no-pretrained", action="store_true",
                         help="random init instead of COCO-pretrained weights (which torchvision downloads "
                              "~170 MB on first use)")
    parser.add_argument("--fold", type=int, default=None,
                         help="train on this fold (from `filament prepare --k-folds`); checkpoint maskrcnn_fold<N>_best.pt")
    parser.add_argument("--annotator-merge", choices=ANNOTATOR_MERGE_MODES, default=ANNOTATOR_MERGE_DEFAULT,
                         help="consensus rule for multi-annotator images (see `prepare --annotator-merge`)")
    parser.add_argument("--limit-train", type=int, default=0, help="smoke tests: use only N training images")
    parser.add_argument("--seed", type=int, default=0)


def _probe_batch_size(model, train_ds, device, candidates=(8, 6, 4, 2, 1)) -> int:
    """Largest batch whose full train step (forward + backward, AMP) fits in
    the GPU memory available *now* - so a run started while another process
    holds part of the card still trains, and one started on a free card uses
    it. Probes with real samples (memory scales with instance count too)."""
    if device.type != "cuda":
        return candidates[-1]
    model.train()
    for b in candidates:
        if b > len(train_ds):
            continue
        try:
            images, targets = collate([train_ds[i] for i in range(b)])
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast(device.type):
                loss = sum(model(images, targets).values())
            loss.backward()
            model.zero_grad(set_to_none=True)
            del images, targets, loss
            torch.cuda.empty_cache()
            return b
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"  batch {b}: out of memory, trying smaller")
    raise RuntimeError("even batch size 1 does not fit in the free GPU memory")


def run(args: argparse.Namespace) -> None:
    paths = Paths()
    paths.ensure()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("device:", device)

    dataset = Dataset(paths.train_ann_path)
    train_ids, val_ids = load_split(paths.split_path_for(args.fold))
    synthetic_img_dir = paths.synthetic_img_dir_for(args.fold)

    train_ds = FilamentInstanceDataset(dataset, paths.train_img_dir, train_ids, imgsz=args.imgsz, augment=True,
                                       annotator_merge=args.annotator_merge, synthetic_img_dir=synthetic_img_dir)
    if args.limit_train:
        train_ds.records = train_ds.records[:args.limit_train]
    print(f"train images: {len(train_ds)}  ({sum(len(r[2]) for r in train_ds.records)} consensus instances)")

    val_ids = sorted(val_ids)
    if args.val_subset and args.val_subset < len(val_ids):
        val_ids = random.Random(args.seed).sample(val_ids, args.val_subset)
    print(f"val records for checkpoint selection: {len(val_ids)}")

    aspect_ratios = tuple(float(r) for r in args.aspect_ratios.split(","))
    anchor_sizes = MaskRCNNConfig().anchor_sizes
    model = build_maskrcnn(pretrained=not args.no_pretrained, imgsz=args.imgsz,
                           anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mask R-CNN parameters: {n_params / 1e6:.1f}M  anchors {anchor_sizes} x ratios {aspect_ratios}")

    batch = args.batch if args.batch > 0 else _probe_batch_size(model, train_ds, device)
    print(f"batch size: {batch}")
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    ckpt_path = maskrcnn_weights_path(paths, args.fold)
    best_pq = -1.0
    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for images, targets in loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(images)
            n += len(images)
        sched.step()

        # checkpoint selection on coarse-mask PQ (the detector alone, no refiner)
        model.eval()
        detector = MaskRCNNDetector(model, device, score_thres=MaskRCNNConfig().score_thres, imgsz=args.imgsz)
        predict_fn = lambda p: predict_instances_from_candidates(p, detector, None, device, refine=False,
                                                                  mask_nms_thresh=0.25)
        gt_df, pred_df = build_val_gt_pred_dataframes(dataset, paths.train_img_dir, val_ids, None, None, device,
                                                       None, None, predict_fn=predict_fn)
        summary = summarize_overlap(get_overlap_df(gt_df, pred_df))
        print(f"epoch {epoch + 1:03d}/{args.epochs}  train_loss={running / max(n, 1):.4f}  "
              f"val(coarse) {format_summary(summary)}")

        if summary["PQ"] > best_pq:
            best_pq = summary["PQ"]
            torch.save({"model": model.state_dict(), "imgsz": args.imgsz, "anchor_sizes": anchor_sizes,
                        "aspect_ratios": aspect_ratios, "epoch": epoch + 1}, ckpt_path)

    print(f"best val coarse-mask PQ: {best_pq:.4f} -> saved to {ckpt_path}")
