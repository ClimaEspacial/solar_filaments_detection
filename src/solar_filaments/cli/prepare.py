import argparse
import json
import random

import numpy as np
import torch

from solar_filaments.config import SEED, Paths
from solar_filaments.data.augment import generate_synthetic_images
from solar_filaments.data.coco import ANNOTATOR_MERGE_MODES, Dataset, kfold_split, save_split, train_val_split
from solar_filaments.data.yolo_dataset import build_yolo_dataset, build_yolo_obb_dataset, build_yolo_seg_dataset


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--val-frac", type=float, default=0.1,
                         help="ignored when --k-folds is set; 0 trains on all data with no held-out val")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--k-folds", type=int, default=0, help="0 = single split (default); >=2 = k-fold")
    parser.add_argument("--augment-copy-paste", type=int, default=0,
                         help="number of synthetic copy-paste training images to generate; 0 = off")
    parser.add_argument("--max-pastes-per-image", type=int, default=2,
                         help="max filaments pasted onto a single synthetic image")
    parser.add_argument("--obb", action="store_true",
                         help="also build an oriented-bounding-box YOLO dataset (yolo_obb_dataset/) "
                              "alongside the axis-aligned one")
    parser.add_argument("--normalize", action="store_true",
                         help="also build a limb-darkening-flattened + CLAHE-preprocessed axis-aligned YOLO "
                              "dataset (yolo_norm_dataset/), for `train-yolo --normalize`. Not combinable with "
                              "--obb this round")
    parser.add_argument("--seg", action="store_true",
                         help="also build an instance-segmentation (polygon-label) YOLO dataset "
                              "(yolo_seg_dataset/), for `train-yolo --seg`")
    parser.add_argument("--annotator-merge", choices=ANNOTATOR_MERGE_MODES, default="majority",
                         help="how to turn 1-3 annotator records per image into one detector target set: "
                              "'majority' = filaments >=50%% of annotators marked (PQ-optimal, default), "
                              "'union' = every filament any annotator marked, 'single' = first record only")


def _prepare_one(dataset, paths, train_ids, val_ids, fold, args):
    if args.augment_copy_paste > 0:
        id_prefix = "synthetic" if fold is None else f"fold{fold}-synthetic"
        synthetic_dir = paths.synthetic_img_dir_for(fold)
        new_ids = generate_synthetic_images(
            dataset, paths.train_img_dir, synthetic_dir, train_ids,
            n_synthetic=args.augment_copy_paste, max_pastes_per_image=args.max_pastes_per_image,
            seed=args.seed, id_prefix=id_prefix,
        )
        print(f"  +{len(new_ids)} synthetic copy-paste images (of {args.augment_copy_paste} requested)")
        train_ids = train_ids | new_ids

    # Save the split *after* augmentation so train-unet (which loads this file
    # directly) also sees the synthetic images - only build_yolo_dataset used
    # to get them, leaving the U-Net stage trained on real data only.
    save_split(paths.split_path_for(fold), train_ids, val_ids)

    common = dict(synthetic_img_dir=paths.synthetic_img_dir_for(fold), annotator_merge=args.annotator_merge)
    yaml_path = build_yolo_dataset(dataset, paths.train_img_dir, paths.yolo_dir_for(fold), train_ids, val_ids, **common)
    print(f"  train records: {len(train_ids)}  val records: {len(val_ids)}  "
          f"(annotator merge: {args.annotator_merge}) -> {yaml_path}")

    if args.obb:
        obb_yaml_path = build_yolo_obb_dataset(dataset, paths.train_img_dir, paths.yolo_obb_dir_for(fold),
                                                train_ids, val_ids, **common)
        print(f"  OBB dataset -> {obb_yaml_path}")

    if args.normalize:
        norm_yaml_path = build_yolo_dataset(dataset, paths.train_img_dir, paths.yolo_norm_dir_for(fold),
                                             train_ids, val_ids, preprocess=True, **common)
        print(f"  normalized dataset -> {norm_yaml_path}")

    if args.seg:
        seg_yaml_path = build_yolo_seg_dataset(dataset, paths.train_img_dir, paths.yolo_seg_dir_for(fold),
                                                train_ids, val_ids, **common)
        print(f"  segmentation dataset -> {seg_yaml_path}")


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = Paths()
    paths.ensure()

    dataset = Dataset(paths.train_ann_path)
    print(dataset.summary())

    if args.k_folds >= 2:
        splits = kfold_split(dataset, args.seed, args.k_folds)
        for i, (train_ids, val_ids) in enumerate(splits):
            print(f"fold {i}:")
            _prepare_one(dataset, paths, train_ids, val_ids, i, args)

        paths.splits_dir.mkdir(parents=True, exist_ok=True)
        with open(paths.kfold_meta_path, "w") as f:
            json.dump({"k": args.k_folds, "seed": args.seed}, f)
        print(f"k-fold metadata written to {paths.kfold_meta_path}")
    elif args.k_folds == 1:
        raise ValueError("--k-folds must be 0 (single split) or >= 2")
    else:
        train_ids, val_ids = train_val_split(dataset, args.seed, args.val_frac)
        _prepare_one(dataset, paths, train_ids, val_ids, None, args)
