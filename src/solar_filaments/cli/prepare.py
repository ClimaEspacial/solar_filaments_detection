import argparse
import random

import numpy as np
import torch

from solar_filaments.config import SEED, Paths
from solar_filaments.data.coco import Dataset, save_split, train_val_split
from solar_filaments.data.yolo_dataset import build_yolo_dataset


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = Paths()
    paths.ensure()

    dataset = Dataset(paths.train_ann_path)
    print(dataset.summary())

    train_ids, val_ids = train_val_split(dataset, args.seed, args.val_frac)
    print(f"train images: {len(train_ids)}  val images: {len(val_ids)}")
    save_split(paths.split_path, train_ids, val_ids)

    yaml_path = build_yolo_dataset(dataset, paths.train_img_dir, paths.yolo_dir, train_ids, val_ids)
    print("YOLO dataset ready at", yaml_path)
