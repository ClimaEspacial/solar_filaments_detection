"""Central configuration: paths and hyperparameters.

Every path is overridable via environment variable so the same code runs
unmodified on Kaggle (`/kaggle/...`) or locally (defaults below).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass
class Paths:
    data_root: Path = field(default_factory=lambda: _env_path(
        "FILAMENT_DATA_ROOT",
        "/data/marchezi/filament_segmentation_2026/MAGFiLO_1.0_Kaggle_2026",
    ))
    work_dir: Path = field(default_factory=lambda: _env_path(
        "FILAMENT_WORK_DIR",
        str(Path(__file__).resolve().parents[2] / "work"),
    ))

    @property
    def train_img_dir(self) -> Path:
        return self.data_root / "train" / "train_images"

    @property
    def test_img_dir(self) -> Path:
        return self.data_root / "test" / "test_images"

    @property
    def train_ann_path(self) -> Path:
        return self.data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    @property
    def yolo_dir(self) -> Path:
        return self.work_dir / "yolo_dataset"

    @property
    def yolo_obb_dir(self) -> Path:
        return self.work_dir / "yolo_obb_dataset"

    @property
    def yolo_norm_dir(self) -> Path:
        return self.work_dir / "yolo_norm_dataset"

    @property
    def yolo_seg_dir(self) -> Path:
        return self.work_dir / "yolo_seg_dataset"

    @property
    def ckpt_dir(self) -> Path:
        return self.work_dir / "checkpoints"

    @property
    def runs_dir(self) -> Path:
        return self.work_dir / "runs"

    @property
    def split_path(self) -> Path:
        return self.work_dir / "split.json"

    @property
    def splits_dir(self) -> Path:
        return self.work_dir / "splits"

    @property
    def kfold_meta_path(self) -> Path:
        return self.splits_dir / "meta.json"

    @property
    def synthetic_img_dir(self) -> Path:
        return self.work_dir / "synthetic_images"

    def split_path_for(self, fold: Optional[int]) -> Path:
        return self.split_path if fold is None else self.splits_dir / f"fold{fold}.json"

    def yolo_dir_for(self, fold: Optional[int]) -> Path:
        return self.yolo_dir if fold is None else self.yolo_dir / f"fold{fold}"

    def yolo_obb_dir_for(self, fold: Optional[int]) -> Path:
        return self.yolo_obb_dir if fold is None else self.yolo_obb_dir / f"fold{fold}"

    def yolo_norm_dir_for(self, fold: Optional[int]) -> Path:
        return self.yolo_norm_dir if fold is None else self.yolo_norm_dir / f"fold{fold}"

    def yolo_seg_dir_for(self, fold: Optional[int]) -> Path:
        return self.yolo_seg_dir if fold is None else self.yolo_seg_dir / f"fold{fold}"

    def synthetic_img_dir_for(self, fold: Optional[int]) -> Path:
        return self.synthetic_img_dir if fold is None else self.synthetic_img_dir / f"fold{fold}"

    def ensure(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CropConfig:
    crop_size: int = 256
    pad_factor: float = 1.35
    min_crop_side: int = 48
    # rotate the crop so the filament's spine lies horizontal before letterboxing,
    # instead of a plain axis-aligned bbox crop. A U-Net trained with this on must
    # also be run with this on at inference (see inference.py) - they're paired.
    # Closed avenue (didn't beat baseline) - kept for reference, off by default.
    use_spine_orientation: bool = False
    # rotate the crop to match the minAreaRect of the GT polygon (same geometry
    # the OBB detector is trained to predict), instead of the spine's PCA axis.
    # Pairs with an OBB-trained YOLO + `predict_instances_obb` at inference.
    use_obb_orientation: bool = False


@dataclass
class YoloConfig:
    base_weights: str = "yolo11s.pt"
    imgsz: int = 1024
    epochs: int = 60
    batch: int = 8
    optimizer: str = "AdamW"
    lr0: float = 1e-3
    patience: int = 15
    name: str = "yolo11s_filament_detect"


@dataclass
class MaskRCNNConfig:
    imgsz: int = 1024
    epochs: int = 20
    batch: int = 2
    lr: float = 1e-4
    num_workers: int = 4
    val_subset: int = 32          # images scored (coarse-mask PQ) each epoch for checkpoint selection
    score_thres: float = 0.5      # inference score cut - re-sweep per model, like conf_thres for YOLO
    aspect_ratios: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)
    anchor_sizes: tuple = (16, 32, 64, 128, 256)


@dataclass
class UNetConfig:
    base_channels: int = 32
    use_edge_attention: bool = True
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class InferenceConfig:
    conf_thres: float = 0.35  # tuned on val: 0.15->PQ 0.394, 0.25->0.410, 0.35->0.419 (peak), 0.45->0.418
    iou_thres: float = 0.5
    imgsz: int = 1024
    mask_thresh: float = 0.5
    min_instance_area: int = 25
    # suppress a candidate mask covered this much by a higher-confidence instance.
    # tuned on val (at conf_thres=0.35): 0.6->PQ 0.430, 0.4->0.433, 0.25->0.438 (plateau;
    # 0.15 and 0.1 give the identical 387 preds / 0.4381 - nothing left to suppress below this).
    mask_nms_thresh: float = 0.25
    # min mean U-Net probability inside a candidate's own thresholded mask;
    # 0.0 = disabled (no filtering). Meant to be swept alongside a *lower*
    # conf_thres than 0.35, to recover recall without the precision hit that
    # lowering conf_thres alone caused (see notebook history for that sweep).
    mask_conf_thres: float = 0.0
    # min elongation (principal/secondary PCA spread ratio) of a rough mask
    # before trusting its estimated orientation in spine-oriented inference;
    # below this the candidate falls back to the axis-aligned mask. Only used
    # when CropConfig.use_spine_orientation is on.
    min_elongation: float = 1.5


SEED = 42
IMAGE_SIZE = 2048  # fixed native resolution for every MAGFiLO image
# how 1-3 annotator records per image become one detector target set (see data/coco.py::merge_annotator_records)
ANNOTATOR_MERGE_DEFAULT = "majority"
