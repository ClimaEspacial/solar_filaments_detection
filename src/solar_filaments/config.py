"""Central configuration: paths and hyperparameters.

Every path is overridable via environment variable so the same code runs
unmodified on Kaggle (`/kaggle/...`) or locally (defaults below).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


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
    def ckpt_dir(self) -> Path:
        return self.work_dir / "checkpoints"

    @property
    def runs_dir(self) -> Path:
        return self.work_dir / "runs"

    @property
    def split_path(self) -> Path:
        return self.work_dir / "split.json"

    def ensure(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CropConfig:
    crop_size: int = 256
    pad_factor: float = 1.35
    min_crop_side: int = 48


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


SEED = 42
IMAGE_SIZE = 2048  # fixed native resolution for every MAGFiLO image
