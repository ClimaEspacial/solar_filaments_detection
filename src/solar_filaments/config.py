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
class FullDiskConfig:
    """Flat U-Net full-disk semantic branch (models/flat_unet.py)."""
    imgsz: int = 512          # the paper's resolution; filaments are thin, so 1024 is worth sweeping
    width: int = 32           # constant channel width -> ~0.26M params
    epochs: int = 60
    batch_size: int = 8
    lr: float = 1e-3
    num_workers: int = 4
    bce_weight: float = 0.5   # filaments are ~1% of full-disk pixels; don't let BCE dominate Dice


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
    # Local validation and the real Kaggle board disagree on this, unresolved as of 2026-09-03:
    # - single 70-image split: 0.35 best (0.394/0.410/0.419/0.418 at .15/.25/.35/.45)
    # - 5-fold OOF (707 images, majority-consensus targets): 0.15 best by a lot (OOF PQ
    #   .15->0.4383, .25->0.4227, .35->0.3604)
    # - real leaderboard: 6 submissions, conf~0.35 scored 0.36 three times, conf~0.15 scored
    #   0.34 three times, across two different model generations.
    #
    # RESOLVED 2026-09-06, and the conclusion is that this knob is dead. Five further
    # submissions spanning conf 0.15-0.30 on the all-data seg@2048 model ALL scored 0.36,
    # and the local sweep agrees (PQ 0.4407/0.4493/0.4472, spread 0.009 against a
    # two-decimal board). Lowering conf trades FN for FP at ~1:1; SQ stays pinned at
    # 0.67-0.68 throughout. We sit on the flat top of the precision/recall curve, so no
    # value of this parameter buys anything - do not spend submissions sweeping it.
    #
    # The transferability failure was the "majority" annotator-merge target, now measured:
    # recall is 27% on support<0.5 filaments vs 70% on support>=0.5, because that target
    # taught the detector to skip them, while PQ scores every annotator record. The
    # marginal analysis puts break-even support at 0.32, below every support level that
    # occurs in MAGFiLO (0.33/0.50/0.67/1.00), so `union` is the correct target and
    # `majority` was a calibration error. See project memory
    # "project-solar-filaments-union-target" and "-conf-threshold-dead".
    conf_thres: float = 0.35
    iou_thres: float = 0.5
    imgsz: int = 1024
    # Probability cut turning the U-Net's output into a binary mask. Never swept until
    # 2026-09-03, when the predicted masks turned out to be 1.14-1.45x the area of the
    # pycocotools-rasterized GT at the old 0.5 default. Swept on fold0 val against
    # pycocotools GT (the competition standard, see data/coco.py::polygon_to_mask):
    # 0.5->PQ 0.3782, 0.6->0.3912, 0.7->0.3993, 0.8->0.4038 (peak), 0.9->0.4032 at
    # conf=0.35; same 0.8 peak at conf=0.15 (0.3995/0.4129/0.4208/0.4234/0.4192). At 0.8 the
    # predicted/GT area ratio is ~1.01, i.e. the optimum is exactly where mask size matches
    # the GT. Independent corroboration: the public "LB 0.70" notebook used 0.70 here.
    mask_thresh: float = 0.8
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
