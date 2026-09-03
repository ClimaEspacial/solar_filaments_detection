"""Mask R-CNN (torchvision, ResNet50-FPN v2) backend.

Configured for filaments rather than COCO objects: small anchors (16-256 px at
the 1024 working resolution) and a wide aspect-ratio set (0.25-4), since
filaments are long and thin and appear at every orientation. Replacing the
aspect ratios re-initializes the RPN head (its output width is
num_anchors-dependent); the COCO-pretrained backbone, FPN and ROI heads are
kept. The image is downsampled 2048 -> 1024 for the network and its masks are
upsampled back, so `coarse_mask` is full-resolution but only 1024-detailed -
the U-Net refiner (inference.refine_candidates) restores the fine structure.
"""
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead

from solar_filaments.detectors.base import Candidate

DEFAULT_ANCHOR_SIZES: Tuple[int, ...] = (16, 32, 64, 128, 256)
DEFAULT_ASPECT_RATIOS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def build_maskrcnn(pretrained: bool = True, imgsz: int = 1024,
                   anchor_sizes: Sequence[int] = DEFAULT_ANCHOR_SIZES,
                   aspect_ratios: Sequence[float] = DEFAULT_ASPECT_RATIOS,
                   detections_per_img: int = 100) -> nn.Module:
    if pretrained:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    else:
        model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)

    # 2 classes: background + filament
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, hidden, 2)

    sizes = tuple((int(s),) for s in anchor_sizes)
    ratios = (tuple(float(r) for r in aspect_ratios),) * len(sizes)
    model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    model.rpn.head = RPNHead(model.backbone.out_channels, len(aspect_ratios), conv_depth=2)

    # more proposals survive for many small objects per image
    model.rpn._pre_nms_top_n = dict(training=3000, testing=2000)
    model.rpn._post_nms_top_n = dict(training=2000, testing=1000)
    model.roi_heads.detections_per_img = detections_per_img

    model.transform.min_size = (imgsz,)
    model.transform.max_size = imgsz
    return model


def to_input_tensor(img_1024: np.ndarray, device) -> torch.Tensor:
    """Grayscale uint8 (already at the working resolution) -> (3, H, W) float in [0, 1]."""
    t = torch.from_numpy(img_1024).float().div_(255.0)
    return t.unsqueeze(0).repeat(3, 1, 1).to(device)


class MaskRCNNDetector:
    def __init__(self, model: nn.Module, device, score_thres: float = 0.5,
                 mask_thresh: float = 0.5, imgsz: int = 1024):
        self.model, self.device = model.eval().to(device), device
        self.score_thres, self.mask_thresh, self.imgsz = score_thres, mask_thresh, imgsz

    @torch.no_grad()
    def detect(self, img: np.ndarray) -> List[Candidate]:
        H, W = img.shape
        small = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        out = self.model([to_input_tensor(small, self.device)])[0]
        keep = out["scores"] >= self.score_thres
        boxes = out["boxes"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        masks = out["masks"][keep, 0].cpu().numpy() >= self.mask_thresh  # (N, imgsz, imgsz)

        sx, sy = W / self.imgsz, H / self.imgsz
        cands = []
        for b, s, m in zip(boxes, scores, masks):
            if not m.any():
                continue
            full = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            box = np.array([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], dtype=np.float32)
            cands.append(Candidate(box, float(s), full))
        return cands
