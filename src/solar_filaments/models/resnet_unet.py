"""Crop refiner with an ImageNet-pretrained ResNet18 encoder.

Motivation is a measurement, not a preference for bigger models. Of the 119 ground-truth
filaments our pipeline fails to match on the validation split, 19.3% sit at IoU 0.40-0.50 -
just under the PQ matching cut - while 56.3% have zero overlap at all. The zero-overlap
group is a detector problem no refiner can touch; the 0.40-0.50 band is a *mask quality*
problem, and converting it is worth about +0.021 PQ. That band is the only positive-
expectation lever left after four recall interventions all came back neutral or negative
(see project memory "recall-routes-exhausted").

The scratch encoder (EdgeAttUNet, 8.8M params) is trained from random init on ~7900 crops.
A pretrained encoder brings edge/texture priors those crops cannot supply, which is exactly
what a near-miss boundary needs.

Grayscale input: rather than repeating the channel three times, the pretrained conv1
kernels are summed across RGB into a single input channel. That preserves the learned
filter shapes at the original scale and keeps the first layer's cost at 1/3.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from solar_filaments.models.unet import ConvBlock, EdgeGuidedAttention


class _Up(nn.Module):
    """Upsample, concatenate the encoder skip, then two convs."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, 2, stride=2)
        self.block = ConvBlock(c_in // 2 + c_skip, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Letterboxed crops are square and power-of-two, so shapes match; interpolate
            # anyway so an odd crop_size degrades gracefully instead of throwing.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResNetUNet(nn.Module):
    """U-Net whose encoder is a ResNet18. Same signature as EdgeAttUNet: (N,1,H,W) -> (N,1,H,W) logits."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, pretrained: bool = True,
                 use_edge_attention: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        w = net.conv1.weight.data
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            # Sum the RGB kernels: a grayscale image is what an RGB one collapses to, so the
            # summed kernel responds to the same structure at the same magnitude.
            self.conv1.weight.data = w.sum(dim=1, keepdim=True) if in_ch == 1 else w
        self.bn1, self.relu, self.maxpool = net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2     # 64ch @/4, 128ch @/8
        self.layer3, self.layer4 = net.layer3, net.layer4     # 256ch @/16, 512ch @/32

        self.use_edge_attention = use_edge_attention
        if use_edge_attention:
            self.edge_attn = EdgeGuidedAttention(512)

        self.up4 = _Up(512, 256, 256)   # /32 -> /16
        self.up3 = _Up(256, 128, 128)   # /16 -> /8
        self.up2 = _Up(128, 64, 64)     # /8  -> /4
        self.up1 = _Up(64, 64, 64)      # /4  -> /2   (skip = pre-maxpool stem)
        self.up0 = _Up(64, 0, 32)       # /2  -> /1
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.relu(self.bn1(self.conv1(x)))   # 64ch  @/2
        e1 = self.layer1(self.maxpool(s0))        # 64ch  @/4
        e2 = self.layer2(e1)                      # 128ch @/8
        e3 = self.layer3(e2)                      # 256ch @/16
        b = self.layer4(e3)                       # 512ch @/32
        if self.use_edge_attention:
            b = self.edge_attn(b, x)

        d = self.up4(b, e3)
        d = self.up3(d, e2)
        d = self.up2(d, e1)
        d = self.up1(d, s0)
        d = self.up0(d)
        return self.head(d)
