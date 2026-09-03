"""Flat U-Net: an ultralightweight full-disk filament segmenter.

Reimplemented from "Flat U-Net: An Efficient Ultralightweight Model for Solar
Filament Segmentation in Full-disk H-alpha Images" (Zhu et al., ApJ 980, 176,
2025; arXiv:2502.07259), which reports DSC 0.79 / precision 0.93 / **recall
0.69** with 0.26M parameters, against 0.69 / 0.98 / **0.53** for a classical
28.95M-parameter U-Net on the same task. Recall is the axis this project is
bottlenecked on, which is why this architecture is worth having.

Two departures from a classical U-Net:

1. **Flat channels.** Width does *not* double with depth - every encoder and
   decoder stage keeps the same channel count. This is what makes the model
   ~100x smaller; the paper argues the usual channel pyramid is redundant for a
   single-class, thin-structure segmentation task.
2. **Channel attention blocks (SCA / CSA).** Query/key/value come from
   bias-free convolutions. CSA computes a full channel-by-channel similarity
   (every channel attends to every other); SCA computes only the within-channel
   term, which is cheaper in memory and is what the paper puts in the backbone,
   reserving CSA for the bottleneck. Both are followed by a residual add, batch
   norm and ReLU.

This is a *semantic* segmenter (one filaments-vs-background mask per image), not
an instance model - see detectors/fulldisk.py for how its output is turned into
instance candidates and into an independent per-candidate confidence signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """SCA (`full=False`) or CSA (`full=True`) block.

    Both project the input to q/k/v with bias-free 1x1 convolutions and reweight
    the value features by a softmax over channel-similarity scores:

    - SCA: score[c] = <q[c], k[c]> / (H*W)              -> one score per channel
    - CSA: A[c, c'] = softmax_c'(<q[c], k[c']> / (H*W))  -> channels mix

    The output is scaled by the channel count so that a uniform attention map is
    the identity in expectation (softmax weights sum to 1 across C).
    """

    def __init__(self, channels: int, full: bool = False):
        super().__init__()
        self.full = full
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        scale = 1.0 / (H * W)

        if self.full:
            qf = q.flatten(2)                                  # B, C, HW
            kf = k.flatten(2)
            attn = torch.softmax(torch.bmm(qf, kf.transpose(1, 2)) * scale, dim=-1)  # B, C, C
            out = torch.bmm(attn, v.flatten(2)).reshape(B, C, H, W) * C
        else:
            score = (q * k).flatten(2).sum(dim=2) * scale       # B, C
            weight = torch.softmax(score, dim=1).view(B, C, 1, 1)
            out = v * weight * C

        return F.relu(self.norm(self.proj(out) + x), inplace=True)


class FlatBlock(nn.Module):
    """Two 3x3 convolutions at constant width, then a channel-attention block."""

    def __init__(self, c_in: int, c_out: int, attention: str = "sca"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
        if attention == "none":
            self.attn = nn.Identity()
        else:
            self.attn = ChannelAttention(c_out, full=(attention == "csa"))

    def forward(self, x):
        return self.attn(self.conv(x))


class FlatUNet(nn.Module):
    """Constant-width U-Net with SCA in the encoder/decoder and CSA at the
    bottleneck (the paper's recommended mixed configuration). `width=32` gives
    ~0.26M parameters."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, width: int = 32, depth: int = 4,
                 bottleneck_attention: str = "csa", attention: str = "sca"):
        super().__init__()
        self.depth = depth
        self.pool = nn.MaxPool2d(2)

        self.encoders = nn.ModuleList(
            [FlatBlock(in_ch if i == 0 else width, width, attention) for i in range(depth)])
        self.bottleneck = FlatBlock(width, width, bottleneck_attention)
        # each decoder stage concatenates its skip connection, hence 2*width in
        self.ups = nn.ModuleList([nn.ConvTranspose2d(width, width, 2, stride=2) for _ in range(depth)])
        self.decoders = nn.ModuleList([FlatBlock(width * 2, width, attention) for _ in range(depth)])
        self.head = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:  # odd input sizes
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        return self.head(x)
