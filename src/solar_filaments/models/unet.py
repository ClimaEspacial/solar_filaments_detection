"""Compact U-Net with an optional edge-guided attention bottleneck.

Plain U-Net decoder/encoder with a small self-attention block at the
bottleneck whose Query/Key are biased by a learned edge map of the *input*
crop (idea from EdgeAttNet, arXiv:2509.02964) - meant to help recover thin
barbs. Channel widths are kept small (32-256) since pipeline efficiency is
also scored.

`norm_layer="instance"` swaps BatchNorm for InstanceNorm, per "A Modern
ConvNet for Solar Filament Detection" (arXiv:2607.24525): "IN focuses more on
local contrast variations, while BN retains global intensity statistics...
particularly suitable for solar filament segmentation because local
intensity variations are critical, and even decisive, indicators on Halpha
images." Default stays "batch" (original behavior, existing checkpoints).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_layer: str, c_out: int) -> nn.Module:
    if norm_layer == "instance":
        return nn.InstanceNorm2d(c_out, affine=True)
    if norm_layer == "batch":
        return nn.BatchNorm2d(c_out)
    raise ValueError(f"unknown norm_layer: {norm_layer!r} (expected 'batch' or 'instance')")


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, norm_layer: str = "batch"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), _make_norm(norm_layer, c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), _make_norm(norm_layer, c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EdgeGuidedAttention(nn.Module):
    """Self-attention over the bottleneck feature map, with Q/K biased by an edge prior."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.edge_conv = nn.Conv2d(1, 1, 3, padding=1)
        self.edge_proj = nn.Conv2d(1, channels, 1)
        self.mhsa = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, feat, input_image):
        B, C, h, w = feat.shape
        edge = torch.sigmoid(self.edge_conv(input_image))
        edge = F.interpolate(edge, size=(h, w), mode="bilinear", align_corners=False)
        edge = self.edge_proj(edge)  # B,C,h,w

        tokens = feat.flatten(2).transpose(1, 2)          # B, h*w, C
        edge_tok = edge.flatten(2).transpose(1, 2)         # B, h*w, C
        q = k = tokens + edge_tok
        attn_out, _ = self.mhsa(q, k, tokens)
        tokens = self.norm(tokens + attn_out)
        return tokens.transpose(1, 2).reshape(B, C, h, w)


class EdgeAttUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32, use_edge_attention=True, norm_layer: str = "batch"):
        super().__init__()
        self.use_edge_attention = use_edge_attention
        self.enc1 = ConvBlock(in_ch, base, norm_layer)
        self.enc2 = ConvBlock(base, base * 2, norm_layer)
        self.enc3 = ConvBlock(base * 2, base * 4, norm_layer)
        self.enc4 = ConvBlock(base * 4, base * 8, norm_layer)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 8, base * 16, norm_layer)
        if use_edge_attention:
            self.edge_attn = EdgeGuidedAttention(base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8, norm_layer)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4, norm_layer)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, norm_layer)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, norm_layer)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        if self.use_edge_attention:
            b = self.edge_attn(b, x)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
