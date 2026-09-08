"""One place to build and load the crop refiner, whichever encoder it uses.

The encoder is inferred from the checkpoint's own keys rather than from a flag, so
existing `unet*_best.pt` files keep loading unchanged and a wrong `--encoder` at
inference time cannot silently load the wrong architecture - it raises instead.
"""
from pathlib import Path
from typing import Optional

import torch

from solar_filaments.models.resnet_unet import ResNetUNet
from solar_filaments.models.unet import EdgeAttUNet

ENCODERS = ("base", "resnet18")


def encoder_of_state_dict(sd: dict) -> str:
    return "resnet18" if any(k.startswith("layer4.") for k in sd) else "base"


def build_refiner(encoder: str = "base", use_edge_attention: bool = True,
                  norm_layer: str = "batch", base_channels: int = 32,
                  pretrained: bool = True) -> torch.nn.Module:
    if encoder == "resnet18":
        return ResNetUNet(use_edge_attention=use_edge_attention, pretrained=pretrained)
    return EdgeAttUNet(base=base_channels, use_edge_attention=use_edge_attention,
                       norm_layer=norm_layer)


def load_refiner(path, device, use_edge_attention: bool = True, norm_layer: str = "batch",
                 expect: Optional[str] = None) -> torch.nn.Module:
    """Build the architecture the checkpoint was saved from, and load it."""
    sd = torch.load(Path(path), map_location=device)
    enc = encoder_of_state_dict(sd)
    if expect is not None and expect != enc:
        raise ValueError(f"{path} holds a '{enc}' refiner but '{expect}' was requested")
    model = build_refiner(enc, use_edge_attention=use_edge_attention, norm_layer=norm_layer,
                          pretrained=False)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model
