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


def out_channels_of_state_dict(sd: dict) -> int:
    """Number of output channels the checkpoint was trained with. Two means an auxiliary
    spine head was used during training; inference reads channel 0 either way."""
    w = sd.get("head.weight")
    return int(w.shape[0]) if w is not None else 1


def build_refiner(encoder: str = "base", use_edge_attention: bool = True,
                  norm_layer: str = "batch", base_channels: int = 32,
                  pretrained: bool = True, out_ch: int = 1) -> torch.nn.Module:
    if encoder == "resnet18":
        return ResNetUNet(use_edge_attention=use_edge_attention, pretrained=pretrained,
                          out_ch=out_ch)
    return EdgeAttUNet(base=base_channels, use_edge_attention=use_edge_attention,
                       norm_layer=norm_layer, out_ch=out_ch)


def load_refiner(path, device, use_edge_attention: bool = True, norm_layer: str = "batch",
                 expect: Optional[str] = None) -> torch.nn.Module:
    """Build the architecture the checkpoint was saved from, and load it."""
    sd = torch.load(Path(path), map_location=device)
    enc = encoder_of_state_dict(sd)
    if expect is not None and expect != enc:
        raise ValueError(f"{path} holds a '{enc}' refiner but '{expect}' was requested")
    model = build_refiner(enc, use_edge_attention=use_edge_attention, norm_layer=norm_layer,
                          pretrained=False, out_ch=out_channels_of_state_dict(sd))
    model.load_state_dict(sd)
    model.to(device).eval()
    return model
