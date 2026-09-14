"""Refiner checkpoints carry their architecture implicitly, so loading must infer it.

Passing the wrong `--encoder` used to be possible and would have loaded a mismatched
architecture; `load_refiner` now raises instead.
"""
import pytest
import torch

from solar_filaments.models.refiner import ENCODERS, build_refiner, encoder_of_state_dict, load_refiner


@pytest.mark.parametrize("encoder", ENCODERS)
def test_encoder_is_recoverable_from_the_state_dict(encoder, tmp_path):
    m = build_refiner(encoder, pretrained=False)
    f = tmp_path / "ckpt.pt"
    torch.save(m.state_dict(), f)
    assert encoder_of_state_dict(torch.load(f, map_location="cpu")) == encoder


@pytest.mark.parametrize("encoder", ENCODERS)
def test_round_trip_through_load_refiner(encoder, tmp_path):
    m = build_refiner(encoder, pretrained=False)
    f = tmp_path / "ckpt.pt"
    torch.save(m.state_dict(), f)
    back = load_refiner(f, torch.device("cpu"))
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        assert back(x).shape == m(x).shape == (1, 1, 128, 128)


def test_requesting_the_wrong_encoder_raises(tmp_path):
    f = tmp_path / "ckpt.pt"
    torch.save(build_refiner("base", pretrained=False).state_dict(), f)
    with pytest.raises(ValueError, match="resnet18"):
        load_refiner(f, torch.device("cpu"), expect="resnet18")
