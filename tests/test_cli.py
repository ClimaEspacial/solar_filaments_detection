"""Every subcommand must at least build its parser and format its help.

A single unescaped '%' in a help string makes argparse raise at --help time, which is how
`train-fulldisk` silently broke.
"""
import argparse
import importlib

import pytest

SUBCOMMANDS = ["prepare", "train_yolo", "train_maskrcnn", "train_unet", "train_rescorer",
               "train_fulldisk", "evaluate", "predict"]


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_parser_builds_and_help_formats(name):
    mod = importlib.import_module(f"solar_filaments.cli.{name}")
    p = argparse.ArgumentParser(prog=name)
    mod.add_args(p)
    assert p.format_help()


def test_main_parser_registers_every_subcommand():
    from solar_filaments.cli.main import build_parser
    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == {n.replace("_", "-") for n in SUBCOMMANDS}
