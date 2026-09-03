import argparse

from solar_filaments.cli import evaluate, predict, prepare, train_maskrcnn, train_unet, train_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="filament", description="Solar filament segmentation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    modules = {
        "prepare": prepare,
        "train-yolo": train_yolo,
        "train-maskrcnn": train_maskrcnn,
        "train-unet": train_unet,
        "evaluate": evaluate,
        "predict": predict,
    }
    for name, module in modules.items():
        sub_parser = sub.add_parser(name)
        module.add_args(sub_parser)
        sub_parser.set_defaults(func=module.run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
