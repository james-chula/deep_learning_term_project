"""CLI: python -m bloodcell_training.cli train --config CONFIG --provider-module MODULE."""
from __future__ import annotations
import argparse
from .config import load_config
from .orchestrate import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="run authorized train/validation-only fine-tuning")
    train.add_argument("--config", required=True)
    train.add_argument(
        "--provider-module",
        required=True,
        help="module exporting canonical build_provider_datasets",
    )
    args = parser.parse_args()
    if args.command == "train":
        print(run_training(load_config(args.config), args.provider_module))


if __name__ == "__main__":
    main()
