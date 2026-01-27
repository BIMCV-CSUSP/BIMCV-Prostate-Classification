from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model using BIMCV-AIKit.")
    parser.add_argument("--config", default="configs/config.json", help="Path to the config JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subprocess.run(["bimcv_train", "-c", args.config], check=True)


if __name__ == "__main__":
    main()
