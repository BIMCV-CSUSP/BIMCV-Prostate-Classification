from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bimcv_prostate.training.average_model import build_new_model_from_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average backbone weights across folds.")
    parser.add_argument("--fold", action="append", required=True, help="Path to a fold checkpoint. Repeat per fold.")
    parser.add_argument(
        "--weight",
        action="append",
        required=True,
        type=float,
        help="Validation weight for the corresponding fold. Repeat per fold.",
    )
    parser.add_argument("--model-name", default="efficientnet-b0")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--out", default="averaged_pretrained_backbone.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_new_model_from_folds(
        fold_ckpt_paths=args.fold,
        fold_validation_scores=args.weight,
        new_num_classes=args.num_classes,
        model_name=args.model_name,
        in_channels=args.in_channels,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)


if __name__ == "__main__":
    main()
