from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from bimcv_prostate.data.dataloaders import ProstateImageDataLoader
from bimcv_prostate.interpretability import (
    ReportConfig,
    build_limited_loader,
    load_all_gt_masks_in_model_space,
    load_models,
    load_validation_dataframe,
    run_interpretability_pages,
    unique_pdf_path,
)
from bimcv_prostate.models.efficientnet import EfficientNet_pretrained


MODEL_PATHS = [
    Path("../logs/classification/Prostate_BIMCV/02-Sep-2025-16:24:58/models/best-model-weights.pth"),
    Path("../logs/classification/Prostate_BIMCV/03-Sep-2025-12:27:57/models/best-model-weights.pth"),
    Path("../logs/classification/Prostate_BIMCV/04-Sep-2025-08:50:36/models/best-model-weights.pth"),
    Path("../logs/classification/Prostate_BIMCV/05-Sep-2025-08:55:26/models/best-model-weights.pth"),
    Path("../logs/classification/Prostate_BIMCV/07-Sep-2025-19:13:33/models/best-model-weights.pth"),
]


REPORT_CONFIG = ReportConfig(
    output_dir=Path("../Results/Patient_Reports_PDF"),
    xnat_root=Path("../Files/data/validation/descargas_xnat_segura/"),
    data_csv=Path("../Files/data_interpretability.csv"),
    model_paths=MODEL_PATHS,
    target_layer="model._blocks.3.27._project_conv",
    class_index=1,
    channel_index=1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interpretability PDF reports (multi-segmentation).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of validation cases to process.")
    parser.add_argument(
        "--start-index",
        dest="start_index",
        type=int,
        default=0,
        help="Validation index to start from.",
    )
    return parser.parse_args()


def _build_model_factory() -> Callable[[Optional[str]], torch.nn.Module]:
    def model_factory(weights: Optional[str]):
        return EfficientNet_pretrained("efficientnet-b7", 2, 3, weights)

    return model_factory


def main() -> None:
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    REPORT_CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> REPORT GENERATOR (limit={args.limit}, start_index={args.start_index})")

    try:
        df_val_all = load_validation_dataframe(
            REPORT_CONFIG.data_csv,
            partition=REPORT_CONFIG.partition,
            path_rewrites=REPORT_CONFIG.path_rewrites,
            rewrite_columns=REPORT_CONFIG.rewrite_columns,
        )
        print(f"Metadata loaded. Validation samples: {len(df_val_all)}")

        dataloader = ProstateImageDataLoader(
            path=str(REPORT_CONFIG.data_csv),
            path_rewrites=REPORT_CONFIG.path_rewrites,
            label_column=REPORT_CONFIG.label_column,
        )
        val_loader, (slice_start, slice_end) = build_limited_loader(
            dataloader,
            partition=REPORT_CONFIG.partition,
            start_index=args.start_index,
            limit=args.limit,
        )
        if val_loader is None:
            print("Validation loader is disabled by configuration.")
            sys.exit(1)

        if args.limit is not None and slice_end is not None:
            df_val = df_val_all.iloc[slice_start:slice_end].reset_index(drop=True)
            print(f"[INFO] Limited loader: {len(df_val)} samples (from {slice_start} to {slice_end - 1}).")
        else:
            df_val = df_val_all
    except Exception as exc:
        print(f"Error initializing data: {exc}")
        sys.exit(1)

    model_factory = _build_model_factory()
    models = load_models(REPORT_CONFIG.model_paths, device=device, model_factory=model_factory)
    if not models:
        print("No models were loaded successfully.")
        sys.exit(1)

    dataset = val_loader.dataset
    global_start_index = args.start_index or 0

    for local_index in tqdm(range(len(dataset)), desc="Processing"):
        global_index = global_start_index + local_index
        try:
            case_id = f"Patient_{global_index:03d}"
            if "ID_XNAT_session" in df_val.columns:
                case_id = str(df_val.iloc[local_index]["ID_XNAT_session"])

            pdf_path = unique_pdf_path(REPORT_CONFIG.output_dir, case_id)
            image_tensor = dataset[local_index]["image"].to(device)

            try:
                gt_masks, _ = load_all_gt_masks_in_model_space(
                    local_index,
                    image_tensor.detach().cpu(),
                    df_val,
                    REPORT_CONFIG.xnat_root,
                )
            except Exception:
                gt_masks = [torch.zeros_like(image_tensor[0:1]).bool()]

            with PdfPages(pdf_path) as pdf:
                run_interpretability_pages(pdf, image_tensor, gt_masks, models, REPORT_CONFIG, device=device)

            plt.close("all")
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - defensive logging for offline scripts
            print(f"Error at index {global_index}: {exc}")
            continue


if __name__ == "__main__":
    main()
