from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ResampleToMatchd,
    Resized,
    ScaleIntensityd,
)
from monai.visualize import GradCAMpp, GuidedBackpropSmoothGrad, OcclusionSensitivity, blend_images

from bimcv_prostate.utils.data_utils import apply_path_rewrites


DEFAULT_REWRITE_COLUMNS: Tuple[str, str, str] = ("t2", "adc", "dwi")
DEFAULT_GT_COLORS: Tuple[str, ...] = (
    "lime",
    "cyan",
    "yellow",
    "magenta",
    "orange",
    "red",
    "blue",
    "white",
)


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path
    xnat_root: Path
    data_csv: Path
    model_paths: Sequence[Path]
    target_layer: str
    class_index: int = 1
    channel_index: int = 1
    occlusion_mask_size: Tuple[int, int, int] = (16, 16, 8)
    occlusion_stride: Tuple[int, int, int] = (12, 12, 4)
    noise_threshold: float = 0.0
    partition: str = "val"
    label_column: str = "csPC"
    path_rewrites: Sequence[Tuple[str, str]] = (
        ("ceib/", ""),
        ("prueba/p0052021_reborn/", "p0052021/"),
        ("prueba/", ""),
    )
    rewrite_columns: Sequence[str] = DEFAULT_REWRITE_COLUMNS
    occlusion_batch_size: int = 10
    guided_backprop_samples: int = 60


def normalize_safe(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor - tensor.min()
    max_value = tensor.max()
    return tensor / (max_value + 1e-8)


def resize_to_match(
    volume_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    mode: str = "trilinear",
) -> torch.Tensor:
    if volume_tensor.ndim == 3:
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
    elif volume_tensor.ndim == 4:
        volume_tensor = volume_tensor.unsqueeze(0)

    target_size = target_tensor.shape[-3:]
    resized = F.interpolate(volume_tensor, size=target_size, mode=mode, align_corners=False)
    return resized.squeeze(0).squeeze(0)


def resize_mask_to_shape(mask_1hwd: torch.Tensor, target_shape_hwd: Sequence[int]) -> torch.Tensor:
    if mask_1hwd.shape[-3:] == tuple(target_shape_hwd):
        return mask_1hwd.bool()

    mask_float = mask_1hwd.float()
    resized = F.interpolate(mask_float.unsqueeze(0), size=tuple(target_shape_hwd), mode="nearest").squeeze(0)
    return resized > 0.5


def load_efficientnet_model(
    weight_path: Path,
    device: torch.device,
    model_factory: Callable[[Optional[str]], nn.Module],
) -> Optional[nn.Module]:
    if not weight_path.exists():
        print(f"[WARN] Weight file not found: {weight_path}")
        return None

    try:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)["state_dict"]
        first_key = next(iter(state_dict.keys()))

        if first_key.startswith("model."):
            model = model_factory(None)
            model.load_state_dict(state_dict)
        else:
            model = model_factory(str(weight_path))

        model.to(device).eval()
        return model
    except Exception as exc:  # pragma: no cover - defensive logging for offline scripts
        print(f"[ERROR] Failed to load model {weight_path}: {exc}")
        return None


def load_models(
    model_paths: Iterable[Path],
    device: torch.device,
    model_factory: Callable[[Optional[str]], nn.Module],
) -> List[nn.Module]:
    models: List[nn.Module] = []
    for path in model_paths:
        model = load_efficientnet_model(path, device=device, model_factory=model_factory)
        if model is not None:
            models.append(model)
    return models


class EnsembleMeanLogits(nn.Module):
    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs: torch.Tensor, **_: object) -> torch.Tensor:
        logits_list = [model(inputs) for model in self.models]
        return torch.stack(logits_list, dim=0).mean(dim=0)


def build_pair_transform() -> Compose:
    """Load an image/label pair and align the label to the image space."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ResampleToMatchd(keys="label", key_dst="image", mode="nearest", padding_mode="zeros"),
            AsDiscreted(keys="label", threshold=0.5),
        ]
    )


def build_label_transform(target_shape_hwd: Tuple[int, int, int]) -> Compose:
    """Resample a segmentation to the T2 space and then resize to the model input size."""
    return Compose(
        [
            LoadImaged(keys=["t2", "label"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["t2", "label"]),
            EnsureTyped(keys=["t2", "label"]),
            ResampleToMatchd(keys="label", key_dst="t2", mode="nearest", padding_mode="zeros"),
            Resized(keys=["label"], spatial_size=target_shape_hwd, mode="nearest"),
            AsDiscreted(keys="label", threshold=0.5),
        ]
    )


def find_scan_with_segmentation(session_dir: Path) -> Path:
    scans = [path for path in session_dir.iterdir() if path.is_dir()]
    if not scans:
        raise FileNotFoundError(f"No scans found in: {session_dir}")

    for scan_dir in scans:
        if (scan_dir / "SEGMENTATION").exists():
            return scan_dir
    return scans[0]


def find_scan_folders_with_segmentation(session_dir: Path) -> List[Path]:
    scans = [path for path in session_dir.iterdir() if path.is_dir()]
    if not scans:
        raise FileNotFoundError(f"No scans found in: {session_dir}")
    return [scan_dir for scan_dir in scans if (scan_dir / "SEGMENTATION").exists()]


def find_anatomical_image(scan_dir: Path) -> str:
    nifti_dir = scan_dir / "NIFTI"
    if nifti_dir.exists():
        nifti_files = list(nifti_dir.glob("*.nii*"))
        if nifti_files:
            return str(nifti_files[0])

    dicom_dir = scan_dir / "DICOM"
    if dicom_dir.exists():
        return str(dicom_dir)

    for item in scan_dir.iterdir():
        if item.is_dir() and item.name not in {"SEGMENTATION", "SEGMENTATIONS"}:
            if any(item.iterdir()):
                return str(item)

    raise FileNotFoundError(f"No anatomical image found in {scan_dir}")


def list_segmentation_sources(segmentation_root: Path) -> List[str]:
    """Collect .dcm, .ima, or .nii files under a SEGMENTATION directory."""
    sources: List[str] = []
    if not segmentation_root.exists():
        return sources

    for subdir in segmentation_root.iterdir():
        if not subdir.is_dir():
            continue

        nifti_files = list(subdir.glob("*.nii*"))
        if nifti_files:
            sources.extend([str(path) for path in nifti_files])
            continue

        dicom_files = sorted(list(subdir.rglob("*.dcm")) + list(subdir.rglob("*.ima")))
        if dicom_files:
            # Use the first file from the series as the entry point.
            sources.append(str(dicom_files[0]))

    return sorted(set(sources))


def list_all_segmentation_sources(session_dir: Path) -> List[Tuple[str, str]]:
    """Return (scan_name, segmentation_path) pairs for all scans in a session."""
    scans_with_seg = find_scan_folders_with_segmentation(session_dir)
    all_sources: List[Tuple[str, str]] = []

    for scan_dir in scans_with_seg:
        seg_root = scan_dir / "SEGMENTATION"
        for source in list_segmentation_sources(seg_root):
            all_sources.append((scan_dir.name, source))

    return all_sources


def _resolve_session_id(df: pd.DataFrame, idx: int) -> str:
    if "ID_XNAT_session" in df.columns:
        return str(df.iloc[idx]["ID_XNAT_session"])
    if "label_MIDS_session" in df.columns:
        return str(df.iloc[idx]["label_MIDS_session"])
    raise KeyError("No session identifier column found in dataframe.")


def load_single_gt_mask_in_model_space(
    idx: int,
    image_tensor_chwd: torch.Tensor,
    df: pd.DataFrame,
    xnat_root: Path,
) -> Tuple[torch.Tensor, str]:
    """Load and merge segmentations for a session, aligned to the model space."""
    session_id = _resolve_session_id(df, idx)
    session_dir = xnat_root / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

    scan_dir = find_scan_with_segmentation(session_dir)
    segmentation_root = scan_dir / "SEGMENTATION"
    anatomical_source = find_anatomical_image(scan_dir)
    segmentation_sources = list_segmentation_sources(segmentation_root)

    print(f"   [INFO] Loading GT for {session_id} from {len(segmentation_sources)} sources.")

    if not segmentation_sources:
        raise FileNotFoundError(f"No valid masks found in {segmentation_root}")

    pair_transform = build_pair_transform()
    merged_mask: Optional[torch.Tensor] = None

    for source in segmentation_sources:
        try:
            sample = pair_transform({"image": anatomical_source, "label": source})
            segmentation = sample["label"]
            merged_mask = (segmentation > 0) if merged_mask is None else (merged_mask | (segmentation > 0))
        except Exception:
            # A bad segmentation should not block the report for the session.
            continue

    if merged_mask is None:
        raise RuntimeError("Failed to load any segmentation for the session.")

    gt_mask = resize_mask_to_shape(merged_mask, image_tensor_chwd.shape[-3:])
    return gt_mask, session_id


def load_all_gt_masks_in_model_space(
    idx: int,
    image_tensor_chwd: torch.Tensor,
    df: pd.DataFrame,
    xnat_root: Path,
) -> Tuple[List[torch.Tensor], str]:
    """Load all available segmentations for a session, aligned to the model space."""
    session_id = _resolve_session_id(df, idx)
    session_dir = xnat_root / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

    segmentation_sources = list_all_segmentation_sources(session_dir)
    if not segmentation_sources:
        raise FileNotFoundError(f"No valid masks found in {session_dir}")

    print(f"   [INFO] Loading GT for {session_id} from {len(segmentation_sources)} sources.")
    for source_index, (scan_name, source) in enumerate(segmentation_sources, start=1):
        print(f"   [INFO] GT source {source_index}/{len(segmentation_sources)} [{scan_name}]: {source}")

    if "t2" not in df.columns:
        raise KeyError("Column 't2' is required to align segmentations.")

    t2_path = df.iloc[idx]["t2"]
    label_transform = build_label_transform(tuple(image_tensor_chwd.shape[-3:]))

    masks: List[torch.Tensor] = []
    for scan_name, source in segmentation_sources:
        try:
            print(f"   [INFO] Reading segmentation: {source}")
            sample = label_transform({"t2": t2_path, "label": source})
            segmentation = sample["label"] > 0
            masks.append(resize_mask_to_shape(segmentation, image_tensor_chwd.shape[-3:]))
        except Exception as exc:
            print(f"   [WARN] Failed to read segmentation {source} (scan {scan_name}): {exc}")
            continue

    if not masks:
        raise RuntimeError("Failed to load any segmentation for the session.")

    return masks, session_id


def _volume_to_plot_array(background_vol_chwd: torch.Tensor) -> np.ndarray:
    # Expected input is (C, H, W, D). We plot as (D, H, W, C).
    return background_vol_chwd.permute(3, 1, 2, 0).detach().cpu().numpy()


def _mask_to_plot_array(mask_1hwd: torch.Tensor) -> np.ndarray:
    # Convert from (1, H, W, D) to (D, H, W) for contour plotting.
    return mask_1hwd.squeeze(0).permute(2, 0, 1).float().detach().cpu().numpy()


def add_grid_page(
    pdf: PdfPages,
    background_vol_chwd: torch.Tensor,
    gt_masks: Optional[Sequence[torch.Tensor]],
    title: str,
    *,
    every_n: int = 1,
    show_gt: bool = False,
    cmap: Optional[str] = "jet",
    gt_colors: Sequence[str] = DEFAULT_GT_COLORS,
) -> None:
    """Add a grid of slices to the PDF, optionally overlaying one or more GT contours."""
    background = _volume_to_plot_array(background_vol_chwd)

    gt_arrays: List[np.ndarray] = []
    if gt_masks:
        gt_arrays = [_mask_to_plot_array(mask) for mask in gt_masks]
        if show_gt and all(gt.sum() == 0 for gt in gt_arrays):
            print(f"   [WARN-PLOT] GT masks for '{title}' are empty.")

    num_slices = background.shape[0]
    slice_indices = list(range(0, num_slices, every_n))
    num_plots = len(slice_indices)
    cols = 6
    rows = max(1, math.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(22, 3.5 * rows))
    fig.suptitle(title, fontsize=20, y=0.98, fontweight="bold")

    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for plot_index, slice_index in enumerate(slice_indices):
        if plot_index >= len(axes_list):
            break
        ax = axes_list[plot_index]

        image_to_show = np.clip(background[slice_index], 0, 1)
        ax.imshow(image_to_show, cmap=cmap if cmap else None)

        if show_gt and gt_arrays:
            any_contour = False
            for gt_index, gt_array in enumerate(gt_arrays):
                if gt_array[slice_index].max() > 0:
                    ax.contour(
                        gt_array[slice_index],
                        colors=gt_colors[gt_index % len(gt_colors)],
                        linewidths=2.0,
                        levels=[0.5],
                    )
                    any_contour = True

            if any_contour:
                suffix = f" [GT x{len(gt_arrays)}]" if len(gt_arrays) > 1 else " [GT]"
                ax.set_title(
                    f"Slice {slice_index}{suffix}",
                    fontsize=10,
                    color="green",
                    fontweight="bold",
                )
            else:
                ax.set_title(f"Slice {slice_index}", fontsize=10)
        else:
            ax.set_title(f"Slice {slice_index}", fontsize=10)

        ax.axis("off")

    for remaining_index in range(plot_index + 1, len(axes_list)):
        axes_list[remaining_index].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def compute_low_res_occlusion(
    image_tensor_chwd: torch.Tensor,
    ensemble_model: nn.Module,
    class_index: int,
    mask_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    batch_size: int,
) -> torch.Tensor:
    occlusion_engine = OcclusionSensitivity(ensemble_model, mask_size=mask_size, n_batch=batch_size)
    occlusion_map, _ = occlusion_engine(image_tensor_chwd.unsqueeze(0), stride=stride)
    if occlusion_map.shape[1] > 1:
        occlusion_map = occlusion_map[:, class_index : class_index + 1]
    occlusion_resized = resize_to_match(occlusion_map.detach().cpu(), image_tensor_chwd)
    return normalize_safe(occlusion_resized)


def _blend_map_with_background(
    map_tensor_hwd: torch.Tensor,
    background_hwd: torch.Tensor,
    noise_threshold: float,
    cmap: str,
    *,
    detail: bool = False,
) -> torch.Tensor:
    if noise_threshold > 0:
        mask = map_tensor_hwd > noise_threshold
        cleaned_map = map_tensor_hwd.clone()
        cleaned_map[~mask] = 0
    else:
        cleaned_map = map_tensor_hwd
        mask = map_tensor_hwd > 0

    alpha = 0.7 if detail else 0.5
    blended = blend_images(
        image=background_hwd.unsqueeze(0),
        label=cleaned_map.unsqueeze(0),
        alpha=alpha,
        cmap=cmap,
        rescale_arrays=True,
    )

    for channel_index in range(3):
        blended[channel_index][~mask] = background_hwd[~mask]

    return blended


def run_interpretability_pages(
    pdf: PdfPages,
    image_tensor_chwd: torch.Tensor,
    gt_masks: Sequence[torch.Tensor],
    models: Sequence[nn.Module],
    config: ReportConfig,
    device: torch.device,
) -> None:
    image_cpu = image_tensor_chwd.detach().cpu()

    background_channel = image_cpu[config.channel_index]
    background_norm = normalize_safe(background_channel)
    background_rgb = torch.stack([background_norm] * 3, dim=0)

    add_grid_page(
        pdf,
        background_rgb,
        None,
        "1. Raw Anatomy (No Overlay)",
        every_n=1,
        show_gt=False,
        cmap="gray",
    )

    add_grid_page(
        pdf,
        background_rgb,
        gt_masks,
        "2. Ground Truth Segmentation(s)",
        every_n=1,
        show_gt=True,
        cmap="gray",
    )

    grad_cams: List[torch.Tensor] = []
    guided_cams: List[torch.Tensor] = []
    guided_backprops: List[torch.Tensor] = []

    for model in models:
        model.zero_grad()
        gradcam_engine = GradCAMpp(model, target_layers=config.target_layer)
        guided_bp_engine = GuidedBackpropSmoothGrad(model, n_samples=config.guided_backprop_samples)

        gradcam_raw = gradcam_engine(x=image_tensor_chwd.unsqueeze(0), class_idx=config.class_index)
        gradcam_norm = normalize_safe(gradcam_raw.detach().cpu())

        guided_bp_raw = guided_bp_engine(x=image_tensor_chwd.unsqueeze(0), index=config.class_index).detach().cpu()

        gradcam_resized = resize_to_match(gradcam_norm, image_tensor_chwd)
        guided_bp_saliency = normalize_safe(torch.abs(guided_bp_raw).mean(dim=1).squeeze(0))
        guided_gradcam = normalize_safe(gradcam_resized * guided_bp_saliency)

        grad_cams.append(gradcam_resized)
        guided_cams.append(guided_gradcam)
        guided_backprops.append(guided_bp_saliency)

    gradcam_mean = normalize_safe(torch.stack(grad_cams).mean(dim=0))
    guided_gradcam_mean = normalize_safe(torch.stack(guided_cams).mean(dim=0))
    guided_bp_mean = normalize_safe(torch.stack(guided_backprops).mean(dim=0))

    ensemble_model = EnsembleMeanLogits(models).to(device).eval()
    occlusion_map = compute_low_res_occlusion(
        image_tensor_chwd,
        ensemble_model,
        class_index=config.class_index,
        mask_size=config.occlusion_mask_size,
        stride=config.occlusion_stride,
        batch_size=config.occlusion_batch_size,
    )

    gradcam_blend = _blend_map_with_background(
        gradcam_mean,
        background_norm,
        config.noise_threshold,
        cmap="jet",
        detail=False,
    )
    add_grid_page(pdf, gradcam_blend, gt_masks, "3. Grad-CAM++ (Localization + GT)", show_gt=True)

    guided_gradcam_blend = _blend_map_with_background(
        guided_gradcam_mean,
        background_norm,
        config.noise_threshold,
        cmap="magma",
        detail=True,
    )
    add_grid_page(pdf, guided_gradcam_blend, gt_masks, "4. Guided Grad-CAM (Fusion Details + GT)", show_gt=True)

    occlusion_blend = _blend_map_with_background(
        occlusion_map,
        background_norm,
        config.noise_threshold,
        cmap="jet",
        detail=False,
    )
    add_grid_page(pdf, occlusion_blend, gt_masks, "5. Occlusion Sensitivity (Validation + GT)", show_gt=True)

    guided_bp_blend = _blend_map_with_background(
        guided_bp_mean,
        background_norm,
        noise_threshold=0.0,
        cmap="magma",
        detail=True,
    )
    add_grid_page(pdf, guided_bp_blend, gt_masks, "6. Guided Backprop (Raw Features + GT)", show_gt=True)


def build_limited_loader(
    loader_obj: object,
    partition: str,
    start_index: int = 0,
    limit: Optional[int] = None,
) -> Tuple[Optional[DataLoader], Tuple[int, Optional[int]]]:
    """Create a partition DataLoader optionally limited to a slice of rows."""
    if limit is None:
        return loader_obj(partition), (0, None)

    if partition in loader_obj._skip_partitions:  # type: ignore[attr-defined]
        return None, (0, 0)

    group = loader_obj._groupby.get_group(partition)  # type: ignore[attr-defined]
    total = len(group)
    start_index = max(0, start_index)
    end_index = min(start_index + limit, total)
    if start_index >= total:
        raise IndexError(f"start_index ({start_index}) is out of range (total={total}).")

    group = group.iloc[start_index:end_index]

    labels = torch.as_tensor(group[loader_obj._label_column].values, dtype=torch.int64)  # type: ignore[attr-defined]
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=loader_obj._num_classes).float()  # type: ignore[attr-defined]

    data = [
        {"t2": t2, "adc": adc, "dwi": dwi, "label": label}
        for t2, adc, dwi, label in zip(
            group[loader_obj._csv_image_columns[0]].values,  # type: ignore[attr-defined]
            group[loader_obj._csv_image_columns[1]].values,  # type: ignore[attr-defined]
            group[loader_obj._csv_image_columns[2]].values,  # type: ignore[attr-defined]
            one_hot_labels,
        )
    ]

    transforms = loader_obj._train_transforms if partition == "train" else loader_obj._val_transforms  # type: ignore[attr-defined]
    dataset = CacheDataset(data=data, transform=transforms, num_workers=7)

    loader_args = dict(loader_obj._config_args)  # type: ignore[attr-defined]
    if partition != "train":
        loader_args["shuffle"] = False

    return DataLoader(dataset, **loader_args), (start_index, end_index)


def load_validation_dataframe(
    data_csv: Path,
    *,
    partition: str,
    path_rewrites: Optional[Sequence[Tuple[str, str]]] = None,
    rewrite_columns: Sequence[str] = DEFAULT_REWRITE_COLUMNS,
) -> pd.DataFrame:
    df = pd.read_csv(data_csv)
    if rewrite_columns and path_rewrites:
        available_columns = [column for column in rewrite_columns if column in df.columns]
        if available_columns:
            apply_path_rewrites(df, available_columns, path_rewrites)
    return df[df["partition"] == partition].reset_index(drop=True)


def unique_pdf_path(output_dir: Path, case_id: str) -> Path:
    base_path = output_dir / f"{case_id}_Report.pdf"
    candidate = base_path
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{case_id}_Report_{counter}.pdf"
        counter += 1
    return candidate
