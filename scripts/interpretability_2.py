# ==============================================================================
# AUTOMATED INTERPRETABILITY REPORT GENERATOR (PDF) - MULTI-SEGMENTATION
#
# Features:
# - Generates a PDF per patient.
# - Visualizes: Raw Anatomy, GT(s), Grad-CAM++, Guided Grad-CAM, Occlusion, Guided BP.
# - Supports multiple segmentations per session (all contours).
# - Uses color palette for each segmentation contour.
# - FIX: Correctly aligns Dataframe indices with Validation Dataloader.
# ==============================================================================

import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    EnsureTyped, ResampleToMatchd, AsDiscreted, Resized
)
from monai.visualize import GradCAMpp, GuidedBackpropSmoothGrad, blend_images, OcclusionSensitivity
from monai.data import CacheDataset, DataLoader

# --- PROJECT IMPORTS ---
try:
    from bimcv_prostate.data.dataloaders import ProstateImageDataLoader as Dataloader
    from bimcv_prostate.models.efficientnet import EfficientNet_pretrained
    from bimcv_prostate.utils.data_utils import apply_path_rewrites
except ImportError:
    print("[WARN] Project modules not found. Ensure you are running from project root.")

# --- CONFIGURATION ---
OUTPUT_FOLDER = "../Results/Patient_Reports_PDF"
XNAT_ROOT = Path("../Files/data/validation/descargas_xnat_segura/")
DATA_CSV = "../Files/data_interpretability.csv" 

# Model & Method Config
TARGET_LAYER = "model._blocks.3.27._project_conv" # EfficientNet-B7
CLASS_TO_EXPLAIN = 1  # 1 = Cancer (csPC)
CHANNEL_TO_SHOW = 1   # 1 = ADC

# Occlusion Config
LOW_RES_MASK_SIZE = (16, 16, 8) 
LOW_RES_STRIDE = (12, 12, 4)    

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- MODEL PATHS ---
MODEL_PATHS = [
    "../logs/classification/Prostate_BIMCV/02-Sep-2025-16:24:58/models/best-model-weights.pth",
    "../logs/classification/Prostate_BIMCV/03-Sep-2025-12:27:57/models/best-model-weights.pth",
    "../logs/classification/Prostate_BIMCV/04-Sep-2025-08:50:36/models/best-model-weights.pth",
    "../logs/classification/Prostate_BIMCV/05-Sep-2025-08:55:26/models/best-model-weights.pth",
    "../logs/classification/Prostate_BIMCV/07-Sep-2025-19:13:33/models/best-model-weights.pth",
]

# ==============================================================================
# 1. MODEL UTILITIES
# ==============================================================================

def load_model(weight_path):
    """Loads EfficientNet-B7 model."""
    if not os.path.exists(weight_path):
        print(f"[WARN] Weight file not found: {weight_path}")
        return None
    try:
        state = torch.load(weight_path, map_location="cpu")["state_dict"]
        if list(state.keys())[0].startswith("model."):
            model = EfficientNet_pretrained("efficientnet-b7", 2, 3, None)
            model.load_state_dict(state)
        else:
            model = EfficientNet_pretrained("efficientnet-b7", 2, 3, weight_path)
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model {weight_path}: {e}")
        return None

class EnsembleMeanLogits(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    def forward(self, x, **kwargs):
        logits_list = [m(x) for m in self.models]
        return torch.stack(logits_list, dim=0).mean(dim=0)

def _norm_safe(x):
    x = x - x.min()
    m = x.max()
    return x / (m + 1e-8)

def resize_to_target(vol_tensor, target_tensor, mode="trilinear"):
    if vol_tensor.ndim == 3: vol_tensor = vol_tensor.unsqueeze(0).unsqueeze(0)
    elif vol_tensor.ndim == 4: vol_tensor = vol_tensor.unsqueeze(0)
    target_size = target_tensor.shape[-3:]
    out = F.interpolate(vol_tensor, size=target_size, mode=mode, align_corners=False)
    return out.squeeze(0).squeeze(0)

# ==============================================================================
# 2. GT LOADING UTILS (MULTI-SEGMENTATION)
# ==============================================================================

def build_label_transform(target_shape_hwd: Tuple[int, int, int]):
    """Misma lógica que el notebook: ResampleToMatch + Resized a input_shape."""
    return Compose([
        LoadImaged(keys=["t2", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["t2", "label"]),
        EnsureTyped(keys=["t2", "label"]),
        ResampleToMatchd(keys="label", key_dst="t2", mode="nearest", padding_mode="zeros"),
        Resized(keys=["label"], spatial_size=target_shape_hwd, mode="nearest"),
        AsDiscreted(keys="label", threshold=0.5),
    ])

def find_scan_folders_with_seg(session_dir: Path) -> List[Path]:
    scans = [p for p in session_dir.iterdir() if p.is_dir()]
    if not scans:
        raise FileNotFoundError(f"No hay scans en: {session_dir}")
    with_seg = [s for s in scans if (s / "SEGMENTATION").exists()]
    return with_seg

def find_anatomical_image(base_scan_folder: Path) -> str:
    nifti_dir = base_scan_folder / "NIFTI"
    if nifti_dir.exists():
        files = list(nifti_dir.glob("*.nii*"))
        if files: return str(files[0])
    dicom_dir = base_scan_folder / "DICOM"
    if dicom_dir.exists(): return str(dicom_dir)
    for item in base_scan_folder.iterdir():
        if item.is_dir() and item.name not in {"SEGMENTATION", "SEGMENTATIONS"}:
            if any(item.iterdir()): return str(item)
    raise FileNotFoundError(f"No se encontró anatomía en {base_scan_folder}")

def list_seg_sources(seg_root: Path):
    """Busca archivos .dcm, .ima o .nii dentro de SEGMENTATION"""
    sources = []
    if not seg_root.exists(): return sources
    for sub in seg_root.iterdir():
        if not sub.is_dir(): continue
        niftis = list(sub.glob("*.nii*"))
        if niftis: 
            sources.extend([str(n) for n in niftis])
            continue
        dcm_files = sorted(list(sub.rglob("*.dcm")) + list(sub.rglob("*.ima")))
        if dcm_files:
            sources.append(str(dcm_files[0])) # Tomamos el primero de la serie
    return sorted(list(set(sources)))

def list_all_seg_sources(session_dir: Path) -> List[Tuple[str, str]]:
    """
    Retorna lista de tuplas (scan_name, seg_path) para todas las segmentaciones
    encontradas en todos los scans de la sesión.
    """
    scans_with_seg = find_scan_folders_with_seg(session_dir)
    all_sources: List[Tuple[str, str]] = []
    for scan_folder in scans_with_seg:
        seg_root = scan_folder / "SEGMENTATION"
        sources = list_seg_sources(seg_root)
        for src in sources:
            all_sources.append((scan_folder.name, src))
    return all_sources

def resize_mask_to_shape(mask_1hwd: torch.Tensor, target_shape_hwd):
    """Asegura que la máscara tenga el mismo tamaño exacto que el tensor del modelo"""
    if mask_1hwd.shape[-3:] == tuple(target_shape_hwd):
        return mask_1hwd.bool()
    m = mask_1hwd.float()
    m_rs = F.interpolate(
        m.unsqueeze(0),
        size=tuple(target_shape_hwd),
        mode="nearest",
    ).squeeze(0)
    return (m_rs > 0.5)

def load_gt_masks_in_model_space(idx: int, pack_img: torch.Tensor, df_merged: pd.DataFrame):
    """
    Carga todas las segmentaciones disponibles en XNAT, alineadas al espacio del modelo.
    Retorna una lista de máscaras (1, H, W, D) y el session_id.
    """
    if "ID_XNAT_session" not in df_merged.columns:
        if "label_MIDS_session" in df_merged.columns:
            session_id = str(df_merged.iloc[idx]["label_MIDS_session"])
        else:
            raise KeyError("No encuentro columna de sesión en df_merged.")
    else:
        session_id = str(df_merged.iloc[idx]["ID_XNAT_session"])

    session_dir = XNAT_ROOT / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"No existe directorio de sesión: {session_dir}")

    seg_sources = list_all_seg_sources(session_dir)
    if not seg_sources:
        raise FileNotFoundError(f"No hay máscaras válidas en {session_dir}")

    print(f"   [INFO] Loading GT for {session_id} from {len(seg_sources)} sources.")
    for i, (scan_name, src) in enumerate(seg_sources, start=1):
        print(f"   [INFO] GT source {i}/{len(seg_sources)} [{scan_name}]: {src}")

    # Usa la anatomía del CSV (T2) para alinear todas las segmentaciones
    if "t2" not in df_merged.columns:
        raise KeyError("No encuentro columna 't2' en df_merged para alinear segmentaciones.")
    t2_path = df_merged.iloc[idx]["t2"]
    label_transform = build_label_transform(tuple(pack_img.shape[-3:]))

    masks = []
    for scan_name, s in seg_sources:
        try:
            print(f"   [INFO] Reading segmentation: {s}")
            sample = label_transform({"t2": t2_path, "label": s})
            seg = sample["label"] # (1, H, W, D)
            seg = (seg > 0)
            seg_model = resize_mask_to_shape(seg, pack_img.shape[-3:]) # (1,H,W,D)
            masks.append(seg_model)
        except Exception as e:
            print(f"   [WARN] Failed to read segmentation {s} (scan {scan_name}): {e}")
            continue

    if len(masks) == 0:
        raise RuntimeError("Fallo total al cargar segmentaciones.")

    return masks, session_id

# ==============================================================================
# 3. VISUALIZATION UTILS (PDF Grid + MULTI-CONTOUR)
# ==============================================================================

def add_grid_page_to_pdf(pdf, background_vol, overlay_vols, title, every_n=1, is_gt=False, cmap="jet"):
    """
    Adds a page to the PDF with a grid of slices + multiple GT contours.
    overlay_vols: None or List[Tensor(1,H,W,D)]
    """
    if background_vol.shape[0] == 3:
        bg = background_vol.permute(3, 1, 2, 0).cpu().numpy()
    else:
        bg = background_vol.permute(3, 1, 2, 0).cpu().numpy()

    gt_list = []
    if overlay_vols is not None:
        for ov in overlay_vols:
            gt = ov.squeeze(0).permute(2, 0, 1).float().cpu().numpy()
            gt_list.append(gt)

    num_slices = bg.shape[0]
    indices = range(0, num_slices, every_n)
    n_plots = len(indices)
    cols = 6
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(22, 3.5 * rows))
    fig.suptitle(title, fontsize=20, y=0.98, fontweight='bold')
    
    if isinstance(axes, np.ndarray): axes = axes.flatten()
    else: axes = [axes]

    # Color cycle for multiple segmentations
    colors = ["lime", "cyan", "yellow", "magenta", "orange", "red", "blue", "white"]

    for i, idx in enumerate(indices):
        if i >= len(axes): break
        ax = axes[i]
        
        img_show = np.clip(bg[idx], 0, 1)
        ax.imshow(img_show, cmap=cmap if cmap else None)
        
        if is_gt and gt_list:
            any_contour = False
            for k, gt in enumerate(gt_list):
                if gt[idx].max() > 0:
                    ax.contour(gt[idx], colors=colors[k % len(colors)], linewidths=1.8, levels=[0.5])
                    any_contour = True
            if any_contour:
                ax.set_title(f"Slice {idx} [GT x{len(gt_list)}]", fontsize=10, color='green', fontweight='bold')
            else:
                ax.set_title(f"Slice {idx}", fontsize=10)
        else:
            ax.set_title(f"Slice {idx}", fontsize=10)
        
        ax.axis('off')

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

# ==============================================================================
# 4. PROCESSING LOGIC
# ==============================================================================

def compute_low_res_occlusion(img_tensor, ensemble_model, class_idx):
    occ_eng = OcclusionSensitivity(ensemble_model, mask_size=LOW_RES_MASK_SIZE, n_batch=10)
    occ_map, _ = occ_eng(img_tensor.unsqueeze(0), stride=LOW_RES_STRIDE)
    if occ_map.shape[1] > 1: occ_map = occ_map[:, class_idx:class_idx+1]
    occ_up = resize_to_target(occ_map.detach().cpu(), img_tensor)
    return _norm_safe(occ_up)

def build_limited_loader(loader_obj, partition: str, start_index: int = 0, limit: Optional[int] = None):
    """
    Crea un DataLoader con un subconjunto del partition.
    Esto evita cachear toda la partición cuando se pide un número pequeño de instancias.
    """
    if limit is None:
        return loader_obj(partition), (0, None)

    if partition in loader_obj._skip_partitions:
        return None, (0, 0)

    group = loader_obj._groupby.get_group(partition)
    total = len(group)
    start_index = max(0, start_index)
    end_index = min(start_index + limit, total)
    if start_index >= total:
        raise IndexError(f"start_index ({start_index}) fuera de rango (total={total}).")

    group = group.iloc[start_index:end_index]

    labels = torch.as_tensor(group[loader_obj._label_column].values, dtype=torch.int64)
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=loader_obj._num_classes).float()

    data = [
        {"t2": t2, "adc": adc, "dwi": dwi, "label": label}
        for t2, adc, dwi, label in zip(
            group[loader_obj._csv_image_columns[0]].values,
            group[loader_obj._csv_image_columns[1]].values,
            group[loader_obj._csv_image_columns[2]].values,
            one_hot_labels,
        )
    ]

    transforms = loader_obj._train_transforms if partition == "train" else loader_obj._val_transforms
    dataset = CacheDataset(data=data, transform=transforms, num_workers=7)

    loader_args = dict(loader_obj._config_args)
    if partition != "train":
        loader_args["shuffle"] = False
    return DataLoader(dataset, **loader_args), (start_index, end_index)

def process_patient(pdf, img_tensor, gt_masks, models, class_idx, noise_thresh=0.0):
    img_cpu = img_tensor.detach().cpu()
    
    # --- A. Raw Anatomy ---
    t2_bg = img_cpu[CHANNEL_TO_SHOW]
    t2_norm = _norm_safe(t2_bg)
    bg_anatomy = torch.stack([t2_norm]*3, dim=0)
    add_grid_page_to_pdf(pdf, bg_anatomy, None, "1. Raw Anatomy (No Overlay)", every_n=1, cmap="gray")
    
    # --- B. Ground Truth(s) ---
    add_grid_page_to_pdf(pdf, bg_anatomy, gt_masks, "2. Ground Truth Segmentations (Multiple Contours)", every_n=1, is_gt=True)

    # --- C. Maps ---
    grad_cams, guided_cams, guided_bps = [], [], []
    for model in models:
        model.zero_grad()
        gcam_eng = GradCAMpp(model, target_layers=TARGET_LAYER)
        gbp_eng = GuidedBackpropSmoothGrad(model, n_samples=60)
        
        gc = _norm_safe(gcam_eng(x=img_tensor.unsqueeze(0), class_idx=class_idx).detach().cpu())
        gbp = gbp_eng(x=img_tensor.unsqueeze(0), index=class_idx).detach().cpu()
        
        gc_up = resize_to_target(gc, img_tensor)
        gbp_saliency = _norm_safe(torch.abs(gbp).mean(dim=1).squeeze(0))
        ggc = _norm_safe(gc_up * gbp_saliency)
        
        grad_cams.append(gc_up)
        guided_cams.append(ggc)
        guided_bps.append(gbp_saliency)

    stack_gc = _norm_safe(torch.stack(grad_cams).mean(dim=0))
    stack_ggc = _norm_safe(torch.stack(guided_cams).mean(dim=0))
    stack_gbp = _norm_safe(torch.stack(guided_bps).mean(dim=0))

    ensemble_wrapper = EnsembleMeanLogits(models).to(device).eval()
    occ_up = compute_low_res_occlusion(img_tensor, ensemble_wrapper, class_idx)

    def make_blend(vol, cmap, thresh=True, is_detail=False):
        if thresh:
            mask = (vol > noise_thresh)
            vol_clean = vol.clone()
            vol_clean[~mask] = 0
        else:
            vol_clean = vol
            mask = (vol > 0)
        alpha = 0.7 if is_detail else 0.5
        blended = blend_images(
            image=t2_norm.unsqueeze(0), label=vol_clean.unsqueeze(0),
            alpha=alpha, cmap=cmap, rescale_arrays=True
        )
        for c in range(3): blended[c][~mask] = t2_norm[~mask]
        return blended

    bg_gc = make_blend(stack_gc, "jet", True)
    add_grid_page_to_pdf(pdf, bg_gc, gt_masks, "3. Grad-CAM++ (Localization + GT)", is_gt=True)

    bg_ggc = make_blend(stack_ggc, "magma", True, is_detail=True)
    add_grid_page_to_pdf(pdf, bg_ggc, gt_masks, "4. Guided Grad-CAM (Fusion Details + GT)", is_gt=True)

    bg_occ = make_blend(occ_up, "jet", True)
    add_grid_page_to_pdf(pdf, bg_occ, gt_masks, "5. Occlusion Sensitivity (Validation + GT)", is_gt=True)
    
    bg_gbp = make_blend(stack_gbp, "magma", False, is_detail=True)
    add_grid_page_to_pdf(pdf, bg_gbp, gt_masks, "6. Guided Backprop (Raw Features + GT)", is_gt=True)

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    print(f"\n>>> REPORT GENERATOR (Limit: {args.limit}, Start: {args.start_index})")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    try:
        df_full = pd.read_csv(DATA_CSV)
        # Aplica los mismos rewrites que el dataloader
        if "t2" in df_full.columns and "adc" in df_full.columns and "dwi" in df_full.columns:
            apply_path_rewrites(df_full, ["t2", "adc", "dwi"], [["ceib/", ""], ["prueba/p0052021_reborn/", "p0052021/"], ["prueba/", ""]])
        df_merged_all = df_full[df_full['partition'] == 'val'].reset_index(drop=True)
        print(f"Metadata loaded. Validation samples: {len(df_merged_all)}")
        
        dataloader_obj = Dataloader(
            path=DATA_CSV,
            path_rewrites=[["ceib/", ""], ["prueba/p0052021_reborn/", "p0052021/"], ["prueba/", ""]],
            label_column="csPC_x",
        )
        val_loader, (s_idx, e_idx) = build_limited_loader(
            dataloader_obj,
            partition="val",
            start_index=args.start_index,
            limit=args.limit,
        )
        if args.limit is not None:
            df_merged = df_merged_all.iloc[s_idx:e_idx].reset_index(drop=True)
            print(f"[INFO] Limited loader: {len(df_merged)} samples (from {s_idx} to {e_idx - 1}).")
        else:
            df_merged = df_merged_all
    except Exception as e:
        print(f"Error initializing data: {e}")
        sys.exit(1)
    
    models_pretrained = []
    for p in tqdm(MODEL_PATHS, desc="Loading Models"):
        m = load_model(p)
        if m: models_pretrained.append(m)
    if not models_pretrained: sys.exit(1)

    dataset = val_loader.dataset

    for idx in tqdm(range(len(dataset)), desc="Processing"):
        try:
            global_idx = (args.start_index or 0) + idx
            case_id = f"Patient_{global_idx:03d}"
            if "ID_XNAT_session" in df_merged.columns:
                case_id = str(df_merged.iloc[idx]["ID_XNAT_session"])

            base_pdf_path = Path(OUTPUT_FOLDER) / f"{case_id}_Report.pdf"
            pdf_path = base_pdf_path
            counter = 1
            while pdf_path.exists():
                pdf_path = Path(OUTPUT_FOLDER) / f"{case_id}_Report_{counter}.pdf"
                counter += 1

            img_tensor = dataset[idx]['image'].to(device)
            
            try:
                gt_masks, _ = load_gt_masks_in_model_space(idx, img_tensor.detach().cpu(), df_merged)
            except Exception:
                gt_masks = [torch.zeros_like(img_tensor[0:1]).bool()]

            with PdfPages(pdf_path) as pdf:
                process_patient(pdf, img_tensor, gt_masks, models_pretrained, CLASS_TO_EXPLAIN)
            
            plt.close('all')
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error {idx}: {e}")
            continue

if __name__ == "__main__":
    main()