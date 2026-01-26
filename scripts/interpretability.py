# ==============================================================================
# AUTOMATED INTERPRETABILITY REPORT GENERATOR (PDF)
#
# Features:
# - Generates a PDF per patient.
# - Visualizes: Raw Anatomy, GT, Grad-CAM++, Guided Grad-CAM, Occlusion, Guided BP.
# - Uses Green Contours (Lime) for GT overlay.
# - FIX: Correctly aligns Dataframe indices with Validation Dataloader.
# ==============================================================================

import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityd, EnsureTyped, ResampleToMatchd, AsDiscreted
)
from monai.visualize import GradCAMpp, GuidedBackpropSmoothGrad, blend_images, OcclusionSensitivity
from monai.data import DataLoader

# --- PROJECT IMPORTS ---
try:
    from bimcv_prostate.data.dataloaders import ProstateImageDataLoader as Dataloader
    from bimcv_prostate.models.efficientnet import EfficientNet_pretrained
except ImportError:
    print("[WARN] Project modules not found. Ensure you are running from project root.")

# --- CONFIGURATION ---
OUTPUT_FOLDER = "../Results/Patient_Reports_PDF"
XNAT_ROOT = Path("../Files/data/validation/descargas_xnat_segura/")
DATA_CSV = "../Files/data_interpretability.csv" 

# Model & Method Config
TARGET_LAYER = "model._blocks.4.37._project_conv" # EfficientNet-B7
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
# 2. GT LOADING UTILS (RESTORED EXACTLY AS REQUESTED)
# ==============================================================================

# Transforms para cargar pares (Imagen, Mascara) y alinearlos
pair_transform = Compose([
    LoadImaged(keys=["image", "label"], reader="ITKReader"),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys="image"),
    EnsureTyped(keys=["image", "label"]),
    # Resamplea la etiqueta al espacio de la imagen usando 'nearest' (para no interpolar clases)
    ResampleToMatchd(keys="label", key_dst="image", mode="nearest", padding_mode="zeros"),
    AsDiscreted(keys="label", threshold=0.5), # Binarizar
])

def find_scan_folder_with_seg(session_dir: Path) -> Path:
    scans = [p for p in session_dir.iterdir() if p.is_dir()]
    if not scans: raise FileNotFoundError(f"No hay scans en: {session_dir}")
    for s in scans:
        if (s / "SEGMENTATION").exists(): return s
    return scans[0]

def find_anatomical_image(base_scan_folder: Path) -> str:
    nifti_dir = base_scan_folder / "NIFTI"
    if nifti_dir.exists():
        files = list(nifti_dir.glob("*.nii*"))
        if files: return str(files[0])
    dicom_dir = base_scan_folder / "DICOM"
    if dicom_dir.exists(): return str(dicom_dir)
    # Fallback
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
        # Nifti
        niftis = list(sub.glob("*.nii*"))
        if niftis: 
            sources.extend([str(n) for n in niftis])
            continue
        # Dicom recursive
        dcm_files = sorted(list(sub.rglob("*.dcm")) + list(sub.rglob("*.ima")))
        if dcm_files:
            sources.append(str(dcm_files[0])) # Tomamos el primero de la serie
    return sorted(list(set(sources)))

def resize_mask_to_shape(mask_1hwd: torch.Tensor, target_shape_hwd):
    """Asegura que la máscara tenga el mismo tamaño exacto que el tensor del modelo"""
    if mask_1hwd.shape[-3:] == tuple(target_shape_hwd):
        return mask_1hwd.bool()
    
    m = mask_1hwd.float()
    # Interpolamos nearest para no perder la naturaleza binaria
    m_rs = F.interpolate(
        m.unsqueeze(0),
        size=tuple(target_shape_hwd),
        mode="nearest",
    ).squeeze(0)
    return (m_rs > 0.5)

def load_gt_mask_in_model_space(idx: int, pack_img: torch.Tensor, df_merged: pd.DataFrame):
    """
    Busca la segmentación en XNAT, la carga y la alinea al espacio del modelo.
    pack_img: Tensor de referencia (C,H,W,D)
    df_merged: DataFrame (filtered) corresponding to the dataset
    """
    # 1. Obtener ID de sesión
    if "ID_XNAT_session" not in df_merged.columns:
        if "label_MIDS_session" in df_merged.columns:
            session_id = str(df_merged.iloc[idx]["label_MIDS_session"])
        else:
            raise KeyError("No encuentro columna de sesión en df_merged.")
    else:
        session_id = str(df_merged.iloc[idx]["ID_XNAT_session"])

    # 2. Rutas
    session_dir = XNAT_ROOT / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"No existe directorio de sesión: {session_dir}")

    scan_folder = find_scan_folder_with_seg(session_dir)
    seg_root = scan_folder / "SEGMENTATION"
    anat_src = find_anatomical_image(scan_folder)
    seg_sources = list_seg_sources(seg_root)

    print(f"   [INFO] Loading GT for {session_id} from {len(seg_sources)} sources.")

    if not seg_sources:
        raise FileNotFoundError(f"No hay máscaras válidas en {seg_root}")

    # 3. Cargar y Fusionar
    total_mask = None
    for s in seg_sources:
        try:
            sample = pair_transform({"image": anat_src, "label": s})
            seg = sample["label"] # (1, H, W, D)
            total_mask = (seg > 0) if total_mask is None else (total_mask | (seg > 0))
        except Exception as e:
            # print(f"[WARN] Error cargando {Path(s).name}: {e}")
            pass

    if total_mask is None:
        raise RuntimeError("Fallo total al cargar segmentaciones.")

    # 4. Redimensionar
    gt_model = resize_mask_to_shape(total_mask, pack_img.shape[-3:]) # (1, H, W, D)
    return gt_model, session_id

# ==============================================================================
# 3. VISUALIZATION UTILS (PDF Grid + DEBUGGING)
# ==============================================================================

def add_grid_page_to_pdf(pdf, background_vol, overlay_vol, title, every_n=1, is_gt=False, cmap="jet"):
    """
    Adds a page to the PDF with a grid of slices + Green Contours for GT.
    Includes Debugging prints.
    """
    # 1. Background Setup (Permute to Axial D-H-W for plotting)
    # Expected Input: (3, H, W, D) -> Output Needed: (D, H, W, 3)
    if background_vol.shape[0] == 3:
        bg = background_vol.permute(3, 1, 2, 0).cpu().numpy()
    else:
        bg = background_vol.permute(3, 1, 2, 0).cpu().numpy()

    # 2. GT Setup (Align dimensions with Background)
    # Input overlay_vol is (1, H, W, D) -> Permute to (D, H, W)
    if overlay_vol is not None:
        gt = overlay_vol.squeeze(0).permute(2, 0, 1).float().cpu().numpy()
        
        # --- DEBUG SECTION ---
        if is_gt and gt.sum() == 0:
            print(f"   [WARN-PLOT] GT Mask for '{title}' is EMPTY (0 pixels).")
        # ---------------------
    else:
        gt = np.zeros(bg.shape[:3])

    num_slices = bg.shape[0]
    indices = range(0, num_slices, every_n)
    n_plots = len(indices)
    cols = 6
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(22, 3.5 * rows))
    fig.suptitle(title, fontsize=20, y=0.98, fontweight='bold')
    
    if isinstance(axes, np.ndarray): axes = axes.flatten()
    else: axes = [axes]

    for i, idx in enumerate(indices):
        if i >= len(axes): break
        ax = axes[i]
        
        # Plot Background
        img_show = np.clip(bg[idx], 0, 1)
        ax.imshow(img_show, cmap=cmap if cmap else None)
        
        # Plot Contour
        if is_gt and gt[idx].max() > 0:
            # Green contour (lime), linewidth=2.0
            ax.contour(gt[idx], colors='lime', linewidths=2.0, levels=[0.5])
            ax.set_title(f"Slice {idx} [GT]", fontsize=10, color='green', fontweight='bold')
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

def process_patient(pdf, img_tensor, gt_mask, models, class_idx, noise_thresh=0.0):
    img_cpu = img_tensor.detach().cpu()
    
    # --- A. Raw Anatomy ---
    t2_bg = img_cpu[CHANNEL_TO_SHOW]
    t2_norm = _norm_safe(t2_bg)
    bg_anatomy = torch.stack([t2_norm]*3, dim=0)
    add_grid_page_to_pdf(pdf, bg_anatomy, None, "1. Raw Anatomy (No Overlay)", every_n=1, cmap="gray")
    
    # --- B. Ground Truth Only ---
    add_grid_page_to_pdf(pdf, bg_anatomy, gt_mask, "2. Ground Truth Segmentation (Green Contour)", every_n=1, is_gt=True)

    # --- C. Maps ---
    grad_cams, guided_cams, guided_bps = [], [], []
    for model in models:
        model.zero_grad()
        # GradCAM++
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

    # Occlusion
    ensemble_wrapper = EnsembleMeanLogits(models).to(device).eval()
    occ_up = compute_low_res_occlusion(img_tensor, ensemble_wrapper, class_idx)

    # Blending Helper
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

    # --- Generate Remaining Pages ---
    bg_gc = make_blend(stack_gc, "jet", True)
    add_grid_page_to_pdf(pdf, bg_gc, gt_mask, "3. Grad-CAM++ (Localization + GT)", is_gt=True)

    bg_ggc = make_blend(stack_ggc, "magma", True, is_detail=True)
    add_grid_page_to_pdf(pdf, bg_ggc, gt_mask, "4. Guided Grad-CAM (Fusion Details + GT)", is_gt=True)

    bg_occ = make_blend(occ_up, "jet", True)
    add_grid_page_to_pdf(pdf, bg_occ, gt_mask, "5. Occlusion Sensitivity (Validation + GT)", is_gt=True)
    
    bg_gbp = make_blend(stack_gbp, "magma", False, is_detail=True)
    add_grid_page_to_pdf(pdf, bg_gbp, gt_mask, "6. Guided Backprop (Raw Features + GT)", is_gt=True)

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
    
    # 1. Load Data
    try:
        df_full = pd.read_csv(DATA_CSV)
        # FILTER DATAFRAME TO MATCH VALIDATION SET
        # This is critical for index alignment
        df_merged = df_full[df_full['partition'] == 'val'].reset_index(drop=True)
        print(f"Metadata loaded. Validation samples: {len(df_merged)}")
        
        dataloader_obj = Dataloader(
            path=DATA_CSV,
            path_rewrites=[["ceib/", ""], ["prueba/p0052021_reborn/", "p0052021/"], ["prueba/", ""]],
            label_column="csPC_x",
        )
        val_loader = dataloader_obj(partition="val")
    except Exception as e:
        print(f"Error initializing data: {e}")
        sys.exit(1)
    
    # 2. Load Models
    models_pretrained = []
    for p in tqdm(MODEL_PATHS, desc="Loading Models"):
        m = load_model(p)
        if m: models_pretrained.append(m)
    if not models_pretrained: sys.exit(1)

    # 3. Loop
    dataset = val_loader.dataset
    start = args.start_index
    end = min(start + args.limit, len(dataset)) if args.limit else len(dataset)

    for idx in tqdm(range(start, end), desc="Processing"):
        try:
            # Now df_merged is aligned with val_loader
            # idx=0 in dataset corresponds to idx=0 in df_merged
            
            # ID Extraction
            case_id = f"Patient_{idx:03d}"
            if "ID_XNAT_session" in df_merged.columns:
                case_id = str(df_merged.iloc[idx]["ID_XNAT_session"])

            # Always generate a PDF; if it exists, append _1, _2, ...
            base_pdf_path = Path(OUTPUT_FOLDER) / f"{case_id}_Report.pdf"
            pdf_path = base_pdf_path
            counter = 1
            while pdf_path.exists():
                pdf_path = Path(OUTPUT_FOLDER) / f"{case_id}_Report_{counter}.pdf"
                counter += 1

            img_tensor = dataset[idx]['image'].to(device)
            
            # Load GT using the aligned dataframe
            try:
                gt_mask, _ = load_gt_mask_in_model_space(idx, img_tensor.detach().cpu(), df_merged)
            except Exception as e:
                # print(f"GT not found for {case_id}: {e}")
                gt_mask = torch.zeros_like(img_tensor[0:1]).bool()

            with PdfPages(pdf_path) as pdf:
                process_patient(pdf, img_tensor, gt_mask, models_pretrained, CLASS_TO_EXPLAIN)
            
            plt.close('all')
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error {idx}: {e}")
            continue

if __name__ == "__main__":
    main()