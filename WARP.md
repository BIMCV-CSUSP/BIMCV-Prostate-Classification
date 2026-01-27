# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This repository contains code for classifying Clinically Significant Prostate Cancer (csPCa) using multiparametric MRI with 3D EfficientNet-B7 models. The project uses transfer learning from the PI-CAI dataset and evaluates on the BIMCV Prostate dataset.

## Development Commands

### Training Models

Pre-train on PI-CAI dataset:
```bash
bimcv_train -c configs/config_picai.json
```

Alternative training command (equivalent):
```bash
python -m bimcv_aikit.training.train -c configs/config_picai.json
```

Train on BIMCV dataset (after pre-training):
```bash
bimcv_train -c configs/config.json
```

Train without pre-training:
```bash
bimcv_train -c configs/config_no_pretrain.json
```

Optional helper wrapper:
```bash
python scripts/train.py --config configs/config.json
```

### Testing Individual Models

Average backbone weights across folds:
```bash
python scripts/average_model.py --fold path/to/fold0.pth --weight 0.25 --fold path/to/fold1.pth --weight 0.25
```

### Analysis and Evaluation

Model evaluation and results analysis are performed in Jupyter notebooks:
- `notebooks/Analize_Results.ipynb` - Main results analysis, ROC curves, model performance metrics
- `notebooks/Analize_models.ipynb` - Model architecture and weight analysis
- `notebooks/Interpretability_Analysis.ipynb` - Model interpretability using techniques like guided backpropagation and occlusion sensitivity
- `notebooks/statistical_analysis.ipynb` - Statistical analysis of model performance
- `notebooks/statistical_comparisons.ipynb` - Comparative statistical tests between models

Optional helper to execute a notebook:
```bash
python scripts/eval.py --notebook notebooks/Analize_Results.ipynb
```

## Architecture Overview

### Two-Stage Training Pipeline

The project follows a two-stage transfer learning approach:

1. **Stage 1 (Pre-training)**: Train on PI-CAI dataset using `configs/config_picai.json`
   - Uses MONAI's EfficientNetBN directly
   - Data loaded via `bimcv_prostate.data.dataloaders.ProstateImageDataLoader` from `Files/picai_cv.csv`
   - 5-fold cross-validation setup
   - Saves best model weights per fold

2. **Stage 2 (Fine-tuning)**: Train on BIMCV dataset using `configs/config.json`
   - Uses custom `EfficientNet_pretrained` from `bimcv_prostate.models`
   - Data loaded via `bimcv_prostate.data.dataloaders.ProstateImageDataLoader` from `Files/data_noder.csv`
   - Loads pre-trained weights from Stage 1 (specified in config's `pretrained_weights_path`)
   - Single train/val/test split

### Model Architecture

**EfficientNet_pretrained** (`src/bimcv_prostate/models/efficientnet.py`):
- Wrapper around MONAI's `EfficientNetBN` that enables loading pre-trained weights
- Supports 3D spatial dimensions for volumetric MRI data
- Default configuration: EfficientNet-B7 with 3 input channels (T2, ADC, DWI) and 2 output classes

**Model Averaging** (`src/bimcv_prostate/training/average_model.py`):
- Implements weighted averaging of model weights across multiple folds
- Uses validation F1 scores as weights
- Excludes classifier head from averaging (re-initialized for target task)
- Function `build_new_model_from_folds()` creates averaged backbone for transfer learning

### Data Loading Pipeline

Both dataloaders follow the same preprocessing pipeline:

1. **Load multi-sequence MRI**: T2-weighted, ADC, and DWI sequences
2. **Resample**: ADC and DWI resampled to match T2 dimensions
3. **Split DWI**: Extract first channel from DWI (high b-value)
4. **Resize**: All sequences to (128, 128, 32)
5. **Normalize**: 
   - ADC: percentile-based scaling (5th-99.5th percentile)
   - T2/DWI: min-max normalization to [0, 1]
6. **Concatenate**: Stack T2, DWI, ADC as 3-channel input
7. **Augmentation** (training only): Random flip, rotation, affine transforms, Gaussian noise

Key differences:
- BIMCV config: Uses `partition_column="split"`, loads from `image_t2`, `image_adc`, `image_dwi` columns
- PI-CAI config: Uses `partition_column="partition"`, loads from `filepath_*_cropped` columns, supports fold-based cross-validation

### Configuration System

All training is driven by JSON config files with the following structure:

- **arch**: Model architecture (module path, class name, initialization args)
- **data_loader**: Data loading specification (module, class, CSV path, batch size)
- **optimizer**: Optimizer type and hyperparameters (Adadelta used throughout)
- **loss**: Loss function (CrossEntropyLoss for binary classification)
- **metrics**: Evaluation metrics (weighted accuracy and F1 score)
- **trainer**: Training configuration (epochs=200, early stopping, TensorBoard logging, save directory)

### Directory Structure

- `src/bimcv_prostate/` - Core library (data, models, training, utils)
- `configs/` - Training configs (JSON)
- `scripts/` - CLI entrypoints (train, eval, average_model)
- `notebooks/` - Analysis notebooks
- `Files/` - Contains all CSV metadata files for datasets
  - `picai_cv.csv` - PI-CAI dataset with fold assignments
  - `data_noder.csv` - BIMCV dataset split information
  - `Statistical_Test/` - Statistical analysis results
- `src/bimcv_prostate/experiments/clinical/` - Clinical variables experiments
- `outputs/results/` - Model evaluation results (JSON files with test set performance)
- `outputs/figures/` - Figures generated by analysis
- `outputs/logs/` - Training logs, model checkpoints, TensorBoard files (gitignored)

### External Dependencies

This project requires the **BIMCV-AIKit** package, which is not included in the repository:

Installation:
```bash
git clone https://github.com/BIMCV-CSUSP/BIMCV-AIKit.git
cd BIMCV-AIKit
pip install -e .
```

BIMCV-AIKit provides:
- `bimcv_aikit.training.train` - Main training loop and ClassificationTrainer
- `bimcv_aikit.monai.transforms.DeleteBlackSlices` - Custom MONAI transform
- Command-line interface: `bimcv_train`

### Key Python Dependencies

- monai == 1.2.0 (medical imaging framework)
- torch == 1.12.1 (deep learning)
- torchvision == 0.13.1
- torchmetrics == 1.1.2 (evaluation metrics)
- pandas == 2.1.0 (data handling)
- numpy == 1.23.4
- nibabel (NIfTI file loading, via MONAI)

## Important Notes

### File Path Handling

The dataloaders support configurable path replacements via `path_rewrites` in the JSON configs. Update those entries to match your environment.

### Configuration Paths

Config files may contain absolute paths (e.g., for pretrained weights). Update those to match your local environment before training.

### Pre-trained Weights

The `configs/config.json` file expects `pretrained_weights_path` to be set to your own PI-CAI fold weights from Stage 1.

### GPU Requirements

Models are trained with:
- Batch size: 16 (PI-CAI) or 32 (BIMCV)
- 3D EfficientNet-B7 architecture
- Input size: (3, 128, 128, 32)

This requires substantial GPU memory (~24GB recommended).

### Test Set Handling

In the BIMCV configs, `skip_partitions` includes `"test"` so test evaluation is handled separately through the analysis notebooks.

## Development Workflow

Typical workflow for reproducing results:

1. Install BIMCV-AIKit package
2. Prepare datasets and update CSV files in `Files/`
3. Update paths in config files to match your environment
4. Pre-train on PI-CAI: `bimcv_train -c configs/config_picai.json`
5. Note best fold weights path from logs
6. Update `pretrained_weights_path` in `configs/config.json`
7. Train on BIMCV: `bimcv_train -c configs/config.json`
8. Analyze results using Jupyter notebooks

For model ensemble approaches, use `scripts/average_model.py` to create averaged weights from multiple folds before Stage 2.
