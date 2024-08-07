# BIMCV-Prostate-Classification
A Repository Containing the code used to classify Clinically Significant Prostate Cancer on BIMCV Prostate MRI Dataset.

<p align="center">
  <img src="Result_Figures/Guided_Backprop_result.png" alt="BIMCV Classification Results" width="90%">
</p>

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Methodology](#methodology)
- [Results](#results)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [Grants and Funding](#grants-and-funding)

## Introduction

Prostate cancer is a leading cause of cancer-related death among men worldwide. Accurate detection and classification of csPCa using MRI are crucial for diagnosis, treatment planning, and monitoring. This project introduces AI-based models designed to improve the detection of csPCa using MRI.

## Dataset

The datasets used in this project include the [PI-CAI Challenge dataset](https://zenodo.org/records/6624726) [1] and MRI images and clinical data from over 8,000 subjects from the BIMCV Prostate dataset. The dataset is structured using the Medical Imaging Data Structuring (MIDS) framework to ensure robustness and diversity.

## Models

### 3D EfficientNet-B7 (Classification)

A 3D EfficientNet-B7 model is adapted and trained for classifying csPCa from multiparametric MRI scans. The model leverages transfer learning from the PI-CAI dataset to enhance performance. It was adapted using MONAI 1.2.0.

## Methodology

1. **Data Preprocessing:** Image normalization, augmentation, and prostate zone segmentation.
2. **Model Training:** Utilization of advanced deep learning techniques, including Vision Transformers and Transfer Learning.
3. **Model Evaluation:** Benchmarking against other models and applying interpretability techniques such as occlusion sensitivity and guided backpropagation.

## Results

The 3D EfficientNet-B7 model achieved an AUC of 0.82 in the validation set.
<p align="center">
  <img src="Result_Figures/AUC_ROC_BIMCV.png" alt="BIMCV Classification Results" width="80%">
</p>

## Dependencies

The current release depends on the following Python libraries:

- monai == 1.2.0
- numpy == 1.23.4
- pandas == 2.1.0
- prettytable == 3.9.0
- ptflops == 0.7
- pygad == 3.2.0
- tensorboard == 2.8.0
- torch == 1.12.1
- torchmetrics == 1.1.2
- torchvision == 0.13.1
- tqdm == 4.62.3

## Installation

To set up the project environment and code you can install [BIMCV_AIKit](https://github.com/BIMCV-CSUSP/BIMCV-AIKit):

```bash
git clone https://github.com/BIMCV-CSUSP/BIMCV-AIKit.git
cd BIMCV-AIKit
pip install -e .
```

## Usage

1. **Data Preparation:** Ensure the dataset is available and structured as required. All data preparation and analysis can be found in the Data Structuring and Data Analysis folders.

2. **Training:** For pre-training with the PI-CAI dataset, download The PI-CAI Challenge: Public Training and Development Dataset. Pretrain using BIMCV-AIKit with:
```bash
bimcv_train -c config_picai.json
```
or
```bash
python -m bimcv_aikit.training.train -c config.json
```
Finally, with the pretrained model, ensure updating the pretrained weights in the config file:
```bash
bimcv_train -c config_bimcv.json
```
3. Evaluation: Use the evaluation scripts to as the [Analize_Results.ipynb](Analize_Results.ipynb) assess model performance on the validation set.

## References
[1] A. Saha, J. S. Bosma, J. J. Twilt, B. van Ginneken, A. Bjartell, A. R. Padhani, D. Bonekamp, G. Villeirs, G. Salomon, G. Giannarini, J. Kalpathy-Cramer, J. Barentsz, K. H. Maier-Hein, M. Rusu, O. Rouvière, R. van den Bergh, V. Panebianco, V. Kasivisvanathan, N. A. Obuchowski, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, and the PI-CAI consortium. “Artificial Intelligence and Radiologists in Prostate Cancer Detection on MRI (PI-CAI): An International, Paired, Non-Inferiority, Confirmatory Study”. The Lancet Oncology 2024; 25(7): 879-887.

## Grants and Funding
Funded by the Spanish Ministry of Economic Affairs and Digital Transformation (Project MIA.2021.M02.0005 TARTAGLIA, from the Recovery, Resilience, and Transformation Plan financed by the European Union through Next Generation EU funds). TARTAGLIA takes place under the R&D Missions in Artificial Intelligence program, which is part of the Spain Digital 2025 Agenda and the Spanish National Artificial Intelligence Strategy.

