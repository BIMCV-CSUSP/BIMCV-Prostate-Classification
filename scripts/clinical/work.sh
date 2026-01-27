#!/bin/bash
#SBATCH --job-name=ClinicalVariablesComparison

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 15

#SBATCH --mem 100G

#SBATCH --output=./outputs/results/clinical/ClinicalVariables.out

#SBATCH --gres=gpu:1

source /projects/ceib/python_enviroments/bimcv_aikit/bin/activate
module load GCC
module load CUDA

export PATH="/home/jaalzate/.local/bin:$PATH"
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/projects/ceib/python_enviroments/bimcv_aikit/lib/python3.10/site-packages:$PYTHONPATH"

cd /home/jaalzate/BIMCV-Prostate-Classification

bimcv_train -c /home/jaalzate/BIMCV-Prostate-Classification/configs/clinical/config_multi.json
