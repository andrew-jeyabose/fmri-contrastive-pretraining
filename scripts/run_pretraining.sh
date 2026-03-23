#!/bin/bash
# scripts/run_pretraining.sh
# --------------------------
# SLURM launch script for single-node multi-GPU pre-training.
# Adjust #SBATCH directives for your cluster.

#SBATCH --job-name=fmri_contrastive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # number of GPUs
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

# ---- environment ----
module load cuda/12.1
source activate fmri_contrastive          # or: conda activate / source venv/bin/activate

mkdir -p logs outputs

# ---- single-GPU fallback ----
# python training/train.py --config configs/default.yaml

# ---- multi-GPU (DDP via torchrun) ----
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    training/train.py \
    --config configs/default.yaml
