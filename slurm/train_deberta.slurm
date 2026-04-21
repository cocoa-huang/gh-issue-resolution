#!/bin/bash
#SBATCH --job-name=deberta-text
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/gh-issue-resolution/logs/deberta_text_%j.out
#SBATCH --error=/scratch/%u/gh-issue-resolution/logs/deberta_text_%j.err

set -e

REPO=/scratch/$USER/gh-issue-resolution

# Activate venv
source $REPO/.venv/bin/activate

# Auth GCS via service account key injected at runtime (set GOOGLE_APPLICATION_CREDENTIALS)
# If using gcloud ADC, run `gcloud auth application-default login` interactively first.

cd $REPO
mkdir -p logs results

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

python3 src/train_deberta_text.py

echo "Job finished: $(date)"
