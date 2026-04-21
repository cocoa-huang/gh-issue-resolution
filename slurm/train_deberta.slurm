#!/bin/bash
#SBATCH --job-name=deberta-text
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/gh-issue-resolution/logs/deberta_text_%j.out
#SBATCH --error=/scratch/%u/gh-issue-resolution/logs/deberta_text_%j.err

set -e

REPO=/scratch/$USER/gh-issue-resolution

source $REPO/.venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=/scratch/$USER/gcp-key.json
export TMPDIR=/scratch/$USER/tmp && mkdir -p $TMPDIR

cd $REPO
mkdir -p logs results

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

python3 src/train_deberta_text.py

echo "Job finished: $(date)"
