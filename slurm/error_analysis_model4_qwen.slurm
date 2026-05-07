#!/bin/bash
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=err_m4_qwen
#SBATCH --output=logs/err_m4_qwen_%j.out
#SBATCH --error=logs/err_m4_qwen_%j.err

set -euo pipefail

cd /scratch/zh2312/gh-issue-resolution
mkdir -p logs
source .venv-qwen/bin/activate
module load google-cloud-sdk/552.0.0

python -u -X faulthandler src/analyze_deberta_errors.py \
  --model-kind signals \
  --checkpoint results/deberta_signals/pytorch_model.bin \
  --qwen-subset \
  --output-dir results/error_analysis/model4_qwen_subset
