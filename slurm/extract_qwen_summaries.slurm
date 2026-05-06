#!/bin/bash
#SBATCH --job-name=qwen-summaries
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/gh-issue-resolution/logs/qwen_summaries_%j.out
#SBATCH --error=/scratch/%u/gh-issue-resolution/logs/qwen_summaries_%j.err

set -e

REPO=/scratch/$USER/gh-issue-resolution

source $REPO/.venv-qwen/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=/scratch/$USER/gcp-key.json
export TMPDIR=/scratch/$USER/tmp && mkdir -p $TMPDIR
export PYTHONUNBUFFERED=1

cd $REPO
mkdir -p logs results

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Python: $(which python)"
python --version
nvidia-smi

python -u src/extract_qwen_summaries.py \
  --max-train-issues 40000 \
  --max-test-issues 10000 \
  --overwrite \
  --clear-output

echo "Job finished: $(date)"
