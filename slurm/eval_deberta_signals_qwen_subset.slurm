#!/bin/bash
#SBATCH --job-name=eval-m4-qwen
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/gh-issue-resolution/logs/eval_deberta_signals_qwen_subset_%j.out
#SBATCH --error=/scratch/%u/gh-issue-resolution/logs/eval_deberta_signals_qwen_subset_%j.err

set -e

REPO=/scratch/$USER/gh-issue-resolution
MODEL4_CHECKPOINT=${MODEL4_CHECKPOINT:-$REPO/results/deberta_signals/pytorch_model.bin}

source $REPO/.venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=/scratch/$USER/gcp-key.json
export TMPDIR=/scratch/$USER/tmp && mkdir -p $TMPDIR
export PYTHONUNBUFFERED=1

cd $REPO
mkdir -p logs results

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Python: $(which python3)"
python3 --version
echo "Model 4 checkpoint: $MODEL4_CHECKPOINT"
test -f "$MODEL4_CHECKPOINT" && ls -lh "$MODEL4_CHECKPOINT"
nvidia-smi

python3 -u src/eval_deberta_signals_qwen_subset.py --checkpoint "$MODEL4_CHECKPOINT"

echo "Job finished: $(date)"
