#!/bin/bash
#SBATCH --job-name=deberta-hybrid
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/gh-issue-resolution/logs/deberta_hybrid_%j.out
#SBATCH --error=/scratch/%u/gh-issue-resolution/logs/deberta_hybrid_%j.err

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
echo "Model 4 checkpoint: $MODEL4_CHECKPOINT"
echo "Python: $(which python3)"
python3 --version
python3 -c "import sys, torch, transformers, gcsfs; print('Executable:', sys.executable); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Transformers:', transformers.__version__); print('gcsfs:', gcsfs.__version__)"
test -f "$GOOGLE_APPLICATION_CREDENTIALS" && echo "GCS credentials found: $GOOGLE_APPLICATION_CREDENTIALS"
test -f "$MODEL4_CHECKPOINT" && ls -lh "$MODEL4_CHECKPOINT"
nvidia-smi

python3 -u src/train_deberta_hybrid.py --warm-start "$MODEL4_CHECKPOINT"

echo "Job finished: $(date)"
