#!/bin/bash
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=err_m5_2k
#SBATCH --output=logs/err_m5_2k_%j.out
#SBATCH --error=logs/err_m5_2k_%j.err

set -euxo pipefail

cd /scratch/zh2312/gh-issue-resolution
mkdir -p logs
source .venv-qwen/bin/activate
module load google-cloud-sdk/552.0.0
export CUDA_VISIBLE_DEVICES=""

date
hostname
pwd
which python
python --version
python -u -c "import torch; print('cuda', torch.cuda.is_available(), flush=True)"
python -u -c "import gcsfs; fs=gcsfs.GCSFileSystem(); print('signal shards', len(fs.glob('gs://gh_issue_ml-data/issues/issues_with_signals/*.parquet')), flush=True)"

python -u -X faulthandler src/analyze_deberta_errors.py \
  --model-kind hybrid \
  --checkpoint results/deberta_hybrid/pytorch_model.bin \
  --qwen-subset \
  --output-dir results/error_analysis/model5_qwen_2k \
  --max-rows 2000
