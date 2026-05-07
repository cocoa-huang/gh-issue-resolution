#!/bin/bash
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=gpu_diag
#SBATCH --output=logs/gpu_diag_%j.out
#SBATCH --error=logs/gpu_diag_%j.err

set -euxo pipefail

cd /scratch/zh2312/gh-issue-resolution
mkdir -p logs
source .venv-qwen/bin/activate
module load google-cloud-sdk/552.0.0

date
hostname
pwd
which python
python --version
env | grep -E "CUDA|NVIDIA|SLURM|LD_LIBRARY_PATH" || true
nvidia-smi

timeout 60s python -u - <<'PY'
import os
print("before torch import", flush=True)
import torch
print("after torch import", flush=True)
print("torch", torch.__version__, flush=True)
print("cuda version", torch.version.cuda, flush=True)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("before is_available", flush=True)
print("is_available", torch.cuda.is_available(), flush=True)
print("before device_count", flush=True)
print("device_count", torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("device name", torch.cuda.get_device_name(0), flush=True)
    x = torch.randn(1024, 1024, device="cuda")
    print("tensor ok", float(x.mean().cpu()), flush=True)
print("done", flush=True)
PY
