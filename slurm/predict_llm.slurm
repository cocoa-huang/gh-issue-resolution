#!/bin/bash
#SBATCH --job-name=llm-issue-predict
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/llm_predict_%j.out
#SBATCH --error=logs/llm_predict_%j.err

set -e

REPO=/scratch/$USER/gh-issue-resolution

cd $REPO
source .venv/bin/activate

mkdir -p logs results

export GOOGLE_APPLICATION_CREDENTIALS=/scratch/$USER/gcp-key.json
export OPENAI_API_KEY=$(cat /scratch/$USER/openai_api_key.txt)

export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=$(cat /scratch/$USER/deepseek_api_key.txt)

python3 src/predict_llm.py \
  --n 500 \
  --model gpt-5.5-mini \
  --mode zero_shot \
  --output results/llm_zero_shot_predictions.jsonl