#!/bin/bash
# Rerun models that had errors (uses the fixed code)
set -e
export HF_HOME=/mnt/data/users/vlad.smirnov/hf_cache

cd /mnt/data/users/vlad.smirnov/vlm-profiler
source activate vlm-profiler

WANDB_ARGS="--wandb-project vlm-profiler --wandb-entity ysda-research"

echo "Rerunning instructblip-vicuna-7b on GPU 1 at $(date)"
python scripts/run_experiments.py \
  --model Salesforce/instructblip-vicuna-7b \
  --gpu-index 1 \
  $WANDB_ARGS

echo "Done at $(date)"
