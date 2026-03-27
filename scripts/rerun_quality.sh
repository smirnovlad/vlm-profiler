#!/bin/bash
# Rerun all models with fixed quality evaluation
set -e
export HF_HOME=/mnt/data/users/vlad.smirnov/hf_cache

cd /mnt/data/users/vlad.smirnov/vlm-profiler
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate vlm-profiler

WANDB_ARGS="--wandb-project vlm-profiler --wandb-entity ysda-research"

# GPU 0: smaller/faster models
run_gpu0() {
    echo "[GPU 0] Starting at $(date)"
    for model in \
        "Salesforce/blip2-opt-2.7b" \
        "Salesforce/blip2-flan-t5-xl" \
        "Salesforce/instructblip-flan-t5-xl" \
        "llava-hf/llava-1.5-7b-hf"; do
        echo "[GPU 0] Running: $model at $(date)"
        python scripts/run_experiments.py --model "$model" --gpu-index 0 $WANDB_ARGS || \
            echo "[GPU 0] FAILED: $model"
    done
    echo "[GPU 0] Done at $(date)"
}

# GPU 1: larger models
run_gpu1() {
    echo "[GPU 1] Starting at $(date)"
    for model in \
        "Salesforce/instructblip-vicuna-7b" \
        "adept/fuyu-8b" \
        "HuggingFaceM4/idefics2-8b" \
        "llava-hf/llava-1.5-13b-hf"; do
        echo "[GPU 1] Running: $model at $(date)"
        python scripts/run_experiments.py --model "$model" --gpu-index 1 $WANDB_ARGS || \
            echo "[GPU 1] FAILED: $model"
    done
    echo "[GPU 1] Done at $(date)"
}

run_gpu0 > outputs/rerun_quality_gpu0.log 2>&1 &
PID0=$!
run_gpu1 > outputs/rerun_quality_gpu1.log 2>&1 &
PID1=$!

echo "Launched GPU 0 (PID $PID0) and GPU 1 (PID $PID1)"
echo "Logs: outputs/rerun_quality_gpu0.log, outputs/rerun_quality_gpu1.log"

wait $PID0
STATUS0=$?
wait $PID1
STATUS1=$?

echo ""
echo "=== DONE ==="
echo "GPU 0 exit: $STATUS0"
echo "GPU 1 exit: $STATUS1"
