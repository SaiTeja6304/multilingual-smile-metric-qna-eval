#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -J SMILE
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH --open-mode=append
#SBATCH -o smile_%A_%a.out
#SBATCH -e smile_%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --array=0-7

requeue_handler() {
    echo "$(date): Received SIGUSR1 — forwarding to children..."
    kill -SIGUSR1 $(jobs -p) 2>/dev/null

    local deadline=$(( $(date +%s) + 90 ))
    while jobs -p | grep -q .; do
        if [ $(date +%s) -gt $deadline ]; then
            echo "$(date): Timeout waiting for children — killing remaining..."
            kill $(jobs -p) 2>/dev/null
            break
        fi
        sleep 2
    done

    echo "$(date): Requesting requeue..."
    scontrol requeue ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
    exit 0
}
trap requeue_handler SIGUSR1

module load python

REPO=/home/ssunku/directed_research/multilingual_smile
source ${REPO}/venv/bin/activate

GT_DIR=/home/ssunku/directed_research/output
PRED_DIR=/home/ssunku/directed_research/workspace/llm_invoke/full_results
OUT_DIR=/home/ssunku/directed_research/workspace/evals/results

MODELS=(
    "google_gemma-3-27b-it"
    "Qwen_Qwen3-30B-A3B"
)

GT_FILES=(
    "xor_dev_full_v1_1.jsonl"
    "xor_dev_retrieve_eng_span_v1_1.jsonl"
    "xor_train_full.jsonl"
    "xor_train_retrieve_eng_span.jsonl"
)

PRED_STEMS=(
    "xor_dev_full_v1_1"
    "xor_dev_retrieve_eng_span_v1_1"
    "xor_train_full"
    "xor_train_retrieve_eng_span"
)

# Array task index (0-7) maps to dataset (0-3) and model (0-1)
# 0-3 for Gemma, 4-7 for Qwen
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))

MODEL=${MODELS[$MODEL_IDX]}
GT=${GT_DIR}/${GT_FILES[$DATASET_IDX]}
PRED=${PRED_DIR}/${MODEL}_${PRED_STEMS[$DATASET_IDX]}_results.jsonl
OUT=${OUT_DIR}/${MODEL}/${PRED_STEMS[$DATASET_IDX]}_results.csv

mkdir -p ${OUT_DIR}/${MODEL}

echo "$(date): ARRAY_TASK=${SLURM_ARRAY_TASK_ID} | Model=${MODEL} | Dataset=${PRED_STEMS[$DATASET_IDX]}"

# Run the evaluation script
python gpu_main.py \
    --input        "${PRED}" \
    --ground-truth "${GT}" \
    --output       "${OUT}" &

wait
echo "$(date): Array task ${SLURM_ARRAY_TASK_ID} complete."

