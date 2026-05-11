#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -J SMILE
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH --open-mode=append
#SBATCH -o smile_%j.out
#SBATCH -e smile_%j.err

module load python

source ./venv/bin/activate

python main.py --input sample_data/test_input.jsonl --ground-truth sample_data/test_ground_truth.jsonl --output sample_data/initial_results.csv