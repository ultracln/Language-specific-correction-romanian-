#!/bin/bash
#SBATCH --job-name=ssl_nlp_prep
#SBATCH --time=00:30:00
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

export HF_HOME=$HOME/.cache/huggingface
mkdir -p $HF_HOME
 
IMAGE_PATH="$HOME/ml_general.sif"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "container not found: $IMAGE_PATH"
    exit 1
fi
 
EXEC_CMD="singularity exec --env HF_HOME=$HF_HOME $IMAGE_PATH"
 
$EXEC_CMD python3 src/data_prep.py \
    --csv data/synthetic.csv \
    --out_dir data/prepared \
    --val_size 0.05 \
    --test_size 0.05 \
    --seed 42 \
    --max_words 120
