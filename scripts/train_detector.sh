#!/bin/bash
#SBATCH --job-name=ssl_nlp_detector
#SBATCH --time=08:00:00
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

export HF_HOME=$HOME/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat $HOME/.hf_token 2>/dev/null)
mkdir -p $HF_HOME

IMAGE_PATH="$HOME/ml_general.sif"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "container not found"
    exit 1
fi
EXEC_CMD="singularity exec --nv --env HF_HOME=$HF_HOME --env HF_TOKEN=$HF_TOKEN --env TOKENIZERS_PARALLELISM=false --env CUDA_HOME=/usr/local/cuda $IMAGE_PATH"

$EXEC_CMD python3 src/detector.py \
    --data_dir data/prepared \
    --out_dir results/detector \
    --model_name readerbench/RoBERT-large \
    --max_length 192 \
    --batch_size 16 \
    --grad_accum 2 \
    --lr 2e-5 \
    --weight_decay 0.005 \
    --epochs 3 \
    --warmup_ratio 0.1 \
    --type_loss_weight 0.5 \
    --num_workers 4 \
    --seed 42