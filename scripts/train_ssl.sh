#!/bin/bash
#SBATCH --job-name=ssl_nlp_train_ssl
#SBATCH --time=04:00:00
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

export HF_HOME=$HOME/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat $HOME/.hf_token 2>/dev/null)
mkdir -p $HF_HOME

IMAGE_PATH="$HOME/ml_general_v5.sif"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "container not found"
    exit 1
fi
EXEC_CMD="singularity exec --nv --env HF_HOME=$HF_HOME --env HF_TOKEN=$HF_TOKEN --env TOKENIZERS_PARALLELISM=false --env CUDA_HOME=/usr/local/cuda $IMAGE_PATH"

# denoising autoencoder pretraining over the base RoBERT-large encoder.
# produces results/ssl_dae/best/; later passed as --model_name to detector.py
# in a separate experiment.
$EXEC_CMD python3 src/ssl_trainer_safe.py \
    --unlabeled_data data/unlabeled_corpus.txt \
    --out_dir results/ssl_dae \
    --model_name readerbench/RoBERT-large \
    --max_length 128 \
    --batch_size 32 \
    --grad_accum 1 \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --epochs 3 \
    --warmup_ratio 0.1 \
    --num_workers 4 \
    --max_examples 100000 \
    --curriculum \
    --min_intensity 0.3 \
    --max_intensity 0.8 \
    --min_edit_distance 2 \
    --save_every 2000 \
    --seed 42
