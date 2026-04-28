#!/bin/bash
#SBATCH --job-name=ssl_nlp_eval
#SBATCH --time=12:00:00
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

export HF_HOME=$HOME/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat $HOME/.hf_token 2>/dev/null)
mkdir -p $HF_HOME

IMAGE_PATH="$HOME/ml_general.sif"
EXEC_CMD="singularity exec --nv --env HF_HOME=$HF_HOME --env HF_TOKEN=$HF_TOKEN --env TOKENIZERS_PARALLELISM=false --env CUDA_HOME=/usr/local/cuda $IMAGE_PATH"

$EXEC_CMD python3 src/eval.py \
    --detector_ckpt results/detector/best.pt \
    --detector_tokenizer results/detector/tokenizer \
    --seq2seq_dir results/seq2seq/best \
    --dataset upb-nlp/gec_ro_cna \
    --out_dir results/eval \
    --beam_size 4 \
    --threshold 0.5

$EXEC_CMD python3 src/eval.py \
    --detector_ckpt results/detector/best.pt \
    --detector_tokenizer results/detector/tokenizer \
    --seq2seq_dir results/seq2seq/best \
    --dataset upb-nlp/gec-ro-comments \
    --out_dir results/eval \
    --beam_size 4 \
    --threshold 0.5