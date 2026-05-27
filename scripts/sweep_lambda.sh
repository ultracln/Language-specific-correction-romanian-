#!/bin/bash
#SBATCH --job-name=ssl_nlp_sweep_lambda
#SBATCH --time=12:00:00
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

export HF_HOME=$HOME/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat $HOME/.hf_token 2>/dev/null)
mkdir -p $HF_HOME

IMAGE_PATH="$HOME/ml_general_v5.sif"
EXEC_CMD="singularity exec --nv --env HF_HOME=$HF_HOME --env HF_TOKEN=$HF_TOKEN --env TOKENIZERS_PARALLELISM=false --env CUDA_HOME=/usr/local/cuda $IMAGE_PATH"

$EXEC_CMD python3 src/sweep_lambda.py \
    --detector_ckpt results/detector/best.pt \
    --detector_tokenizer results/detector/tokenizer \
    --seq2seq_dir results/seq2seq/best \
    --rescore_lm readerbench/RoGPT2-base \
    --dataset upb-nlp/gec_ro_cna \
    --out_dir results/eval_rescore \
    --threshold 0.3 \
    --lambdas 0.1,0.25,0.5,1.0,2.0 \
    --beam_size 4 \
    --lowercase \
    --errant_bin_dir /opt/conda/bin

$EXEC_CMD python3 src/sweep_lambda.py \
    --detector_ckpt results/detector/best.pt \
    --detector_tokenizer results/detector/tokenizer \
    --seq2seq_dir results/seq2seq/best \
    --rescore_lm readerbench/RoGPT2-base \
    --dataset upb-nlp/gec-ro-comments \
    --out_dir results/eval_rescore \
    --threshold 0.3 \
    --lambdas 0.1,0.25,0.5,1.0,2.0 \
    --beam_size 4 \
    --lowercase \
    --errant_bin_dir /opt/conda/bin
