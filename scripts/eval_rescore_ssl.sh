#!/bin/bash
#SBATCH --job-name=ssl_nlp_eval_rescore
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

# canonical rescored eval. config:
#   detector threshold: 0.3 (from threshold sweep)
#   rescoring LM: RoGPT2-medium (from LM size experiment)
#   rescore lambda: 2.0 (from lambda sweep)
#   beam search: standard (the diverse beam search experiment regressed CNA F0.5
#   from 0.488 to 0.283; the flag is retained but off by default).
$EXEC_CMD python3 src/eval.py \
    --detector_ckpt results/detector_ssl/best.pt \
    --detector_tokenizer results/detector_ssl/tokenizer \
    --seq2seq_dir results/seq2seq/best \
    --dataset upb-nlp/gec_ro_cna \
    --out_dir results/eval_rescore_ssl \
    --beam_size 4 \
    --threshold 0.3 \
    --lowercase \
    --errant_bin_dir /opt/conda/bin \
    --rescore_lm readerbench/RoGPT2-medium \
    --rescore_lambda 2.0

$EXEC_CMD python3 src/eval.py \
    --detector_ckpt results/detector_ssl/best.pt \
    --detector_tokenizer results/detector_ssl/tokenizer \
    --seq2seq_dir results/seq2seq/best \
    --dataset upb-nlp/gec-ro-comments \
    --out_dir results/eval_rescore_ssl \
    --beam_size 4 \
    --threshold 0.3 \
    --lowercase \
    --errant_bin_dir /opt/conda/bin \
    --rescore_lm readerbench/RoGPT2-medium \
    --rescore_lambda 2.0
