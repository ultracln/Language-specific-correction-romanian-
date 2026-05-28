#!/bin/bash
# srun --pty is used to allocate a GPU interactively and hold it until the user exits
srun --partition=dgxa100 --gres=gpu:1 --time=01:00:00 \
    --cpus-per-task=4 --mem=16GB --pty \
    singularity exec --nv \
    --env HF_HOME=$HOME/.cache/huggingface \
    --env TOKENIZERS_PARALLELISM=false \
    --env CUDA_HOME=/usr/local/cuda \
    $HOME/ml_general_v5.sif \
    python3 src/demo_interactive.py \
        --rescore_lm readerbench/RoGPT2-medium \
        --rescore_lambda 2.0
