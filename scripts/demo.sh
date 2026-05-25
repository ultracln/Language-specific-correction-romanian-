#!/bin/bash
#SBATCH --job-name=ssl_nlp_demo
#SBATCH --time=12:00:00
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

export HF_HOME=$HOME/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
mkdir -p $HF_HOME

IMAGE_PATH="$HOME/ml_general_v4.sif"
EXEC_CMD="singularity exec --nv --env HF_HOME=$HF_HOME --env TOKENIZERS_PARALLELISM=false --env CUDA_HOME=/usr/local/cuda $IMAGE_PATH"

echo "=== demo 1: diacritics error ==="
$EXEC_CMD python3 src/pipeline.py \
    --text "13 aprilie: Al Doilea Razboi Mondial: Trupele Germaniei au ocupat Belgradul."

echo ""
echo "=== demo 2: noun form error ==="
$EXEC_CMD python3 src/pipeline.py \
    --text "Eu mergem la magazi pentru paine."

echo ""
echo "=== demo 3: spelling error ==="
$EXEC_CMD python3 src/pipeline.py \
    --text "Industria muzicala e un loc dur in care sati faci o cariera."

echo ""
echo "=== demo 4: agreement error ==="
$EXEC_CMD python3 src/pipeline.py \
    --text "Copii a venit ieri la scoala dupa ce ploua."

echo ""
echo "=== demo 5: clean sentence (should not change much) ==="
$EXEC_CMD python3 src/pipeline.py \
    --text "Astăzi este o zi frumoasă și soarele strălucește puternic."