#!/bin/bash
#SBATCH --job-name=ssl_nlp_prep_ssl_corpus
#SBATCH --time=00:30:00
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

export HF_HOME=$HOME/.cache/huggingface
mkdir -p $HF_HOME

IMAGE_PATH="$HOME/ml_general_v5.sif"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "container not found: $IMAGE_PATH"
    exit 1
fi

EXEC_CMD="singularity exec --env HF_HOME=$HF_HOME $IMAGE_PATH"

# extracts the 'correct' column (674k clean romanian sentences) from synthetic.csv
# into a one-sentence-per-line text file consumed by ssl_trainer.py.
$EXEC_CMD python3 src/prepare_unlabeled_corpus.py \
    --input data/synthetic.csv \
    --output data/unlabeled_corpus.txt \
    --columns correct
