import argparse

from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset


DEFAULT_MODELS = [
    "readerbench/RoBERT-large",
    "google/mt5-small",
]

DEFAULT_DATASETS = [
    "upb-nlp/gec-ro-comments",
    "upb-nlp/gec_ro_cna",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--skip_models", action="store_true")
    p.add_argument("--skip_datasets", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.skip_models:
        for name in args.models:
            print(f"downloading model: {name}")
            AutoTokenizer.from_pretrained(name)
            if "mt5" in name or "t5" in name:
                AutoModelForSeq2SeqLM.from_pretrained(name)
            else:
                AutoModel.from_pretrained(name)
            print(f"  done.")

    if not args.skip_datasets:
        for name in args.datasets:
            print(f"downloading dataset: {name}")
            load_dataset(name)
            print(f"  done.")

    print("all assets cached.")


if __name__ == "__main__":
    main()