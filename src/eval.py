import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from utils import normalize_romanian, word_tokenize
from pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_ckpt", type=str, default="results/detector/best.pt")
    p.add_argument("--detector_tokenizer", type=str, default="results/detector/tokenizer")
    p.add_argument("--seq2seq_dir", type=str, default="results/seq2seq/best")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--out_dir", type=str, default="results/eval")
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_examples", type=int, default=-1)
    return p.parse_args()


def make_pairs(data):
    n = len(data)
    if n % 2 != 0:
        print(f"warning: dataset has odd length ({n}), dropping last row")
        n -= 1
    pairs = []
    for i in range(0, n, 2):
        pairs.append({"incorrect": data[i]["text"], "correct": data[i + 1]["text"]})
    return pairs


def normalize_for_match(s: str) -> str:
    return " ".join(word_tokenize(normalize_romanian(s)))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading dataset: {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset)
    if args.split not in ds:
        split = list(ds.keys())[0]
        print(f"split '{args.split}' not found, using '{split}'")
    else:
        split = args.split
    data = ds[split]
    print(f"raw split '{split}': {len(data)} rows")

    pairs = make_pairs(data)
    print(f"sentence pairs: {len(pairs)}")

    pipe = Pipeline(args.detector_ckpt, args.detector_tokenizer, args.seq2seq_dir,
                    args.max_length, args.beam_size, args.threshold)

    n = len(pairs) if args.max_examples <= 0 else min(len(pairs), args.max_examples)
    correct = changed = spurious = stayed_same = total = 0
    samples = []

    for i in tqdm(range(n)):
        inc = pairs[i]["incorrect"]
        cor = pairs[i]["correct"]

        result = pipe(inc)
        pred = normalize_for_match(result["output"])
        truth = normalize_for_match(cor)
        inc_norm = normalize_for_match(inc)
        has_err = inc_norm != truth

        is_correct = pred == truth
        was_changed = pred != inc_norm

        total += 1
        if is_correct:
            correct += 1
        if was_changed:
            changed += 1
        if not has_err and was_changed:
            spurious += 1
        if has_err and not was_changed:
            stayed_same += 1

        if len(samples) < 50:
            samples.append({
                "input": inc, "target": cor, "output": result["output"],
                "is_correct": is_correct,
                "flagged": result.get("flagged_tokens", []),
                "predicted_types": result.get("predicted_types", []),
            })

    summary = {
        "dataset": args.dataset,
        "split": split,
        "n_pairs": total,
        "exact_match_acc": correct / max(total, 1),
        "changed_rate": changed / max(total, 1),
        "spurious_rate": spurious / max(total, 1),
        "stayed_same_rate": stayed_same / max(total, 1),
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    name = args.dataset.replace("/", "_")
    with (out_dir / f"{name}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / f"{name}_samples.json").open("w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nwrote eval to {out_dir}")


if __name__ == "__main__":
    main()