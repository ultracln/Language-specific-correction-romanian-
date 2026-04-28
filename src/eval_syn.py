import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from utils import normalize_romanian, read_jsonl, word_tokenize
from pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_ckpt", type=str, default="results/detector/best.pt")
    p.add_argument("--detector_tokenizer", type=str, default="results/detector/tokenizer")
    p.add_argument("--seq2seq_dir", type=str, default="results/seq2seq/best")
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--detector_test_jsonl", type=str, default="data/prepared/detector_test.jsonl")
    p.add_argument("--out_dir", type=str, default="results/eval")
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_examples", type=int, default=2000)
    return p.parse_args()


def load_test(args):
    if args.test_csv:
        df = pd.read_csv(args.test_csv)
        rows = [
            {"correct": r["correct"], "incorrect": r["incorrect"], "error_type": r["error_type"], "has_error": int(r["has_error"])}
            for _, r in df.iterrows()
        ]
    else:
        rows = []
        for r in read_jsonl(args.detector_test_jsonl):
            rows.append({
                "correct": " ".join(_recover_correct(r)),
                "incorrect": " ".join(r["tokens"]),
                "error_type": r["error_type"],
                "has_error": int(r["has_error"]),
            })
    return rows


def _recover_correct(r):
    return r["tokens"]


def f_beta(p, r, beta=0.5):
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def normalize_for_match(s: str) -> str:
    return " ".join(word_tokenize(normalize_romanian(s)))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(args.detector_ckpt, args.detector_tokenizer, args.seq2seq_dir,
                    args.max_length, args.beam_size, args.threshold)

    if not args.test_csv:
        print("WARNING: --test_csv not provided. detector_test.jsonl does not retain corrected ground truth. provide --test_csv with the original csv (filtered to test ids) for accurate correction metrics.")

    test_rows = load_test(args)
    if args.max_examples > 0:
        test_rows = test_rows[: args.max_examples]
    print(f"evaluating on {len(test_rows)} examples")

    by_type = defaultdict(lambda: {"correct": 0, "total": 0, "changed": 0, "spurious": 0, "stayed_same": 0})
    overall = {"correct": 0, "total": 0, "changed": 0, "spurious": 0, "stayed_same": 0}
    samples = []

    for row in tqdm(test_rows):
        inc = row["incorrect"]
        cor = row["correct"]
        etype = row["error_type"]
        has_err = row["has_error"]

        result = pipe(inc)
        pred = normalize_for_match(result["output"])
        truth = normalize_for_match(cor)
        inc_norm = normalize_for_match(inc)

        is_correct = pred == truth
        changed = pred != inc_norm
        spurious = (has_err == 0) and changed
        stayed_same = (has_err == 1) and not changed

        by_type[etype]["total"] += 1
        overall["total"] += 1
        if is_correct:
            by_type[etype]["correct"] += 1
            overall["correct"] += 1
        if changed:
            by_type[etype]["changed"] += 1
            overall["changed"] += 1
        if spurious:
            by_type[etype]["spurious"] += 1
            overall["spurious"] += 1
        if stayed_same:
            by_type[etype]["stayed_same"] += 1
            overall["stayed_same"] += 1

        if len(samples) < 50:
            samples.append({
                "input": inc, "target": cor, "output": result["output"],
                "error_type": etype, "is_correct": is_correct,
                "flagged": result.get("flagged_tokens", []),
                "predicted_types": result.get("predicted_types", []),
            })

    print("\n=== per error type ===")
    print(f"{'type':<14} {'n':>6} {'acc':>7} {'changed':>9} {'spurious':>10} {'no_change_when_should':>22}")
    summary = {}
    for etype, s in sorted(by_type.items()):
        acc = s["correct"] / max(s["total"], 1)
        summary[etype] = {
            "n": s["total"],
            "exact_match_acc": acc,
            "changed_rate": s["changed"] / max(s["total"], 1),
            "spurious_rate": s["spurious"] / max(s["total"], 1),
            "stayed_same_rate": s["stayed_same"] / max(s["total"], 1),
        }
        print(f"{etype:<14} {s['total']:>6} {acc:>7.4f} {s['changed']:>9} {s['spurious']:>10} {s['stayed_same']:>22}")

    overall_acc = overall["correct"] / max(overall["total"], 1)
    print(f"\noverall exact-match accuracy: {overall_acc:.4f}  (n={overall['total']})")

    with (out_dir / "summary.json").open("w") as f:
        json.dump({"by_type": summary, "overall_acc": overall_acc, "n": overall["total"]}, f, indent=2)
    with (out_dir / "samples.json").open("w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nwrote eval to {out_dir}")


if __name__ == "__main__":
    main()