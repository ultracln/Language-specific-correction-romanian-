"""run the eval pipeline at multiple detector thresholds on a single dataset,
caching the detector forward pass per sentence so only the corrector re-runs
per threshold."""
import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from utils import normalize_romanian, word_tokenize
from pipeline import Pipeline
from eval import make_pairs, normalize_for_match
from errant_eval import errant_score


def parse_thresholds(s: str) -> list[float]:
    try:
        thresholds = [float(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid --thresholds value: {e}")
    if not thresholds:
        raise argparse.ArgumentTypeError("--thresholds must contain at least one value")
    return thresholds


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
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--max_examples", type=int, default=-1)
    p.add_argument("--thresholds", type=parse_thresholds,
                   default=parse_thresholds("0.2,0.3,0.4,0.5"),
                   help="comma-separated list of detection thresholds")
    p.add_argument("--errant", action=argparse.BooleanOptionalAction, default=True,
                   help="compute span-based F0.5 using manually-emitted M2 + upstream errant_compare")
    p.add_argument("--errant_bin_dir", type=str, default=None,
                   help="directory containing errant_compare; defaults to PATH lookup")
    p.add_argument("--keep_tmp", action="store_true")
    p.add_argument("--rescore_lm", type=str, default=None,
                   help="HF causal LM id; enables top-k beam rescoring (loaded once)")
    p.add_argument("--rescore_lambda", type=float, default=0.1)
    p.add_argument("--rescore_topk", type=int, default=None)
    return p.parse_args()


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
    print(f"thresholds: {args.thresholds}")

    # self.threshold is unused in the sweep loop; pipeline calls go through
    # detect_probs + flags_from_probs with explicit thresholds. the lm
    # (if any) is loaded once here and reused across all sentences/thresholds.
    pipe = Pipeline(args.detector_ckpt, args.detector_tokenizer, args.seq2seq_dir,
                    args.max_length, args.beam_size, threshold=0.5,
                    lowercase=args.lowercase,
                    rescore_lm=args.rescore_lm, rescore_lambda=args.rescore_lambda,
                    rescore_topk=args.rescore_topk)
    if args.rescore_lm:
        topk = args.rescore_topk if args.rescore_topk is not None else args.beam_size
        print(f"rescoring: enabled ({args.rescore_lm}, lambda={args.rescore_lambda}, topk={topk})")

    n = len(pairs) if args.max_examples <= 0 else min(len(pairs), args.max_examples)
    thresholds = args.thresholds

    per_thr = {
        thr: {
            "correct": 0, "changed": 0, "spurious": 0, "stayed_same": 0, "total": 0,
            "sources": [], "hypotheses": [], "references": [],
        }
        for thr in thresholds
    }

    for i in tqdm(range(n)):
        inc = pairs[i]["incorrect"]
        cor = pairs[i]["correct"]
        sentence = normalize_romanian(inc)
        tokens = word_tokenize(sentence)

        # single detector forward pass shared across all thresholds for this sentence.
        det_probs, word_ids, type_pred = pipe.detect_probs(tokens)

        truth = normalize_for_match(cor)
        inc_norm = normalize_for_match(inc)
        has_err = inc_norm != truth

        for thr in thresholds:
            flags, types = pipe.flags_from_probs(det_probs, word_ids, type_pred, thr, len(tokens))
            if not any(flags):
                output = sentence
            else:
                tagged = pipe.tag(tokens, flags, types)
                output = pipe.correct(tagged, sentence)
            pred = normalize_for_match(output)
            is_correct = pred == truth
            was_changed = pred != inc_norm

            s = per_thr[thr]
            s["total"] += 1
            if is_correct:
                s["correct"] += 1
            if was_changed:
                s["changed"] += 1
            if not has_err and was_changed:
                s["spurious"] += 1
            if has_err and not was_changed:
                s["stayed_same"] += 1
            s["sources"].append(inc)
            s["hypotheses"].append(output)
            s["references"].append(cor)

    threshold_results = []
    for thr in thresholds:
        s = per_thr[thr]
        result = {
            "threshold": thr,
            "exact_match_acc": s["correct"] / max(s["total"], 1),
            "changed_rate": s["changed"] / max(s["total"], 1),
            "spurious_rate": s["spurious"] / max(s["total"], 1),
            "stayed_same_rate": s["stayed_same"] / max(s["total"], 1),
        }
        if args.errant:
            errant_dir = out_dir / f"errant_tmp_thr_{thr:.3f}"
            er = errant_score(s["sources"], s["hypotheses"], s["references"], errant_dir,
                              keep_tmp=args.keep_tmp, bin_dir=args.errant_bin_dir)
            if er is not None:
                result["errant_precision"] = er["precision"]
                result["errant_recall"] = er["recall"]
                result["errant_f05"] = er["f05"]
                result["errant_n"] = er["n"]
            else:
                result["errant_status"] = "failed; see stdout for details"
        threshold_results.append(result)

    best_by_em = max(threshold_results, key=lambda r: r["exact_match_acc"])["threshold"]
    ranked_errant = [r for r in threshold_results if "errant_f05" in r]
    best_by_errant = max(ranked_errant, key=lambda r: r["errant_f05"])["threshold"] if ranked_errant else None

    summary = {
        "dataset": args.dataset,
        "split": split,
        "n_pairs": n,
        "thresholds": threshold_results,
        "best_by_errant_f05": best_by_errant,
        "best_by_exact_match": best_by_em,
    }

    name = args.dataset.replace("/", "_")
    out_path = out_dir / f"{name}_sweep.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # comparison table
    print()
    print(f"{'threshold':>10} {'em':>7} {'changed':>9} {'spurious':>10} {'stayed_same':>13} {'err_p':>7} {'err_r':>7} {'err_f05':>8}")
    for r in threshold_results:
        mark = " *" if r["threshold"] == best_by_errant else ""
        ep = r.get("errant_precision", float("nan"))
        er_ = r.get("errant_recall", float("nan"))
        ef = r.get("errant_f05", float("nan"))
        print(
            f"{r['threshold']:>10.3f} {r['exact_match_acc']:>7.4f} "
            f"{r['changed_rate']:>9.4f} {r['spurious_rate']:>10.4f} "
            f"{r['stayed_same_rate']:>13.4f} "
            f"{ep:>7.4f} {er_:>7.4f} {ef:>8.4f}{mark}"
        )
    print(f"\nbest by errant_f05: {best_by_errant}  |  best by exact match: {best_by_em}")
    print(f"wrote sweep to {out_path}")


if __name__ == "__main__":
    main()
