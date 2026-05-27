"""run the eval pipeline at a fixed detector threshold over multiple
rescore_lambda values. per sentence, the detector pass, the corrector
generate, the lm scoring, and the edit distance are computed once and cached;
only the argmax repeats per lambda."""
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


def parse_lambdas(s: str) -> list[float]:
    try:
        lambdas = [float(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid --lambdas value: {e}")
    if not lambdas:
        raise argparse.ArgumentTypeError("--lambdas must contain at least one value")
    return lambdas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_ckpt", type=str, default="results/detector/best.pt")
    p.add_argument("--detector_tokenizer", type=str, default="results/detector/tokenizer")
    p.add_argument("--seq2seq_dir", type=str, default="results/seq2seq/best")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--out_dir", type=str, default="results/eval_rescore")
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--max_examples", type=int, default=-1)
    p.add_argument("--threshold", type=float, default=0.3,
                   help="fixed detector threshold (sweep optimum from sweep_threshold)")
    p.add_argument("--lambdas", type=parse_lambdas,
                   default=parse_lambdas("0.1,0.25,0.5,1.0,2.0"),
                   help="comma-separated list of rescore_lambda values")
    p.add_argument("--rescore_lm", type=str, required=True,
                   help="HF causal LM id; required (lambda sweep only makes sense with rescoring)")
    p.add_argument("--rescore_topk", type=int, default=None)
    p.add_argument("--diverse_beams", action="store_true",
                   help="enable diverse beam search in the corrector")
    p.add_argument("--diversity_penalty", type=float, default=0.5)
    p.add_argument("--errant", action=argparse.BooleanOptionalAction, default=True,
                   help="compute span-based F0.5 using manually-emitted M2 + upstream errant_compare")
    p.add_argument("--errant_bin_dir", type=str, default=None,
                   help="directory containing errant_compare; defaults to PATH lookup")
    p.add_argument("--keep_tmp", action="store_true")
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
    print(f"threshold: {args.threshold}")
    print(f"lambdas: {args.lambdas}")
    topk = args.rescore_topk if args.rescore_topk is not None else args.beam_size
    print(f"rescoring: {args.rescore_lm} (topk={topk})")

    # pipeline.rescore_lambda is unused in the sweep loop — pick_best is called
    # with explicit lambdas. lm + corrector + detector are all loaded once here.
    pipe = Pipeline(args.detector_ckpt, args.detector_tokenizer, args.seq2seq_dir,
                    args.max_length, args.beam_size, threshold=args.threshold,
                    lowercase=args.lowercase,
                    rescore_lm=args.rescore_lm, rescore_lambda=0.0,
                    rescore_topk=args.rescore_topk,
                    diverse_beams=args.diverse_beams, diversity_penalty=args.diversity_penalty)
    if args.diverse_beams:
        print(f"diverse beams: enabled (penalty={args.diversity_penalty})")

    n = len(pairs) if args.max_examples <= 0 else min(len(pairs), args.max_examples)
    lambdas = args.lambdas

    per_lam = {
        lam: {
            "correct": 0, "changed": 0, "spurious": 0, "stayed_same": 0, "total": 0,
            "sources": [], "hypotheses": [], "references": [],
        }
        for lam in lambdas
    }

    for i in tqdm(range(n)):
        inc = pairs[i]["incorrect"]
        cor = pairs[i]["correct"]
        sentence = normalize_romanian(inc)
        tokens = word_tokenize(sentence)

        flags, types = pipe.detect(tokens)

        truth = normalize_for_match(cor)
        inc_norm = normalize_for_match(inc)
        has_err = inc_norm != truth

        if not any(flags):
            # no flagged spans → output is the input for every lambda, no lm calls.
            cands = None
        else:
            tagged = pipe.tag(tokens, flags, types)
            # single corrector generate + K lm forwards + K edit distances per sentence.
            cands = pipe.score_candidates(tagged, sentence)

        for lam in lambdas:
            if cands is None:
                output = sentence
            else:
                output = pipe.pick_best(cands, lam)
            pred = normalize_for_match(output)
            is_correct = pred == truth
            was_changed = pred != inc_norm

            s = per_lam[lam]
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

    lambda_results = []
    for lam in lambdas:
        s = per_lam[lam]
        result = {
            "lambda": lam,
            "exact_match_acc": s["correct"] / max(s["total"], 1),
            "changed_rate": s["changed"] / max(s["total"], 1),
            "spurious_rate": s["spurious"] / max(s["total"], 1),
            "stayed_same_rate": s["stayed_same"] / max(s["total"], 1),
        }
        if args.errant:
            errant_dir = out_dir / f"errant_tmp_lam_{lam:.3f}"
            er = errant_score(s["sources"], s["hypotheses"], s["references"], errant_dir,
                              keep_tmp=args.keep_tmp, bin_dir=args.errant_bin_dir)
            if er is not None:
                result["errant_precision"] = er["precision"]
                result["errant_recall"] = er["recall"]
                result["errant_f05"] = er["f05"]
                result["errant_n"] = er["n"]
            else:
                result["errant_status"] = "failed; see stdout for details"
        lambda_results.append(result)

    best_by_em = max(lambda_results, key=lambda r: r["exact_match_acc"])["lambda"]
    ranked_errant = [r for r in lambda_results if "errant_f05" in r]
    best_by_errant = max(ranked_errant, key=lambda r: r["errant_f05"])["lambda"] if ranked_errant else None

    summary = {
        "dataset": args.dataset,
        "split": split,
        "threshold": args.threshold,
        "rescore_lm": args.rescore_lm,
        "rescore_topk": topk,
        "n_pairs": n,
        "lambdas": lambda_results,
        "best_by_errant_f05": best_by_errant,
        "best_by_exact_match": best_by_em,
    }

    name = args.dataset.replace("/", "_")
    out_path = out_dir / f"{name}_lambda_sweep.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # comparison table
    print()
    print(f"{'lambda':>8} {'em':>7} {'changed':>9} {'spurious':>10} {'stayed_same':>13} {'err_p':>7} {'err_r':>7} {'err_f05':>8}")
    for r in lambda_results:
        mark = " *" if r["lambda"] == best_by_errant else ""
        ep = r.get("errant_precision", float("nan"))
        er_ = r.get("errant_recall", float("nan"))
        ef = r.get("errant_f05", float("nan"))
        print(
            f"{r['lambda']:>8.3f} {r['exact_match_acc']:>7.4f} "
            f"{r['changed_rate']:>9.4f} {r['spurious_rate']:>10.4f} "
            f"{r['stayed_same_rate']:>13.4f} "
            f"{ep:>7.4f} {er_:>7.4f} {ef:>8.4f}{mark}"
        )
    print(f"\nbest by errant_f05: {best_by_errant}  |  best by exact match: {best_by_em}")
    print(f"wrote sweep to {out_path}")


if __name__ == "__main__":
    main()
