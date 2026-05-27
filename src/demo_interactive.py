"""interactive terminal demo of the gec pipeline.

reads one sentence per line from stdin, prints a labeled trace of each
pipeline stage (input → tokens → flagged → tagged → output). reuses
Pipeline directly; no tokenization or inference logic is duplicated here.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from utils import ID_TO_ERROR_TYPE, normalize_romanian, word_tokenize
from pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_ckpt", type=str, default="results/detector/best.pt")
    p.add_argument("--detector_tokenizer", type=str, default="results/detector/tokenizer")
    p.add_argument("--seq2seq_dir", type=str, default="results/seq2seq/best")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--rescore_lm", type=str, default=None)
    p.add_argument("--rescore_lambda", type=float, default=2.0)
    p.add_argument("--lowercase", action="store_true",
                   help="if not passed, the pipeline inherits the value the detector checkpoint was trained with")
    return p.parse_args()


def _fmt_token(t: str) -> str:
    # alphanumeric tokens print bare; punctuation gets python repr quoting.
    return t if t.isalnum() else repr(t)


def _fmt_tokens(tokens: list[str]) -> str:
    return "[" + ", ".join(_fmt_token(t) for t in tokens) + "]"


def _fmt_flagged(tokens: list[str], flags: list[int], types: list[int]) -> str:
    parts = []
    for tok, f, t in zip(tokens, flags, types):
        if f:
            parts.append(f"{tok} ({ID_TO_ERROR_TYPE.get(t, 'no_change')})")
    return ", ".join(parts) if parts else "(none)"


def trace(pipe: Pipeline, sentence: str) -> None:
    sentence = normalize_romanian(sentence)
    tokens = word_tokenize(sentence)
    det_probs, word_ids, type_pred = pipe.detect_probs(tokens)
    flags, types = pipe.flags_from_probs(
        det_probs, word_ids, type_pred, pipe.threshold, len(tokens)
    )

    print(f"INPUT:    {sentence}")
    print(f"TOKENS:   {_fmt_tokens(tokens)}")
    print(f"FLAGGED:  {_fmt_flagged(tokens, flags, types)}")

    if not any(flags):
        print(f"OUTPUT:   {sentence}")
        return

    tagged = pipe.tag(tokens, flags, types)
    output = pipe.correct(tagged, sentence)
    print(f"TAGGED:   {tagged}")
    print(f"OUTPUT:   {output}")


def main():
    args = parse_args()
    pipe = Pipeline(
        args.detector_ckpt, args.detector_tokenizer, args.seq2seq_dir,
        max_length=128, beam_size=4, threshold=args.threshold,
        lowercase=args.lowercase,
        rescore_lm=args.rescore_lm, rescore_lambda=args.rescore_lambda,
    )

    lc_str = "on" if pipe.lowercase else "off"
    rescore_str = "on" if pipe.lm is not None else "off"
    print(f"pipeline loaded: threshold={args.threshold}, lowercase={lc_str}, rescore={rescore_str}")
    print("type a Romanian sentence and press enter. type q to exit.")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            break
        try:
            trace(pipe, line)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
