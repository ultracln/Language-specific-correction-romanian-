import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))

from utils import (
    ERROR_TYPE_TO_ID,
    normalize_romanian,
    set_seed,
    token_error_labels,
    word_tokenize,
    write_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/prepared")
    p.add_argument("--val_size", type=float, default=0.05)
    p.add_argument("--test_size", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_words", type=int, default=120)
    return p.parse_args()


def build_examples(df: pd.DataFrame):
    detector_rows, corrector_rows, skipped = [], [], 0

    for idx, row in df.iterrows():
        correct = normalize_romanian(str(row["correct"]))
        incorrect = normalize_romanian(str(row["incorrect"]))
        err_type = str(row["error_type"]).strip()
        has_error = int(row["has_error"])

        c_tokens = word_tokenize(correct)
        i_tokens = word_tokenize(incorrect)

        if not i_tokens or not c_tokens:
            skipped += 1
            continue

        word_labels = token_error_labels(i_tokens, c_tokens)
        type_id = ERROR_TYPE_TO_ID.get(err_type, ERROR_TYPE_TO_ID["no_change"])

        if has_error == 0:
            word_type_labels = [0] * len(i_tokens)
        else:
            word_type_labels = [type_id if lbl == 1 else 0 for lbl in word_labels]

        detector_rows.append({
            "id": int(idx),
            "tokens": i_tokens,
            "labels": word_labels,
            "type_labels": word_type_labels,
            "error_type": err_type,
            "has_error": has_error,
        })

        if has_error == 1:
            tagged = tag_errors(i_tokens, word_labels)
            target = " ".join(c_tokens)
            corrector_rows.append({
                "id": int(idx),
                "input": tagged,
                "target": target,
                "error_type": err_type,
            })

    return detector_rows, corrector_rows, skipped


def tag_errors(tokens: list[str], labels: list[int]) -> str:
    out, in_err = [], False
    for tok, lbl in zip(tokens, labels):
        if lbl == 1 and not in_err:
            out.append("<e>")
            in_err = True
        elif lbl == 0 and in_err:
            out.append("</e>")
            in_err = False
        out.append(tok)
    if in_err:
        out.append("</e>")
    return " ".join(out)


def stratified_split(rows, val_size, test_size, seed):
    strata = [r["error_type"] for r in rows]
    train_rows, temp_rows, _, temp_strata = train_test_split(
        rows, strata, test_size=val_size + test_size, random_state=seed, stratify=strata
    )
    rel = test_size / (val_size + test_size)
    val_rows, test_rows = train_test_split(
        temp_rows, test_size=rel, random_state=seed, stratify=temp_strata
    )
    return train_rows, val_rows, test_rows


def report(name, rows):
    counts = Counter(r["error_type"] for r in rows)
    total = len(rows)
    print(f"  {name}: {total} examples")
    for k in sorted(counts):
        pct = 100 * counts[k] / total if total else 0
        print(f"    {k:<14} {counts[k]:>7}  ({pct:5.2f}%)")


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows")
    print("error_type distribution:")
    for k, v in df["error_type"].value_counts().items():
        print(f"  {k:<14} {v}")

    df = df[df["correct"].astype(str).map(lambda s: len(s.split()) <= args.max_words)]
    print(f"after length filter: {len(df)} rows")

    print("building aligned examples...")
    detector_rows, corrector_rows, skipped = build_examples(df)
    print(f"detector examples: {len(detector_rows)}")
    print(f"corrector examples: {len(corrector_rows)}")
    print(f"skipped (empty): {skipped}")

    print("splitting...")
    det_train, det_val, det_test = stratified_split(
        detector_rows, args.val_size, args.test_size, args.seed
    )

    train_ids = {r["id"] for r in det_train}
    val_ids = {r["id"] for r in det_val}
    test_ids = {r["id"] for r in det_test}
    cor_train = [r for r in corrector_rows if r["id"] in train_ids]
    cor_val = [r for r in corrector_rows if r["id"] in val_ids]
    cor_test = [r for r in corrector_rows if r["id"] in test_ids]

    print("\ndetector splits:")
    report("train", det_train)
    report("val", det_val)
    report("test", det_test)

    print("\ncorrector splits (errored only):")
    report("train", cor_train)
    report("val", cor_val)
    report("test", cor_test)

    write_jsonl(out_dir / "detector_train.jsonl", det_train)
    write_jsonl(out_dir / "detector_val.jsonl", det_val)
    write_jsonl(out_dir / "detector_test.jsonl", det_test)
    write_jsonl(out_dir / "corrector_train.jsonl", cor_train)
    write_jsonl(out_dir / "corrector_val.jsonl", cor_val)
    write_jsonl(out_dir / "corrector_test.jsonl", cor_test)

    test_ids_set = {r["id"] for r in det_test}
    df_test = df.iloc[list(test_ids_set)] if False else df.loc[df.index.isin(test_ids_set)]
    df_test.to_csv(out_dir / "test.csv", index=False)
    print(f"wrote test.csv with {len(df_test)} rows for end-to-end eval")

    print(f"\nwrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
