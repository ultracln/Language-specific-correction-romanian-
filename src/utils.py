import re
import json
import random
import unicodedata
from pathlib import Path

import numpy as np
import torch


CEDILLA_TO_COMMA = {
    "\u015f": "\u0219",  # ş -> ș
    "\u015e": "\u0218",  # Ş -> Ș
    "\u0163": "\u021b",  # ţ -> ț
    "\u0162": "\u021a",  # Ţ -> Ț
}

ERROR_TYPES = [
    "no_change",
    "diacritics",
    "spelling",
    "orthographic",
    "punctuation",
    "morphology",
    "agreement",
]
ERROR_TYPE_TO_ID = {t: i for i, t in enumerate(ERROR_TYPES)}
ID_TO_ERROR_TYPE = {i: t for t, i in ERROR_TYPE_TO_ID.items()}

# mapping from raw dataset error_type strings to the 7-class detector vocabulary.
# the 400k synthetic dataset emits 10 noiser categories; this collapse keeps the
# type head learnable. "mixed" rows have multiple errors but no per-token type;
# they fall back to "spelling" as a generic span label.
ERROR_TYPE_MAP = {
    "no_change":               "no_change",
    "diacritics":              "diacritics",
    "spelling":                "spelling",
    "social_media":            "spelling",
    "orthographic":            "orthographic",
    "punctuation":             "punctuation",
    "noun_form":               "morphology",
    "inflection":              "morphology",
    "agreement":               "agreement",
    "morphological_agreement": "agreement",
    "mixed":                   "spelling",
}


def collapse_error_type(raw: str) -> str:
    """map a raw dataset error_type onto the 7-class detector vocabulary.
    unknown labels default to 'no_change'."""
    return ERROR_TYPE_MAP.get(raw.strip(), "no_change")


# typed span tags. one pair per linguistic error type that can be a span target.
# no_change is intentionally excluded (it's not a span).
ERROR_TYPE_TO_TAG = {
    "diacritics":   ("<e_diac>",  "</e_diac>"),
    "spelling":     ("<e_spell>", "</e_spell>"),
    "orthographic": ("<e_ortho>", "</e_ortho>"),
    "punctuation":  ("<e_punct>", "</e_punct>"),
    "morphology":   ("<e_morph>", "</e_morph>"),
    "agreement":    ("<e_agr>",   "</e_agr>"),
}
SPAN_TAGS_OPEN = [open_t for open_t, _ in ERROR_TYPE_TO_TAG.values()]
SPAN_TAGS_CLOSE = [close_t for _, close_t in ERROR_TYPE_TO_TAG.values()]
ALL_SPAN_TAGS = SPAN_TAGS_OPEN + SPAN_TAGS_CLOSE


def normalize_romanian(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for src, dst in CEDILLA_TO_COMMA.items():
        text = text.replace(src, dst)
    return text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def word_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def lowercase_tokens(tokens: list[str]) -> list[str]:
    """opt-in casing normalization for the detector input only."""
    return [t.lower() for t in tokens]


def levenshtein_align(src: list[str], tgt: list[str]) -> list[tuple[str, int, int]]:
    n, m = len(src), len(tgt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if src[i - 1] == tgt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src[i - 1] == tgt[j - 1]:
            ops.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    ops.reverse()
    return ops


def write_m2(source_path: Path, target_path: Path, out_path: Path, normalize: bool = True) -> None:
    """emit an m2 file from two parallel text files (one sentence per line).

    each block in the output is:
        S <space-joined source tokens>
        A <start> <end>|||OTHER|||<correction>|||REQUIRED|||-NONE-|||0
        ... (one A line per merged edit; noop if no edits)
        <blank line>

    spans are 0-indexed half-open intervals over word_tokenize(source). edits
    merge consecutive non-match ops (sub/del/ins) into a single span; a match
    op breaks the span. insertions have start == end. deletions emit an empty
    correction string. error type is always 'OTHER' — errant_compare's default
    span-based mode ignores categories.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open(encoding="utf-8") as fs, \
         target_path.open(encoding="utf-8") as ft, \
         out_path.open("w", encoding="utf-8") as fo:
        for src_line, tgt_line in zip(fs, ft):
            src = src_line.rstrip("\n")
            tgt = tgt_line.rstrip("\n")
            if normalize:
                src = normalize_romanian(src)
                tgt = normalize_romanian(tgt)
            src_tokens = word_tokenize(src)
            tgt_tokens = word_tokenize(tgt)

            fo.write("S " + " ".join(src_tokens) + "\n")

            if src_tokens == tgt_tokens:
                fo.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
                fo.write("\n")
                continue

            ops = levenshtein_align(src_tokens, tgt_tokens)
            # merge consecutive non-match ops into spans; match breaks the span.
            groups: list[list[tuple[str, int, int]]] = []
            current: list[tuple[str, int, int]] = []
            for op in ops:
                if op[0] == "match":
                    if current:
                        groups.append(current)
                        current = []
                else:
                    current.append(op)
            if current:
                groups.append(current)

            for group in groups:
                src_idxs = [o[1] for o in group if o[0] in ("sub", "del")]
                correction_tokens = [tgt_tokens[o[2]] for o in group if o[0] in ("sub", "ins")]
                if src_idxs:
                    start = min(src_idxs)
                    end = max(src_idxs) + 1
                else:
                    # pure insertion: ins ops within a group share the same src index (the insertion point).
                    start = group[0][1]
                    end = start
                correction = " ".join(correction_tokens)
                fo.write(f"A {start} {end}|||OTHER|||{correction}|||REQUIRED|||-NONE-|||0\n")

            fo.write("\n")


def token_error_labels(incorrect_tokens: list[str], correct_tokens: list[str]) -> list[int]:
    labels = [0] * len(incorrect_tokens)
    ops = levenshtein_align(incorrect_tokens, correct_tokens)
    for op, i, _ in ops:
        if op == "sub" and 0 <= i < len(incorrect_tokens):
            labels[i] = 1
        elif op == "del" and 0 <= i < len(incorrect_tokens):
            labels[i] = 1
        elif op == "ins":
            if i < len(incorrect_tokens):
                labels[i] = 1
            if i - 1 >= 0:
                labels[i - 1] = 1
    return labels


def align_to_subwords(
    word_tokens: list[str],
    word_labels: list[int],
    tokenizer,
    max_length: int,
):
    encoding = tokenizer(
        word_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    word_ids = encoding.word_ids()
    sub_labels = []
    seen = set()
    for wid in word_ids:
        if wid is None:
            sub_labels.append(-100)
        elif wid in seen:
            sub_labels.append(-100)
        else:
            seen.add(wid)
            sub_labels.append(word_labels[wid])
    encoding["labels"] = sub_labels
    return encoding


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)