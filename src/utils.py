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
    "noun_form",
    "agreement",
]
ERROR_TYPE_TO_ID = {t: i for i, t in enumerate(ERROR_TYPES)}
ID_TO_ERROR_TYPE = {i: t for t, i in ERROR_TYPE_TO_ID.items()}


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
