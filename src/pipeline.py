import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

try:
    from Levenshtein import distance as _lev_distance
except ImportError:
    _lev_distance = None

sys.path.append(str(Path(__file__).resolve().parent))

from utils import (
    ERROR_TYPE_TO_TAG,
    ERROR_TYPES,
    ID_TO_ERROR_TYPE,
    lowercase_tokens,
    normalize_romanian,
    word_tokenize,
)
from detector import TwoHeadDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_ckpt", type=str, default="results/detector/best.pt")
    p.add_argument("--detector_tokenizer", type=str, default="results/detector/tokenizer")
    p.add_argument("--seq2seq_dir", type=str, default="results/seq2seq/best")
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--input_file", type=str, default=None)
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--rescore_lm", type=str, default=None,
                   help="HF causal LM id (e.g. readerbench/RoGPT2-base); enables top-k beam rescoring")
    p.add_argument("--rescore_lambda", type=float, default=0.1,
                   help="weight on the char edit-distance penalty against the input")
    p.add_argument("--rescore_topk", type=int, default=None,
                   help="number of beam candidates to rescore; defaults to beam_size")
    return p.parse_args()


class Pipeline:
    def __init__(self, det_ckpt, det_tok, s2s_dir, max_length, beam_size, threshold, lowercase=False,
                 rescore_lm=None, rescore_lambda=0.1, rescore_topk=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.beam_size = beam_size
        self.threshold = threshold

        ckpt = torch.load(det_ckpt, map_location=self.device)
        model_name = ckpt["model_name"]
        # cli flag wins if set; otherwise pick up the value the checkpoint was trained with.
        self.lowercase = lowercase or ckpt.get("args", {}).get("lowercase", False)
        self.det_tok = AutoTokenizer.from_pretrained(det_tok)
        self.detector = TwoHeadDetector(model_name, num_types=len(ERROR_TYPES))
        self.detector.load_state_dict(ckpt["state_dict"])
        self.detector.to(self.device).eval()

        self.s2s_tok = AutoTokenizer.from_pretrained(s2s_dir)
        self.s2s = AutoModelForSeq2SeqLM.from_pretrained(s2s_dir).to(self.device).eval()

        # optional lm rescoring. loaded once here; reused for every sentence.
        self.rescore_lm_name = rescore_lm
        self.rescore_lambda = rescore_lambda
        self.rescore_topk = rescore_topk if rescore_topk is not None else beam_size
        if rescore_lm is not None:
            self.lm_tok = AutoTokenizer.from_pretrained(rescore_lm)
            self.lm = AutoModelForCausalLM.from_pretrained(rescore_lm).to(self.device).eval()
        else:
            self.lm_tok = None
            self.lm = None

    @torch.no_grad()
    def detect_probs(self, tokens):
        """run the detector once and return raw per-subword arrays. no
        thresholding here — the caller can apply any threshold against the
        cached probs via flags_from_probs."""
        # detector sees lowercased input when enabled; the caller's `tokens`
        # list is preserved so the corrector keeps casing.
        det_input = lowercase_tokens(tokens) if self.lowercase else tokens
        enc = self.det_tok(det_input, is_split_into_words=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
        det_logits, type_logits = self.detector(enc["input_ids"], enc["attention_mask"])
        det_probs = torch.softmax(det_logits, -1)[0, :, 1].cpu().numpy()
        type_pred = type_logits.argmax(-1)[0].cpu().numpy()
        word_ids = enc.word_ids(0)
        return det_probs, word_ids, type_pred

    def flags_from_probs(self, det_probs, word_ids, type_pred, threshold, n_tokens=None):
        """apply a threshold to cached subword probs, propagate to word-level
        via the first-subword-of-each-word rule. n_tokens defaults to the
        unique word-id count, which is correct only when no truncation
        happened; pass len(tokens) explicitly to keep trailing truncated
        tokens flagged as 0 (matches old detect() behavior)."""
        if n_tokens is None:
            non_none = [w for w in word_ids if w is not None]
            n_tokens = (max(non_none) + 1) if non_none else 0
        word_flags = [0] * n_tokens
        word_types = [0] * n_tokens
        seen = set()
        for sub_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen or wid >= n_tokens:
                continue
            seen.add(wid)
            if det_probs[sub_idx] >= threshold:
                word_flags[wid] = 1
                word_types[wid] = int(type_pred[sub_idx])
        return word_flags, word_types

    def detect(self, tokens):
        """backward-compatible wrapper: forwards through detect_probs +
        flags_from_probs using self.threshold."""
        det_probs, word_ids, type_pred = self.detect_probs(tokens)
        return self.flags_from_probs(det_probs, word_ids, type_pred, self.threshold, len(tokens))

    def tag(self, tokens, flags, types):
        out, in_err = [], False
        open_tag, close_tag = "<e_spell>", "</e_spell>"  # fallback, overwritten on span open
        for tok, lbl, tid in zip(tokens, flags, types):
            if lbl == 1 and not in_err:
                type_name = ID_TO_ERROR_TYPE.get(tid, "no_change")
                if type_name == "no_change":
                    open_tag, close_tag = ERROR_TYPE_TO_TAG["spelling"]
                else:
                    open_tag, close_tag = ERROR_TYPE_TO_TAG[type_name]
                out.append(open_tag)
                in_err = True
            elif lbl == 0 and in_err:
                out.append(close_tag)
                in_err = False
            out.append(tok)
        if in_err:
            out.append(close_tag)
        return " ".join(out)

    @torch.no_grad()
    def correct(self, tagged):
        enc = self.s2s_tok(tagged, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.device)
        gen = self.s2s.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=self.beam_size,
            max_length=self.max_length,
        )
        return self.s2s_tok.decode(gen[0], skip_special_tokens=True)

    def __call__(self, sentence, threshold=None):
        thr = self.threshold if threshold is None else threshold
        sentence = normalize_romanian(sentence)
        tokens = word_tokenize(sentence)
        det_probs, word_ids, type_pred = self.detect_probs(tokens)
        flags, types = self.flags_from_probs(det_probs, word_ids, type_pred, thr, len(tokens))
        if not any(flags):
            return {
                "input": sentence,
                "output": sentence,
                "any_error": False,
                "flagged_tokens": [],
                "predicted_types": [],
            }
        flagged_words = [tokens[i] for i, f in enumerate(flags) if f]
        flagged_types = [ID_TO_ERROR_TYPE[t] for i, t in enumerate(types) if flags[i]]
        tagged = self.tag(tokens, flags, types)
        output = self.correct(tagged)
        return {
            "input": sentence,
            "tagged": tagged,
            "output": output,
            "any_error": True,
            "flagged_tokens": flagged_words,
            "predicted_types": flagged_types,
        }


def main():
    args = parse_args()
    pipe = Pipeline(args.detector_ckpt, args.detector_tokenizer, args.seq2seq_dir,
                    args.max_length, args.beam_size, args.threshold, lowercase=args.lowercase)

    if args.text:
        result = pipe(args.text)
        print(f"input:  {result['input']}")
        print(f"output: {result['output']}")
        if result["any_error"]:
            print(f"flagged: {result['flagged_tokens']}")
            print(f"types:   {result['predicted_types']}")
        return

    if args.input_file:
        out_lines = []
        with open(args.input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = pipe(line)
                out_lines.append(result["output"])
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines))
            print(f"wrote {len(out_lines)} corrections to {args.output_file}")
        else:
            for o in out_lines:
                print(o)


if __name__ == "__main__":
    main()
