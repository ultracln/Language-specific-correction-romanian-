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
    p.add_argument("--diverse_beams", action="store_true",
                   help="enable diverse beam search in the corrector (one group per beam)")
    p.add_argument("--diversity_penalty", type=float, default=0.5,
                   help="inter-group diversity penalty for diverse beam search")
    return p.parse_args()


class Pipeline:
    def __init__(self, det_ckpt, det_tok, s2s_dir, max_length, beam_size, threshold, lowercase=False,
                 rescore_lm=None, rescore_lambda=0.1, rescore_topk=None,
                 diverse_beams=False, diversity_penalty=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.beam_size = beam_size
        self.threshold = threshold

        # diverse beam search needs num_beams >= 2 (num_beam_groups == num_beams).
        # the divisibility constraint (num_beams % num_beam_groups == 0) holds
        # trivially since we set num_beam_groups = num_beams.
        if diverse_beams and beam_size < 2:
            raise ValueError(
                "diverse_beams requires beam_size >= 2; got beam_size="
                f"{beam_size}. diverse beam search is meaningless with a single beam."
            )
        self.diverse_beams = diverse_beams
        self.diversity_penalty = diversity_penalty
        if diverse_beams:
            # transformers >= 4.59 moved group beam search to a custom_generate
            # repo; both keys are required to route generate() there.
            self._beam_kwargs = {
                "num_beams": beam_size,
                "num_beam_groups": beam_size,
                "diversity_penalty": diversity_penalty,
                "do_sample": False,
                "custom_generate": "transformers-community/group-beam-search",
                "trust_remote_code": True,
            }
        else:
            self._beam_kwargs = {"num_beams": beam_size}

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
    def _lm_logprob(self, text: str) -> float:
        """mean per-token log-likelihood under the rescoring lm. mean (not
        sum) so candidates of different lengths are comparable: a shorter
        candidate that happens to be high-probability per-token shouldn't
        win automatically. empty input returns -inf so empty candidates lose."""
        if not text:
            return float("-inf")
        enc = self.lm_tok(text, return_tensors="pt", truncation=True,
                          max_length=self.max_length).to(self.device)
        if enc["input_ids"].shape[1] < 2:
            # need at least one prediction target after the shift
            return float("-inf")
        out = self.lm(input_ids=enc["input_ids"],
                      attention_mask=enc.get("attention_mask"),
                      labels=enc["input_ids"])
        # hf causal lm returns mean cross-entropy over predicted positions.
        return -out.loss.item()

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """char-level levenshtein. prefers python-Levenshtein if installed,
        else a small rolling-array dp."""
        if a == b:
            return 0
        if _lev_distance is not None:
            return _lev_distance(a, b)
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[m]

    @torch.no_grad()
    def score_candidates(self, tagged, input_sentence):
        """rescoring-only path: generate top-k beams, score each with the lm
        and the char edit-distance penalty. returns one dict per candidate:
            {"text": str, "lm_score": float, "edit_penalty": float}
        all computation here is lambda-independent, so the result can be
        cached across many rescore_lambda values."""
        enc = self.s2s_tok(tagged, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.device)
        k = min(self.rescore_topk, self.beam_size)
        gen = self.s2s.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_return_sequences=k,
            max_length=self.max_length,
            early_stopping=True,
            **self._beam_kwargs,
        )
        candidates_text = self.s2s_tok.batch_decode(gen, skip_special_tokens=True)
        norm = max(len(input_sentence), 1)
        out = []
        for cand in candidates_text:
            lm_score = self._lm_logprob(cand)
            edit_penalty = self._edit_distance(cand, input_sentence) / norm
            out.append({"text": cand, "lm_score": lm_score, "edit_penalty": edit_penalty})
        return out

    @staticmethod
    def pick_best(candidates, rescore_lambda):
        """argmax of lm_score - rescore_lambda * edit_penalty. ties broken by
        higher lm_score (more fluent under the rescoring lm) — deterministic
        because max() with a tuple key is stable on ties via tuple ordering."""
        best = max(
            candidates,
            key=lambda c: (c["lm_score"] - rescore_lambda * c["edit_penalty"], c["lm_score"]),
        )
        return best["text"]

    @torch.no_grad()
    def correct(self, tagged, input_sentence):
        if self.lm is None:
            enc = self.s2s_tok(tagged, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.device)
            gen = self.s2s.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=self.max_length,
                **self._beam_kwargs,
            )
            return self.s2s_tok.decode(gen[0], skip_special_tokens=True)

        # rescoring path: keep top-k beams, rerank by lm fluency minus a small edit penalty.
        cands = self.score_candidates(tagged, input_sentence)
        return self.pick_best(cands, self.rescore_lambda)

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
        output = self.correct(tagged, sentence)
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
                    args.max_length, args.beam_size, args.threshold, lowercase=args.lowercase,
                    rescore_lm=args.rescore_lm, rescore_lambda=args.rescore_lambda,
                    rescore_topk=args.rescore_topk,
                    diverse_beams=args.diverse_beams, diversity_penalty=args.diversity_penalty)

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
