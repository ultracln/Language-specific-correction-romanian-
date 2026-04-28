import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.append(str(Path(__file__).resolve().parent))

from utils import (
    ERROR_TYPES,
    ID_TO_ERROR_TYPE,
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
    return p.parse_args()


class Pipeline:
    def __init__(self, det_ckpt, det_tok, s2s_dir, max_length, beam_size, threshold):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.beam_size = beam_size
        self.threshold = threshold

        ckpt = torch.load(det_ckpt, map_location=self.device)
        model_name = ckpt["model_name"]
        self.det_tok = AutoTokenizer.from_pretrained(det_tok)
        self.detector = TwoHeadDetector(model_name, num_types=len(ERROR_TYPES))
        self.detector.load_state_dict(ckpt["state_dict"])
        self.detector.to(self.device).eval()

        self.s2s_tok = AutoTokenizer.from_pretrained(s2s_dir)
        self.s2s = AutoModelForSeq2SeqLM.from_pretrained(s2s_dir).to(self.device).eval()

    @torch.no_grad()
    def detect(self, tokens):
        enc = self.det_tok(tokens, is_split_into_words=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
        det_logits, type_logits = self.detector(enc["input_ids"], enc["attention_mask"])
        det_probs = torch.softmax(det_logits, -1)[0, :, 1]
        type_pred = type_logits.argmax(-1)[0]
        word_ids = enc.word_ids(0)

        word_flags = [0] * len(tokens)
        word_types = [0] * len(tokens)
        seen = set()
        for sub_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            if det_probs[sub_idx].item() >= self.threshold:
                word_flags[wid] = 1
                word_types[wid] = type_pred[sub_idx].item()
        return word_flags, word_types

    def tag(self, tokens, flags):
        out, in_err = [], False
        for tok, lbl in zip(tokens, flags):
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

    def __call__(self, sentence):
        sentence = normalize_romanian(sentence)
        tokens = word_tokenize(sentence)
        flags, types = self.detect(tokens)
        if not any(flags):
            return {
                "input": sentence,
                "output": " ".join(tokens),
                "any_error": False,
                "flagged_tokens": [],
                "predicted_types": [],
            }
        flagged_words = [tokens[i] for i, f in enumerate(flags) if f]
        flagged_types = [ID_TO_ERROR_TYPE[t] for i, t in enumerate(types) if flags[i]]
        tagged = self.tag(tokens, flags)
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
                    args.max_length, args.beam_size, args.threshold)

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
