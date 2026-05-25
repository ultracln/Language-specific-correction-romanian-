import argparse
import os
import random
import re

import pandas as pd


class RomanianSSLNoiser:
    def __init__(self):
        # diacritics removal map
        self.diacritics_map = str.maketrans("ăâîșțĂÂÎȘȚ", "aaistAAIST")
        # qwerty neighbors for typo simulation
        self.kb_neighbors = {
            'a': 'qwsz', 's': 'qwedcxaz', 'd': 'werfvcxs', 'f': 'ertgbvcd',
            'g': 'rtyhnbvf', 'h': 'tyujmnbg', 'j': 'yuiokmnh', 'k': 'uioplmj',
            'l': 'opk', 'm': 'njk', 'n': 'bhj', 'b': 'vgh', 'v': 'cfgb',
            'c': 'xdfv', 'x': 'zsdc', 'z': 'asx', 'r': 'edftg', 't': 'rfghy',
            'y': 'tghju', 'u': 'yhjki', 'i': 'ujklo', 'o': 'iklp', 'p': 'ol'
        }

    def remove_diacritics(self, text):
        return str(text).translate(self.diacritics_map)

    def add_typos(self, text, prob=0.15):
        # swap two adjacent letters
        if len(text) < 4:
            return text
        words = text.split()
        new_words = []
        for word in words:
            if len(word) > 3 and random.random() < prob:
                idx = random.randint(1, len(word) - 2)
                chars = list(word)
                chars[idx], chars[idx - 1] = chars[idx - 1], chars[idx]
                word = "".join(chars)
            new_words.append(word)
        return " ".join(new_words)

    def phonetic_errors(self, text, prob=0.1):
        # final i / ii confusion; n in place of m before p/b
        if random.random() < prob:
            if text.endswith('ii'):
                text = text[:-1]
            elif text.endswith('i'):
                text += 'i'

        # one randomly chosen occurrence of mp -> np or mb -> nb
        if random.random() < prob:
            positions = [m.start() for m in re.finditer(r'm[pb]', text)]
            if positions:
                pos = random.choice(positions)
                text = text[:pos] + 'n' + text[pos + 1:]
        return text

    def lowercase_noise(self, text):
        return text.lower()

    def generate_ssl_data(self, input_path, output_path, seed):
        random.seed(seed)

        if not os.path.exists(input_path):
            print(f"error: input file not found at {input_path}")
            return

        df = pd.read_csv(input_path)
        col_name = 'correct' if 'correct' in df.columns else ('text_curat' if 'text_curat' in df.columns else None)

        if not col_name:
            print(f"error: columns found: {df.columns.tolist()}")
            return

        clean_texts = df[col_name].dropna().tolist()
        ssl_rows = []

        for text in clean_texts:
            # diacritics only
            ssl_rows.append({
                'correct': text,
                'incorrect': self.remove_diacritics(text),
                'error_type': 'diacritics',
                'has_error': 1
            })

            # diacritics + neighbor swap
            noisy_text = self.remove_diacritics(text)
            noisy_text = self.add_typos(noisy_text)
            ssl_rows.append({
                'correct': text,
                'incorrect': noisy_text,
                'error_type': 'spelling',
                'has_error': 1
            })

            # phonetic / final-i errors
            ssl_rows.append({
                'correct': text,
                'incorrect': self.phonetic_errors(text),
                'error_type': 'orthographic',
                'has_error': 1
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(ssl_rows).to_csv(output_path, index=False)
        print(f"ssl dataset saved to {output_path} ({len(ssl_rows)} rows)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, default="data/synthetic.csv")
    p.add_argument("--output_path", type=str, default="data/train_ssl.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    noiser = RomanianSSLNoiser()
    noiser.generate_ssl_data(args.input_path, args.output_path, args.seed)
