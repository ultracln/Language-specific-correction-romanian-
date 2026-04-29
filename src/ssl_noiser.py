import pandas as pd
import random
import os

class RomanianSSLNoiser:
    def __init__(self):
        # Mapare pentru eliminarea diacriticelor
        self.diacritics_map = str.maketrans("ăâîșțĂÂÎȘȚ", "aaistAAIST")
        # Litere vecine pe tastatura QWERTY pentru simularea de typos
        self.kb_neighbors = {
            'a': 'qwsz', 's': 'qwedcxaz', 'd': 'werfvcxs', 'f': 'ertgbvcd',
            'g': 'rtyhnbvf', 'h': 'tyujmnbg', 'j': 'yuiokmnh', 'k': 'uIoplmj',
            'l': 'opk', 'm': 'njk', 'n': 'bhj', 'b': 'vgh', 'v': 'cfgb',
            'c': 'xdfv', 'x': 'zsdc', 'z': 'asx', 'r': 'edftg', 't': 'rfghy',
            'y': 'tghju', 'u': 'yhjki', 'i': 'ujklo', 'o': 'iklp', 'p': 'ol'
        }

    def remove_diacritics(self, text):
        return str(text).translate(self.diacritics_map)

    def add_typos(self, text, prob=0.15):
        """Inversa doua litere vecine (Swap)"""
        if len(text) < 4: return text
        words = text.split()
        new_words = []
        for word in words:
            if len(word) > 3 and random.random() < prob:
                idx = random.randint(1, len(word) - 2)
                chars = list(word)
                chars[idx], chars[idx-1] = chars[idx-1], chars[idx]
                word = "".join(chars)
            new_words.append(word)
        return " ".join(new_words)

    def phonetic_errors(self, text, prob=0.1):
        """Erori specifice: i-uri finale, n in loc de m inainte de p/b"""
        # Exemplu: membri -> membrii sau invers
        if random.random() < prob:
            if text.endswith('i'): text += 'i'
            elif text.endswith('ii'): text = text[:-1]
        
        # n in loc de m inainte de p/b (inpreuna)
        text = text.replace('mp', 'np').replace('mb', 'nb') if random.random() < prob else text
        return text

    def lowercase_noise(self, text):
        return text.lower()

    def generate_ssl_data(self, input_path, output_path):
        if not os.path.exists(input_path):
            print(f"EROARE: Nu gasesc fisierul de intrare la {input_path}")
            return

        df = pd.read_csv(input_path)
        col_name = 'correct' if 'correct' in df.columns else ('text_curat' if 'text_curat' in df.columns else None)
        
        if not col_name:
            print(f"EROARE: Coloane gasite: {df.columns.tolist()}")
            return

        clean_texts = df[col_name].dropna().tolist()
        ssl_rows = []

        for text in clean_texts:
            # 1. Doar Diacritice (Task de baza)
            ssl_rows.append({
                'correct': text,
                'incorrect': self.remove_diacritics(text),
                'error_type': 'diacritics',
                'has_error': 1
            })
            
            # 2. Typos + Diacritice (Mix)
            noisy_text = self.remove_diacritics(text)
            noisy_text = self.add_typos(noisy_text)
            ssl_rows.append({
                'correct': text,
                'incorrect': noisy_text,
                'error_type': 'mixed_typo',
                'has_error': 1
            })

            # 3. Erori fonetice/gramaticale usoare
            ssl_rows.append({
                'correct': text,
                'incorrect': self.phonetic_errors(text),
                'error_type': 'phonetic',
                'has_error': 1
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(ssl_rows).to_csv(output_path, index=False)
        print(f"Dataset SSL complex salvat in {output_path} ({len(ssl_rows)} randuri)")

if __name__ == '__main__':
    noiser = RomanianSSLNoiser()
    noiser.generate_ssl_data('data/synthetic.csv', 'data/train_ssl.csv')