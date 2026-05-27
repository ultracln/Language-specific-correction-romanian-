"""
SSL Corruption Module - Generates corrupted versions of clean text for Denoising Autoencoder

This module provides corruption strategies for self-supervised learning:
- The corruption patterns are applied to clean text
- The DAE learns to reconstruct the original from corrupted input
- This is TRUE SSL: self-supervision from unlabeled data

Improvements:
- Multi-error mixing: multiple corruption types per sentence
- Social media domain augmentation (letter repetition, abbreviations, missing spaces)
- Difficulty filtering support via edit_distance()
- Curriculum-ready: intensity parameter controls progression
"""

import random
import unicodedata


def edit_distance(s1: str, s2: str) -> int:
    """Simple edit distance for difficulty filtering."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]


class TextCorruptor:
    """Applies various corruption strategies for SSL pre-training"""
    
    def __init__(self, seed=42):
        random.seed(seed)
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
        # Abrevieri comune românești pentru augmentare social media
        self.ro_abbreviations = {
            'pentru': 'ptr', 'despre': 'dsp', 'împreună': 'împr',
            'acesta': 'ăsta', 'aceasta': 'asta', 'această': 'asta',
            'bineînțeles': 'bînț', 'mulțumesc': 'mulțam', 'mulțumesc': 'ms',
            'salutare': 'sal', 'bună': 'bn', 'mâine': 'mâin',
            'serios': 'srs', 'oricum': 'orc',
        }

    def corrupt_diacritics(self, text, prob=0.15):
        """Remove diacritics with given probability"""
        if random.random() < prob:
            return str(text).translate(self.diacritics_map)
        return text

    def corrupt_character_swap(self, text, prob=0.1):
        """Swap adjacent characters (typo simulation)"""
        if len(text) < 4 or random.random() > prob:
            return text
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

    def corrupt_character_substitute(self, text, prob=0.05):
        """Replace character with keyboard neighbor"""
        if random.random() > prob:
            return text
        words = text.split()
        new_words = []
        for word in words:
            chars = list(word.lower())
            # Apply substitution to 1-2 random chars
            n_subs = random.randint(0, min(2, len(chars)))
            for _ in range(n_subs):
                idx = random.randint(0, len(chars) - 1)
                if chars[idx] in self.kb_neighbors:
                    chars[idx] = random.choice(self.kb_neighbors[chars[idx]])
            new_words.append("".join(chars))
        return " ".join(new_words)

    def corrupt_phonetic(self, text, prob=0.1):
        """Common phonetic/orthographic errors in Romanian"""
        if random.random() > prob:
            return text
        
        corruptions = [
            ('ț', 'tț'),  # doubled consonant
            ('ș', 'sș'),
            ('ii', 'i'),  # double i removal
            ('mp', 'np'),  # n/m confusion before p/b
            ('mb', 'nb'),
        ]
        
        for src, dst in corruptions:
            if random.random() < 0.3:
                text = text.replace(src, dst)
        return text

    def corrupt_case(self, text, prob=0.05):
        """Random case errors"""
        if random.random() > prob:
            return text
        words = text.split()
        new_words = []
        for word in words:
            if random.random() < 0.3 and len(word) > 1:
                # Randomly change case of first letter
                word = word[0].swapcase() + word[1:]
            new_words.append(word)
        return " ".join(new_words)

    def corrupt_token_drop(self, text, prob=0.05):
        """Drop random tokens (simulates OCR/transcription errors)"""
        if random.random() > prob:
            return text
        words = text.split()
        if len(words) <= 2:
            return text
        # Drop 1 random word
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
        return " ".join(words)

    def corrupt_whitespace(self, text, prob=0.05):
        """Add/remove extra whitespace"""
        if random.random() > prob:
            return text
        words = text.split()
        if random.random() < 0.5:
            # Double spaces
            return "  ".join(words)
        return text

    def corrupt_letter_repetition(self, text, prob=0.1):
        """Repeat random vowels in words - social media style (e.g. 'draaaag')"""
        if random.random() > prob:
            return text
        vowels = set('aeiouăâîAEIOUĂÂÎ')
        words = text.split()
        new_words = []
        for word in words:
            if len(word) > 3 and random.random() < 0.25:
                chars = list(word)
                # Find a vowel to repeat
                vowel_indices = [i for i, c in enumerate(chars) if c in vowels]
                if vowel_indices:
                    idx = random.choice(vowel_indices)
                    repeat = random.randint(2, 4)
                    chars = chars[:idx] + [chars[idx]] * repeat + chars[idx+1:]
                word = "".join(chars)
            new_words.append(word)
        return " ".join(new_words)

    def corrupt_abbreviation(self, text, prob=0.1):
        """Replace common Romanian words with social media abbreviations"""
        if random.random() > prob:
            return text
        for full, abbrev in self.ro_abbreviations.items():
            if full in text and random.random() < 0.4:
                text = text.replace(full, abbrev, 1)
        return text

    def corrupt_missing_space(self, text, prob=0.05):
        """Remove space between two random adjacent words"""
        if random.random() > prob:
            return text
        words = text.split()
        if len(words) < 3:
            return text
        idx = random.randint(0, len(words) - 2)
        words[idx] = words[idx] + words[idx + 1]
        del words[idx + 1]
        return " ".join(words)

    # All individual corruption methods as a named registry
    _ALL_METHODS = [
        'diacritics', 'char_swap', 'char_substitute', 'phonetic',
        'case', 'token_drop', 'whitespace',
        'letter_repetition', 'abbreviation', 'missing_space',
    ]

    def _apply_single(self, text, method, intensity):
        """Dispatch a single named corruption method."""
        if method == 'diacritics':
            return self.corrupt_diacritics(text, prob=0.4 * intensity)
        elif method == 'char_swap':
            return self.corrupt_character_swap(text, prob=0.15 * intensity)
        elif method == 'char_substitute':
            return self.corrupt_character_substitute(text, prob=0.1 * intensity)
        elif method == 'phonetic':
            return self.corrupt_phonetic(text, prob=0.15 * intensity)
        elif method == 'case':
            return self.corrupt_case(text, prob=0.2 * intensity)
        elif method == 'token_drop':
            return self.corrupt_token_drop(text, prob=0.08 * intensity)
        elif method == 'whitespace':
            return self.corrupt_whitespace(text, prob=0.1 * intensity)
        elif method == 'letter_repetition':
            return self.corrupt_letter_repetition(text, prob=0.15 * intensity)
        elif method == 'abbreviation':
            return self.corrupt_abbreviation(text, prob=0.15 * intensity)
        elif method == 'missing_space':
            return self.corrupt_missing_space(text, prob=0.08 * intensity)
        return text

    def apply_corruption(self, text, corruption_type='mixed', intensity=0.5, n_corruptions=None):
        """
        Apply corruption to text for DAE training.

        Args:
            text: Clean text to corrupt
            corruption_type: 'light', 'medium', 'heavy', 'social', or 'mixed'
            intensity: Probability multiplier (0.0-1.0); used for curriculum scheduling
            n_corruptions: How many distinct corruption types to chain (None = type-based default)

        Returns:
            Corrupted text for reconstruction target
        """
        # Select method pool based on type
        if corruption_type == 'light':
            pool = ['diacritics', 'char_swap']
            default_n = 1
        elif corruption_type == 'medium':
            pool = ['diacritics', 'char_swap', 'char_substitute', 'phonetic']
            default_n = 2
        elif corruption_type == 'heavy':
            pool = ['char_substitute', 'phonetic', 'case', 'token_drop', 'whitespace', 'missing_space']
            default_n = 3
        elif corruption_type == 'social':
            pool = ['diacritics', 'letter_repetition', 'abbreviation', 'missing_space', 'char_swap']
            default_n = 2
        else:  # mixed - all methods
            pool = self._ALL_METHODS
            default_n = 2

        n = n_corruptions if n_corruptions is not None else default_n
        # Sample without replacement up to pool size
        chosen = random.sample(pool, min(n, len(pool)))

        corrupted = text
        for method in chosen:
            corrupted = self._apply_single(corrupted, method, intensity)
        return corrupted


def create_dae_dataset(texts, corruptor, output_path, intensity=0.5):
    """
    Create a dataset for Denoising Autoencoder training
    
    Args:
        texts: List of clean text samples
        corruptor: TextCorruptor instance
        output_path: Where to save (jsonl format)
        intensity: Corruption intensity
    """
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            if not text or len(text.strip()) < 3:
                continue
            corrupted = corruptor.apply_corruption(text, corruption_type='mixed', intensity=intensity)
            if corrupted == text:  # Skip if no corruption applied
                continue
            row = {
                'clean': text,
                'corrupted': corrupted
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # Simple test
    corruptor = TextCorruptor()
    sample = "Aceasta este o propoziție de testare în limba română cu diacritice."
    print(f"Original:  {sample}")
    print(f"Corrupted: {corruptor.apply_corruption(sample)}")
