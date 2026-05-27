"""
Utility to prepare unlabeled text corpus for SSL pre-training

Usage:
    python3 prepare_unlabeled_corpus.py --input <source.csv> --output data/unlabeled_corpus.txt
    
This script extracts clean text from various sources to create an unlabeled corpus
for self-supervised learning with Denoising Autoencoder.
"""

import argparse
from pathlib import Path
import pandas as pd


def extract_from_csv(csv_path, column_names=None, output_path=None):
    """
    Extract text from CSV file columns
    
    Args:
        csv_path: Path to CSV file
        column_names: List of column names to extract (auto-detect if None)
        output_path: Where to save (uses csv_path.txt if None)
    """
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Auto-detect text columns if not specified
    if column_names is None:
        # Look for common column names
        possible_cols = ['text', 'content', 'sentence', 'correct', 'clean', 'data', 'description']
        column_names = [col for col in possible_cols if col in df.columns]
        
        if not column_names:
            column_names = [df.columns[0]]  # Use first column as fallback
    
    print(f"Using columns: {column_names}")
    
    # Extract unique texts
    texts = set()
    for col in column_names:
        if col in df.columns:
            col_texts = df[col].dropna().astype(str)
            texts.update(col_texts[col_texts.str.len() > 5])  # Minimum 5 chars
    
    texts = sorted(list(texts))
    print(f"Extracted {len(texts)} unique texts")
    
    # Save to file
    if output_path is None:
        output_path = Path(csv_path).stem + ".txt"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            text = text.strip()
            if text and len(text) > 3:
                f.write(text + '\n')
    
    print(f"✓ Saved {len(texts)} texts to {output_path}")
    return output_path


def create_sample_corpus():
    """Create a sample unlabeled corpus for testing"""
    sample_texts = [
        "Aceasta este o propoziție de testare în limba română cu diacritice.",
        "Corectarea gramaticală este o sarcină complexă în procesarea limbajului natural.",
        "Modelele de învățare automată pot detecta și corecta erori în texte.",
        "Învățarea auto-supervizată se bazează pe date neetichetate.",
        "Autoencoderele cu zgomot sunt utile pentru pre-antrenament pe corpusuri mari.",
        "Transformatorii sunt arhitecturi puternice pentru procesarea textului.",
        "Limba română are caractere speciale: ă, â, î, ș, ț.",
        "Erorile gramaticale includ diacritice, ortografie și acorduri.",
        "Corpusurile de text sunt resurse valoroase pentru învățarea în domeniu.",
        "Evaluarea modelelor NLP necesită metrice standardizate și seturi de test bine definite.",
    ]
    
    output_path = Path("data/unlabeled_corpus_sample.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print(f"✓ Created sample corpus: {output_path}")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Prepare unlabeled corpus for SSL training")
    p.add_argument("--input", type=str, help="Input CSV file")
    p.add_argument("--output", type=str, help="Output text file")
    p.add_argument("--columns", type=str, nargs="+", help="Column names to extract")
    p.add_argument("--sample", action="store_true", help="Create sample corpus")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.sample:
        create_sample_corpus()
    elif args.input:
        extract_from_csv(args.input, column_names=args.columns, output_path=args.output)
    else:
        print("Usage:")
        print("  python3 prepare_unlabeled_corpus.py --input data.csv --output corpus.txt")
        print("  python3 prepare_unlabeled_corpus.py --sample")
