"""
Diagnose tokenizer coverage for Ukrainian characters.
This script checks if the Russian TrOCR tokenizer properly handles Ukrainian-specific characters.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from pathlib import Path
from collections import Counter
from transformers import TrOCRProcessor
import unicodedata

def analyze_tokenizer_coverage():
    print("=" * 80)
    print("TOKENIZER COVERAGE ANALYSIS")
    print("=" * 80)

    # Load processor
    model_name = "kazars24/trocr-base-handwritten-ru"
    processor = TrOCRProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # Load training data
    train_csv = Path("data/ukrainian_train_normalized/train.csv")
    val_csv = Path("data/ukrainian_val_normalized/val.csv")

    if not train_csv.exists():
        print(f"[ERROR] Training CSV not found: {train_csv}")
        return

    df_train = pd.read_csv(train_csv, names=['image_path', 'text'])
    df_val = pd.read_csv(val_csv, names=['image_path', 'text']) if val_csv.exists() else pd.DataFrame()

    # Collect all text
    all_texts = list(df_train['text']) + (list(df_val['text']) if not df_val.empty else [])

    print(f"\n[OK] Loaded {len(df_train)} training + {len(df_val)} validation samples")

    # 1. Character frequency analysis
    print("\n" + "=" * 80)
    print("1. CHARACTER FREQUENCY ANALYSIS")
    print("=" * 80)

    char_counter = Counter()
    for text in all_texts:
        char_counter.update(text)

    # Ukrainian-specific characters
    ua_chars = ['і', 'ї', 'є', 'ґ', 'І', 'Ї', 'Є', 'Ґ']

    print("\nUkrainian-specific characters:")
    for char in ua_chars:
        count = char_counter.get(char, 0)
        print(f"  {char}: {count:,} occurrences")

    print("\nTop 30 most frequent characters:")
    for char, count in char_counter.most_common(30):
        unicode_name = unicodedata.name(char, 'UNKNOWN')
        print(f"  '{char}' (U+{ord(char):04X} {unicode_name}): {count:,}")

    # 2. Tokenization quality check
    print("\n" + "=" * 80)
    print("2. TOKENIZATION QUALITY CHECK")
    print("=" * 80)

    # Test Ukrainian-specific strings
    test_strings = [
        "і ї є ґ",  # Individual UA chars
        "Київ",     # Kiev
        "дівчина",  # girl (has і)
        "їжа",      # food (has ї)
        "єдність",  # unity (has є)
        "ґрунт",    # soil (has ґ)
        "Україна",  # Ukraine
        "об'єкт",   # object (apostrophe + є)
    ]

    print("\nTokenizing Ukrainian test strings:")
    unk_count = 0
    total_tokens = 0

    for text in test_strings:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids)

        has_unk = tokenizer.unk_token in tokens if tokenizer.unk_token else False
        unk_in_ids = tokenizer.unk_token_id in token_ids if tokenizer.unk_token_id else False

        status = "[WARNING] UNK!" if (has_unk or unk_in_ids) else "[OK]"

        print(f"\n  Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Match: {text == decoded} {status}")

        if has_unk or unk_in_ids:
            unk_count += 1
        total_tokens += len(tokens)

    # 3. Sample real training data
    print("\n" + "=" * 80)
    print("3. REAL TRAINING DATA SAMPLES")
    print("=" * 80)

    print("\nTokenizing 10 random training samples:")
    samples = df_train.sample(min(10, len(df_train)))

    unk_samples = 0
    total_char_loss = []

    for idx, row in samples.iterrows():
        text = row['text']
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids)

        has_unk = tokenizer.unk_token in tokens if tokenizer.unk_token else False
        unk_in_ids = tokenizer.unk_token_id in token_ids if tokenizer.unk_token_id else False

        match = text == decoded
        char_loss = len(text) - len(decoded) if not match else 0
        total_char_loss.append(abs(char_loss))

        status = "[WARNING]" if (has_unk or unk_in_ids or not match) else "[OK]"

        print(f"\n  Original: '{text[:80]}{'...' if len(text) > 80 else ''}'")
        print(f"  Decoded:  '{decoded[:80]}{'...' if len(decoded) > 80 else ''}'")
        print(f"  Tokens: {len(tokens)}, Match: {match}, Char loss: {char_loss} {status}")

        if has_unk or unk_in_ids or not match:
            unk_samples += 1

    # 4. Vocabulary check
    print("\n" + "=" * 80)
    print("4. VOCABULARY CHARACTER COVERAGE")
    print("=" * 80)

    vocab = tokenizer.get_vocab()
    print(f"\nTokenizer vocab size: {len(vocab)}")

    # Check if UA chars are in vocab
    print("\nUkrainian character presence in vocabulary:")
    for char in ua_chars:
        in_vocab = char in vocab
        # Also check if it's part of any token
        tokens_with_char = [token for token in vocab.keys() if char in token]
        print(f"  {char}: Direct={in_vocab}, In {len(tokens_with_char)} tokens")
        if tokens_with_char[:5]:  # Show first 5 examples
            print(f"       Examples: {tokens_with_char[:5]}")

    # 5. Summary
    print("\n" + "=" * 80)
    print("5. SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    print(f"\nTokenization issues found:")
    print(f"  - Test strings with UNK: {unk_count}/{len(test_strings)}")
    print(f"  - Training samples with issues: {unk_samples}/10")
    print(f"  - Average character loss: {sum(total_char_loss)/len(total_char_loss):.1f} chars")

    if unk_count > 0 or unk_samples > 2:
        print("\n[CRITICAL] Tokenizer coverage is POOR for Ukrainian!")
        print("RECOMMENDATION: Train a Ukrainian-specific tokenizer")
        print("  1. Extract all training transcriptions")
        print("  2. Train BPE or SentencePiece tokenizer")
        print("  3. Resize model decoder embeddings")
        print("  4. Retrain with new tokenizer")
        print("\nThis is likely the PRIMARY cause of high CER (23%)!")
    else:
        print("\n[OK] Tokenizer coverage seems adequate")
        print("Look for other issues (image preprocessing, normalization)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_tokenizer_coverage()
