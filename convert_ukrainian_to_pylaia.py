#!/usr/bin/env python3
"""
Convert Ukrainian V2 CSV format to PyLaia format.

Input:
- data/pylaia_ukrainian_v2_train/train.csv (21,944 lines)
- data/pylaia_ukrainian_v2_val/train.csv (814 lines)

Output:
- data/pylaia_ukrainian_v2_combined/lines.txt (training set)
- data/pylaia_ukrainian_v2_combined/lines_val.txt (validation set)
- data/pylaia_ukrainian_v2_combined/syms.txt (vocabulary)
- data/pylaia_ukrainian_v2_combined/dataset_info.json (metadata)
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter

def main():
    # Paths
    train_csv = Path('data/pylaia_ukrainian_v2_train/train.csv')
    val_csv = Path('data/pylaia_ukrainian_v2_val/train.csv')
    output_dir = Path('data/pylaia_ukrainian_v2_combined')
    output_dir.mkdir(exist_ok=True)

    # Read CSVs
    print(f"Reading training CSV: {train_csv}")
    train_df = pd.read_csv(train_csv, names=['image_path', 'text'])
    print(f"  Loaded {len(train_df)} training lines")

    print(f"Reading validation CSV: {val_csv}")
    val_df = pd.read_csv(val_csv, names=['image_path', 'text'])
    print(f"  Loaded {len(val_df)} validation lines")

    # Convert paths to relative paths from output directory
    # Training: ../pylaia_ukrainian_v2_train/line_images/...
    # Validation: ../pylaia_ukrainian_v2_val/line_images/...

    lines_train = []
    for _, row in train_df.iterrows():
        img_path = f"../pylaia_ukrainian_v2_train/{row['image_path']}"
        text = row['text']
        lines_train.append(f"{img_path} {text}")

    lines_val = []
    for _, row in val_df.iterrows():
        img_path = f"../pylaia_ukrainian_v2_val/{row['image_path']}"
        text = row['text']
        lines_val.append(f"{img_path} {text}")

    # Write lines.txt
    lines_txt = output_dir / 'lines.txt'
    print(f"\nWriting training lines to {lines_txt}")
    with open(lines_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_train))
    print(f"  Wrote {len(lines_train)} lines")

    # Write lines_val.txt
    lines_val_txt = output_dir / 'lines_val.txt'
    print(f"Writing validation lines to {lines_val_txt}")
    with open(lines_val_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_val))
    print(f"  Wrote {len(lines_val)} lines")

    # Build vocabulary from both train and val
    print("\nBuilding vocabulary...")
    char_counter = Counter()
    for text in train_df['text']:
        char_counter.update(text)
    for text in val_df['text']:
        char_counter.update(text)

    # Sort by frequency (most common first)
    symbols = [char for char, _ in char_counter.most_common()]

    # Add <SPACE> token at the beginning
    if ' ' in symbols:
        symbols.remove(' ')
    symbols = ['<SPACE>'] + symbols

    # Write syms.txt
    syms_txt = output_dir / 'syms.txt'
    print(f"Writing vocabulary to {syms_txt}")
    with open(syms_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(symbols))
    print(f"  Wrote {len(symbols)} symbols")

    # Write metadata
    metadata = {
        'num_train': len(train_df),
        'num_val': len(val_df),
        'total_lines': len(train_df) + len(val_df),
        'vocab_size': len(symbols),
        'background_normalized': False,
        'preserve_aspect_ratio': True,
        'target_height': 128,
        'use_polygon_mask': True,
        'source_train': str(train_csv),
        'source_val': str(val_csv)
    }

    metadata_json = output_dir / 'dataset_info.json'
    print(f"\nWriting metadata to {metadata_json}")
    with open(metadata_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE")
    print("="*60)
    print(f"Training lines:   {len(train_df):,}")
    print(f"Validation lines: {len(val_df):,}")
    print(f"Total lines:      {len(train_df) + len(val_df):,}")
    print(f"Vocabulary size:  {len(symbols)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - lines.txt:       Training set")
    print(f"  - lines_val.txt:   Validation set")
    print(f"  - syms.txt:        Vocabulary ({len(symbols)} symbols)")
    print(f"  - dataset_info.json: Metadata")
    print("="*60)

if __name__ == '__main__':
    main()
