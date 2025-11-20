#!/usr/bin/env python
"""
Quick script to create PyLaia format files (gt/, lines.txt, syms.txt) from TrOCR CSV.
Images are already copied with rsync.
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_pylaia_files_from_csv.py input.csv output_dir")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_dir = Path(sys.argv[2])

    print(f"Reading CSV: {csv_file}")
    df = pd.read_csv(csv_file, header=None, names=['image_path', 'text'])

    print(f"Found {len(df)} samples")

    # Create gt/ directory
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Generate lines.txt and gt/ files
    lines_file = output_dir / "lines.txt"
    all_chars = set()

    with open(lines_file, 'w', encoding='utf-8') as f_lines:
        for idx, row in df.iterrows():
            # Extract image filename (without path and extension)
            img_path = Path(row['image_path'])
            sample_id = img_path.stem

            # Write to lines.txt
            f_lines.write(f"{sample_id}\n")

            # Write ground truth text
            text = str(row['text'])
            gt_file = gt_dir / f"{sample_id}.txt"
            with open(gt_file, 'w', encoding='utf-8') as f_gt:
                f_gt.write(text)

            # Collect characters for vocabulary
            all_chars.update(text)

    print(f"Wrote {len(df)} entries to {lines_file}")
    print(f"Created {len(df)} ground truth files in {gt_dir}")

    # Generate syms.txt (vocabulary in KALDI format: <space> instead of actual space)
    syms_file = output_dir / "symbols.txt"

    # Remove space character and add <SPACE> token
    if ' ' in all_chars:
        all_chars.remove(' ')

    # Remove non-breaking space (U+00A0) if present - it causes empty string bug
    if '\xa0' in all_chars:
        all_chars.remove('\xa0')
        print("  - Removed non-breaking space (U+00A0)")

    # Sort characters (excluding space)
    sorted_chars = sorted(all_chars)

    # Filter out empty strings (should not happen, but safety check)
    sorted_chars = [char for char in sorted_chars if len(char) > 0]

    # Write vocabulary with <SPACE> at the beginning
    with open(syms_file, 'w', encoding='utf-8') as f_syms:
        f_syms.write("<SPACE>\n")
        for char in sorted_chars:
            f_syms.write(f"{char}\n")

    print(f"Wrote vocabulary with {len(sorted_chars) + 1} symbols to {syms_file}")
    print(f"  - <SPACE> token")
    print(f"  - {len(sorted_chars)} unique characters")

    # Preview vocabulary
    print("\nVocabulary preview (first 30 characters):")
    print("<SPACE>", " ".join(sorted_chars[:30]))

    print("\nDone!")

if __name__ == "__main__":
    main()
