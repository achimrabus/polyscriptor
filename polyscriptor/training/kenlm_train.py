"""
Train a character-level n-gram language model for PyLaia post-correction.

This script trains a KenLM language model on Ukrainian/Cyrillic text for
use with PyLaia CTC beam search decoding.

Usage:
    python train_character_lm.py --input corpus.txt --output ukrainian_char.arpa --order 5

    # Or train from multiple Transkribus PAGE XML files:
    python train_character_lm.py --input_dir data/processed --output ukrainian_char.arpa

Note: Requires KenLM to be installed. On Windows, this is challenging.
      Consider using WSL or pre-built binaries.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET


def extract_text_from_pagexml(xml_path: Path) -> str:
    """Extract ground truth text from Transkribus PAGE XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle namespace
        ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        # Extract all TextLine elements
        lines = []
        for textline in root.findall('.//ns:TextLine', ns):
            unicode_elem = textline.find('.//ns:Unicode', ns)
            if unicode_elem is not None and unicode_elem.text:
                lines.append(unicode_elem.text.strip())

        return '\n'.join(lines)

    except Exception as e:
        print(f"Warning: Failed to parse {xml_path}: {e}")
        return ""


def prepare_corpus(input_path: Path, output_path: Path, input_dir: Path = None):
    """
    Prepare character-level corpus from text file or PAGE XML directory.

    For character-level LM, we separate each character with spaces so KenLM
    treats them as "words".
    """
    all_text = []

    if input_dir and input_dir.exists():
        # Extract text from all PAGE XML files
        print(f"Extracting text from PAGE XML files in {input_dir}...")
        xml_files = list(input_dir.glob('**/*.xml'))
        for xml_file in xml_files:
            text = extract_text_from_pagexml(xml_file)
            if text:
                all_text.append(text)
        print(f"Extracted text from {len(xml_files)} PAGE XML files")

    elif input_path and input_path.exists():
        # Read from plain text file
        print(f"Reading text from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            all_text.append(f.read())

    else:
        raise ValueError("Must provide either --input or --input_dir")

    # Combine all text
    corpus_text = '\n'.join(all_text)

    # Convert to character-level (space-separated characters)
    char_lines = []
    for line in corpus_text.split('\n'):
        if line.strip():
            # Separate each character with space, preserve actual spaces as <SPACE>
            char_line = ' '.join(
                '<SPACE>' if c == ' ' else c
                for c in line.strip()
            )
            char_lines.append(char_line)

    # Write character-level corpus
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(char_lines))

    print(f"Prepared character-level corpus: {len(char_lines)} lines")
    return len(char_lines)


def train_kenlm(corpus_path: Path, output_path: Path, order: int = 5):
    """
    Train KenLM language model.

    Args:
        corpus_path: Path to character-level corpus
        output_path: Path to output .arpa file
        order: N-gram order (default: 5 for character-level)
    """
    print(f"Training {order}-gram character-level language model...")

    try:
        # Try to run lmplz (KenLM binary)
        cmd = [
            'lmplz',
            '-o', str(order),
            '--text', str(corpus_path),
            '--arpa', str(output_path),
            '--discount_fallback',
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"\nLanguage model saved to: {output_path}")

        # Optionally convert to binary format for faster loading
        binary_path = output_path.with_suffix('.bin')
        try:
            subprocess.run(
                ['build_binary', str(output_path), str(binary_path)],
                check=True
            )
            print(f"Binary model saved to: {binary_path}")
        except subprocess.CalledProcessError:
            print("Note: Could not create binary model (build_binary not found)")

        return True

    except FileNotFoundError:
        print("\nERROR: KenLM not found!")
        print("\nKenLM installation instructions:")
        print("  Linux/WSL: sudo apt-get install kenlm")
        print("  macOS: brew install kenlm")
        print("  Windows: Use WSL or download pre-built binaries from:")
        print("           https://kheafield.com/code/kenlm/")
        print("\nAlternative: Use the web-based LM trainer at:")
        print("  https://github.com/kpu/kenlm#for-machine-translation")
        return False

    except subprocess.CalledProcessError as e:
        print(f"ERROR: KenLM training failed: {e}")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train character-level language model for PyLaia'
    )
    parser.add_argument(
        '--input', type=Path,
        help='Input text file (one line per training example)'
    )
    parser.add_argument(
        '--input_dir', type=Path,
        help='Directory with Transkribus PAGE XML files'
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output ARPA language model file'
    )
    parser.add_argument(
        '--order', type=int, default=5,
        help='N-gram order (default: 5 for character-level)'
    )
    parser.add_argument(
        '--keep_corpus', action='store_true',
        help='Keep intermediate character-level corpus file'
    )

    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Prepare character-level corpus
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        suffix='.txt',
        delete=not args.keep_corpus
    ) as corpus_file:
        corpus_path = Path(corpus_file.name)

        # Prepare corpus
        num_lines = prepare_corpus(
            args.input,
            corpus_path,
            args.input_dir
        )

        if num_lines == 0:
            print("ERROR: No text extracted from input")
            return 1

        # Train language model
        success = train_kenlm(corpus_path, args.output, args.order)

        if args.keep_corpus:
            kept_path = args.output.with_suffix('.corpus.txt')
            corpus_path.rename(kept_path)
            print(f"\nCharacter-level corpus saved to: {kept_path}")

    if success:
        print("\n✓ Language model training complete!")
        print(f"\nTo use this model with PyLaia:")
        print(f"  from inference_pylaia_lm import PyLaiaInferenceLM")
        print(f"  model = PyLaiaInferenceLM(")
        print(f"      model_path='models/pylaia_ukrainian',")
        print(f"      lm_path='{args.output}'")
        print(f"  )")
        return 0
    else:
        print("\n✗ Language model training failed")
        print("\nConsider using WSL for easier KenLM installation:")
        print("  wsl")
        print("  sudo apt-get install kenlm")
        print("  python3 train_character_lm.py ...")
        return 1


if __name__ == '__main__':
    exit(main())
