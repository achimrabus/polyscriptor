"""
Convert TrOCR-format dataset to PyLaia format.

PyLaia expects:
- Line images in a single directory
- Text files with same basename as images
- A symbols file with vocabulary
- Train/val list files

Usage:
    python convert_to_pylaia.py --input_csv data/efendiev_3/train.csv --output_dir data/pylaia_efendiev_train
"""

import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Set
import shutil


def normalize_height(image: Image.Image, target_height: int = 64) -> Image.Image:
    """
    Normalize image height while preserving aspect ratio.
    
    Args:
        image: Input PIL Image
        target_height: Target height in pixels
        
    Returns:
        Resized image
    """
    width, height = image.size
    if height == 0:
        return image
    new_width = int(width * target_height / height)
    if new_width == 0:
        new_width = 1
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)


def convert_dataset(
    csv_path: str,
    output_dir: str,
    data_root: str = "data/efendiev_3",
    target_height: int = 64,
    normalize_images: bool = True,
    grayscale: bool = True
):
    """
    Convert TrOCR CSV dataset to PyLaia format.
    
    Args:
        csv_path: Path to input CSV (image_path,text)
        output_dir: Output directory for PyLaia dataset
        data_root: Root directory containing line images
        target_height: Target height for normalized images
        normalize_images: Whether to normalize image heights
        grayscale: Convert images to grayscale
    """
    # Read CSV
    df = pd.read_csv(csv_path, names=['image_path', 'text'], header=None, encoding='utf-8')
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    gt_dir = output_path / "gt"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Track character set and successfully converted samples
    char_set: Set[str] = set()
    list_entries = []
    failed = []
    
    print(f"\nProcessing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Construct full image path
            img_path = Path(data_root) / row['image_path']
            
            if not img_path.exists():
                failed.append(f"{idx}: Image not found: {img_path}")
                continue
            
            # Load image
            img = Image.open(img_path)
            
            # Convert to grayscale if requested
            if grayscale:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            
            # Normalize height if requested
            if normalize_images:
                img = normalize_height(img, target_height)
            
            # Generate output filename (zero-padded index)
            img_id = f"{idx:06d}"
            output_img_path = images_dir / f"{img_id}.png"
            
            # Save image
            img.save(output_img_path, "PNG")
            
            # Get ground truth text
            text = str(row['text']).strip()
            
            if not text:
                failed.append(f"{idx}: Empty text")
                continue
            
            # Save ground truth
            gt_path = gt_dir / f"{img_id}.txt"
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Update character set (preserve spaces)
            char_set.update(text)
            
            # Add to list
            list_entries.append(img_id)
            
        except Exception as e:
            failed.append(f"{idx}: {str(e)}")
    
    print(f"\nSuccessfully converted: {len(list_entries)} samples")
    if failed:
        print(f"Failed: {len(failed)} samples")
        failed_log = output_path / "failed.log"
        with open(failed_log, 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed))
        print(f"Failed samples logged to: {failed_log}")
    
    # Write list file
    list_file = output_path / "lines.txt"
    with open(list_file, 'w', encoding='utf-8') as f:
        for entry in list_entries:
            f.write(f"{entry}\n")
    print(f"\nWrote list file: {list_file}")
    
    # Create symbols file (vocabulary)
    # PyLaia convention: one symbol per line, special tokens first
    symbols = ['<SPACE>']  # Space token
    
    # Add regular characters (sorted, excluding actual spaces)
    regular_chars = sorted(char_set - {' '})
    symbols.extend(regular_chars)
    
    symbols_file = output_path / "symbols.txt"
    with open(symbols_file, 'w', encoding='utf-8') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    print(f"Wrote symbols file: {symbols_file}")
    print(f"Vocabulary size: {len(symbols)} characters")
    
    # Print character set info
    print(f"\nCharacter set preview:")
    preview = symbols[:50]
    print(' '.join(preview))
    if len(symbols) > 50:
        print(f"... and {len(symbols) - 50} more")
    
    # Create a summary file
    summary_file = output_path / "conversion_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"PyLaia Dataset Conversion Summary\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Source CSV: {csv_path}\n")
        f.write(f"Data root: {data_root}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Target height: {target_height}px\n")
        f.write(f"Grayscale: {grayscale}\n")
        f.write(f"Normalize heights: {normalize_images}\n\n")
        f.write(f"Converted samples: {len(list_entries)}\n")
        f.write(f"Failed samples: {len(failed)}\n")
        f.write(f"Vocabulary size: {len(symbols)} characters\n\n")
        f.write(f"Files created:\n")
        f.write(f"  - {len(list_entries)} images in images/\n")
        f.write(f"  - {len(list_entries)} text files in gt/\n")
        f.write(f"  - lines.txt (list file)\n")
        f.write(f"  - symbols.txt (vocabulary)\n")
    
    print(f"\nConversion complete! Summary saved to: {summary_file}")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/           ({len(list_entries)} PNG files)")
    print(f"    ├── gt/               ({len(list_entries)} TXT files)")
    print(f"    ├── lines.txt         (list of sample IDs)")
    print(f"    ├── symbols.txt       (vocabulary)")
    print(f"    └── conversion_summary.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TrOCR dataset to PyLaia format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert training set
  python convert_to_pylaia.py --input_csv data/efendiev_3/train.csv --output_dir data/pylaia_efendiev_train
  
  # Convert validation set
  python convert_to_pylaia.py --input_csv data/efendiev_3/val.csv --output_dir data/pylaia_efendiev_val
  
  # Convert with custom height and keep color
  python convert_to_pylaia.py --input_csv data/efendiev_3/train.csv --output_dir data/pylaia_efendiev_train --height 96 --no-grayscale
        """
    )
    
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Input CSV file (image_path,text format)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for PyLaia dataset'
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/efendiev_3',
        help='Root directory containing line images (default: data/efendiev_3)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=64,
        help='Target image height in pixels (default: 64)'
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Do not normalize image heights (keep original sizes)'
    )
    
    parser.add_argument(
        '--no-grayscale',
        action='store_true',
        help='Keep RGB images instead of converting to grayscale'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PyLaia Dataset Conversion")
    print("=" * 60)
    print(f"Input CSV:     {args.input_csv}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Data root:     {args.data_root}")
    print(f"Target height: {args.height}px")
    print(f"Normalize:     {not args.no_normalize}")
    print(f"Grayscale:     {not args.no_grayscale}")
    print("=" * 60)
    
    convert_dataset(
        csv_path=args.input_csv,
        output_dir=args.output_dir,
        data_root=args.data_root,
        target_height=args.height,
        normalize_images=not args.no_normalize,
        grayscale=not args.no_grayscale
    )


if __name__ == '__main__':
    main()