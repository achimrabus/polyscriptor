#!/usr/bin/env python3
"""
Re-extract Ukrainian V2c training data WITH EXIF fix.

This script ensures:
1. EXIF rotation is applied (transkribus_parser.py line 232)
2. All 773 pages from training_set are processed (including 99 Лист files)
3. All 28 pages from validation_set are processed (including 5 Лист files)
4. Parallel processing using all CPU cores

Critical: This fixes the V2b bug where Лист files were excluded due to
missing EXIF rotation, causing coordinate misalignment.

Input:
- /home/achimrabus/htr_gui/Ukrainian_Data/training_set/ (773 images)
- /home/achimrabus/htr_gui/Ukrainian_Data/validation_set/ (28 images)

Output:
- data/pylaia_ukrainian_v2c_train/ (expected: ~23,000+ lines from 773 pages)
- data/pylaia_ukrainian_v2c_val/ (expected: ~900+ lines from 28 pages)
"""

import subprocess
import sys
from pathlib import Path
from multiprocessing import cpu_count

def run_extraction(input_dir: str, output_dir: str, description: str):
    """Run transkribus_parser.py with full parallelism."""
    
    print("\n" + "="*70)
    print(f"EXTRACTING: {description}")
    print("="*70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: 24 (limited for system stability)")
    print("="*70)
    
    # Check if output directory already exists
    output_path = Path(output_dir)
    if output_path.exists():
        csv_file = output_path / 'train.csv'
        if csv_file.exists():
            print(f"⚠️  WARNING: Output directory already exists: {output_dir}")
            print(f"   This will use existing data instead of re-extracting!")
            print(f"   Delete the directory first to force fresh extraction.")
            return 0, 0
    
    # Verify EXIF fix is present
    parser_file = Path('transkribus_parser.py')
    if not parser_file.exists():
        print("❌ ERROR: transkribus_parser.py not found!")
        sys.exit(1)
    
    with open(parser_file) as f:
        content = f.read()
        if 'ImageOps.exif_transpose' not in content:
            print("❌ ERROR: EXIF fix not found in transkribus_parser.py!")
            print("   Expected: ImageOps.exif_transpose(page_image)")
            sys.exit(1)
    
    print("✅ EXIF fix verified in transkribus_parser.py")
    
    # Count input images
    input_path = Path(input_dir)
    images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.JPG'))
    print(f"✅ Found {len(images)} input images")
    
    # Check for Лист files
    list_files = [img for img in images if 'Лист' in img.name or 'List' in img.name]
    if list_files:
        print(f"✅ Found {len(list_files)} Лист files (will be included!)")
    
    # Run extraction with polygon masking and aspect ratio preservation
    cmd = [
        '/home/achimrabus/htr_gui/dhlab-slavistik/htr_gui/bin/python',
        'transkribus_parser.py',
        '--input_dir', input_dir,
        '--output_dir', output_dir,
        '--use-polygon-mask',
        '--preserve-aspect-ratio',
        '--target-height', '128',
        '--num-workers', '24',
        '--train_ratio', '1.0'  # Don't split, we're doing train/val separately
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Extraction failed with exit code {result.returncode}")
        sys.exit(1)
    
    # Verify output
    output_path = Path(output_dir)
    csv_file = output_path / 'train.csv'
    line_images_dir = output_path / 'line_images'
    
    if not csv_file.exists():
        print(f"\n❌ ERROR: Expected output file not found: {csv_file}")
        sys.exit(1)
    
    # Count lines
    with open(csv_file) as f:
        num_lines = sum(1 for _ in f)
    
    # Count line images
    num_images = len(list(line_images_dir.glob('*.png'))) if line_images_dir.exists() else 0
    
    print(f"\n✅ Extraction complete!")
    print(f"   Lines extracted: {num_lines:,}")
    print(f"   Line images: {num_images:,}")
    
    # Count pages processed
    pages = set()
    with open(csv_file) as f:
        for line in f:
            img_path = line.split(',')[0]
            page = img_path.split('_')[0].replace('line_images/', '')
            pages.add(page)
    
    print(f"   Pages processed: {len(pages)}")
    
    # Check for Лист lines
    with open(csv_file) as f:
        list_lines = sum(1 for line in f if 'Лист' in line or 'List' in line)
    
    if list_lines > 0:
        print(f"   ✅ Лист lines: {list_lines:,} (SUCCESS - Лист files included!)")
    else:
        print(f"   ⚠️  WARNING: No Лист lines found!")
    
    return num_lines, len(pages)

def main():
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Ukrainian V2c Data Re-extraction" + " "*21 + "║")
    print("║" + " "*15 + "WITH EXIF FIX + Лист FILES" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    
    # Extract training set - use NEW directory name to avoid conflict
    train_lines, train_pages = run_extraction(
        input_dir='/home/achimrabus/htr_gui/Ukrainian_Data/training_set',
        output_dir='data/pylaia_ukrainian_v2c_train_fresh',
        description='Training Set (773 pages expected)'
    )
    
    # Extract validation set - use NEW directory name to avoid conflict
    val_lines, val_pages = run_extraction(
        input_dir='/home/achimrabus/htr_gui/Ukrainian_Data/validation_set',
        output_dir='data/pylaia_ukrainian_v2c_val_fresh',
        description='Validation Set (28 pages expected)'
    )
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "EXTRACTION SUMMARY" + " "*30 + "║")
    print("╠" + "="*68 + "╣")
    print(f"║  Training:   {train_lines:,} lines from {train_pages} pages" + " "*(68-len(f"  Training:   {train_lines:,} lines from {train_pages} pages")) + "║")
    print(f"║  Validation: {val_lines:,} lines from {val_pages} pages" + " "*(68-len(f"  Validation: {val_lines:,} lines from {val_pages} pages")) + "║")
    print(f"║  Total:      {train_lines + val_lines:,} lines from {train_pages + val_pages} pages" + " "*(68-len(f"  Total:      {train_lines + val_lines:,} lines from {train_pages + val_pages} pages")) + "║")
    print("╠" + "="*68 + "╣")
    
    if train_pages >= 773 and val_pages >= 28:
        print("║  ✅ SUCCESS: All pages processed (including Лист files!)" + " "*10 + "║")
    else:
        print(f"║  ⚠️  WARNING: Expected 773 train + 28 val = 801 total pages" + " "*7 + "║")
        print(f"║             Got {train_pages} train + {val_pages} val = {train_pages + val_pages} total pages" + " "*(68-len(f"             Got {train_pages} train + {val_pages} val = {train_pages + val_pages} total pages")) + "║")
    
    print("╠" + "="*68 + "╣")
    print("║  Next steps:" + " "*55 + "║")
    print("║  1. Run: python3 convert_ukrainian_v2c_to_pylaia.py" + " "*15 + "║")
    print("║  2. Inspect: jupyter notebook inspect_ukrainian_v2c.ipynb" + " "*9 + "║")
    print("║  3. Train: python3 train_pylaia_ukrainian_v2c.py" + " "*19 + "║")
    print("╚" + "="*68 + "╝")

if __name__ == '__main__':
    main()
