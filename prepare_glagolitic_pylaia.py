"""
Prepare Glagolitic line-level data for PyLaia training.

PyLaia expects:
- Line images in a directory
- Corresponding .gt.txt files with ground truth transcriptions
- partition files (train.lst, val.lst) listing the basenames
"""

import os
import sys
import csv
from pathlib import Path
from shutil import copy2

def prepare_pylaia_dataset(input_csv: Path, output_dir: Path, split_name: str):
    """
    Convert Qwen3-format CSV to PyLaia format.

    Args:
        input_csv: Path to CSV file with columns: image_path, transcription
        output_dir: Output directory for PyLaia dataset
        split_name: 'train' or 'val'
    """
    print(f"\nPreparing {split_name} set from {input_csv}...")

    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  Found {len(rows)} samples in CSV")

    # Process each sample
    basenames = []
    success_count = 0
    skip_count = 0

    for idx, row in enumerate(rows):
        image_rel_path = row['image_path']
        transcription = row['transcription']

        # Construct source image path
        # image_path format: "train/images/filename.png"
        source_image = Path("data/glagolitic_qwen_lines") / image_rel_path

        if not source_image.exists():
            skip_count += 1
            continue

        # Create basename (filename without extension)
        basename = source_image.stem

        # Copy image to PyLaia images directory
        dest_image = images_dir / f"{basename}.png"
        copy2(source_image, dest_image)

        # Create ground truth text file
        gt_file = images_dir / f"{basename}.gt.txt"
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(transcription)

        basenames.append(basename)
        success_count += 1

    print(f"  Processed {success_count} samples successfully")
    print(f"  Skipped {skip_count} samples (missing images)")

    # Write partition file (list of basenames)
    partition_file = output_dir / f"{split_name}.lst"
    with open(partition_file, 'w', encoding='utf-8') as f:
        for basename in basenames:
            f.write(f"{basename}\n")

    print(f"  Wrote partition file: {partition_file} ({len(basenames)} lines)")

    return success_count

def main():
    # Input CSVs from Qwen3 data preparation
    train_csv = Path("data/glagolitic_qwen_lines/train.csv")
    val_csv = Path("data/glagolitic_qwen_lines/val.csv")

    # Output directory for PyLaia
    output_dir = Path("data/pylaia_glagolitic")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Glagolitic Data Preparation for PyLaia")
    print("=" * 80)
    print(f"Input directory: data/glagolitic_qwen_lines")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Check input files
    if not train_csv.exists():
        print(f"ERROR: Training CSV not found: {train_csv}")
        sys.exit(1)
    if not val_csv.exists():
        print(f"ERROR: Validation CSV not found: {val_csv}")
        sys.exit(1)

    # Prepare training set
    train_count = prepare_pylaia_dataset(train_csv, output_dir, "train")

    # Prepare validation set
    val_count = prepare_pylaia_dataset(val_csv, output_dir, "val")

    # Summary
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Total samples: {train_count + val_count}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  Images: {output_dir}/images/")
    print(f"  Partitions: train.lst, val.lst")
    print("\nNext step: Start PyLaia training with:")
    print(f"  python start_pylaia_glagolitic_training.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
