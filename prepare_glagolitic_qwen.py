"""
Prepare Glagolitic Transkribus Export for Qwen3 VLM Training

This script processes the Glagolitic Transkribus export to create full-page
image-text pairs for Qwen3 VLM fine-tuning.

Unlike line-level HTR (TrOCR, PyLaia), Qwen3 processes entire pages and
generates complete transcriptions.
"""

import os
import sys
import csv
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from lxml import etree
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial

# Paths
TRAIN_INPUT = Path("C:/Users/Achim/Documents/TrOCR/Glagolitic/Glagolitic_train")
VAL_INPUT = Path("C:/Users/Achim/Documents/TrOCR/Glagolitic/Glagolitic_val")
OUTPUT_ROOT = Path("./data/glagolitic_qwen")

# Namespaces for PAGE XML
NS = {
    'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
}


def extract_page_transcription(xml_path: Path) -> str:
    """
    Extract full-page transcription from PAGE XML.

    Args:
        xml_path: Path to PAGE XML file

    Returns:
        Full page transcription (all lines concatenated)
    """
    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        # Find all TextLine elements
        text_lines = root.findall('.//page:TextLine', namespaces=NS)

        transcriptions = []
        for line in text_lines:
            # Get Unicode text
            unicode_elem = line.find('.//page:Unicode', namespaces=NS)
            if unicode_elem is not None and unicode_elem.text:
                text = unicode_elem.text.strip()
                if text:
                    transcriptions.append(text)

        # Join lines with newlines to preserve page structure
        return '\n'.join(transcriptions)

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return ""


def process_single_page(xml_file: Path, input_dir: Path, images_dir: Path, split_name: str) -> Tuple[bool, str, str, str]:
    """
    Process a single page (to be called in parallel).

    Returns:
        Tuple of (success, image_rel_path, transcription, error_msg)
    """
    try:
        # Extract transcription
        transcription = extract_page_transcription(xml_file)

        if not transcription:
            return (False, "", "", "Empty transcription")

        # Find corresponding image file
        image_base = xml_file.stem

        # Try to find image
        image_path = None
        for ext in ['.tif', '.jpg', '.jpeg', '.png']:
            candidate = input_dir / f"{image_base}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if not image_path:
            return (False, "", "", f"No image found for {xml_file.name}")

        # Copy image to output directory
        output_image = images_dir / f"{image_base}.png"

        # Convert to PNG for consistency
        img = Image.open(image_path)
        img.save(output_image, "PNG")

        # Return relative path and transcription
        image_rel_path = f"{split_name}/images/{output_image.name}"
        return (True, image_rel_path, transcription, "")

    except Exception as e:
        return (False, "", "", str(e))


def process_dataset(input_dir: Path, output_dir: Path, split_name: str) -> Tuple[int, int]:
    """
    Process a dataset (train or val) to create Qwen3-compatible format.

    Args:
        input_dir: Input directory with Transkribus export
        output_dir: Output directory for processed data
        split_name: 'train' or 'val'

    Returns:
        Tuple of (total_pages, valid_pages)
    """
    print(f"\n[{split_name.upper()}] Processing {input_dir}")

    # Create output directories
    images_dir = output_dir / split_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV output
    csv_path = output_dir / f"{split_name}.csv"

    page_dir = input_dir / "page"
    if not page_dir.exists():
        print(f"  Error: No 'page' directory found in {input_dir}")
        return 0, 0

    # Get all PAGE XML files
    xml_files = list(page_dir.glob("*.xml"))
    print(f"  Found {len(xml_files)} PAGE XML files")

    # Use all CPU cores for parallel processing
    num_workers = cpu_count()
    print(f"  Using {num_workers} CPU cores for parallel processing")

    # Create partial function with fixed arguments
    process_func = partial(process_single_page,
                          input_dir=input_dir,
                          images_dir=images_dir,
                          split_name=split_name)

    # Process in parallel
    valid_pages = []
    skipped = 0

    with Pool(num_workers) as pool:
        results = pool.map(process_func, xml_files)

    # Collect results
    for success, image_rel_path, transcription, error_msg in results:
        if success:
            valid_pages.append((image_rel_path, transcription))
        else:
            skipped += 1
            if error_msg and error_msg != "Empty transcription":
                print(f"  Warning: {error_msg}")

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for image_path, text in valid_pages:
            writer.writerow([image_path, text])

    print(f"  Processed: {len(valid_pages)} pages")
    print(f"  Skipped: {skipped} pages")
    print(f"  Output CSV: {csv_path}")

    return len(xml_files), len(valid_pages)


def create_dataset_info(output_dir: Path, train_count: int, val_count: int):
    """Create dataset_info.json with metadata."""
    info = {
        "dataset": "Glagolitic Manuscripts",
        "description": "Handwritten Glagolitic script from Transkribus",
        "format": "full-page",
        "preprocessing": {
            "page_level": True,
            "line_segmentation": False
        },
        "splits": {
            "train": {
                "samples": train_count,
                "csv": "train.csv"
            },
            "val": {
                "samples": val_count,
                "csv": "val.csv"
            }
        },
        "notes": [
            "Full-page images with complete transcriptions",
            "Suitable for Qwen3 VLM training",
            "Lines preserved with newlines in transcriptions"
        ]
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nDataset info saved to: {info_path}")


def main():
    print("=" * 80)
    print("Glagolitic Dataset Preparation for Qwen3 VLM Training")
    print("=" * 80)

    # Check input directories
    if not TRAIN_INPUT.exists():
        print(f"Error: Training data not found at {TRAIN_INPUT}")
        sys.exit(1)

    if not VAL_INPUT.exists():
        print(f"Error: Validation data not found at {VAL_INPUT}")
        sys.exit(1)

    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_ROOT}")

    # Process training set
    train_total, train_valid = process_dataset(TRAIN_INPUT, OUTPUT_ROOT, "train")

    # Process validation set
    val_total, val_valid = process_dataset(VAL_INPUT, OUTPUT_ROOT, "val")

    # Create dataset info
    create_dataset_info(OUTPUT_ROOT, train_valid, val_valid)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training set:   {train_valid}/{train_total} pages processed")
    print(f"Validation set: {val_valid}/{val_total} pages processed")
    print(f"\nDataset ready for Qwen3 training!")
    print(f"Use with: python finetune_qwen_ukrainian.py \\")
    print(f"    --data_root \"{OUTPUT_ROOT}\" \\")
    print(f"    --train_csv train.csv \\")
    print(f"    --val_csv val.csv \\")
    print(f"    --output_dir ./models/Qwen3-VL-8B-glagolitic")
    print("=" * 80)


if __name__ == "__main__":
    main()
