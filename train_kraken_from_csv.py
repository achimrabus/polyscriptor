"""
Kraken Training from CSV + Line Images (Arrow Format)

This script converts already extracted line images + CSV to Arrow format
and trains a Kraken model. Uses the same format as your ChSl notebook.

Usage:
    python train_kraken_from_csv.py
"""

import os
import csv
from pathlib import Path
from PIL import Image
import numpy as np

# Data paths
DATA_ROOT = "C:/Users/Achim/Documents/TrOCR/dhlab-slavistik/data/ukrainian_train_aspect_ratio"
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
OUTPUT_DIR = "./models/kraken_ukrainian_arrow"
MODEL_NAME = "ukrainian_htr"

# Training parameters (optimized for 2x RTX 4090 @ 24GB VRAM each)
LEARNING_RATE = 0.0001  # -r
BATCH_SIZE = 8          # -B (increased from 1 - Kraken HTR uses ~500-800MB per sample, 8 = ~6GB safe)
LAG = 20                # --lag (early stopping patience)
DEVICE = "cuda:0"       # -d (Kraken doesn't support multi-GPU, using GPU 0)
WORKERS = 20            # --workers (parallel CPU data loading)

# Model architecture (matching your ChSl notebook - adapted for height 120)
# Input height 120px (good for Ukrainian lines which are ~128px after aspect ratio preservation)
SPEC = "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]"

def load_csv_with_validation(csv_path, data_root):
    """
    Load CSV and validate that image files exist.
    Gracefully handles missing files.
    """
    valid_samples = []
    missing_files = []

    csv_full_path = os.path.join(data_root, csv_path)

    if not os.path.exists(csv_full_path):
        print(f"ERROR: CSV file not found: {csv_full_path}")
        return [], []

    with open(csv_full_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) < 2:
                continue

            image_rel_path = row[0]
            text = row[1]

            # Full path to image
            image_path = os.path.join(data_root, image_rel_path)

            # Check if file exists
            if os.path.exists(image_path):
                valid_samples.append((image_path, text))
            else:
                missing_files.append(image_path)

    return valid_samples, missing_files

def convert_to_arrow(samples, output_path):
    """
    Convert CSV samples to Arrow format for Kraken.
    Arrow format: binary dataset with image + text pairs.
    """
    try:
        from pyarrow import Table, schema, field, binary, string, RecordBatchStreamWriter
    except ImportError:
        print("ERROR: pyarrow not installed. Install with: pip install pyarrow")
        return False

    print(f"Converting {len(samples)} samples to Arrow format...")

    # Prepare data
    images = []
    texts = []

    for idx, (image_path, text) in enumerate(samples):
        if idx % 1000 == 0:
            print(f"  Processing {idx}/{len(samples)}...")

        try:
            # Read image as bytes
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            images.append(image_bytes)
            texts.append(text)

        except Exception as e:
            print(f"  Warning: Failed to read {image_path}: {e}")
            continue

    # Create Arrow table
    arrow_schema = schema([
        field('image', binary()),
        field('text', string())
    ])

    table = Table.from_arrays(
        [images, texts],
        schema=arrow_schema
    )

    # Write to Arrow file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'wb') as f:
        writer = RecordBatchStreamWriter(f, table.schema)
        writer.write_table(table)
        writer.close()

    print(f"OK: Arrow file created: {output_path}")
    print(f"  Total samples: {len(images)}")

    return True

def main():
    print("=" * 80)
    print("Kraken Training from CSV + Line Images (Arrow Format)")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load and validate training data
    print("\n[1/4] Loading and validating training data...")
    train_samples, train_missing = load_csv_with_validation(TRAIN_CSV, DATA_ROOT)

    if train_missing:
        print(f"  Warning: Warning: {len(train_missing)} training files not found (will be skipped)")
        if len(train_missing) <= 10:
            for f in train_missing:
                try:
                    print(f"    - {f}")
                except UnicodeEncodeError:
                    print(f"    - [filename with unicode characters]")
        else:
            print(f"    First 10 missing files (count only, filenames may contain unicode)")

    print(f"  OK: Loaded {len(train_samples)} valid training samples")

    if len(train_samples) == 0:
        print("ERROR: No valid training samples found!")
        return

    # Step 2: Load and validate validation data
    print("\n[2/4] Loading and validating validation data...")
    val_samples, val_missing = load_csv_with_validation(VAL_CSV, DATA_ROOT)

    if val_missing:
        print(f"  Warning: Warning: {len(val_missing)} validation files not found (will be skipped)")

    print(f"  OK: Loaded {len(val_samples)} valid validation samples")

    # Step 3: Convert to Arrow format
    print("\n[3/4] Converting to Arrow format...")
    train_arrow = os.path.join(OUTPUT_DIR, "train.arrow")
    val_arrow = os.path.join(OUTPUT_DIR, "val.arrow")

    if not convert_to_arrow(train_samples, train_arrow):
        return

    if val_samples:
        if not convert_to_arrow(val_samples, val_arrow):
            return

    # Step 4: Build and run ketos train command
    print("\n[4/4] Preparing training command...")
    output_model = os.path.join(OUTPUT_DIR, MODEL_NAME)

    # Find ketos executable
    import sys
    # The ketos.exe is in LocalCache/local-packages/Python311/Scripts
    ketos_path = r"C:\Users\Achim\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\ketos.exe"

    if not os.path.exists(ketos_path):
        # Fallback to just "ketos" if in PATH
        ketos_path = "ketos"
        print(f"  Warning: ketos.exe not found at expected location, using PATH")

    cmd = [
        ketos_path, "train",
        "-o", output_model,
        "-r", str(LEARNING_RATE),
        "-B", str(BATCH_SIZE),
        "--lag", str(LAG),
        "-s", SPEC,
        "-N", "50",  # Number of epochs
        train_arrow,  # Training data (positional argument)
    ]

    # Validation is enabled automatically if present in Arrow file
    # Kraken 6.0 uses positional arguments for data files

    print("\nTraining configuration:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Early stopping lag: {LAG} epochs")
    print(f"  Epochs: 50")
    print(f"  Device: Auto-detected (CUDA if available)")
    print(f"  Output: {output_model}_best.mlmodel")

    # Run training
    print("\n" + "=" * 80)
    print("Starting Kraken training...")
    print("=" * 80 + "\n")

    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 80)
        print("OK: Training completed successfully!")
        print(f"Model saved to: {output_model}_best.mlmodel")
        print("=" * 80)

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with exit code {e.returncode}")
        print("Check the error messages above for details.")

    except FileNotFoundError:
        print("\nERROR: 'ketos' command not found!")
        print("Make sure Kraken is installed: pip install kraken")

if __name__ == "__main__":
    main()
