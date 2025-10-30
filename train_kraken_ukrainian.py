"""
Kraken Training Script for Ukrainian Handwriting Recognition

This script trains a Kraken recognition model on Ukrainian manuscript data
using PAGE XML format with ground truth transcriptions.

Usage:
    python train_kraken_ukrainian.py
"""

import os
import subprocess
import glob
from pathlib import Path

# Training configuration
TRAINING_DATA_DIR = "C:/Users/Achim/Documents/TrOCR/Ukrainian_Data/training_set"
VALIDATION_DATA_DIR = "C:/Users/Achim/Documents/TrOCR/Ukrainian_Data/validation_set"
OUTPUT_DIR = "./models/kraken_ukrainian"
MODEL_NAME = "ukrainian_htr"

# Training parameters
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
DEVICE = "cuda:0"  # Use first GPU
WORKERS = 4

# Model architecture (VGSL spec)
# This is a standard HTR architecture for Kraken
# Format: [input] [conv layers] [LSTM layers] [output]
# Example: [1,48,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]
SPEC = "[1,48,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]"

def collect_xml_files(data_dir):
    """Collect all PAGE XML files from the data directory."""
    page_dir = os.path.join(data_dir, "page")
    if os.path.exists(page_dir):
        xml_files = glob.glob(os.path.join(page_dir, "*.xml"))
    else:
        xml_files = glob.glob(os.path.join(data_dir, "*.xml"))

    # Filter out mets.xml and metadata.xml
    xml_files = [f for f in xml_files if not f.endswith(('mets.xml', 'metadata.xml'))]

    print(f"Found {len(xml_files)} XML files in {data_dir}")
    return xml_files

def create_file_list(xml_files, output_file):
    """Create a text file listing all training files."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for xml_file in xml_files:
            # Kraken expects paths to XML files
            f.write(f"{xml_file}\n")
    print(f"Created file list: {output_file}")

def main():
    print("=" * 80)
    print("Kraken Training Setup for Ukrainian HTR")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect training and validation files
    print("\n[1/5] Collecting data files...")
    train_files = collect_xml_files(TRAINING_DATA_DIR)
    val_files = collect_xml_files(VALIDATION_DATA_DIR)

    if len(train_files) == 0:
        print("ERROR: No training files found!")
        return

    if len(val_files) == 0:
        print("WARNING: No validation files found! Training without validation.")

    # Create file lists
    print("\n[2/5] Creating file lists...")
    train_list = os.path.join(OUTPUT_DIR, "train_files.txt")
    val_list = os.path.join(OUTPUT_DIR, "val_files.txt")

    create_file_list(train_files, train_list)
    if val_files:
        create_file_list(val_files, val_list)

    # Build ketos train command
    print("\n[3/5] Preparing training command...")
    output_model = os.path.join(OUTPUT_DIR, MODEL_NAME)

    cmd = [
        "ketos", "train",
        "-o", output_model,
        "--device", DEVICE,
        "-f", "page",  # Input format is PAGE XML
        "--workers", str(WORKERS),
        "-N", str(EPOCHS),
        "-B", str(BATCH_SIZE),
        "-r", str(LEARNING_RATE),
        "--spec", SPEC,
        "--augment",  # Enable data augmentation
        "--threads", str(WORKERS),
    ]

    # Add training files
    cmd.append("-t")
    cmd.extend(train_files)

    # Add validation files if available
    if val_files:
        cmd.append("-e")
        cmd.extend(val_files)

    print("\n[4/5] Training command prepared")
    print(f"Training {len(train_files)} files, validating on {len(val_files)} files")
    print(f"Output: {output_model}_best.mlmodel")

    # Run training
    print("\n[5/5] Starting training...")
    print("=" * 80)

    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print(f"Model saved to: {output_model}_best.mlmodel")
        print("=" * 80)

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with exit code {e.returncode}")
        print("Check the error messages above for details.")
        return

    except FileNotFoundError:
        print("\nERROR: 'ketos' command not found!")
        print("Make sure Kraken is installed: pip install kraken")
        return

if __name__ == "__main__":
    main()
