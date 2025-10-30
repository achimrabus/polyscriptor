"""
Start PyLaia training for Glagolitic manuscripts.

Uses the PyLaia CTC-based HTR approach with line-level images.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 80)
    print("PyLaia Training - Glagolitic Manuscripts")
    print("=" * 80)

    # Data paths
    data_dir = Path("data/pylaia_glagolitic")
    images_dir = data_dir / "images"
    train_lst = data_dir / "train.lst"
    val_lst = data_dir / "val.lst"
    output_dir = Path("models/pylaia_glagolitic")

    # Check data exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run prepare_glagolitic_pylaia.py first!")
        sys.exit(1)

    if not train_lst.exists() or not val_lst.exists():
        print(f"ERROR: Partition files not found!")
        print(f"  Expected: {train_lst}")
        print(f"  Expected: {val_lst}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count samples
    with open(train_lst) as f:
        train_count = len(f.readlines())
    with open(val_lst) as f:
        val_count = len(f.readlines())

    print(f"\nDataset: Glagolitic Manuscripts (LINE-LEVEL)")
    print(f"  Training samples: {train_count:,}")
    print(f"  Validation samples: {val_count:,}")
    print(f"  Images directory: {images_dir}")
    print(f"  Output directory: {output_dir}")

    print("\n" + "=" * 80)
    print("Starting PyLaia Training")
    print("=" * 80)

    # PyLaia training command
    # Using pylaia-htr-train-ctc for CTC-based HTR
    cmd = [
        "pylaia-htr-train-ctc",
        str(images_dir),  # Image directory
        str(train_lst),    # Training partition
        str(val_lst),      # Validation partition
        "--train_path", str(images_dir),
        "--valid_path", str(images_dir),
        "--output_dir", str(output_dir),
        "--max_epochs", "100",
        "--batch_size", "16",
        "--learning_rate", "0.0003",
        "--gpu", "1",  # Use first GPU
        "--height", "128",  # Fixed height for line images
        "--early_stopping_patience", "20",
        "--checkpoint_save_interval", "5",
    ]

    print("\nCommand:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)

    try:
        # Run training
        result = subprocess.run(cmd, check=True)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nModel saved to: {output_dir}")
        print("\nTo use the model for inference:")
        print(f"  pylaia-htr-decode-ctc {images_dir} test.lst {output_dir}/model.ckpt")

    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print("TRAINING FAILED!")
        print("=" * 80)
        print(f"\nExit code: {e.returncode}")
        print("\nCheck the error messages above for details.")
        sys.exit(1)

    except FileNotFoundError:
        print("\n" + "=" * 80)
        print("ERROR: pylaia-htr-train-ctc not found!")
        print("=" * 80)
        print("\nMake sure PyLaia is installed:")
        print("  pip install pylaia")
        print("\nOr activate the PyLaia virtual environment:")
        print("  .\\venv_pylaia\\Scripts\\activate")
        sys.exit(1)

if __name__ == "__main__":
    main()
