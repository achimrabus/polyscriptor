"""
Training wrapper with guardrails for Qwen3-VL-8B Ukrainian fine-tuning.
Monitors GPU memory, training progress, and handles errors gracefully.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_prerequisites():
    """Check all prerequisites before starting training."""
    print("=" * 80)
    print("Pre-flight checks for Qwen3-VL-8B training")
    print("=" * 80)

    # Check GPUs
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available!")
            return False

        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")

        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {name} ({total_mem:.1f} GB)")

            if total_mem < 20:
                print(f"WARNING: GPU {i} has less than 20GB VRAM. Training may fail!")

    except Exception as e:
        print(f"ERROR checking GPUs: {e}")
        return False

    # Check dataset
    data_root = Path("C:/Users/Achim/Documents/TrOCR/dhlab-slavistik/data/ukrainian_train_aspect_ratio")

    if not data_root.exists():
        print(f"ERROR: Dataset not found at {data_root}")
        return False

    train_csv = data_root / "train.csv"
    val_csv = data_root / "val.csv"

    if not train_csv.exists():
        print(f"ERROR: train.csv not found at {train_csv}")
        return False

    if not val_csv.exists():
        print(f"ERROR: val.csv not found at {val_csv}")
        return False

    # Count samples
    with open(train_csv, 'r', encoding='utf-8') as f:
        train_lines = sum(1 for _ in f)

    with open(val_csv, 'r', encoding='utf-8') as f:
        val_lines = sum(1 for _ in f)

    print(f"Dataset: {train_lines} train, {val_lines} val samples")

    if train_lines < 100:
        print("ERROR: Too few training samples!")
        return False

    # Check dependencies
    try:
        from transformers import Qwen3VLForConditionalGeneration
        from peft import LoraConfig
        from trl import SFTTrainer
        from qwen_vl_utils import process_vision_info
        print("All dependencies OK")
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        return False

    # Check disk space for checkpoints
    output_dir = Path("./models/Qwen3-VL-8B-ukrainian")
    output_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / 1024**3
    print(f"Free disk space: {free_gb:.1f} GB")

    if free_gb < 50:
        print("WARNING: Less than 50GB free disk space. Checkpoints may fill disk!")

    print("=" * 80)
    print("All checks passed!")
    print("=" * 80)
    return True


def start_training():
    """Start training with torchrun for multi-GPU."""

    # Training command
    cmd = [
        "torchrun",
        "--nproc_per_node=2",  # Use both GPUs
        "finetune_qwen_ukrainian.py",
        "--data_root", "C:/Users/Achim/Documents/TrOCR/dhlab-slavistik/data/ukrainian_train_aspect_ratio",
        "--train_csv", "train.csv",
        "--val_csv", "val.csv",
        "--output_dir", "./models/Qwen3-VL-8B-ukrainian",
        "--epochs", "5",
        "--batch_size", "4",
        "--gradient_accumulation", "8",
        "--learning_rate", "5e-5",
        "--early_stopping_patience", "3",
        "--early_stopping_threshold", "0.01",
    ]

    print("\nStarting training with command:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)
    print("Training started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    print("\nIMPORTANT:")
    print("- Training will run for up to 5 epochs (~2-3 hours)")
    print("- Early stopping enabled (stops if no improvement for 3 evaluations)")
    print("- Checkpoints saved every 100 steps to ./models/Qwen3-VL-8B-ukrainian")
    print("- Monitor with: nvidia-smi -l 1")
    print("- TensorBoard: tensorboard --logdir ./models/Qwen3-VL-8B-ukrainian")
    print("\nPress Ctrl+C to stop training (model will save current checkpoint)")
    print("=" * 80 + "\n")

    # Run training
    try:
        process = subprocess.run(cmd, check=True)
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"Training failed with exit code {e.returncode}")
        print("=" * 80)
        return e.returncode
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print("Latest checkpoint saved in ./models/Qwen3-VL-8B-ukrainian")
        print("=" * 80)
        return 130


def main():
    """Main entry point with guardrails."""

    print("\n" + "=" * 80)
    print("Qwen3-VL-8B Ukrainian Manuscript Fine-tuning")
    print("=" * 80 + "\n")

    # Check prerequisites
    if not check_prerequisites():
        print("\nERROR: Prerequisites check failed!")
        print("Please fix the errors above before starting training.")
        sys.exit(1)

    # Confirm with user
    print("\nReady to start training. This will:")
    print("  - Download Qwen3-VL-8B-Instruct model (~16GB)")
    print("  - Train for up to 5 epochs (~2-3 hours)")
    print("  - Use ~20-22GB VRAM per GPU")
    print("  - Save checkpoints to ./models/Qwen3-VL-8B-ukrainian\n")

    # Auto-start (user already confirmed)
    print("Starting in 3 seconds...")
    time.sleep(3)

    # Start training
    exit_code = start_training()

    if exit_code == 0:
        print("\n" + "=" * 80)
        print("Next steps:")
        print("  1. Test model: python inference_qwen.py --model ./models/Qwen3-VL-8B-ukrainian/final_model --image test.jpg")
        print("  2. Use in GUI: Load model in Qwen3 tab")
        print("  3. Evaluate: Compare CER with TrOCR baseline")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("Training failed. Check errors above.")
        print("Common issues:")
        print("  - OOM: Reduce --batch_size to 2")
        print("  - Missing model: Check internet connection")
        print("  - Dataset errors: Verify data paths")
        print("=" * 80 + "\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
