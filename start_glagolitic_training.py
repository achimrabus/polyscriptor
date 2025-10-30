"""
Start Glagolitic Qwen3 Training with GPU Detection and Fallback

Attempts multi-GPU training first, falls back to single GPU if issues occur.
"""

import subprocess
import sys
import os
import torch

# Set PyTorch memory management environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def check_gpu_availability():
    """Check available GPUs."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs available!")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s):")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    return num_gpus

def run_multi_gpu_training():
    """Attempt multi-GPU training with torchrun."""
    print("\n" + "=" * 80)
    print("ATTEMPTING MULTI-GPU TRAINING (2x RTX 4090)")
    print("=" * 80)

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "finetune_qwen_ukrainian.py",
        "--data_root", "data/glagolitic_qwen_lines",
        "--train_csv", "train.csv",
        "--val_csv", "val.csv",
        "--output_dir", "./models/Qwen3-VL-8B-glagolitic",
        "--epochs", "10",
        "--batch_size", "4",  # Restored to 4 for line-level images
        "--gradient_accumulation", "4",  # Reduced for multi-GPU
        "--learning_rate", "5e-5",
        "--early_stopping_patience", "3",
        "--early_stopping_threshold", "0.01"
    ]

    print("\nCommand:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nMulti-GPU training failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\nERROR: 'torchrun' not found. Make sure PyTorch distributed is installed.")
        return False

def run_single_gpu_training():
    """Fallback to single GPU training."""
    print("\n" + "=" * 80)
    print("FALLING BACK TO SINGLE-GPU TRAINING")
    print("=" * 80)

    cmd = [
        sys.executable,  # python
        "finetune_qwen_ukrainian.py",
        "--data_root", "data/glagolitic_qwen_lines",
        "--train_csv", "train.csv",
        "--val_csv", "val.csv",
        "--output_dir", "./models/Qwen3-VL-8B-glagolitic",
        "--epochs", "10",
        "--batch_size", "4",  # Restored to 4 for line-level images
        "--gradient_accumulation", "8",  # Standard accumulation
        "--learning_rate", "5e-5",
        "--early_stopping_patience", "3",
        "--early_stopping_threshold", "0.01"
    ]

    print("\nCommand:")
    print(" ".join(cmd))
    print("\nNote: Using batch_size=4 with gradient_accumulation=8 (line-level images)")
    print("Effective batch size: 4 × 8 × 1 = 32")
    print("Memory optimization: gradient_checkpointing=True, expandable_segments=True")
    print("\n" + "=" * 80)

    # Pass environment variables to subprocess
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        result = subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nSingle-GPU training also failed with exit code {e.returncode}")
        return False

def main():
    print("=" * 80)
    print("Glagolitic Qwen3-VL Fine-tuning Launcher")
    print("=" * 80)

    # Check GPU availability
    num_gpus = check_gpu_availability()

    # Dataset info
    print("\nDataset: Glagolitic Manuscripts (LINE-LEVEL)")
    print("  Training samples: 23,203 text lines")
    print("  Validation samples: 1,361 text lines")
    print("  Base model: Qwen/Qwen3-VL-8B-Instruct")

    # Training config
    print("\nTraining Configuration:")
    print("  Epochs: 10")
    print("  Batch size per device: 4")
    print("  Learning rate: 5e-5")
    print("  Early stopping: patience=3, threshold=0.01")

    if num_gpus >= 2:
        print("\nStrategy: Try multi-GPU first, fallback to single GPU if needed")
        success = run_multi_gpu_training()

        if not success:
            print("\nMulti-GPU failed. Retrying with single GPU...")
            success = run_single_gpu_training()
    else:
        print("\nStrategy: Single GPU training (only 1 GPU detected)")
        success = run_single_gpu_training()

    if success:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nModel saved to: ./models/Qwen3-VL-8B-glagolitic")
        print("\nTo use the model:")
        print("  python inference_qwen3.py \\")
        print("    --model_path ./models/Qwen3-VL-8B-glagolitic/final_model \\")
        print("    --image_path your_glagolitic_page.jpg")
    else:
        print("\n" + "=" * 80)
        print("TRAINING FAILED!")
        print("=" * 80)
        print("\nCheck the error messages above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
