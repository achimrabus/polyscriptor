"""
Multi-GPU training launcher for 2x RTX 4090

This script sets up environment variables and launches training
on both GPUs using PyTorch DistributedDataParallel.

Usage:
    python train_multi_gpu.py --config config_efendiev.yaml
"""

import os
import sys
import torch
import subprocess

def main():
    # Check GPUs
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs available!")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs:")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if num_gpus < 2:
        print("\nWARNING: Less than 2 GPUs found. Training will use single GPU.")
        print("Running standard training script...")
        # Fall back to single GPU training
        cmd = ["python", "optimized_training.py"] + sys.argv[1:]
        subprocess.run(cmd)
        return

    print(f"\n{'='*60}")
    print(f"MULTI-GPU TRAINING: Using {num_gpus} GPUs")
    print(f"{'='*60}\n")

    # Set environment variables for optimal performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use both GPUs
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings

    # Use torchrun for distributed training (better than python -m torch.distributed.launch)
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",  # Number of GPUs
        "--master_port=29500",  # Port for communication
        "optimized_training.py"
    ] + sys.argv[1:]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\nERROR: 'torchrun' not found. Installing accelerate...")
        subprocess.run(["pip", "install", "-U", "accelerate"], check=True)
        print("\nRetrying with torchrun...")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
