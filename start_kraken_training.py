"""
Smart Kraken Training Starter with OOM Protection

Automatically detects available GPU VRAM and adjusts batch size.
Retries with reduced batch size if OOM occurs.
"""

import subprocess
import sys

def get_gpu_memory():
    """Get free GPU memory in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        # Get first GPU's free memory
        free_mem = int(result.stdout.strip().split('\n')[0])
        return free_mem
    except Exception as e:
        print(f"Warning: Could not detect GPU memory: {e}")
        return 8000  # Conservative default

def calculate_safe_batch_size(free_mem_mb):
    """
    Calculate safe batch size based on free VRAM.

    Kraken HTR memory usage (empirical):
    - Base model: ~2GB
    - Per sample: ~500-800MB (depends on image width)
    - Safe buffer: 4GB
    """
    usable_mem = free_mem_mb - 4000  # Leave 4GB buffer

    if usable_mem < 2000:
        return 1  # Minimum

    # Conservative estimate: 800MB per sample
    batch_size = int(usable_mem / 800)

    # Cap at reasonable maximum
    batch_size = min(batch_size, 16)
    batch_size = max(batch_size, 1)  # At least 1

    return batch_size

def main():
    print("=" * 80)
    print("Kraken Training with OOM Protection")
    print("=" * 80)

    # Check GPU memory
    free_mem = get_gpu_memory()
    print(f"\nGPU 0 Free Memory: {free_mem} MB")

    # Calculate batch size
    batch_size = calculate_safe_batch_size(free_mem)
    print(f"Recommended Batch Size: {batch_size}")
    print(f"Expected VRAM Usage: ~{2000 + batch_size * 800} MB")
    print(f"Safety Margin: ~{free_mem - (2000 + batch_size * 800)} MB")

    # Try training with automatic retry on OOM
    max_retries = 3
    current_batch = batch_size

    for attempt in range(max_retries):
        print("\n" + "=" * 80)
        print(f"Attempt {attempt + 1}/{max_retries} - Batch Size: {current_batch}")
        print("=" * 80 + "\n")

        # Update batch size in training script
        import train_kraken_from_csv
        train_kraken_from_csv.BATCH_SIZE = current_batch

        try:
            # Run training
            train_kraken_from_csv.main()

            # If we get here, training succeeded
            print("\n" + "=" * 80)
            print("OK: Training completed successfully!")
            print("=" * 80)
            return

        except subprocess.CalledProcessError as e:
            error_output = str(e)

            # Check if it's an OOM error
            if "out of memory" in error_output.lower() or "oom" in error_output.lower():
                print("\nWarning: Out of Memory Error Detected!")

                if attempt < max_retries - 1:
                    # Reduce batch size and retry
                    current_batch = max(1, current_batch // 2)
                    print(f"Retrying with reduced batch size: {current_batch}")
                else:
                    print("Maximum retries reached. Try manually reducing batch size.")
                    sys.exit(1)
            else:
                # Not an OOM error, don't retry
                print(f"Training failed: {e}")
                sys.exit(1)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            sys.exit(0)

if __name__ == "__main__":
    main()
