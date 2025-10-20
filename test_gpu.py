"""Quick GPU test for multi-GPU training."""
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

print("="*60)
print("GPU Test")
print("="*60)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# Test model loading
print("\nLoading model...")
try:
    model = VisionEncoderDecoderModel.from_pretrained("kazars24/trocr-base-handwritten-ru")
    print(f"Model loaded successfully")
    print(f"Model device: {next(model.parameters()).device}")

    # Try moving to GPU
    if torch.cuda.is_available():
        print("\nMoving model to GPU...")
        model = model.to('cuda:0')
        print(f"Model on GPU: {next(model.parameters()).device}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        # Try DataParallel
        if torch.cuda.device_count() > 1:
            print(f"\nTrying DataParallel with {torch.cuda.device_count()} GPUs...")
            model = torch.nn.DataParallel(model)
            print("DataParallel successful!")
            print(f"Devices: {model.device_ids}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete!")
print("="*60)
