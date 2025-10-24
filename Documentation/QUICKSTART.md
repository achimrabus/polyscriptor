# Quick Start Guide - TrOCR Cyrillic Training

⚡ **Get training in under 5 minutes!**

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Transkribus export (PAGE XML + images)

## 1. Install (2 minutes)

```bash
# Clone/navigate to repo
cd dhlab-slavistik

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if using GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 2. Prepare Data (1 minute)

```bash
# Parse your Transkribus export
python transkribus_parser.py \
    --input_dir /path/to/your/transkribus_export \
    --output_dir ./my_data

# Output shows:
# ✓ Extracted 5234 text lines
# ✓ Train: 4187 lines -> my_data/train.csv
# ✓ Val: 1047 lines -> my_data/val.csv
```

## 3. Configure (1 minute)

```bash
# Copy example config
cp example_config.yaml my_config.yaml

# Edit the config file - change this line:
# data_root: "./my_data"  # Your parsed data location
```

Minimal `my_config.yaml`:
```yaml
data_root: "./my_data"
output_dir: "./models/my_model"
batch_size: 16
epochs: 10
cache_images: true
```

## 4. Train (1 minute to start)

```bash
# Start training
python optimized_training.py --config my_config.yaml

# Monitor with TensorBoard (optional, in another terminal)
tensorboard --logdir ./models/my_model
```

## 5. Done! ✓

Training will take 2-4 hours depending on data size.

Your model will be saved to: `./models/my_model/`

---

## Troubleshooting

### "Out of memory" error
Edit config: `batch_size: 8` or `cache_images: false`

### "No XML files found"
Check your `--input_dir` path contains `.xml` files

### Training is slow
- Enable `cache_images: true` (requires RAM)
- Check GPU is being used: `torch.cuda.is_available()`

### Poor results
- Need more training data (>1000 lines recommended)
- Increase epochs: `epochs: 20`
- Try lower learning rate: `learning_rate: 1e-5`

---

## Next Steps

- 📖 Read [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed docs
- 📊 Check [CHANGES.md](CHANGES.md) for performance improvements
- 🔧 Adjust hyperparameters in your config file
- 🎯 Export more data from Transkribus and retrain

---

## Example: Full Workflow

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Parse Transkribus export
python transkribus_parser.py \
    --input_dir ~/Downloads/my_manuscript_export \
    --output_dir ./manuscript_data

# 3. Configure
cat > config.yaml << EOF
data_root: "./manuscript_data"
output_dir: "./models/manuscript_model"
model_name: "kazars24/trocr-base-handwritten-ru"
batch_size: 16
gradient_accumulation_steps: 4
epochs: 10
cache_images: true
use_augmentation: true
EOF

# 4. Train
python optimized_training.py --config config.yaml

# 5. Monitor (in another terminal)
tensorboard --logdir ./models/manuscript_model

# Done! Model ready in ./models/manuscript_model/
```

---

## Performance Comparison

| Pipeline | Training Time | Data Loading | Batch Size |
|----------|--------------|--------------|------------|
| **Old Notebook** | ~10-15 hours | 0.5-2s/batch | 4 |
| **Optimized** | ~2-4 hours | 0.01-0.05s/batch | 64 (effective) |
| **Speedup** | **3-4x faster** | **10-50x faster** | **16x larger** |

---

## Common Config Options

```yaml
# Small GPU (4-6GB VRAM)
batch_size: 4
gradient_accumulation_steps: 16
cache_images: false
gradient_checkpointing: true

# Medium GPU (8-12GB VRAM)
batch_size: 16
gradient_accumulation_steps: 4
cache_images: true
gradient_checkpointing: false

# Large GPU (16+ GB VRAM)
batch_size: 32
gradient_accumulation_steps: 2
cache_images: true
gradient_checkpointing: false
```

---

**Questions?** See [USAGE_GUIDE.md](USAGE_GUIDE.md) or check issues on GitHub.

Happy training! 🚀
