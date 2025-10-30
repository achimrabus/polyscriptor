# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrOCR fine-tuning pipeline for Cyrillic handwriting recognition (Russian, Ukrainian, Church Slavonic manuscripts). The project uses Transformer-based OCR (TrOCR) from HuggingFace with an optimized training pipeline that includes Transkribus PAGE XML integration and multi-GPU support.

**Key Achievements**:
- 10-50x faster training through image caching and batch optimization (from ~11 hours to ~1 hour on 2x RTX 4090)
- **CRITICAL: Aspect ratio preservation** - Ukrainian line images are typically 4077×357px. Without aspect ratio preservation, TrOCR's ViTImageProcessor brutally resizes to 384×384, causing 10.6x width downsampling (characters shrink from ~80px to ~7px width). Using `--preserve-aspect-ratio` with target height 128px preserves 4x more character resolution.

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 (required for RTX 4090)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Data Preparation
```bash
# Parse Transkribus PAGE XML export to create training dataset
# CRITICAL: Use --preserve-aspect-ratio to avoid brutal downsampling (see below)
python transkribus_parser.py \
    --input_dir /path/to/transkribus_export \
    --output_dir ./processed_data \
    --train_ratio 0.8 \
    --preserve-aspect-ratio \
    --target-height 128 \
    --normalize-background

# Preprocessing flags (order of importance):
# --preserve-aspect-ratio: (CRITICAL) Resize to target height while preserving aspect ratio
#                          Without this, TrOCR's ViT brutally resizes to 384x384, causing
#                          10.6x width downsampling for Ukrainian lines (4077x357 → 384x384)
#                          Characters shrink from ~80px to ~7px width → poor recognition
#                          With this flag: 4077x357 → 128px height → 1467x128 → characters
#                          remain ~28px width (4x improvement)
# --target-height 128: Target height in pixels (default: 128, recommended: 96-150)
#                      Best practice for line HTR with ViT encoder
# --normalize-background: Apply CLAHE normalization (useful for aged/colored paper)
# --use-polygon-mask: Use polygon segmentation instead of bounding boxes (slower)
# --num-workers N: Number of CPU cores for parallel processing (default: all cores)
```

### Training
```bash
# Single GPU training
python optimized_training.py --config config.yaml

# Multi-GPU training (automatic detection)
python optimized_training.py --config config_efendiev.yaml

# Multi-GPU with torchrun (explicit control)
torchrun --nproc_per_node=2 optimized_training.py --config config.yaml

# Monitor training
tensorboard --logdir ./models/<model_name>
```

### Inference
```bash
# Whole-page inference with automatic segmentation
python inference_page.py \
    --image page.jpg \
    --checkpoint models/ukrainian_model/checkpoint-3000 \
    --num_beams 4

# With PAGE XML segmentation (more accurate)
python inference_page.py \
    --image page.jpg \
    --xml page.xml \
    --checkpoint models/ukrainian_model/checkpoint-3000

# With background normalization (REQUIRED if model was trained with it)
python inference_page.py \
    --image page.jpg \
    --checkpoint models/ukrainian_model/checkpoint-3000 \
    --normalize-background

# GUI application for interactive transcription
python transcription_gui_qt.py

# Party GUI for proof-of-concept (experimental)
python transcription_gui_party.py
```

### Party OCR Integration

**Party** is a multilingual transformer-based HTR system that operates on whole pages using PAGE XML format.

```bash
# Party inference via WSL (required on Windows)
wsl bash -c "cd /path/to/image/directory && \
source /mnt/c/Users/Achim/Documents/TrOCR/dhlab-slavistik/venv_party_wsl/bin/activate && \
party -d cuda:0 ocr -i input.xml output.xml \
-mi /path/to/model.safetensors --language chu"

# Example with full paths
wsl bash -c "cd /mnt/c/Users/Achim/Desktop && \
source /mnt/c/Users/Achim/Documents/TrOCR/dhlab-slavistik/venv_party_wsl/bin/activate && \
party -d cuda:0 ocr -i page.xml page_output.xml \
-mi /mnt/c/Users/Achim/Documents/TrOCR/dhlab-slavistik/models/party_models/party_european_langs.safetensors \
--language chu"
```

**Important Notes**:
- Party requires PAGE XML with line segmentation (use `page_xml_exporter.py` or Transkribus export)
- Model: `models/party_models/party_european_langs.safetensors`
- Language code: `chu` (Church Slavonic), `rus` (Russian), `ukr` (Ukrainian)
- **CRITICAL BUG FIX**: `party_repo/party/tokenizer.py:236` - Added `(1, 'big')` to `to_bytes()` for Python 3.10+ compatibility
- Party runs in WSL environment (`venv_party_wsl`) to avoid dependency conflicts
- Image and XML must be in same directory for Party to find the image file

### GPU Monitoring
```bash
# Check GPU availability
nvidia-smi

# Monitor GPU usage in real-time
nvidia-smi -l 1

# Check PyTorch GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Evaluation
```bash
# Detailed checkpoint evaluation
python eval_checkpoint_detailed.py --checkpoint models/ukrainian_model/checkpoint-3000
```

## Architecture Overview

### Core Components

**1. Data Pipeline (`transkribus_parser.py`)**
- Parses Transkribus PAGE XML exports (supports nested `page/` subdirectory structure)
- Extracts line images with bounding boxes or polygon masks
- **CRITICAL PREPROCESSING - Aspect Ratio Preservation**:
  - **Problem**: TrOCR's ViTImageProcessor brutally resizes all images to 384×384
    - Ukrainian manuscript lines are typically 4077×357 pixels (11.4:1 aspect ratio)
    - Direct resize to 384×384 causes 10.6x width downsampling
    - Characters shrink from ~80px width to ~7px width → unreadable for model
  - **Solution**: `--preserve-aspect-ratio` flag (in `transkribus_parser.py`)
    - Resizes to target height (default 128px) while maintaining aspect ratio
    - Example: 4077×357 → 1467×128 (height-normalized)
    - Characters remain ~28px width (4x better than brutal resize)
    - TrOCR's processor will handle final padding to square during training
  - **Implementation**: `resize_with_aspect_ratio()` method at lines 106-145
    - Uses LANCZOS resampling for quality
    - Saves properly sized images to disk
    - Training script doesn't need to do anything special
- **Other preprocessing options**:
  - Background normalization (CLAHE + LAB color space) for aged/colored paper
  - Polygon masking vs bounding box extraction
- Parallel processing with multiprocessing Pool (uses all CPU cores by default)
- Outputs: `train.csv`, `val.csv`, `line_images/`, `dataset_info.json`
- **IMPORTANT**: Dataset metadata includes preprocessing flags (`preserve_aspect_ratio`, `target_height`, `background_normalized`)

**2. Optimized Training (`optimized_training.py`)**
- Image caching in RAM (10-50x faster data loading)
- Large effective batch sizes via gradient accumulation (typically 64-128)
- Mixed precision training (FP16)
- Robust error handling in dataset (filters missing images, handles empty text)
- Multi-GPU support (automatic via HuggingFace Trainer)
- YAML-based configuration system
- Memory monitoring and periodic garbage collection
- **Key classes**:
  - `OptimizedTrainingConfig`: Dataclass with training parameters, loads from YAML
  - `OptimizedOCRDataset`: Dataset with image caching, augmentation, and fallback handling
  - `MemoryMonitorCallback`: Clears GPU memory periodically

**3. Inference System (`inference_page.py`)**
- **Line Segmentation**:
  - `LineSegmenter`: Automatic segmentation using horizontal projection with multiple strategies
    - Combines Otsu + adaptive thresholding
    - Morphological operations to connect broken characters
    - Configurable sensitivity (0.01-0.1) for tight line spacing
    - Smart gap detection and line merging
  - `PageXMLSegmenter`: Uses existing Transkribus PAGE XML annotations (more accurate)
- **OCR Engine**:
  - `TrOCRInference`: Wraps VisionEncoderDecoderModel + TrOCRProcessor
    - Supports both local checkpoints and HuggingFace Hub models
    - Optional background normalization (MUST match training preprocessing)
    - Beam search for quality (num_beams=1 for speed, 4 for quality)
- **Preprocessing**: `normalize_background()` function applies CLAHE normalization (must match training)

**4. GUI Applications**
- **`transcription_gui_qt.py`**: Main PyQt6-based GUI
  - Features: zoom/pan, drag & drop, model comparison, export to TXT/CSV
  - Integrates with TrOCR inference pipeline
  - Supports both local and HuggingFace models
- **`transcription_gui_plugin.py`**: Plugin-based GUI with HTREngine architecture
  - Modular engine system (TrOCR, Qwen3-VL, PyLaia, Kraken)
  - Supports line-level and page-level recognition
  - PAGE XML export capability
- **`transcription_gui_party.py`**: Proof-of-concept for Party OCR integration
  - Left-image, right-transcription split view
  - Kraken line segmentation with green box visualization
  - Automatic PAGE XML generation for Party processing
  - WSL subprocess integration with Party OCR

**5. PAGE XML Integration (`page_xml_exporter.py`)**
- Exports line segmentation data to PAGE XML 2013-07-15 format
- Compatible with Transkribus, Party, and other PAGE XML processors
- Handles optional attributes safely (`confidence`, `coords`, `text`)
- Supports both bounding box and polygon coordinates
- Used by GUIs for exporting segmentation results

### Configuration System

Training is controlled via YAML config files (see `example_config.yaml`):

```yaml
# Model selection
model_name: "kazars24/trocr-base-handwritten-ru"

# Data paths (output from transkribus_parser.py)
data_root: "./processed_data"
train_csv: "train.csv"
val_csv: "val.csv"

# Performance optimization
cache_images: true        # Cache images in RAM (2-4GB per 10k images)
batch_size: 16            # Per-device batch size
gradient_accumulation_steps: 4  # Effective batch = batch_size * accumulation * num_gpus
dataloader_num_workers: 4

# Training parameters
epochs: 10
learning_rate: 3e-5
fp16: true                # Mixed precision (faster on RTX GPUs)

# Augmentation (enabled by default)
use_augmentation: true
aug_rotation_degrees: 2
aug_brightness: 0.3
aug_contrast: 0.3
```

### Data Flow

1. **Transkribus Export** → PAGE XML + Images
2. **`transkribus_parser.py`** → Cropped line images + CSV metadata
3. **`optimized_training.py`** → Fine-tuned TrOCR model
4. **`inference_page.py`** or **`transcription_gui_qt.py`** → Transcriptions

### Critical Implementation Details

**1. ASPECT RATIO PRESERVATION (Most Critical)**
- **When to use**: ALWAYS for high aspect ratio line images (>3:1), especially Ukrainian manuscripts
- **Where it happens**: During data preparation in `transkribus_parser.py`
  - Images are resized to target height (128px default) while preserving aspect ratio
  - Saved to disk in `line_images/` directory
  - Training script loads already-resized images
- **Why it matters**: Without this, TrOCR's ViT encoder brutal-resizes to 384×384
  - Ukrainian lines: 4077×357 → direct resize to 384×384 = 10.6x width compression
  - Characters: ~80px → ~7px width (unreadable)
  - With aspect ratio preservation: characters remain ~28px width (4x better)
- **Impact on CER**: Aspect ratio preservation can reduce CER by 10-20 percentage points
- **How to check**: Look at `dataset_info.json` → `"preserve_aspect_ratio": true`
- **Inference**: No special handling needed - images are already properly sized

**2. Background Normalization (Secondary)**
- If training data uses `--normalize-background`, inference MUST use `--normalize-background`
- Useful for aged paper with color variations or uneven lighting
- Less critical than aspect ratio for most datasets

**3. Preprocessing Consistency**
- Dataset metadata (`dataset_info.json`) records all preprocessing flags
- Always check this file before inference to match training preprocessing

**4. Multi-GPU Training**
- HuggingFace Trainer automatically detects all GPUs
- Effective batch size = `batch_size * gradient_accumulation_steps * num_gpus`
- For 2x RTX 4090: typical config uses batch_size=32, accumulation=2 → effective batch=128
- On Windows, DDP uses `gloo` backend (set via `TORCH_DISTRIBUTED_BACKEND` env var)

**5. Memory Management**
- Image caching trades RAM for speed (2-4GB per 10k images)
- If OOM: reduce `batch_size`, disable `cache_images`, or enable `gradient_checkpointing`
- `MemoryMonitorCallback` clears GPU cache every 500 steps

**6. Dataset Validation**
- `OptimizedOCRDataset` filters missing image files at initialization
- Returns fallback samples (blank image + ".") for corrupt data to prevent training crashes
- Validates text is not empty/NaN and images meet minimum size requirements

**7. Model Compatibility**
- Base model: `kazars24/trocr-base-handwritten-ru` (Russian-pretrained)
- Processor and tokenizer loaded from base model
- Checkpoints can be loaded from local path or HuggingFace Hub
- Generation config set on model: `decoder_start_token_id`, `eos_token_id`, `max_length`, etc.

## Project Structure

```
.
├── transkribus_parser.py       # Transkribus PAGE XML → training dataset
├── optimized_training.py       # Main training script (fast, cached)
├── inference_page.py           # Whole-page inference with segmentation
├── transcription_gui_qt.py     # PyQt6 GUI application
├── example_config.yaml         # Template training configuration
├── config_*.yaml               # Experiment-specific configs
│
├── models/                     # Trained models and checkpoints
│   ├── efendiev_3_model/       # Example: Efendiev dataset model
│   ├── ukrainian_model/        # Example: Ukrainian dataset model
│   └── */training_config.yaml  # Auto-saved config for each run
│
├── data/                       # Processed datasets (gitignored)
│
├── party/
│   └── divideInTestTrainLst.py # Legacy dataset splitting utility
│
├── requirements.txt            # Python dependencies
├── README.md                   # High-level overview
├── USAGE_GUIDE.md             # Detailed usage instructions
└── MULTI_GPU_TRAINING.md      # Multi-GPU setup guide
```

## Hardware Configuration

**Primary Setup**: 2x NVIDIA RTX 4090 (24GB VRAM each, 48GB total)

**Performance**: ~1 hour for 15 epochs on 2K lines (vs ~11 hours on old single-GPU pipeline)

## Model Registry

**Best Model**: `kazars24/trocr-base-handwritten-ru` (Russian-pretrained, CER 0.253 on combined Cyrillic dataset)

**Training Data**: Russian (1.36M lines) + Ukrainian (6.47M lines) + Church Slavonic

## Windows-Specific Notes

- Use `.venv\Scripts\activate` (not `source venv/bin/activate`)
- DDP backend: `gloo` (NCCL not available on Windows)
- Batch scripts: `run_training.bat`, `run_training_ddp.bat` (for convenience)
- PowerShell scripts: `run_training_ddp.ps1` (may require execution policy change)
- Unicode console encoding: some print statements wrapped in try/except to handle errors

## Development Workflow

1. **Export data from Transkribus** (PAGE XML format)
2. **Parse with preprocessing**: `transkribus_parser.py`
   - **ALWAYS use `--preserve-aspect-ratio`** (critical for high aspect ratio lines)
   - Use `--normalize-background` if dealing with aged/colored paper
   - Check output in `line_images/` directory to verify aspect ratios look correct
3. **Verify preprocessing**: Check `dataset_info.json` for `preserve_aspect_ratio: true`
4. **Create config**: Copy `example_config.yaml`, adjust `data_root` and hyperparameters
5. **Train**: `python optimized_training.py --config config.yaml` (multi-GPU automatic)
6. **Monitor**: TensorBoard at `http://localhost:6006`
7. **Evaluate**: Test on validation set, check CER metric
8. **Inference**: Use `inference_page.py` or GUI (match preprocessing from training if needed)
9. **Iterate**: Add more data, adjust hyperparameters, repeat

## Troubleshooting

**Out of Memory**:
- Reduce `batch_size` (e.g., 32 → 16 → 8)
- Disable `cache_images: false`
- Enable `gradient_checkpointing: true`

**Slow Training**:
- Enable `cache_images: true` (requires RAM)
- Increase `dataloader_num_workers` (but not too high, 4-8 is typical)
- Use multi-GPU setup

**Poor OCR Quality**:
- **FIRST CHECK**: Was training data prepared with `--preserve-aspect-ratio`?
  - Look at `dataset_info.json` → should have `"preserve_aspect_ratio": true`
  - Inspect `line_images/` → images should have preserved aspect ratios (not square)
  - If missing, re-run `transkribus_parser.py` with `--preserve-aspect-ratio` and retrain
- Ensure inference preprocessing matches training (background normalization)
- Increase `num_beams` at inference (4 for quality, 1 for speed)
- Try more epochs or different learning rate
- Check dataset quality and transcription accuracy

**Segmentation Issues**:
- Use PAGE XML segmentation instead of automatic (more accurate)
- Adjust `min_line_height` or `sensitivity` for `LineSegmenter`
- Manually correct segmentation in Transkribus before export

## Key Files for Reference

- **`USAGE_GUIDE.md`**: Comprehensive step-by-step usage guide
- **`MULTI_GPU_TRAINING.md`**: Multi-GPU setup and performance details
- **`example_config.yaml`**: Annotated configuration template
- **`README.md`**: Project overview and quick start
