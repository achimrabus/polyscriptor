# TrOCR Transcription GUI - User Guide

## Overview

Professional PyQt6-based GUI for transcribing handwritten text using TrOCR models. Supports both local fine-tuned models and HuggingFace Hub models.

## Features

- **Seamless Image Viewing**: Zoom, pan, and navigate through multiple images
- **Automatic Line Segmentation**: Configurable detection with visual overlay
- **Flexible Model Loading**: Local checkpoints or HuggingFace Hub models
- **Device Selection**: CPU or GPU inference
- **Batch Processing**: Load and process multiple images sequentially
- **Abort Capability**: Stop OCR processing mid-execution
- **Background Normalization**: Optional preprocessing for aged documents
- **Export Options**: Save transcriptions to TXT format

## Getting Started

### Launch the GUI

```bash
python transcription_gui_qt.py
```

### Basic Workflow

1. **Load Image**: Click "Open Image" or drag & drop an image file
2. **Detect Lines**: Click "Detect Lines" to segment the image into text lines
3. **Adjust Parameters** (if needed): Modify threshold, min height, or morph ops
4. **Select Model**: Choose a local model or enter HuggingFace model ID
5. **Process**: Click "Process All Lines" to transcribe
6. **Save**: Export the transcription to a text file

## Line Segmentation

### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Threshold** | 0.5-15% | 5% | Detection threshold. Higher = more selective (detects only clear lines). Lower = more sensitive (detects faint lines). |
| **Min Height** | 5-50px | 10px | Minimum line height in pixels. Reduce for tightly-spaced text. |
| **Morph. Ops** | On/Off | On | Morphological operations to connect broken/faded characters. |

### Troubleshooting Line Detection

**Problem: No lines detected**
- **Solution**: DECREASE Threshold (try 1-2%) to detect fainter lines
- Lower Min Height if text is very small
- Enable Morph. Ops checkbox

**Problem: Only 1 line detected (but page has multiple lines)**
- **Solution**: INCREASE Threshold (try 8-10%) to be more selective
- Reduce Min Height if lines are close together
- Enable Morph. Ops checkbox

**Problem: Too many fragments detected**
- **Solution**: INCREASE Threshold to merge fragments
- INCREASE Min Height to filter out noise
- Enable Morph. Ops to connect broken characters

## Model Selection

### Local Models

1. Navigate to **Local** tab
2. Select from dropdown (automatically scans `./models/` directory)
3. Or click "Browse..." to select a custom checkpoint directory

**Requirements**: Directory must contain `config.json`, `pytorch_model.bin`, and tokenizer files.

### HuggingFace Models

1. Navigate to **HuggingFace** tab
2. Enter model ID (e.g., `kazars24/trocr-base-handwritten-ru`, `dh-unibe/trocr-kurrent`)
3. Click **Validate** to check model availability
4. Model will be downloaded automatically on first use (cached locally)

**Recommended Models**:
- `microsoft/trocr-base-handwritten`: English handwriting (baseline)
- `kazars24/trocr-base-handwritten-ru`: Russian/Cyrillic handwriting
- `dh-unibe/trocr-kurrent`: German Kurrent script

## Device Selection: CPU vs GPU

### CPU Inference

**Advantages:**
- ✅ More stable for long batch processing
- ✅ No CUDA memory management issues
- ✅ Works on any machine
- ✅ Can handle arbitrarily large images

**Disadvantages:**
- ❌ 3-5x slower per line than GPU

### GPU Inference

**Advantages:**
- ✅ 3-5x faster per line than CPU
- ✅ Best for single-image workflows

**Disadvantages:**
- ❌ CUDA out-of-memory errors possible with large images
- ❌ Requires NVIDIA GPU with CUDA support
- ❌ May have initialization overhead

### Recommendation

**Default: CPU** - More reliable for typical use cases

**Use GPU when:**
- Processing single images interactively
- Working with small-to-medium sized images
- Speed is critical and you have a powerful GPU

**Use CPU when:**
- Processing batches of images
- Working with very large images (>4000px)
- Experiencing CUDA memory errors
- Running on a machine without CUDA

## Inference Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Beam Search** | 1-10 | 4 | Number of beams for beam search. Higher = better quality but slower. |
| **Max Length** | 64-256 | 128 | Maximum sequence length. Increase for very long lines. |

## Background Normalization

**When to enable:**
- Aged or yellowed paper
- Varying illumination across the page
- Colored backgrounds

**When to disable:**
- Modern documents with white backgrounds
- Already normalized/preprocessed images

**Note:** If your model was trained WITH background normalization, you MUST enable it during inference.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open image |
| `Ctrl+S` | Save transcription |
| `Ctrl+0` | Fit image to window |
| `Ctrl++` | Zoom in |
| `Ctrl+-` | Zoom out |
| `Left Arrow` / `Page Up` | Previous image |
| `Right Arrow` / `Page Down` | Next image |

## Aborting Processing

If you accidentally start processing with the wrong model or settings:

1. Click the red **Abort** button
2. Processing will stop after completing the current line
3. Partial results may be available in the text editor
4. Select the correct model/settings and try again

## Common Issues

### Issue: "No models found" in dropdown

**Solution:** Place model checkpoints in `./models/` directory with this structure:
```
models/
├── ukrainian_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── preprocessor_config.json
│   └── tokenizer.json
```

### Issue: HuggingFace model fails with "expected str but got NoneType"

**Solution:** This model doesn't include a processor config. The GUI now automatically falls back to using the base model's processor. If it still fails, the model may not be compatible with TrOCR.

### Issue: Line segmentation is slow

**Solution:** Line segmentation runs on CPU and is generally fast (<5 seconds for typical images). If it's slow:
- Image might be extremely large (>8000px) - consider resizing
- Check system CPU usage

### Issue: OCR processing is slow

**Solution:**
- Switch to **GPU** mode if available (3-5x faster)
- Reduce **Beam Search** from 4 to 1 or 2 (faster but lower quality)
- Process fewer lines at once

### Issue: CUDA out of memory

**Solution:**
- Switch to **CPU** mode
- Close other GPU-intensive applications
- Reduce image size before loading

## Tips & Tricks

1. **Batch Processing**: Load multiple images with "Load Images..." and use arrow keys to navigate

2. **Fine-tune Parameters**: After detecting lines, adjust parameters and re-detect without reloading the image

3. **Model Comparison**: Process the same image with different models (Local vs HuggingFace) to compare quality

4. **Font Customization**: Click "Select Font..." to use fonts that better display your script (e.g., Unicode fonts for Cyrillic)

5. **Partial Processing**: If you abort processing, the text editor will contain partial results up to the last completed line

## Performance Benchmarks

**Test Setup**: 20 lines, 4000×3000px image, Intel i7-10700K, RTX 3080

| Device | Time per Line | Total Time (20 lines) |
|--------|--------------|---------------------|
| **CPU** | ~2.5s | ~50s |
| **GPU** | ~0.7s | ~14s |

**Note:** GPU has ~5s initialization overhead, so CPU might be faster for <5 lines.

## Advanced: Command-Line Usage

For batch processing without GUI, use `inference_page.py`:

```bash
# Single image
python inference_page.py --image page.jpg --checkpoint models/ukrainian_model

# With HuggingFace model
python inference_page.py --image page.jpg --checkpoint kazars24/trocr-base-handwritten-ru --huggingface

# With background normalization
python inference_page.py --image page.jpg --checkpoint models/ukrainian_model --normalize-background
```

## Support

For issues, feature requests, or questions:
- Check the [GitHub Issues](https://github.com/yourusername/yourrepo/issues)
- Review the [Training Guide](TRAINING_GUIDE.md) for model fine-tuning
- See [GUI_IMPROVEMENTS.md](GUI_IMPROVEMENTS.md) for recent changes

## Version History

### v2.0 (Current)
- Added abort button for OCR processing
- Fixed HuggingFace model loading (processor fallback)
- Improved line segmentation with configurable parameters
- Enhanced error messages and user guidance
- Changed default device to CPU (more stable)
- Updated threshold parameter label for clarity

### v1.0
- Initial release with PyQt6 GUI
- Local and HuggingFace model support
- Automatic line segmentation
- CPU/GPU device selection
