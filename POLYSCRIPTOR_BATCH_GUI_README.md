# Polyscriptor Batch GUI - Usage Guide

A minimal Qt6 GUI launcher for batch HTR processing. Builds and displays `batch_processing.py` commands.

## Quick Start

```bash
# Launch the GUI
python polyscriptor_batch_gui.py
```

## Features

### 1. Input/Output Configuration
- **Input Folder**: Folder containing manuscript images (JPEG, PNG, TIFF)
- **Output Folder**: Where transcriptions will be saved
- Browse buttons for easy folder selection

### 2. Engine Selection
Choose from 6 HTR engines:
- **PyLaia**: Best for Church Slavonic, Ukrainian, Glagolitic (CTC-based)
- **TrOCR**: HuggingFace transformer models (good for Russian)
- **Qwen3-VL**: Vision-language model (slow but accurate, for complex layouts)
- **Party**: Page-level transformer HTR (requires PAGE XML)
- **Kraken**: Mature OCR engine with pre-trained models
- **Churro**: VLM-based HTR

**Model Configuration**:
- **Model Path**: Local model file (`.pt`, `.pth`, `.safetensors`, `.mlmodel`)
- **Model ID (HF)**: HuggingFace Hub ID (e.g., `kazars24/trocr-base-handwritten-ru`)
- GUI shows relevant fields based on engine selection

### 3. Segmentation Options
- **Method**:
  - `kraken`: Neural segmentation (best accuracy)
  - `hpp`: Horizontal projection profile (fast)
  - `none`: For pre-segmented line images
- **Use PAGE XML**: Auto-detect PAGE XML files from `page/` subfolder
- **Sensitivity**: Segmentation threshold (0.01-1.0)

### 4. Output Options
- **Formats**: TXT, CSV, PAGE XML (check multiple)
- **Resume**: Skip already processed images (useful for interrupted batches)
- **Verbose**: Detailed logging output

### 5. Presets System
**Built-in presets**:
- Church Slavonic (PyLaia + Kraken)
- Ukrainian (PyLaia + Kraken)
- Glagolitic (PyLaia + Kraken)
- Russian (TrOCR HF)

**Custom presets**:
- Configure settings → Click "Save" → Enter name
- Saved to: `~/.config/polyscriptor/presets.json`
- Click "Load" to restore saved presets

### 6. Command Preview
- Live preview of generated `batch_processing.py` command
- Updates automatically as you change settings
- Copy-paste friendly

### 7. Execution
- **Dry Run (Test First)**: Test with first image, see time estimate
- **Start Batch Processing**: Shows command to execute
- **Copy to Clipboard**: Easy copy-paste to terminal

## Workflow Example

1. **Select folders**:
   - Input: `HTR_Images/my_manuscripts`
   - Output: `output/my_batch`

2. **Choose preset**: "Church Slavonic (PyLaia + Kraken)"
   - Or configure manually

3. **Adjust options**:
   - ☑ Use PAGE XML
   - ☑ Resume
   - Segmentation: Kraken

4. **Dry Run**: Click "Dry Run (Test First)"
   - Shows estimated time
   - Tests first image
   - Verifies configuration

5. **Execute**: Click "Start Batch Processing"
   - Copy command to clipboard
   - Paste in terminal
   - Monitor progress

## Design Philosophy

**"CLI wrapper, not reimplementation"**

This GUI builds commands for `batch_processing.py` rather than duplicating its logic. This ensures:
- **CLI/GUI parity**: Both run identical code
- **Easy maintenance**: No duplicate implementation
- **Reliability**: Leverages battle-tested CLI
- **Flexibility**: Advanced users can modify commands

## Tips

1. **Start with Dry Run**: Always test first to catch configuration errors
2. **Use Presets**: Save common configurations for quick reuse
3. **Check Command Preview**: Verify command looks correct before running
4. **Resume Mode**: Enable for large batches that might be interrupted
5. **PAGE XML**: Auto-detection saves ~40% time if XML files available

## Troubleshooting

**"Model file does not exist"**:
- Use Browse button to select valid model file
- Check path in command preview

**"Input folder does not exist"**:
- Verify folder path is correct
- Use Browse button to select folder

**Engine-specific warnings**:
- Qwen3-VL: Very slow (~1-2 min/page), use only for small batches
- Party: Requires PAGE XML files

## Model Paths (Examples)

```
PyLaia Church Slavonic:
models/pylaia_church_slavonic_20251103_162857/best_model.pt

PyLaia Ukrainian:
models/pylaia_ukrainian_pagexml_20251101_182736/best_model.pt

PyLaia Glagolitic:
models/pylaia_glagolitic_with_spaces_20251102_182103/best_model.pt

TrOCR Russian (HF):
kazars24/trocr-base-handwritten-ru
```

## Advanced: Command Line Integration

The GUI generates standard `batch_processing.py` commands. You can:
1. Copy command from GUI
2. Modify in terminal (add flags, change values)
3. Save as shell script for automation

Example generated command:
```bash
python batch_processing.py \
  --input-folder HTR_Images/my_folder \
  --engine PyLaia \
  --model-path models/pylaia_church_slavonic_20251103_162857/best_model.pt \
  --segmentation-method kraken \
  --use-pagexml \
  --device cuda:0 \
  --output-format txt \
  --resume \
  --verbose
```

## Dependencies

- PyQt6 (already installed in `htr_gui` venv)
- All batch_processing.py dependencies

No additional installation required!
