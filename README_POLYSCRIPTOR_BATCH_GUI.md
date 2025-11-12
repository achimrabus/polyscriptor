# Polyscriptor Batch GUI

Minimal Qt6 GUI launcher for batch HTR processing with live output monitoring.

## Overview

Polyscriptor Batch GUI is a **CLI wrapper** - it builds and executes `batch_processing.py` commands with a user-friendly interface. This ensures CLI/GUI parity (they run identical code).

## Features

- ✅ **Engine Selection**: PyLaia, TrOCR, Qwen3-VL, Party, Kraken, Churro
- ✅ **Model Configuration**: Local file paths or HuggingFace model IDs
- ✅ **Segmentation Options**: Kraken (neural), HPP (fast), None (pre-segmented)
- ✅ **PAGE XML Auto-Detection**: Automatically uses PAGE XML if available
- ✅ **Output Formats**: TXT, CSV, PAGE XML, JSON (multi-select)
- ✅ **Preset System**: Save/load common configurations
- ✅ **Live Execution**: Real-time stdout/stderr streaming in dialog
- ✅ **Command Preview**: See exact command before execution
- ✅ **Validation**: Input validation before running

## Usage

### Launch GUI

```bash
source htr_gui/bin/activate
python polyscriptor_batch_gui.py
```

### Quick Start

1. **Select Input Folder**: Click "Browse..." to choose folder with manuscript images
2. **Choose Engine**: Select from dropdown (e.g., PyLaia for Church Slavonic)
3. **Load Preset** (Optional): Select built-in preset like "Church Slavonic (PyLaia + Kraken)"
4. **Configure Model**: Browse to local `.pt` file or enter HuggingFace model ID
5. **Set Segmentation**: Choose method (Kraken recommended) and enable PAGE XML if available
6. **Select Output Formats**: Check TXT, CSV, or PAGE XML
7. **Test First**: Click "Dry Run (Test First)" to validate configuration
8. **Start Processing**: Click "Start Batch Processing" and watch live output

## Built-in Presets

- **Church Slavonic (PyLaia + Kraken)**: `models/pylaia_church_slavonic_20251103_162857/best_model.pt`
- **Ukrainian (PyLaia + Kraken)**: `models/pylaia_ukrainian_pagexml_20251101_182736/best_model.pt`
- **Glagolitic (PyLaia + Kraken)**: `models/pylaia_glagolitic_with_spaces_20251102_182103/best_model.pt`
- **Russian (TrOCR HF)**: `kazars24/trocr-base-handwritten-ru`

## Saving Custom Presets

1. Configure your settings
2. Click "Save" button
3. Enter preset name
4. Preset saved to `~/.config/polyscriptor/presets.json`
5. Reload GUI to see new preset in dropdown

## Engine-Specific Notes

### PyLaia
- Requires local model path (`.pt` file)
- Best for trained historical manuscripts
- Supports both local models and vocabulary files

### TrOCR
- Supports both local checkpoints and HuggingFace models
- Num Beams control appears (1=fast, 4=quality)
- Default segmentation: Kraken

### Qwen3-VL
- Requires HuggingFace model ID
- **WARNING**: Very slow (~1-2 min/page)
- Only use for small batches or complex layouts
- Default segmentation: None (whole page)

### Party
- Requires local `.safetensors` model file
- Works best with PAGE XML input
- Batch-optimized for speed
- Default segmentation: None (uses PAGE XML)

## Live Execution

When you click "Dry Run" or "Start Batch Processing", the GUI:

1. Validates your configuration
2. Opens execution dialog showing:
   - Full command being executed
   - Live stdout/stderr output (auto-scrolling)
   - Color-coded error messages (red)
   - Success/failure indicator on completion
3. Allows you to copy command to clipboard
4. Keeps process alive until you close dialog

## Command Preview

The bottom panel shows the exact command that will be executed:

```bash
python batch_processing.py --input-folder HTR_Images/my_folder \
  --engine PyLaia --model-path models/pylaia_church_slavonic.../best_model.pt \
  --device cuda:0 --segmentation-method kraken --use-pagexml \
  --output-format txt --output-format csv
```

This updates in real-time as you change settings.

## Troubleshooting

**"Model file does not exist"**
- Check that model path is absolute or relative to project root
- Use "Browse..." button to ensure correct path

**"Input folder does not exist"**
- Verify folder path is correct
- Use "Browse..." button to select folder

**"Please select at least one output format"**
- Check at least one format: TXT, CSV, or PAGE XML

**GUI doesn't open**
- Ensure PyQt6 is installed: `pip install PyQt6`
- Check virtual environment is activated

**Process output not showing**
- Check that `batch_processing.py` is in same directory
- Verify Python path is correct in execution

## Architecture

**Design Philosophy**: "CLI wrapper, not reimplementation"

The GUI:
- Builds command-line arguments from form inputs
- Executes `batch_processing.py` via QProcess
- Streams output to dialog in real-time
- Does NOT duplicate batch processing logic

This ensures:
- CLI and GUI always behave identically
- Easy maintenance (only one implementation)
- Full access to all CLI features
- No code duplication

## File Locations

- **GUI Script**: `polyscriptor_batch_gui.py`
- **Presets**: `~/.config/polyscriptor/presets.json`
- **Batch Processor**: `batch_processing.py` (backend)

## See Also

- [batch_processing.py](batch_processing.py) - CLI batch processor (backend)
- [POLYSCRIPTOR_BATCH_GUI_PLAN.md](POLYSCRIPTOR_BATCH_GUI_PLAN.md) - Design documentation
- [CLAUDE.md](CLAUDE.md) - Project overview and recent improvements
