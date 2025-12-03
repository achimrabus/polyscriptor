# TrOCR Inference Guide

This guide explains how to use the inference scripts for whole-page transcription of Ukrainian handwritten documents.

## Scripts Overview

### 1. `inference_page.py` - Command-Line Interface
A command-line tool for batch processing and scripting.

### 2. `inference_page_gui.py` - Graphical Interface
A user-friendly GUI for interactive transcription.

## Installation

Make sure you have all dependencies installed:

```bash
pip install torch transformers pillow scipy numpy tqdm
```

## Usage

### Command-Line Interface

**Basic usage:**
```bash
python inference_page.py \
    --image path/to/page.jpg \
    --checkpoint models/ukrainian_model/checkpoint-3000
```

**With all options:**
```bash
python inference_page.py \
    --image page.jpg \
    --checkpoint models/ukrainian_model/checkpoint-3000 \
    --num_beams 4 \
    --max_length 128 \
    --output transcription.txt \
    --debug
```

**Using existing Transkribus segmentation:**
```bash
python inference_page.py \
    --image page.jpg \
    --xml page/page.xml \
    --checkpoint models/ukrainian_model/checkpoint-3000
```

**Note:** Transkribus exports PAGE XML files to a `page/` subdirectory by default. If processing a folder exported from Transkribus, your XML files will be in `<export_folder>/page/*.xml` while images are in `<export_folder>/*.jpg`.

**Command-line arguments:**
- `--image`: Path to input page image (required)
- `--checkpoint`: Path to TrOCR checkpoint directory (required)
- `--xml`: Optional PAGE XML file for line segmentation
- `--output`: Output text file (default: `<image_name>_transcription.txt`)
- `--num_beams`: Beam search parameter (default: 4, higher=better quality but slower)
- `--max_length`: Maximum sequence length (default: 128)
- `--min_line_height`: Minimum line height for auto-segmentation (default: 20)
- `--debug`: Visualize line segmentation
- `--device`: Force CPU or CUDA (default: auto-detect)

### Graphical Interface

**Launch the GUI:**
```bash
python inference_page_gui.py
```

**Steps:**
1. Click "Browse..." next to Image to select your page image
2. (Optional) Select a PAGE XML file if you have existing line segmentation
3. Click "Browse..." next to Model Checkpoint to select your trained model
4. Adjust settings if needed:
   - Beam Search: Higher values (4-10) give better quality but are slower
   - Max Length: Maximum characters per line (128 is usually fine)
5. Click "Process Page" to start transcription
6. View results in the right panel
7. Click "Save Transcription" to save to a text file

## Features

### Automatic Line Segmentation
If you don't provide a PAGE XML file, the script automatically segments the page into lines using horizontal projection profile analysis.

**How it works:**
1. Converts image to grayscale
2. Applies adaptive thresholding
3. Computes horizontal projection (sum of pixels per row)
4. Identifies gaps between lines
5. Extracts individual line images

**Limitations:**
- Works best for clean, well-spaced lines
- May struggle with:
  - Overlapping lines
  - Skewed pages
  - Very tight line spacing
  - Complex layouts

**Recommendation:** For best results, use Transkribus PAGE XML with manual line corrections.

### Using PAGE XML Segmentation
If you have Transkribus PAGE XML annotations, you can use them for more accurate segmentation:

```bash
python inference_page.py \
    --image page.jpg \
    --xml page/page.xml \
    --checkpoint checkpoint-3000
```

This uses the exact line coordinates from your Transkribus export.

**Transkribus Export Structure:**
```
transkribus_export/
├── image1.jpg
├── image2.jpg
└── page/
    ├── image1.xml
    └── image2.xml
```

Transkribus always exports PAGE XML files to a `page/` subdirectory. The batch processor automatically looks for XML files in this location.

## Choosing a Checkpoint

Use the best checkpoint from your training run:

1. Check TensorBoard or training logs for the checkpoint with lowest CER
2. Common locations:
   - `models/ukrainian_model/checkpoint-2500` (best from previous run: 25.96% CER)
   - `models/ukrainian_model/checkpoint-3000` (25.52% CER)

Example:
```bash
# Find available checkpoints
ls models/ukrainian_model/

# Use the best one
python inference_page.py \
    --image page.jpg \
    --checkpoint models/ukrainian_model/checkpoint-3000
```

## Performance Tips

### Speed vs Quality Trade-offs

**Fast inference (for testing):**
```bash
python inference_page.py \
    --image page.jpg \
    --checkpoint checkpoint-3000 \
    --num_beams 1 \
    --device cuda
```

**High quality (for final transcription):**
```bash
python inference_page.py \
    --image page.jpg \
    --checkpoint checkpoint-3000 \
    --num_beams 4 \
    --device cuda
```

**Batch processing with Transkribus XML:**
```bash
# Process Transkribus export with PAGE XML
for img in transkribus_export/*.jpg; do
    base=$(basename "$img" .jpg)
    python inference_page.py \
        --image "$img" \
        --xml "transkribus_export/page/${base}.xml" \
        --checkpoint models/ukrainian_model/checkpoint-3000 \
        --num_beams 4
done
```

**Or use the dedicated batch processor (recommended):**
```bash
python batch_processing.py \
    --input transkribus_export \
    --output results \
    --checkpoint models/ukrainian_model/checkpoint-3000 \
    --use_pagexml
```
The batch processor automatically finds PAGE XML files in the `page/` subdirectory.

### GPU Acceleration
The script automatically uses CUDA if available. To force CPU:
```bash
python inference_page.py --image page.jpg --checkpoint checkpoint-3000 --device cpu
```

## Example Workflow

**1. Prepare your data (Transkribus export):**
```
project/
├── transkribus_export/
│   ├── page001.jpg
│   ├── page002.jpg
│   ├── page003.jpg
│   └── page/              ← XML files in subdirectory
│       ├── page001.xml
│       ├── page002.xml
│       └── page003.xml
└── models/
    └── ukrainian_model/
        └── checkpoint-3000/
```

**2. Process with batch processor (recommended):**
```bash
python batch_processing.py \
    --input transkribus_export \
    --output results \
    --checkpoint models/ukrainian_model/checkpoint-3000 \
    --use_pagexml
```

The batch processor will:
- Automatically find PAGE XML files in `transkribus_export/page/`
- Use XML coordinates for accurate line extraction
- Fall back to auto-segmentation if XML is invalid/missing
- Handle EXIF rotation automatically
- Generate transcriptions and optionally PAGE XML output

**3. Check results:**
```bash
ls results/transcriptions/
cat results/transcriptions/page001.txt
```

## Troubleshooting

### "PAGE XML dimensions mismatch" warnings
If you see dimension mismatch errors like `XML: 2848x4272, actual: 4272x2848`, this is caused by EXIF rotation metadata in your images. The batch processor automatically handles this using `ImageOps.exif_transpose()`. If you encounter this:
- **Update to latest version** (EXIF fix added Nov 2025)
- The processor will automatically rotate images to match PAGE XML
- About 18% of images may have EXIF rotation tags

### "No lines detected"
- Try adjusting `--min_line_height` (default: 20)
- Use `--debug` to visualize segmentation
- Consider using PAGE XML instead of automatic segmentation

### "Out of memory" errors
- Reduce `--num_beams` (try 1 or 2)
- Use `--device cpu` to process on CPU instead
- Process smaller images or crop pages

### Poor transcription quality
- Make sure you're using the best checkpoint (check CER in training logs)
- Try increasing `--num_beams` to 6 or 8
- Verify image quality (should be at least 300 DPI)
- Check if line segmentation is accurate with `--debug`

### Unicode errors on Windows
The scripts handle Ukrainian text correctly. If you see encoding errors:
- Make sure your terminal supports UTF-8
- On Windows PowerShell: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`

## Next Steps

1. **Improve segmentation:** Create PAGE XML annotations in Transkribus for better line detection
2. **Post-processing:** Add spell-checking or language model correction
3. **Batch processing:** Write a script to process entire document collections
4. **Evaluation:** Compare with ground truth using CER/WER metrics

## Future Enhancements (TODO)

- [ ] Confidence scores per line
- [ ] Export to various formats (JSON, XML, CSV)
- [ ] Line-level visualization with bounding boxes
- [ ] Integration with post-processing tools
- [ ] Support for different output formats (PAGE XML, ALTO)
- [ ] Multi-column layout handling
- [ ] Rotation correction for skewed pages
