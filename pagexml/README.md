# PAGE XML Batch Segmentation (Standalone)

This module batch-creates Transkribus/PAGE XML files from a folder of page images without touching the existing GUI transcription plugin.

- Input: a directory of page images (PNG/JPG/TIF)
- Output: PAGE XML files in `pagexml/xml/` (default), and visual overlays in `pagexml/overlays/` (optional)
- Segmentation backend: Kraken classical `pageseg` for line detection + heuristic region clustering (1–4 columns typical)
- Reading order: Regions left→right; lines within region top→bottom

## Segmentation Modes

This tool supports three segmentation modes:

1. **Classical** (default for CPU): Uses Kraken's `pageseg.segment()` with convex hull region polygons and heuristic column clustering (1–4 columns)
2. **Neural**: Uses Kraken's `blla.mlmodel` for advanced layout analysis with tighter polygons and refined baselines (GPU recommended)
3. **Auto**: Tries neural first, automatically falls back to classical if segmentation quality is insufficient

## Why a separate module?
To keep your current GUI pipeline stable, this is a separate tool with its own CLI and no changes to `transcription_gui_plugin.py`.

## CLI Usage

```bash
# Activate your venv first
source htr_gui/bin/activate

# Classical mode (convex hull polygons)
python -m pagexml.pagexml_batch_segmenter \
  --input ./pages \
  --output ./pagexml/xml \
  --overlays ./pagexml/overlays \
  --mode classical \
  --device cpu \
  --max-columns 4

# Neural mode with QC metrics export
python -m pagexml.pagexml_batch_segmenter \
  --input ./pages \
  --output ./pagexml/xml \
  --overlays ./pagexml/overlays \
  --mode neural \
  --neural-model blla.mlmodel \
  --device cuda \
  --qc-csv ./qc_metrics.csv

# Auto mode (neural with fallback)
python -m pagexml.pagexml_batch_segmenter \
  --input ./pages \
  --output ./pagexml/xml \
  --mode auto \
  --neural-model blla.mlmodel \
  --device cuda \
  --max-columns 4 \
  --qc-csv ./qc_metrics.csv
```

Options:
- `--input`: Path to folder with images
- `--output`: Output folder for PAGE XML (default: `./pagexml/xml`)
- `--overlays`: Folder for rendered overlays (default: `./pagexml/overlays`); omit to disable overlays
- `--mode`: Segmentation mode: `classical`, `neural`, or `auto` (default: `classical`)
- `--neural-model`: Path to Kraken neural model file (default: `blla.mlmodel`; required for neural/auto modes)
- `--qc-csv`: Path to export quality control metrics CSV (optional)
- `--device`: `cpu` or `cuda` (CUDA recommended for neural mode)
- `--max-columns`: Upper bound for column clustering in classical mode (default: 4)
- `--min-line-height`: Filter out lines smaller than this (px; default 8)
- `--deskew`: If present, attempt a light deskew before segmentation

## Output
- **PAGE XML**: Conforms to PAGE schema 2019-07-15 structure with `<TextRegion>` polygonal `<Coords>` and `<TextLine>` with `<Baseline>`
- **ReadingOrder**: Included for multi-column pages (left→right, top→bottom within regions)
- **Overlays**: PNG images with colored region polygons, line boxes, and baselines for quick QC
- **QC Metrics CSV** (optional): Per-page quality metrics including:
  - `filename`: Image base name
  - `mode`: Segmentation mode used (classical/neural)
  - `regions`: Number of text regions detected
  - `lines`: Number of text lines detected  
  - `mean_line_height`: Average line height (px)
  - `height_variance`: Line height variance (consistency indicator)
  - `baseline_ratio`: Avg baseline length / bbox width (quality indicator; <0.5 suggests truncation)
  - `time_sec`: Processing time per page
  - `fallback`: Whether auto mode fell back to classical (yes/no)

## Notes / Limitations
- **Classical mode**: Heuristic column clustering (1–4 columns); marginalia grouped into side regions when clearly separated
- **Neural mode**: Requires `blla.mlmodel` (shipped with Kraken or downloadable); trained primarily on Latin scripts but provides good baseline for Church Slavonic
- **Auto mode**: Fallback triggers if neural returns <3 lines; logs warning and uses classical segmentation
- For pages with heavy skew, enable `--deskew`
- Convex hull polygons (classical) may over-approximate concave ornamental regions; neural mode provides tighter geometry

## Upgrade Roadmap
✅ **Step A**: Convex hull region polygons (DONE)  
✅ **Step B**: Neural segmentation with blla.mlmodel (DONE)  
✅ **Step C**: QC metrics CSV export (DONE)  
⏳ **Step D**: Fine-tuning pipeline for Church Slavonic/Glagolitic (PLANNED)

See `SEGMENTATION_UPGRADE_PLAN.md` for detailed implementation roadmap.

## GUI Usage

You can launch a minimal GUI to run the batch segmenter without touching the existing plugin.

Option A (module):

```bash
source htr_gui/bin/activate
python -m pagexml.pagexml_gui
```

Option B (launcher script):

```bash
source htr_gui/bin/activate
python run_pagexml_gui.py
```

Features:
- Input/output/overlays folder pickers with auto-suggested paths
- **Segmentation mode selector**: Classical, Neural, or Auto
- **Neural model path**: Specify `blla.mlmodel` location (for neural/auto modes)
- **QC metrics export**: Optional CSV path for per-page quality tracking
- Device selection (CPU/CUDA)
- Max columns, min line height, deskew toggle
- Start/Stop buttons with graceful thread shutdown
- Progress bar and live log with mode and timing information
