# Multi-Engine HTR Training & Comparison Tool

A comprehensive toolkit for training and comparing different Handwritten Text Recognition (HTR) engines on historical manuscript datasets. Supports TrOCR, CRNN-CTC, Qwen3-VL, LightOnOCR, Party, and Kraken engines with a unified GUI interface.

**Primary Focus:** Cyrillic manuscripts (Russian, Ukrainian, Church Slavonic, Glagolitic)

---

## 🎯 Features

### Multiple HTR Engines
- **TrOCR**: Transformer-based OCR (line-level)
- **CRNN-CTC**: Puigcerver CRNN with CTC decoding (line-level, PyLaia-inspired)
- **Qwen3-VL**: Vision-Language Model (line/page-level, custom prompts)
- **LightOnOCR**: Lightweight VLM (~4GB VRAM, line-level, fine-tuned variants)
- **Churro**: Qwen fork, experimental (line/page-level, custom prompts)
- **Party**: Transformer-based HTR (line-level, multilingual)
- **Kraken**: Segmentation & recognition
- **PaddleOCR**: Multi-language printed/mixed text detection + recognition (whole-page, subprocess-isolated)

### Commercial & Local Vision Models
- **Commercial APIs**: Google Gemini, Anthropic Claude Vision (via API keys)
- **Local LLMs**: OpenWebUI integration for local vision models
- **Unified interface**: All models accessible through same engine plugin system

### Core Capabilities
- **Plugin GUI**: Compare engines side-by-side with unified interface
- **Model management**: Easy switching between trained models and API providers
- **Export formats**: TXT, CSV, PAGE XML

### Training Pipelines (GPU required)
- **CRNN-CTC**: Custom CRNN training with PAGE XML support
- **TrOCR**: Fine-tuning pipeline with image caching (10-50x faster)
- **Data preparation**: Transkribus PAGE XML parser

### Key Capabilities
- Line segmentation (automatic or PAGE XML-based)
- Custom prompt support (Qwen3-VL)
- Batch processing
- PAGE XML import/export

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/achimrabus/polyscriptor.git
cd polyscriptor

# Create virtual environment
python3 -m venv htr_env
source htr_env/bin/activate  # Linux/Mac
# or: htr_env\Scripts\activate  # Windows
```

**GPU install (CUDA 12.1 — Linux/Windows with NVIDIA GPU):**
```bash
# Install CUDA torch first, then the rest
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**CPU-only install (no GPU required):**
```bash
# Linux/Mac:
pip install -r requirements.txt

# Windows (avoids a torch DLL load error on CPU-only machines):
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Launch GUI for inference

**Local usage (Linux/Mac):**
```bash
source htr_env/bin/activate
python3 transcription_gui_plugin.py
```

**Local usage (Windows):**
```bat
htr_env\Scripts\activate
python transcription_gui_plugin.py
```

> **Note:** The plugin GUI requires `PyQt6` (included in `requirements.txt`). The web UI (`uvicorn web.polyscriptor_server:app`) works without PyQt6.

**Remote server usage (GUI over X11):**
```bash
# See REMOTE_GUI_GUIDE.md for detailed setup
# Quick test: X11 forwarding with MobaXterm
ssh -X user@server
cd ~/htr_gui/dhlab-slavistik
source htr_env/bin/activate
python3 transcription_gui_plugin.py
```

**Recommended for remote: CLI batch processing**
```bash
# More efficient than GUI for server workflows
python3 batch_processing.py \
    --input-folder HTR_Images/my_folder \
    --engine crnn-ctc \
    --model-path models/crnn_ctc_model/best_model.pt \
    --use-pagexml
```

📖 **See [REMOTE_GUI_GUIDE.md](REMOTE_GUI_GUIDE.md)** for comprehensive remote access options (X11, VNC, CLI workflows)

### 3. Train a Model (CLI, CRNN-CTC Example)

```bash
# Step 1: Parse Transkribus PAGE XML export → CSV format
python3 transkribus_parser.py \
    --input_dir /path/to/transkribus_export \
    --output_dir ./data/my_dataset \
    --preserve-aspect-ratio \
    --target-height 128

# Step 2: Convert CSV → CRNN-CTC format (required!)
python3 convert_to_pylaia.py \
    --input_csv ./data/my_dataset/train.csv \
    --output_dir ./data/crnn_train

python3 convert_to_pylaia.py \
    --input_csv ./data/my_dataset/val.csv \
    --output_dir ./data/crnn_val

# Step 3: Train CRNN-CTC model
python3 train_pylaia.py \
    --train_dir ./data/crnn_train \
    --val_dir ./data/crnn_val \
    --output_dir ./models/my_model \
    --batch_size 32 \
    --epochs 250
```

---

## 🤖 Pre-trained Models

Polyscriptor works with models from several sources. The table below lists tested, ready-to-use options for each engine.

### CRNN-CTC

Ready-to-use models for Cyrillic and Glagolitic manuscripts, trained with Polyscriptor, are published on HuggingFace:

**[https://huggingface.co/achimrabus](https://huggingface.co/achimrabus)**

| Model | Script / Language | CER | Description |
|-------|-------------------|-----|-------------|
| [crnn-ctc-church-slavonic](https://huggingface.co/achimrabus/crnn-ctc-church-slavonic) | Church Slavonic | 2.89% | QuantiSlav corpus (Elena Renje) |
| [crnn-ctc-prosta-mova](https://huggingface.co/achimrabus/crnn-ctc-prosta-mova) | Prosta Mova (Old Ruthenian) | 3.77% | Continslav corpus (Martin Meindl) |
| [crnn-ctc-ukrainian](https://huggingface.co/achimrabus/crnn-ctc-ukrainian) | Ukrainian (Cyrillic) | 4.76% | 19th–20th c. manuscripts (MultiHTR) |
| [crnn-ctc-glagolitic](https://huggingface.co/achimrabus/crnn-ctc-glagolitic) | Croatian Glagolitic | 5.33% | Outputs Latin transliteration |

All models run on CPU (no GPU required) and can be loaded directly in the CRNN-CTC engine.

### TrOCR

A collection of TrOCR models for Cyrillic handwriting (Russian, Ukrainian, Church Slavonic) is maintained here:

**[https://huggingface.co/cyrillic-trocr](https://huggingface.co/cyrillic-trocr)**

These can be loaded in the TrOCR engine by entering the HuggingFace model ID (e.g. `cyrillic-trocr/trocr-base-handwritten-ru`).

### Qwen3-VL

Vision-language models fine-tuned for historical Slavic manuscripts (Church Slavonic, Glagolitic):

**[https://huggingface.co/wjbmattingly](https://huggingface.co/wjbmattingly)**

Load any compatible Qwen3-VL model via its HuggingFace ID in the Qwen3-VL engine.

### Kraken

Segmentation and recognition models for historical manuscripts are available on Zenodo. The default `blla` layout analysis model for neural line segmentation:

**[https://zenodo.org/records/7755483](https://zenodo.org/records/7755483)**

### Party

Party OCR models with Church Slavonic in the pretraining data:

**[https://zenodo.org/records/15075344](https://zenodo.org/records/15075344)**

---

## 📁 Repository Structure

```
.
├── train_pylaia.py                  # CRNN-CTC training script
├── inference_pylaia_native.py       # CRNN-CTC inference (native Linux)
├── inference_page.py                # Line segmentation + OCR pipeline
├── transcription_gui_plugin.py      # Main GUI application
├── polyscriptor_batch_gui.py        # Batch processing GUI
├── batch_processing.py              # Batch processing CLI
├── htr_engine_base.py              # HTR engine interface
│
├── engines/                         # HTR engine plugins
│   ├── trocr_engine.py             # TrOCR transformer
│   ├── pylaia_engine.py            # CRNN-CTC (Puigcerver CRNN)
│   ├── qwen3_engine.py             # Qwen3-VL (local)
│   ├── lighton_ocr_engine.py       # LightOnOCR VLM (lightweight)
│   ├── churro_engine.py            # Churro (Qwen fork)
│   ├── party_engine.py             # Party multilingual HTR
│   ├── kraken_engine.py            # Kraken segmentation
│   ├── commercial_api_engine.py    # Google Gemini, OpenAI GPT & Anthropic Claude APIs
│   ├── openwebui_engine.py         # OpenWebUI local LLMs
│   ├── paddle_engine.py            # PaddleOCR (subprocess, isolated venv)
│   └── paddle_worker.py            # PaddleOCR worker (runs inside venv_paddle)
│
├── optimized_training.py            # TrOCR fine-tuning script
├── transkribus_parser.py            # PAGE XML data preparation
├── alto_parser.py                   # ALTO XML data preparation
├── page_xml_exporter.py             # Export results to PAGE XML
├── qwen3_prompts.py                 # Custom prompts for Qwen3-VL
│
├── requirements.txt                 # Python dependencies
│
├── web/                             # Browser-based web interface
│   ├── polyscriptor_server.py      # FastAPI backend (SSE streaming)
│   ├── static/
│   │   ├── index.html              # Single-page app
│   │   ├── app.js                  # State management, event bus
│   │   ├── app.css                 # Styles
│   │   └── components/             # ES6 modules (engine, viewer, transcription, batch)
│   └── tests/
│       └── test_server.py          # API tests (pytest + FastAPI TestClient)
│
└── models/                          # Trained models (excluded from git)
    ├── pylaia_*/                    # CRNN-CTC model checkpoints
    └── trocr_*/                     # TrOCR fine-tuned models
```

---

## 🎓 Typical Workflow

### Training a CRNN-CTC Model

1. **Export data from Transkribus** (PAGE XML format)
2. **Parse with preprocessing**:
   ```bash
   python3 transkribus_parser.py \
       --input_dir ./transkribus_export \
       --output_dir ./data/my_dataset \
       --preserve-aspect-ratio \
       --target-height 128 \
       # add --flip-rtl for Ottoman/Arabic/Hebrew manuscripts
   ```
3. **Convert to CRNN-CTC format**:
   ```bash
   python3 convert_to_pylaia.py \
       --input_csv ./data/my_dataset/train.csv \
       --output_dir ./data/crnn_train
   python3 convert_to_pylaia.py \
       --input_csv ./data/my_dataset/val.csv \
       --output_dir ./data/crnn_val
   ```
4. **Train model**:
   ```bash
   python3 train_pylaia.py \
       --train_dir ./data/crnn_train \
       --val_dir ./data/crnn_val \
       --output_dir ./models/my_model \
       --batch_size 32 \
       --epochs 250
   ```
5. **Use in GUI**: Model will appear in the CRNN-CTC engine dropdown

### Using Trained Models

Trained models can be loaded in the GUI:
- CRNN-CTC models: Select from dropdown or browse to model directory
- TrOCR models: Specify HuggingFace Hub ID or local checkpoint path
- Commercial APIs: Enter API keys in engine configuration

---

## 🛠️ Command-Line Inference

### CRNN-CTC (Single Line)

```bash
python3 inference_pylaia_native.py \
    --checkpoint models/my_model/best_model.pt \
    --syms models/my_model/symbols.txt \
    --image line_image.png
```

### CRNN-CTC (Full Page with Segmentation)

```bash
python3 inference_page.py \
    --image page.jpg \
    --checkpoint models/my_model/best_model.pt \
    --num-beams 4
```

---

## 📦 Batch Processing

### Batch Processing GUI

For processing multiple images or folders, use the batch processing GUI:

```bash
python3 polyscriptor_batch_gui.py
```

**Features:**
- Process entire folders of images
- Automatic PAGE XML detection (uses existing segmentation if available)
- Progress tracking with live output
- Export results to TXT, CSV, or PAGE XML
- Resume interrupted processing

### Batch Processing CLI

For scripted/automated workflows:

```bash
python3 batch_processing.py \
    --input-folder ./images \
    --engine crnn-ctc \
    --model-path models/my_model/best_model.pt \
    --segmentation-method kraken \
    --output-folder ./output \
    --use-pagexml
```

**Key options:**
- `--engine`: crnn-ctc, TrOCR, Qwen3-VL, LightOnOCR, Party, Kraken, PaddleOCR
- `--segmentation-method`: kraken (recommended), hpp (fast), none (pre-segmented)
- `--use-pagexml`: Auto-detect and use existing PAGE XML segmentation
- `--resume`: Skip already-processed files
- `--dry-run`: Test without writing output
- `--flip-rtl`: Flip line images horizontally for RTL scripts (see below)

---

## ↔️ RTL Manuscript Support (Ottoman, Arabic, Hebrew)

Some manuscript traditions are written right-to-left (RTL), but are transcribed with a Latin or Cyrillic alphabet that is read left-to-right (LTR). A key example is **Ottoman Turkish manuscripts**: the image is scanned RTL, but the ground truth transcription is a Latin-alphabet transliteration read LTR.

**The problem:** CRNN-CTC uses CTC alignment, which maps spatial position left-to-right to sequence position. If the image is RTL but the label is LTR, training fails. TrOCR (attention decoder) has the same issue. The fix is simple: **flip all line images horizontally** before training and during inference. This makes the visual input LTR, matching the LTR transcription.

### What to do

**Step 1 — Data preparation** (flip images when parsing Transkribus export):
```bash
python3 transkribus_parser.py \
    --input_dir ./transkribus_export \
    --output_dir ./data/ottoman_dataset \
    --preserve-aspect-ratio \
    --target-height 128 \
    --flip-rtl          # <-- flip all line images horizontally
```
This saves flipped images to disk. The `dataset_info.json` will record `"flip_rtl": true`.

**Step 2 — Training:** No change needed. Train normally on the flipped images:
```bash
# CRNN-CTC
python3 train_pylaia.py \
    --train_dir ./data/crnn_train \
    --val_dir ./data/crnn_val \
    --output_dir ./models/ottoman_model \
    --epochs 250

# TrOCR
python3 optimized_training.py --config config_ottoman.yaml
```

**Step 3 — Inference:** You must use `--flip-rtl` at inference time too, because the model was trained on flipped images and expects the same orientation.

```bash
# Batch processing (CRNN-CTC or TrOCR)
python3 batch_processing.py \
    --input-folder HTR_Images/ottoman \
    --engine crnn-ctc \
    --model-path models/ottoman_model/best_model.pt \
    --flip-rtl

# Single-page inference (TrOCR)
python3 inference_page.py \
    --image page.jpg \
    --checkpoint models/ottoman_model/checkpoint-3000 \
    --flip-rtl
```

**GUI (PyQt):** Enable the "RTL manuscript (flip line images)" checkbox in the engine's settings panel before loading the model.

**Web UI:** Enable "RTL manuscript (flip line images)" in the CRNN-CTC or TrOCR config form. For TrOCR, this setting is applied at model load time (same as "Normalize Background") — reload the model after changing it.

### Which engines support `--flip-rtl`?

| Engine | Supported | Notes |
|--------|-----------|-------|
| **CRNN-CTC** | Yes | Flip applied per inference call — works as a runtime toggle |
| **TrOCR** | Yes | Flip applied at model load time — must reload if changed |
| **API engines** (Gemini, Claude, GPT) | Not needed | Vision models handle image orientation internally |
| **Qwen3-VL / LightOnOCR** | Not needed | Same — full-image vision models |

### Kraken Neural (blla) — reading order (`text_direction`)

When using Kraken Neural segmentation on RTL manuscripts, set the reading direction so that columns are returned in the correct order (right-most column first):

- **Web UI**: "Reading direction" dropdown in Segmentation → Kraken Neural options → select `RTL (Arabic, Ottoman, Hebrew, …)`
- **PyQt GUI**: "Direction" combo box in the blla segmentation settings
- **CLI**: not yet exposed as a flag; defaults to `horizontal-lr`

This controls **column ordering only** — which column is listed first. It is independent of `--flip-rtl`, which flips the pixel content of each line. For Ottoman manuscripts you typically want **both**.

---

## 🖨️ PaddleOCR Engine

PaddleOCR performs its own text **detection + recognition** on whole pages — no pre-segmented lines needed. It runs in an isolated `venv_paddle` to avoid OpenCV conflicts with the main environment.

### Setup

```bash
python3 -m venv venv_paddle
source venv_paddle/bin/activate
# CPU only:
pip install paddlepaddle paddleocr
# GPU (CUDA 12.x):
pip install paddleocr
pip install paddlepaddle-gpu==3.0.0 -f https://www.paddlepaddle.org.cn/packages/stable/cu126/
deactivate
```

### Language codes

PaddleOCR uses **ISO language codes** (not script names). Enter the code in the "Language code" field of the engine config. Common examples:

| Code | Language / Script |
|------|-------------------|
| `en` | English |
| `ch` | Chinese + English (strongest general model) |
| `de` or `german` | German |
| `fr` or `french` | French |
| `ru` | Russian (Cyrillic) |
| `uk` | Ukrainian (Cyrillic) |
| `bg` | Bulgarian (Cyrillic) |
| `la` | Latin (classical) |
| `ar` | Arabic |
| `japan` | Japanese |
| `korean` | Korean |

> **Note:** Models download automatically on first use (~50–200 MB per script group). Only `en` is fetched during initial setup. Other language models cache in `~/.paddlex/official_models/`.
> Full language list: https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/blog/multi_languages.html

---

## 🌐 Web UI (Browser-Based Interface)

**Polyscriptor includes a browser-based web interface** — run inference locally or on a remote server and interact from any browser. No X11 forwarding needed; when running on a remote server, no local Python install is required either.

### Quick Start

```bash
# Web dependencies are included in requirements.txt — no extra install needed.

# Activate your virtual environment first:
source htr_env/bin/activate    # Linux/Mac
# or: htr_env\Scripts\activate  # Windows

# Start the server (run from the project root)
uvicorn web.polyscriptor_server:app --host 0.0.0.0 --port 8765

# Open in browser
# Local: http://localhost:8765
# Remote: use SSH tunnel (see below)
```

> **Works without a GPU.** Commercial APIs (Gemini, Claude, OpenAI) and TrOCR run on CPU. CRNN-CTC also runs on CPU — inference is slower (~1–2 min/page) but fully functional, and our published Church Slavonic, Ukrainian and Glagolitic models all work this way. Only Qwen3-VL and LightOnOCR require a GPU.

### Remote Access via SSH Tunnel

```bash
# On your laptop — tunnel port 8765 through SSH
ssh -L 8765:localhost:8765 user@your.server.edu

# Then open: http://localhost:8765
# No firewall issues — works on any university network
```

### Features

- **Engine selection** with dynamic configuration forms (CRNN-CTC, TrOCR, Kraken, etc.)
- **Image upload** — drag-and-drop, file picker, or PDF upload (multi-page PDFs become batch items)
- **Segmentation** — Kraken (neural blla or classical) with color-coded region overlay
- **Live transcription** — Server-Sent Events stream, lines appear as processed
- **Batch queue** — multi-image queue, drag-to-reorder, cancel, prev/next navigation
- **Inline editing** — double-click any transcription line to correct it
- **Confidence filter** — slider to dim low-confidence lines
- **Export** — TXT, CSV, PAGE XML (single image or ZIP for entire batch)
- **Font selector** — Monomakh Unicode (recommended for Church Slavonic), Old Standard TT, and others
- **Kraken model presets** — 12 Zenodo community models with one-click download
- **Resizable panels** — drag handles to adjust column widths, saved across sessions

### Running Tests

```bash
source htr_env/bin/activate
pip install pytest httpx
pytest web/tests/test_server.py -v
```

---

## 🖥️ Remote Server Usage

Running on a remote Linux server without GUI? You have several options:

### Option 1: CLI Batch Processing

**Best for**: Production workflows, processing many images

```bash
# Process entire folders efficiently
python3 batch_processing.py \
    --input-folder HTR_Images/manuscripts \
    --engine crnn-ctc \
    --model-path models/crnn_ctc_model/best_model.pt \
    --use-pagexml \
    --output-folder output
```

**Benefits**: faster than GUI methods, no display overhead, scriptable

### Option 2: X11 Forwarding (Interactive Work)

**Best for**: Interactive GUI work, visual parameter tuning, model comparison

**Using MobaXterm on Windows:**
1. Install MobaXterm (X server auto-starts)
2. SSH with X11 forwarding enabled
3. Test: `xclock &` (should show clock window)
4. Launch GUI: `python3 transcription_gui_plugin.py`

**Performance**: Good over LAN/local WiFi, slower over internet connections. Enable compression for best results.

### Option 3: VNC (Alternative for Slow Connections)

**Best for**: When X11 is too slow (poor internet), extended GUI sessions, session persistence

```bash
# On server
vncserver :1 -geometry 1920x1080

# Connect from Windows using VNC viewer to: server:5901
```

**Benefits**: Better compression than X11, survives disconnects, works well over internet

### Comparison

| Method | Speed | Best For | Network Type |
|--------|-------|----------|--------------|
| CLI Batch Processing | ⚡⚡⚡ | Production, automation | Any |
| **Web UI** | ⚡⚡⚡ | **Interactive work, no install needed** | Any (SSH tunnel) |
| X11 Forwarding | ⚡⚡ | Interactive GUI work | LAN/Local WiFi |
| X11 Forwarding | ⚡ | Light use only | Internet |
| VNC/NoMachine | ⚡⚡ | Extended sessions, poor connections | Any |

---

## ⚙️ Configuration

### CRNN-CTC Training Parameters

Key hyperparameters for optimal performance:

```python
{
    "img_height": 128,           # Target image height
    "batch_size": 32,            # GPU-optimized (44GB VRAM)
    "num_epochs": 250,           # With early stopping
    "learning_rate": 0.0003,
    "early_stopping_patience": 15,
    "augment_train": True,       # Data augmentation
    "device": "cuda:0"
}
```

### TrOCR Training Configuration

```yaml
model_name: "kazars24/trocr-base-handwritten-ru"
data_root: "./processed_data"
batch_size: 16
epochs: 10
cache_images: true             # 10-50x faster data loading
fp16: true                     # Mixed precision training
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

1. **New HTR engines**: Add plugins for other HTR systems
2. **Model training**: Share trained models for new scripts/languages
3. **Bug fixes**: Especially inference/GUI issues
4. **Documentation**: Improve guides and examples

---

## 📝 License

MIT License

Copyright (c) 2025 Achim Rabus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🙏 Acknowledgments

- **PyLaia** (architecture inspiration for CRNN-CTC engine): https://github.com/jpuigcerver/PyLaia
- **TrOCR**: Microsoft's Transformer-based OCR: https://huggingface.co/microsoft/trocr-base-handwritten
- **LightOnOCR**: Lightweight VLM for OCR: https://huggingface.co/lightonai/LightOnOCR-2-1B-base
- **Party**: PAge-wise Recognition of Text-y https://github.com/mittagessen/party/
- **Transkribus**: Transcription, training, and inference plattform: https://app.transkribus.org/
- **Qwen3-VL**: Alibaba's Vision-Language Model: https://github.com/QwenLM/Qwen3-VL
- **William Mattingly**: Support with VLM fine-tuning and Church Slavonic models: https://huggingface.co/wjbmattingly

---

## 📧 Contact

For questions, bug reports, or collaboration inquiries:
- GitHub Issues: [Create an issue](https://github.com/achimrabus/polyscriptor/issues)

---

## 🔬 Technical Notes

### Critical Preprocessing for CRNN-CTC

**Aspect Ratio Preservation** is CRITICAL for high aspect ratio line images:

```bash
# ALWAYS use --preserve-aspect-ratio for manuscript lines
python3 transkribus_parser.py \
    --preserve-aspect-ratio \
    --target-height 128 \
    # ...other args
```

Without this, TrOCR's ViT encoder brutally resizes to 384×384, causing 10.6x width compression for Ukrainian lines (4077×357 → 384×384). Characters shrink from ~80px to ~7px width, making recognition nearly impossible.

### Known Bugs (Fixed)

1. **KALDI Format Vocabulary**: Train/inference scripts now auto-detect format
2. **`<space>` vs `<SPACE>`**: Both cases handled correctly
3. **Vocabulary File Mismatch**: Training scripts auto-copy vocabulary to model directory
