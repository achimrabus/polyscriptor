# Multi-Engine HTR Training & Comparison Tool

A comprehensive toolkit for training and comparing different Handwritten Text Recognition (HTR) engines on historical manuscript datasets. Supports TrOCR, PyLaia, Qwen3-VL, LightOnOCR, Party, and Kraken engines with a unified GUI interface.

**Primary Focus:** Cyrillic manuscripts (Russian, Ukrainian, Church Slavonic, Glagolitic)

---

## 🎯 Features

### Multiple HTR Engines
- **TrOCR**: Transformer-based OCR (line-level)
- **PyLaia**: CTC-based CRNN (line-level)
- **Qwen3-VL**: Vision-Language Model (line/page-level, custom prompts)
- **LightOnOCR**: Lightweight VLM (~4GB VRAM, line-level, fine-tuned variants)
- **Churro**: Qwen fork, experimental (line/page-level, custom prompts)
- **Party**: Transformer-based HTR (line-level, multilingual)
- **Kraken**: Segmentation & recognition

### Commercial & Local Vision Models
- **Commercial APIs**: Google Gemini, Anthropic Claude Vision (via API keys)
- **Local LLMs**: OpenWebUI integration for local vision models
- **Unified interface**: All models accessible through same engine plugin system

### Core Capabilities
- **Plugin GUI**: Compare engines side-by-side with unified interface
- **Model management**: Easy switching between trained models and API providers
- **Export formats**: TXT, CSV, PAGE XML

### Training Pipelines (GPU required)
- **PyLaia**: Custom CRNN training with PAGE XML support
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

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (if you have a GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Launch GUI for inference

**Local usage:**
```bash
source htr_env/bin/activate
python3 transcription_gui_plugin.py
```

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
    --engine PyLaia \
    --model-path models/pylaia_model/best_model.pt \
    --use-pagexml
```

📖 **See [REMOTE_GUI_GUIDE.md](REMOTE_GUI_GUIDE.md)** for comprehensive remote access options (X11, VNC, CLI workflows)

### 3. Train a Model (CLI, PyLaia Example)

```bash
# Step 1: Parse Transkribus PAGE XML export → CSV format
python3 transkribus_parser.py \
    --input_dir /path/to/transkribus_export \
    --output_dir ./data/my_dataset \
    --preserve-aspect-ratio \
    --target-height 128

# Step 2: Convert CSV → PyLaia format (required!)
python3 convert_to_pylaia.py \
    --input_csv ./data/my_dataset/train.csv \
    --output_dir ./data/pylaia_train

python3 convert_to_pylaia.py \
    --input_csv ./data/my_dataset/val.csv \
    --output_dir ./data/pylaia_val

# Step 3: Train PyLaia model
python3 train_pylaia.py \
    --train_dir ./data/pylaia_train \
    --val_dir ./data/pylaia_val \
    --output_dir ./models/my_model \
    --batch_size 32 \
    --epochs 250
```

---

## 📁 Repository Structure

```
.
├── train_pylaia.py                  # PyLaia CRNN training script
├── inference_pylaia_native.py       # PyLaia inference (native Linux)
├── inference_page.py                # Line segmentation + OCR pipeline
├── transcription_gui_plugin.py      # Main GUI application
├── polyscriptor_batch_gui.py        # Batch processing GUI
├── batch_processing.py              # Batch processing CLI
├── htr_engine_base.py              # HTR engine interface
│
├── engines/                         # HTR engine plugins
│   ├── trocr_engine.py             # TrOCR transformer
│   ├── pylaia_engine.py            # PyLaia CRNN
│   ├── qwen3_engine.py             # Qwen3-VL (local)
│   ├── lighton_ocr_engine.py       # LightOnOCR VLM (lightweight)
│   ├── churro_engine.py            # Churro (Qwen fork)
│   ├── party_engine.py             # Party multilingual HTR
│   ├── kraken_engine.py            # Kraken segmentation
│   ├── commercial_api_engine.py    # Google Gemini, OpenAI GPT & Anthropic Claude APIs
│   └── openwebui_engine.py         # OpenWebUI local LLMs
│
├── optimized_training.py            # TrOCR fine-tuning script
├── transkribus_parser.py            # PAGE XML data preparation
├── alto_parser.py                   # ALTO XML data preparation
├── page_xml_exporter.py             # Export results to PAGE XML
├── qwen3_prompts.py                 # Custom prompts for Qwen3-VL
│
├── requirements.txt                 # Python dependencies
│
└── models/                          # Trained models (excluded from git)
    ├── pylaia_*/                    # PyLaia model checkpoints
    └── trocr_*/                     # TrOCR fine-tuned models
```

---

## 🎓 Typical Workflow

### Training a PyLaia Model

1. **Export data from Transkribus** (PAGE XML format)
2. **Parse with preprocessing**:
   ```bash
   python3 transkribus_parser.py \
       --input_dir ./transkribus_export \
       --output_dir ./data/my_dataset \
       --preserve-aspect-ratio \
       --target-height 128
   ```
3. **Convert to PyLaia format**:
   ```bash
   python3 convert_to_pylaia.py \
       --input_csv ./data/my_dataset/train.csv \
       --output_dir ./data/pylaia_train
   python3 convert_to_pylaia.py \
       --input_csv ./data/my_dataset/val.csv \
       --output_dir ./data/pylaia_val
   ```
4. **Train model**:
   ```bash
   python3 train_pylaia.py \
       --train_dir ./data/pylaia_train \
       --val_dir ./data/pylaia_val \
       --output_dir ./models/my_model \
       --batch_size 32 \
       --epochs 250
   ```
5. **Use in GUI**: Model will appear in PyLaia engine dropdown

### Using Trained Models

Trained models can be loaded in the GUI:
- PyLaia models: Select from dropdown or browse to model directory
- TrOCR models: Specify HuggingFace Hub ID or local checkpoint path
- Commercial APIs: Enter API keys in engine configuration

---

## 🛠️ Command-Line Inference

### PyLaia (Single Line)

```bash
python3 inference_pylaia_native.py \
    --checkpoint models/my_model/best_model.pt \
    --syms models/my_model/symbols.txt \
    --image line_image.png
```

### PyLaia (Full Page with Segmentation)

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
    --engine PyLaia \
    --model-path models/my_model/best_model.pt \
    --segmentation-method kraken \
    --output-folder ./output \
    --use-pagexml
```

**Key options:**
- `--engine`: PyLaia, TrOCR, Qwen3-VL, LightOnOCR, Party, Kraken
- `--segmentation-method`: kraken (recommended), hpp (fast), none (pre-segmented)
- `--use-pagexml`: Auto-detect and use existing PAGE XML segmentation
- `--resume`: Skip already-processed files
- `--dry-run`: Test without writing output

---

## 🖥️ Remote Server Usage

Running on a remote Linux server without GUI? You have several options:

### Option 1: CLI Batch Processing

**Best for**: Production workflows, processing many images

```bash
# Process entire folders efficiently
python3 batch_processing.py \
    --input-folder HTR_Images/manuscripts \
    --engine PyLaia \
    --model-path models/pylaia_model/best_model.pt \
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
| X11 Forwarding | ⚡⚡ | Interactive GUI work | LAN/Local WiFi |
| X11 Forwarding | ⚡ | Light use only | Internet |
| VNC/NoMachine | ⚡⚡ | Extended sessions, poor connections | Any |

---

## ⚙️ Configuration

### PyLaia Training Parameters

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

- **PyLaia**: CTC-based HTR system: https://github.com/jpuigcerver/PyLaia
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

### Critical Preprocessing for PyLaia

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
