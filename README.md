# Multi-Engine HTR Training & Comparison Tool

A comprehensive toolkit for training and comparing different Handwritten Text Recognition (HTR) engines on historical manuscript datasets. Supports TrOCR, PyLaia, Qwen3-VL, Party, and Kraken engines with a unified GUI interface.

**Primary Focus:** Cyrillic manuscripts (Russian, Ukrainian, Church Slavonic, Glagolitic)

---

## 🎯 Features

### Multiple HTR Engines
- **TrOCR**: Transformer-based OCR (line-level, English/Cyrillic)
- **PyLaia**: CTC-based CRNN (line-level, excellent for manuscripts)
- **Qwen3-VL**: Vision-Language Model (line/page-level, multilingual, custom prompts)
- **Party**: Transformer-based HTR (page-level, multilingual)
- **Kraken**: Traditional segmentation & recognition

### Commercial & Local Vision Models
- **Commercial APIs**: Google Gemini, Anthropic Claude Vision (via API keys)
- **Local LLMs**: OpenWebUI integration for local vision models
- **Unified interface**: All models accessible through same engine plugin system

### Core Capabilities
- **Plugin GUI**: Compare engines side-by-side with unified interface
- **Model management**: Easy switching between trained models and API providers
- **Export formats**: TXT, CSV, PAGE XML

### Training Pipelines
- **PyLaia**: Custom CRNN training with PAGE XML support
- **TrOCR**: Fine-tuning pipeline with image caching (10-50x faster)
- **Data preparation**: Transkribus PAGE XML parser

### Key Capabilities
- Line segmentation (automatic or PAGE XML-based)
- Confidence scoring and statistics
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

### 2. Launch GUI

```bash
source htr_env/bin/activate
python3 transcription_gui_plugin.py
```

### 3. Train a Model (PyLaia Example)

```bash
# Prepare data from Transkribus PAGE XML export
python3 transkribus_parser.py \
    --input_dir /path/to/transkribus_export \
    --output_dir ./data/my_dataset \
    --preserve-aspect-ratio \
    --target-height 128

# Train PyLaia model
python3 train_pylaia.py \
    --train-dir ./data/my_dataset/train \
    --val-dir ./data/my_dataset/val \
    --output-dir ./models/my_model \
    --batch-size 32 \
    --num-epochs 250 \
    --device cuda:0
```

---

## 📁 Repository Structure

```
.
├── train_pylaia.py                  # PyLaia CRNN training script
├── inference_pylaia_native.py       # PyLaia inference (native Linux)
├── inference_page.py                # Line segmentation + OCR pipeline
├── transcription_gui_plugin.py      # Main GUI application
├── htr_engine_base.py              # HTR engine interface
│
├── engines/                         # HTR engine plugins
│   ├── trocr_engine.py             # TrOCR transformer
│   ├── pylaia_engine.py            # PyLaia CRNN
│   ├── qwen3_engine.py             # Qwen3-VL (local)
│   ├── party_engine.py             # Party multilingual HTR
│   ├── kraken_engine.py            # Kraken segmentation
│   ├── gemini_engine.py            # Google Gemini API
│   ├── claude_engine.py            # Anthropic Claude API
│   └── openwebui_engine.py         # OpenWebUI local LLMs
│
├── optimized_training.py            # TrOCR fine-tuning script
├── transkribus_parser.py            # PAGE XML data preparation
├── page_xml_exporter.py             # Export results to PAGE XML
├── qwen3_prompts.py                 # Custom prompts for Qwen3-VL
│
├── example_config.yaml              # TrOCR training config template
├── requirements.txt                 # Python dependencies
│
├── models/                          # Trained models (excluded from git)
│   ├── pylaia_*/                    # PyLaia model checkpoints
│   ├── trocr_*/                     # TrOCR fine-tuned models
│   └── README.md                    # Links to downloadable models
│
└── docs/
    ├── PYLAIA_TRAINING_STATUS.md    # PyLaia training results & bug fixes
    └── LINUX_SERVER_MIGRATION.md    # Linux setup guide
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
3. **Train model**:
   ```bash
   python3 train_pylaia.py \
       --train-dir ./data/my_dataset/train \
       --val-dir ./data/my_dataset/val \
       --output-dir ./models/my_model \
       --batch-size 32 \
       --device cuda:0
   ```
4. **Use in GUI**: Model will appear in PyLaia engine dropdown

### Using Trained Models

Trained models can be loaded in the GUI:
- PyLaia models: Select from dropdown or browse to model directory
- TrOCR models: Specify HuggingFace Hub ID or local checkpoint path
- Commercial APIs: Enter API keys in engine configuration

See `models/README.md` for links to downloadable pre-trained models.

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

### TrOCR Training (example_config.yaml)

```yaml
model_name: "kazars24/trocr-base-handwritten-ru"
data_root: "./processed_data"
batch_size: 16
epochs: 10
cache_images: true             # 10-50x faster data loading
fp16: true                     # Mixed precision training
```

---

## 📖 Documentation

- **[PYLAIA_TRAINING_STATUS.md](PYLAIA_TRAINING_STATUS.md)**: Training results, bug fixes, and insights
- **[LINUX_SERVER_MIGRATION.md](LINUX_SERVER_MIGRATION.md)**: Server setup guide

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
- **Party**: PAge-wise Recognition of Text-y https://github.com/mittagessen/party/
- **Transkribus**: Transcription, training, and inference plattform: https://app.transkribus.org/
- **Qwen3-VL**: Alibaba's Vision-Language Model: https://github.com/QwenLM/Qwen3-VL

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

See [PYLAIA_TRAINING_STATUS.md](PYLAIA_TRAINING_STATUS.md) for detailed bug analysis.
