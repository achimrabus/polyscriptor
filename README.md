# Polyscriptor: Multi-Engine HTR and OCR for Historical Documents

A comprehensive toolkit for training and comparing text recognition engines on historical manuscripts and prints. Supports TrOCR, CRNN-CTC, Qwen3-VL, LightOnOCR, Party, Kraken, Kraken 7 (PP-OCR), GLM-OCR, dots.mocr, commercial and local VLMs via API and more through a browser-based web interface (plus a PyQt6 desktop GUI and a batch CLI).

**Scope:** Polyscriptor is not tied to one script. Its engines cover Latin-script print and handwriting (including Fraktur and early modern print), right-to-left manuscripts and Cyrillic. Its **own trained models** are primarily for Cyrillic (Russian, Ukrainian, Church Slavonic) and Glagolitic. Which engine suits which material: see [Which engine for which material](#-which-engine-for-which-material).

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
- **Kraken 7 (PP-OCR)**: kraken 7.1 with the multilingual PP-OCRv6 base models (line-level, fast, also on CPU; subprocess-isolated)
- **PaddleOCR**: Multi-language printed/mixed text detection + recognition (whole-page, subprocess-isolated)
- **dots.mocr**: Page-level OCR VLM (3B); strong on Latin-script print, reads whole pages without segmentation (subprocess-isolated)
- **GLM-OCR**: Page-level OCR VLM (1.3B); faster than dots.mocr and close behind it on Latin-script print (subprocess-isolated)

### Commercial & Local Vision Models
- **Commercial APIs**: Google Gemini, OpenAI GPT, Anthropic Claude (via your own API keys)
- **Local LLMs**: OpenWebUI integration for local vision models
- **Unified interface**: All models accessible through same engine plugin system

### Core Capabilities
- **Web interface**: Upload, segment, transcribe, correct and export in the browser; runs locally or on a server
- **Engine comparison**: Compare engines side-by-side (web interface and desktop GUI)
- **Model management**: Easy switching between trained models and API providers
- **Line segmentation**: Kraken (neural or classical), a fast projection-based method, or existing PAGE XML
- **Batch processing**: desktop batch GUI and CLI
- **Import/export**: PAGE XML in and out; export to TXT and CSV
- **Custom prompts** for the VLM engines (Qwen3-VL, commercial APIs)

### Training Pipelines (GPU required)
- **CRNN-CTC**: Custom CRNN training with PAGE XML support
- **TrOCR**: Fine-tuning pipeline with image caching in RAM
- **Data preparation**: PAGE XML parser (Transkribus and eScriptorium exports), ALTO XML parser

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/achimrabus/polyscriptor.git
cd polyscriptor

# Create virtual environment
# Requires Python 3.10–3.12 (Python 3.13+ may lack prebuilt wheels for some
# packages; check your version with: python --version)
python3 -m venv htr_env
source htr_env/bin/activate  # Linux/Mac
# or: htr_env\Scripts\activate  # Windows
```

Install everything in **one** pip call into the fresh environment. `requirements-kraken.txt`
adds Kraken segmentation; it is optional but recommended (without it, the fast but much weaker
HPP segmentation is used).

**GPU install (CUDA 12.8 — Linux/Windows with NVIDIA GPU):**
```bash
pip install -r requirements-gpu.txt -r requirements.txt -r requirements-kraken.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

**CPU-only install (no GPU required):**
```bash
# Linux/Mac:
pip install -r requirements.txt -r requirements-kraken.txt

# Windows (avoids a torch DLL load error on CPU-only machines):
pip install -r requirements.txt -r requirements-kraken.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

> **Do not add Kraken to an existing environment later** (`pip install kraken` or
> `pip install -r requirements-kraken.txt` as a second step). Kraken pins several shared packages
> tightly; installed afterwards, pip downgrades them and `transformers` stops importing. If you
> need Kraken in an environment that was set up without it, create a new environment and use
> the one-call install above.

### 2. Start the web interface

```bash
source htr_env/bin/activate        # Windows: htr_env\Scripts\activate
uvicorn web.polyscriptor_server:app --host 0.0.0.0 --port 8765
```

Open **http://localhost:8765** in your browser. On a remote server, tunnel the port from your
laptop with `ssh -L 8765:localhost:8765 user@server` and open the same address locally. No
display, X11 or Qt is needed. Details: [Web Interface](#-web-interface).

**Alternative: desktop GUI (PyQt6)**
```bash
python3 transcription_gui_plugin.py      # single pages
python3 polyscriptor_batch_gui.py        # folders
```

> **Linux:** PyQt6 (Qt ≥ 6.5) needs the system library `libxcb-cursor0`. If the GUI fails with *"Could not load the Qt platform plugin "xcb""*, install it: `sudo apt install libxcb-cursor0` (Debian/Ubuntu).

**For many pages: batch CLI**
```bash
python3 batch_processing.py \
    --input-folder HTR_Images/my_folder \
    --engine crnn-ctc \
    --model-path models/crnn_ctc_model/best_model.pt \
    --segmentation-method kraken-blla \
    --use-pagexml
```

### 3. Train a Model (CLI, CRNN-CTC Example)

```bash
# Step 1: Parse PAGE XML export → CSV format
# Works with Transkribus exports (namespace 2013-07-15) and
# eScriptorium/Kraken exports (namespace 2019-07-15) — both are auto-detected.
python3 transkribus_parser.py \
    --input_dir /path/to/pagexml_export \
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

## 🌐 Web Interface

**The web interface is the main way to use Polyscriptor** — run inference locally or on a remote server and interact from any browser. No X11 forwarding needed; when running on a remote server, no local Python install is required either.

### Starting the server

```bash
source htr_env/bin/activate    # Windows: htr_env\Scripts\activate
uvicorn web.polyscriptor_server:app --host 0.0.0.0 --port 8765   # run from the project root
# open http://localhost:8765
```

> **Works without a GPU.** Commercial APIs (Gemini, Claude, OpenAI) need no local GPU. CRNN-CTC runs on CPU at ~1–2 min/page, and all published CRNN-CTC models work this way. TrOCR runs on CPU too, but slowly: about 10 s per line with beam 4 and 4 s with beam 1 (8 CPU threads), i.e. several minutes per page. Kraken and PaddleOCR also run on CPU, and so does Kraken 7 (PP-OCR) at about 0.2 s per line on a server CPU. The local VLMs (Qwen3-VL, LightOnOCR, GLM-OCR, dots.mocr) need a GPU in practice.

### Remote Access via SSH Tunnel

```bash
# On your laptop — tunnel port 8765 through SSH
ssh -L 8765:localhost:8765 user@your-server

# Then open: http://localhost:8765
# Only SSH access to the server is needed; port 8765 need not be open
```

### Features

- **Engine selection** with dynamic configuration forms (CRNN-CTC, TrOCR, Kraken, etc.)
- **Image upload** — drag-and-drop, file picker, or PDF upload (multi-page PDFs become batch items)
- **Segmentation** — Kraken (neural blla or classical) with color-coded region overlay; blla splits multi-column pages into columns ("Max columns", "Split width %", default 20 %) and keeps single-column pages whole
- **Live transcription** — Server-Sent Events stream, lines appear as processed
- **Batch queue** — multi-image queue, drag-to-reorder, cancel, prev/next navigation
- **Inline editing** — double-click any transcription line to correct it
- **Confidence filter** — slider to dim low-confidence lines
- **Export** — TXT, CSV, PAGE XML (single image or ZIP for entire batch)
- **Font selector** — Monomakh Unicode (recommended for Church Slavonic), Old Standard TT, and others
- **Kraken model presets** — local blla segmentation model and CATMuS-Print (Zenodo, one-click download); any other Kraken `.mlmodel` can be loaded from a local path
- **Resizable panels** — drag handles to adjust column widths, saved across sessions

### Running Tests

```bash
source htr_env/bin/activate
pip install pytest httpx
pytest web/tests/test_server.py -v
```

---

## 🤖 Pre-trained Models

Polyscriptor works with models from several sources. The sections below list ready-to-use models for each engine. CER figures we did not measure ourselves are marked as model-card figures.

### CRNN-CTC

Ready-to-use models for Cyrillic and Glagolitic manuscripts, trained with Polyscriptor, are published on HuggingFace:

**[https://huggingface.co/achimrabus](https://huggingface.co/achimrabus)**

| Model | Script / Language | CER | Description |
|-------|-------------------|-----|-------------|
| [crnn-ctc-church-slavonic](https://huggingface.co/achimrabus/crnn-ctc-church-slavonic) | Church Slavonic | 2.89% | QuantiSlav corpus (Elena Renje) |
| [crnn-ctc-prosta-mova](https://huggingface.co/achimrabus/crnn-ctc-prosta-mova) | Prosta Mova (Old Ruthenian) | 3.77% | Continslav corpus (Martin Meindl) |
| [crnn-ctc-ukrainian](https://huggingface.co/achimrabus/crnn-ctc-ukrainian) | Ukrainian (Cyrillic) | 4.76% | 19th–20th c. manuscripts (MultiHTR) |
| [crnn-ctc-glagolitic](https://huggingface.co/achimrabus/crnn-ctc-glagolitic) | Croatian Glagolitic | 5.33% | Outputs Latin transliteration |
| [crnn-ctc-russian](https://huggingface.co/achimrabus/crnn-ctc-russian) | Russian (Cyrillic) | 6.92% | 18th–20th c. manuscripts, print and typescript; pre- and post-reform orthography |

All models run on CPU (no GPU required). They are presets of the CRNN-CTC engine: in a fresh
clone they appear as "[Hugging Face, auto-download]" and are downloaded on first load (then
cached); no `models/` folder needs to be set up. A model placed under the registry path in
`models/` is used instead of the download.

### TrOCR

A collection of TrOCR models for Cyrillic handwriting (Russian, Ukrainian, Church Slavonic) is maintained here:

**[https://huggingface.co/cyrillic-trocr](https://huggingface.co/cyrillic-trocr)**

Built-in presets of the TrOCR engine (downloaded on first load). CER measured on 8 October 2026 with
the decoding Polyscriptor uses (beam 4, no repeat block, `max_length` 128) on held-out validation pages:

| Model | Script / Language | Validation CER | Note |
|-------|-------------------|-----|------|
| [cyrillic-trocr/trocr-church-slavonic-handwritten](https://huggingface.co/cyrillic-trocr/trocr-church-slavonic-handwritten) | Church Slavonic | 3.80% | 20,679 lines; same data as the CRNN-CTC model |
| [cyrillic-trocr/trocr-ukrainian-handwritten](https://huggingface.co/cyrillic-trocr/trocr-ukrainian-handwritten) | Ukrainian | 4.99% | v2.0; 970 lines, 19th–20th c. manuscripts |
| [achimrabus/trocr-glagolitic-handwritten](https://huggingface.co/achimrabus/trocr-glagolitic-handwritten) | Croatian Glagolitic | 5.72% | 1,551 lines; outputs Latin transliteration |

Further presets from other providers, for Latin script and Cyrillic. CER as stated in the model cards, not measured by us:

| Model | Script / Language | CER (model card) | Note |
|-------|-------------------|-----|------|
| [dh-unibe/trocr-kurrent](https://huggingface.co/dh-unibe/trocr-kurrent) | German Kurrent, 19th c. | 2.66% | test split |
| [dh-unibe/trocr-kurrent-XVI-XVII](https://huggingface.co/dh-unibe/trocr-kurrent-XVI-XVII) | German Kurrent, 16th–18th c. | 5.42% | test split from the same hands |
| [Kansallisarkisto/cyrillic-htr-model](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model) | Cyrillic, 17th–20th c. | ~8% | |
| [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten), [-printed](https://huggingface.co/microsoft/trocr-base-printed) (also *large*) | English handwriting / print | — | Microsoft's original TrOCR models |

Any other TrOCR model can be loaded by entering its HuggingFace ID or a local path.

**Decoding.** Polyscriptor decodes TrOCR without the n-gram repeat block (`no_repeat_ngram_size=0`)
and with `max_length=128`. TrOCR's tokenizer splits Cyrillic into single letters or smaller pieces,
so a repeat block suppresses ordinary letter sequences; switching it off improved every Cyrillic and
Glagolitic model we tested. The block can be set as "Repeat block" in the web interface and the
desktop GUI, or with `--no-repeat-ngram-size`; for Latin-script models it is untested.

### LightOnOCR

Lightweight VLM models (~4 GB VRAM) for line-level HTR, compatible with the LightOnOCR engine:

| Model | Script / Language | Description |
|-------|-------------------|-------------|
| [wjbmattingly/LightOnOCR-2-1B-old-church-slavonic-line](https://huggingface.co/wjbmattingly/LightOnOCR-2-1B-old-church-slavonic-line) | Church Slavonic | Fine-tuned on QuantiSlav + Polyscriptor training data |
| [wjbmattingly/LightOnOCR-2-1B-german-shorthand-line](https://huggingface.co/wjbmattingly/LightOnOCR-2-1B-german-shorthand-line) | German shorthand | CER 21.9% (model card) |

Load via HuggingFace model ID in the LightOnOCR engine.

### Qwen3-VL

Vision-language models fine-tuned for historical Slavic manuscripts (Church Slavonic, Glagolitic):

**[https://huggingface.co/wjbmattingly](https://huggingface.co/wjbmattingly)**

Load any compatible Qwen3-VL model via its HuggingFace ID in the Qwen3-VL engine.

### Kraken

The default `blla` model for neural line segmentation ships with Kraken itself; the repository
carries an identical copy as `pagexml/blla.mlmodel`. Further segmentation and recognition models
are published on Zenodo, for example:

- **[CATMuS-Print](https://zenodo.org/records/10592716)** — printed text, multilingual (built-in preset)
- **[Generic HTR model for Old Cyrillic uncial and semi-uncial](https://zenodo.org/records/7755483)** — Church Slavonic, 11th–16th c.

Load a downloaded `.mlmodel` in the Kraken engine via its local path.

### Kraken 7 (PP-OCR)

kraken 7.1 recognises lines with a PP-OCRv6 architecture. Its multilingual base models
(44 languages, 10 scripts, Apache-2.0) are presets of the Kraken 7 (PP-OCR) engine;
`./setup_ocr_vlm_venvs.sh kraken7` downloads them (see [Kraken 7 (PP-OCR) Engine](#kraken-7-pp-ocr-engine)):

| Model | Size | Zenodo |
|-------|------|--------|
| PP-OCRv6 medium (default) | 64 MB | [10.5281/zenodo.21788410](https://doi.org/10.5281/zenodo.21788410) |
| PP-OCRv6 small | 13 MB | [10.5281/zenodo.21788405](https://doi.org/10.5281/zenodo.21788405) |
| PP-OCRv6 tiny | 3 MB | [10.5281/zenodo.21788403](https://doi.org/10.5281/zenodo.21788403) |

medium is the largest published base model and the most accurate; small and tiny are clearly
weaker (Russian handwriting: medium 14.1 %, small 18.0 %, tiny 27.1 % CER). Any other kraken 7
recognition model, e.g. one fine-tuned with `ketos`, can be loaded as a custom model file.

### Party

Party OCR models with Church Slavonic in the pretraining data:

**[https://zenodo.org/records/15075344](https://zenodo.org/records/15075344)**

---

## 🧭 Which engine for which material

Polyscriptor's own models cover Cyrillic and Glagolitic. For other material, use an external engine, or train a CRNN-CTC or TrOCR model on your own ground truth.

| Material | Start with | Also worth trying | Hardware |
|---|---|---|---|
| Cyrillic handwriting (Russian, Ukrainian), Church Slavonic, Prosta Mova | CRNN-CTC or TrOCR models from [Pre-trained Models](#-pre-trained-models) | Kraken 7 (PP-OCR) base model; Party, LightOnOCR and Qwen3-VL models fine-tuned for Church Slavonic | CPU for CRNN-CTC; TrOCR on CPU is slow |
| Croatian Glagolitic (Latin transliteration) | CRNN-CTC or TrOCR Glagolitic model | Qwen3-VL Glagolitic models | CPU for CRNN-CTC; TrOCR on CPU is slow |
| Latin-script print and typewriting, Fraktur, early modern print | Kraken 7 (PP-OCR) with a line segmentation, or dots.mocr for whole pages | GLM-OCR (faster than dots.mocr); Kraken with a print model such as CATMuS-Print (CPU) | CPU or GPU for Kraken 7; GPU for dots.mocr and GLM-OCR |
| German Kurrent, 16th–19th c. | TrOCR Kurrent presets (dh-unibe) | — | GPU recommended; TrOCR on CPU is slow |
| Other Latin-script handwriting | a CRNN-CTC or TrOCR model trained on your ground truth | Kraken 7 (PP-OCR) as a baseline without training | GPU for training |
| Right-to-left manuscripts transcribed in Latin or Cyrillic letters (e.g. Ottoman Turkish) | a CRNN-CTC or TrOCR model trained with `--flip-rtl` (see RTL Manuscript Support below) | — | GPU for training |
| Anything else, without a suitable model | commercial APIs (Gemini, GPT, Claude) | local VLMs via OpenWebUI | API key |

Check any engine on a sample of your own pages before processing a whole collection.

### Measured CER

Character error rate (CER, %) on line images, November 2026, default settings of each engine. Lower is better; — means no model for this material or not measured. Kraken 7 is the PP-OCRv6 medium base model without fine-tuning. dots.mocr and GLM-OCR are page models: on a single line crop they sometimes also read part of a neighbouring line, so their figures on whole pages may be better.

| Material | Test set | Lines | Polyscriptor model | Kraken McCATMuS | Kraken 7 (PP-OCR) | dots.mocr | GLM-OCR |
|---|---|---|---|---|---|---|---|
| Ukrainian handwriting, 19th–20th c. | validation pages of the Ukrainian models | 970 | **5.0** (TrOCR) | — | 9.9 | 44.2 | 60.2 |
| Russian handwriting, print and typescript | validation lines of the Russian model | 500 | —¹ | — | 14.1 | 69.8 | 75.2 |
| Middle High German texts in printed editions, 19th–20th c. | 4 editions, unpublished transcriptions | 117 | — | — | **2.7** | 5.4 | 6.8 |
| Printed text, 16th–21st c. (mostly French) | [CATMuS-Modern](https://huggingface.co/datasets/CATMuS/modern) test split | 300 | — | 3.8 | 3.0 | **2.9** | 4.0 |
| Typewriting | CATMuS-Modern test split | 57 | — | 8.6 | **3.3** | 3.5 | 3.9 |
| Latin-script handwriting | CATMuS-Modern test split | 300 | — | 22.1 | **13.1** | 20.6 | 26.4 |
| German Fraktur, 19th c. | [GT4HistOCR](https://zenodo.org/records/1344132) dta19 | 200 | — | 21.3 | **3.9** | 4.6 | 6.3 |
| Early Modern Latin print, 1471–1686 | GT4HistOCR EarlyModernLatin | 200 | — | 13.5 | **4.1** | 9.9 | 10.0 |
| Herbals, 1487–1870 | GT4HistOCR RIDGES | 200 | — | 32.6 | **4.9** | 10.3 | 11.6 |
| *Narrenschiff* editions, 1488–1509 | GT4HistOCR Kallimachos | 200 | — | 22.1 | **5.7** | 12.7 | 14.4 |
| German incunabula | GT4HistOCR RefCorpus-ENHG | 200 | — | 28.7 | **5.5** | 13.3 | 16.4 |

¹ Not measured on these 500 lines. `crnn-ctc-russian` reads the full validation set at 6.9 %.

Notes:
- **Kraken 7 figures may be too favourable.** According to its model card, the PP-OCRv6 training data include three Church Slavonic, two Russian and one Ukrainian dataset that are not named, and public Latin-script ground truth including CATMuS and Fraktur sets; the authors' test split is a random 5 % that does not separate documents. Lines of these test sets may therefore have been seen in training. On Cyrillic handwriting, the models trained for the material still read clearly better: Ukrainian TrOCR 5.0 % against 9.9 % on the same lines, `crnn-ctc-russian` 6.9 % on the full Russian validation set against 14.1 % on 500 of its lines.
- GT4HistOCR transcriptions are diplomatic (ſ, ꝛ, superscript e, abbreviations), and the figures count a modernised `s` for `ſ` as an error. With these variants folded, all 1,000 GT4HistOCR lines read at 3.0 % with Kraken 7, 6.8 % with dots.mocr and 9.7 % with GLM-OCR. GT4HistOCR has been public since 2018 and is common OCR training data, so these models may have seen its lines; this applies most to dta19.
- McCATMuS (Kraken) is a reference trained on CATMuS, not a Fraktur model. It was run on cut line images, which handicaps it most on handwriting.
- Neither page VLM is usable on Cyrillic.

---

## 📁 Repository Structure

```
.
├── train_pylaia.py                  # CRNN-CTC training script
├── inference_pylaia_native.py       # CRNN-CTC inference (native Linux)
├── inference_page.py                # Line segmentation + OCR pipeline
├── transcription_gui_plugin.py      # Desktop GUI (PyQt6)
├── polyscriptor_batch_gui.py        # Desktop batch GUI (PyQt6)
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
│   ├── paddle_worker.py            # PaddleOCR worker (runs inside venv_paddle)
│   ├── glm_ocr_engine.py           # GLM-OCR (subprocess, venv_glm)
│   ├── dots_ocr_engine.py          # dots.mocr (subprocess, venv_dots)
│   ├── kraken7_engine.py           # Kraken 7 PP-OCR (persistent subprocess, venv_kraken7)
│   └── *_worker.py                 # workers that run inside the isolated venvs
│
├── setup_ocr_vlm_venvs.sh           # creates venv_glm, venv_dots, venv_kraken7
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

1. **Export PAGE XML** from Transkribus or eScriptorium
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
5. **Use it**: Model will appear in the CRNN-CTC engine dropdown (web interface and desktop GUI)

### Using Trained Models

Trained models can be loaded in the web interface or the desktop GUI:
- CRNN-CTC models: Select from dropdown or browse to model directory
- TrOCR models: Specify HuggingFace Hub ID or local checkpoint path
- Commercial APIs: Enter API keys in engine configuration

---

## 🛠️ Command-Line Inference

### CRNN-CTC and all other engines

Use the [batch CLI](#batch-processing-cli); for a single page, put it in a folder of its own.

### TrOCR (single page)

```bash
python3 inference_page.py \
    --image page.jpg \
    --checkpoint models/my_trocr_model/checkpoint-3000 \
    --num_beams 4
# add --xml page.xml to use an existing PAGE XML segmentation
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
- `--engine`: crnn-ctc, TrOCR, Qwen3-VL, LightOnOCR, Party, Kraken, kraken7, PaddleOCR, GLM-OCR, dots.mocr (the names shown in the web interface, or short names such as `kraken7` for Kraken 7 (PP-OCR))
- `--segmentation-method`: kraken-blla (neural, handles multiple columns), kraken (classical), hpp (fast, default), none (pre-segmented lines); page engines (GLM-OCR, dots.mocr, PaddleOCR) ignore it
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

# TrOCR (config: a copy of example_config.yaml pointing to the flipped dataset)
python3 optimized_training.py --config my_ottoman_config.yaml
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

**Web interface:** Enable "RTL manuscript (flip line images)" in the CRNN-CTC or TrOCR config form. For TrOCR, this setting is applied at model load time (same as "Normalize Background") — reload the model after changing it.

**Desktop GUI (PyQt):** Enable the "RTL manuscript (flip line images)" checkbox in the engine's settings panel before loading the model.

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

## 🖨️ Page OCR VLMs: GLM-OCR and dots.mocr

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (`zai-org/GLM-OCR`, 1.3B, MIT) and [dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) (`rednote-hilab/dots.mocr`, 3B, MIT + use policy) are page-level OCR models. Each runs as a subprocess in its own venv, because they need different `transformers` versions than the main environment. They read whole pages and need no line segmentation.

### Setup

```bash
./setup_ocr_vlm_venvs.sh            # creates venv_glm and venv_dots
./setup_ocr_vlm_venvs.sh dots       # only one of them

# dots.mocr weights: the directory name must not contain periods
huggingface-cli download rednote-hilab/dots.mocr --local-dir models/DotsMOCR
```

- GLM-OCR: `transformers>=5.3`, torch ≥ 2.6 (cu126 wheels); ~4 GB VRAM. Weights download on first use.
- dots.mocr: `transformers==4.56.1`, torch 2.5 (cu121) and **flash-attn** (its model code imports it unconditionally); ~9 GB VRAM. The setup script installs a prebuilt flash-attn wheel for Linux x86_64 and Python 3.12; on other platforms or Python versions, install a matching flash-attn yourself.

The setup script is a bash script for Linux (on Windows: WSL). Both engines appear automatically in the web interface and the desktop GUI once their venv exists.

- **GLM-OCR** prompt modes: Text Recognition (default), Formula/Table Recognition, Free.
- **dots.mocr** modes: OCR / plain text (default), Layout (JSON), Scene spotting. It samples as in its model card (temperature 0.1, top_p 0.9) with a fixed seed, so runs on the same hardware repeat.

How both compare with the other engines: [Which engine for which material](#-which-engine-for-which-material).

---

## Kraken 7 (PP-OCR) Engine

kraken 7.1 added a PP-OCRv6 line recognition architecture and multilingual base models
(see [Pre-trained Models](#kraken-7-pp-ocr)). kraken 7 cannot share the main environment, which needs
kraken 6 for blla segmentation and the Kraken engine, so the engine runs a worker process in its
own `venv_kraken7`. The worker loads the model once and then reads one line per request.

### Setup

```bash
./setup_ocr_vlm_venvs.sh kraken7    # venv_kraken7 (kraken 7.1.1, torch cu128) + base models
```

The script links the base models as `models/kraken7/ppocr_{medium,small,tiny}.safetensors`.
The engine appears in the web interface and the desktop GUI once `venv_kraken7` exists.

### Use

- It reads **line images**: choose a line segmentation (Kraken blla for manuscripts and
  multi-column pages) or use an existing PAGE XML.
- Settings: model (medium, small, tiny or a custom kraken 7 `.safetensors` file) and device
  (GPU or CPU). On a GPU a line takes about 0.05 s, on a server CPU about 0.2 s.
- Batch CLI: `python3 batch_processing.py --input-folder pages/ --engine kraken7 --segmentation-method kraken-blla`
  (`--model-path ppocr_small` or a model file for another model).
- The base models follow mixed transcription conventions (see their model card). For a fixed
  convention or a script they do not cover, fine-tune with kraken's `ketos` and load the result
  as a custom model file.

---

## 🖥️ Remote Server Usage

Running on a remote Linux server without a display? You have several options.

### Option 0: Web interface via SSH tunnel (recommended)

**Best for**: Interactive work, model comparison, correcting transcriptions; nothing to install locally.
See [Web Interface](#-web-interface): start the server, then `ssh -L 8765:localhost:8765 user@server` and open http://localhost:8765.

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

You need a local X server: built in on Linux, XQuartz on macOS, MobaXterm or VcXsrv on Windows.

**Example: MobaXterm on Windows:**
1. Install MobaXterm (X server auto-starts)
2. SSH with X11 forwarding enabled (on Linux/macOS: `ssh -X user@your-server`)
3. Test: `xclock &` (should show clock window)
4. Launch GUI: `python3 transcription_gui_plugin.py`

**Performance**: Good over LAN/local WiFi, slower over internet connections. Enable compression for best results.

### Option 3: VNC (Alternative for Slow Connections)

**Best for**: When X11 is too slow (poor internet), extended GUI sessions, session persistence

```bash
# On server
vncserver :1 -geometry 1920x1080

# Connect with any VNC viewer to: your-server:5901
# (or tunnel it: ssh -L 5901:localhost:5901 user@your-server)
```

**Benefits**: Better compression than X11, survives disconnects, works well over internet

### Comparison

| Method | Responsiveness | Best For | Network Type |
|--------|-------|----------|--------------|
| CLI Batch Processing | high | Production, automation | Any |
| **Web UI** | high | **Interactive work, no install needed** | Any (SSH tunnel) |
| X11 Forwarding | medium | Interactive GUI work | LAN/Local WiFi |
| X11 Forwarding | low | Light use only | Internet |
| VNC/NoMachine | medium | Extended sessions, poor connections | Any |

---

## ⚙️ Configuration

### CRNN-CTC Training Parameters

Key hyperparameters for optimal performance:

```python
{
    "img_height": 128,           # Target image height
    "batch_size": 32,            # for a GPU with ~40 GB VRAM; reduce on smaller GPUs
    "num_epochs": 250,           # With early stopping
    "learning_rate": 0.0003,
    "early_stopping_patience": 15,
    "augment_train": True,       # Data augmentation
    "device": "cuda:0"
}
```

### TrOCR Training Configuration

```yaml
model_name: "kazars24/trocr-base-handwritten-ru"   # base model; for Latin script e.g. microsoft/trocr-base-handwritten
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

Copyright (c) 2025–2026 Achim Rabus

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
- **Kraken**: Segmentation and recognition: https://github.com/mittagessen/kraken
- **kraken 7 PP-OCRv6 base models** (Apache-2.0): https://doi.org/10.5281/zenodo.21788410; the architecture follows PaddleOCR's PP-OCR: https://github.com/PaddlePaddle/PaddleOCR
- **Transkribus**: Transcription, training, and inference platform: https://app.transkribus.org/
- **Qwen3-VL**: Alibaba's Vision-Language Model: https://github.com/QwenLM/Qwen3-VL
- **GLM-OCR**: https://huggingface.co/zai-org/GLM-OCR
- **dots.mocr**: https://huggingface.co/rednote-hilab/dots.mocr
- **Test data** for the engine comparison: CATMuS-Modern (https://huggingface.co/datasets/CATMuS/modern) and GT4HistOCR (Springmann et al. 2018, https://zenodo.org/records/1344132)
- **William Mattingly**: Support with VLM fine-tuning and Church Slavonic models: https://huggingface.co/wjbmattingly

---

## 📧 Contact

For questions, bug reports, or collaboration inquiries:
- GitHub Issues: [Create an issue](https://github.com/achimrabus/polyscriptor/issues)

---

## 🔬 Technical Notes

### Critical Preprocessing for Line Models (CRNN-CTC, TrOCR)

**Aspect Ratio Preservation** is CRITICAL for high aspect ratio line images:

```bash
# ALWAYS use --preserve-aspect-ratio for manuscript lines
python3 transkribus_parser.py \
    --preserve-aspect-ratio \
    --target-height 128 \
    # ...other args
```

Without this, TrOCR's ViT encoder brutally resizes to 384×384, causing 10.6x width compression for Ukrainian lines (4077×357 → 384×384). Characters shrink from ~80px to ~7px width, making recognition nearly impossible.
