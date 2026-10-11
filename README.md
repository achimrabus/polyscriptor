# Polyscriptor

Multi-engine HTR toolkit for historical manuscripts: segment, transcribe,
compare and export through a modern web interface, a unified CLI, or reusable
Python modules. Built around Cyrillic handwriting — Russian, Ukrainian, Church
Slavonic, Glagolitic — but usable for any script the bundled engines support.

**Web UI** · **14 pluggable engines** · **TrOCR & CRNN-CTC training pipelines**
· **Transkribus/ALTO data preparation** · **PAGE XML in and out**

---

## Features

**Web interface** (`polyscriptor serve`) — a single local process, no Node.js
needed at runtime:

- Upload images or PDFs, segment with Kraken neural (blla), Kraken classical,
  or HPP — or import PAGE XML segmentation
- Draw-a-box re-segmentation: mark a bad region and re-run blla just inside it
- Engine pool with per-engine configuration forms and VRAM-aware loading
- Live transcription progress over SSE, per-line confidence, inline correction
- Editable text, dim-by-confidence view, export to TXT, CSV or PAGE XML
- Batch queue (upload → segment → transcribe a folder, ZIP export)
- Engine comparison with ground-truth scoring (CER/WER leaderboard per engine)
- API-key protection for shared deployments, dark and light themes

**CLI** — one entry point for everything (`polyscriptor --help`):

- `polyscriptor batch` — batch-transcribe folders with resume, dry-run, presets
- `polyscriptor segment` — batch line segmentation to PAGE XML
- `polyscriptor train-crnn` / `polyscriptor train-trocr` — GPU training pipelines
- `polyscriptor transkribus` / `polyscriptor alto` / `polyscriptor pylaia-convert` — dataset preparation
- `polyscriptor infer-page` — segment + transcribe a single page

**Trained model presets** — the engine forms offer published Cyrillic/Church
Slavonic TrOCR checkpoints (e.g. `cyrillic-trocr/trocr-church-slavonic-handwritten`,
val CER 4.99 %) and Kraken `.mlmodel` recognizers (Zenodo presets auto-download).

## Installation

Requires Python 3.10–3.12 (3.13+ may lack prebuilt wheels for some packages).

```bash
git clone https://github.com/achimrabus/polyscriptor.git
cd polyscriptor
python3 -m venv htr_env
source htr_env/bin/activate        # Windows: htr_env\Scripts\activate
pip install -e ".[kraken]"
```

The optional `kraken` extra adds Kraken neural (blla) segmentation — strongly
recommended; without it only the fast but much weaker HPP segmenter is available.

**GPU (CUDA 12.8 wheels):**

```bash
pip install -e ".[kraken]" --extra-index-url https://download.pytorch.org/whl/cu128
```

**CPU-only on Windows** (avoids a torch DLL load error):

```bash
pip install -e ".[kraken]" --extra-index-url https://download.pytorch.org/whl/cpu
```

For running the test suite add the `dev` extra: `pip install -e ".[kraken,dev]"`.

## Quick start

```bash
polyscriptor serve
```

Open <http://127.0.0.1:8765>, upload a page image, click **Segment**, pick an
engine under *Engine*, load a model, and press **Ctrl+Enter** (or the Transcribe
button) to transcribe every line. Then export, compare against another engine,
or fix lines by hand.

### Running on a server (SSH tunnel)

The server binds to `127.0.0.1` by default — on a remote machine, keep it that
way and forward the port instead of exposing it:

```bash
ssh -L 8765:127.0.0.1:8765 user@yourserver
```

Then open <http://127.0.0.1:8765> locally. Use `--host 0.0.0.0 --port 8765`
only behind a reverse proxy or firewall, together with API keys
(see [Protecting a shared deployment](#protecting-a-shared-deployment)).

### Keyboard shortcuts

| Keys | Action |
|---|---|
| `Ctrl+Enter` | Run transcription on the current image |
| `Esc` | Clear drawn re-segmentation boxes |

## The web interface

- **Toolbar** — file upload (images and PDFs), segmentation method, device
  (`cpu`/`cuda:0`), reading direction (`horizontal-lr/-rl`, `vertical`), and an
  optional custom Kraken `.mlmodel` path.
- **Engine panel** (left) — engine select, its configuration form (beam search,
  prompts, decoding options…), *Load model* / *Unload*. Models load on demand
  and are pooled with VRAM-aware eviction.
- **Page view** (centre) — zoom/pan canvas with region (green) and line (blue)
  overlays. **Box re-seg** lets you drag rectangles; only regions inside them
  are re-segmented with blla (the drawn rectangle replaces them entirely).
  Regions can also be deleted wholesale from the dropdown.
- **Transcription panel** (right) — one editable field per line with a
  confidence dot, dim-by-confidence slider, TXT/CSV/PAGE-XML export.
- **Tabs** — *Workspace*, *Batch* (folder queue + ZIP export), and *Compare*
  (run the current segmentation with several engines, upload ground-truth
  PAGE XML and score them on a CER/WER leaderboard).

The previous single-file UI remains reachable at `/static/index.html`.

## Engines

| Engine | Type | Notes |
|---|---|---|
| **TrOCR** | line HTR | fine-tuned Cyrillic presets on Hugging Face, auto-download |
| **CRNN-CTC (PyLaia-inspired)** | line HTR | Puigcerver CRNN, beam/CTC decoding, LM support |
| **Kraken** | line HTR | `.mlmodel` recognizers, Zenodo presets (e.g. CATMuS-Print) |
| **Qwen3-VL** | page/line VLM | custom prompt presets, sampling controls |
| **Churro VLM** | page/line VLM | Qwen fork, experimental |
| **LightOnOCR** | line VLM | lightweight (~4 GB VRAM) |
| **Party** | line HTR | multilingual transformer |
| **OpenWebUI** | VLM via API | any vision model on a local OpenWebUI |
| **Commercial APIs** | VLM via API | Google Gemini, Anthropic Claude, OpenAI |
| **DeepSeek-OCR** | page OCR | runs in its own venv (subprocess-isolated) |
| **PaddleOCR** | page OCR/detection | runs in its own venv (subprocess-isolated) |
| **PaddleOCR-VL** | page VLM OCR | runs in its own venv (subprocess-isolated) |
| **Chandra** | document VLM | runs in its own venv (subprocess-isolated) |
| **LapaOCR** | document VLM | 8-bit/4-bit quantization options |

Segmentation engines (used by both UI and CLI): **Kraken neural (blla)** — the
bundled `blla.mlmodel` for multilingual text + column layout; **Kraken
classical** (CPU, no download); **HPP** projection profiles (fast, weak).

## CLI usage

All commands print full flags with `polyscriptor <command> --help`.

```bash
# Batch-transcribe a folder with TrOCR using existing PAGE XML segmentation
polyscriptor batch --input-folder pages/ --engine TrOCR \
    --model-id cyrillic-trocr/trocr-ukrainian-handwritten \
    --use-pagexml --device cuda --output-format txt pagexml --resume

# Segment a folder to PAGE XML with the blla model (+ overlay images for QC)
polyscriptor segment --input pages/ --output pagexml/ --overlays overlays/ --device cuda

# Fine-tune a TrOCR model from a YAML config
polyscriptor train-trocr --config config.yaml

# Prepare a training dataset from Transkribus exports
polyscriptor transkribus --input_dir export/ --output_dir dataset/
```

`batch` supports `--dry-run` (inspect the plan before running), `--resume`
(skip already-processed images), `--segmentation-method`, per-engine decoding
flags (`--num-beams`, `--no-repeat-ngram-size`, `--temperature`…), and Rich
progress bars.

## Training & datasets

- **CRNN-CTC**: `polyscriptor train-crnn` — CRNN + CTC training from a line
  dataset (PAGE XML or the PyLaia format), word list + optional KenLM LM.
- **TrOCR**: `polyscriptor train-trocr` — fine-tuning via YAML config with
  dataset image caching (10–50× faster epochs) and validation CER reporting.
- **Datasets**: `polyscriptor transkribus` (PAGE XML) and `polyscriptor alto`
  (ALTO XML) crop line images + metadata; `polyscriptor pylaia-convert` turns
  CSV datasets into the CRNN-CTC format. GPU required for training.

## Protecting a shared deployment

Set `--host 0.0.0.0` behind a TLS reverse proxy and manage API keys:

```bash
python -m polyscriptor.web.generate_api_key --name alice       # create a key
python -m polyscriptor.web.generate_api_key --list             # list keys
```

Only SHA-256 hashes are stored (`web/api_keys.yaml`, override with
`POLYSCRIPTOR_API_KEYS_FILE`). Requests carry the key in the `X-API-Key`
header — the web UI asks for it once and keeps it in the browser's
localStorage.

## REST API

`polyscriptor serve` exposes a JSON API with server-sent-events streaming
(OpenAPI schema at `/openapi.json`, interactive docs at `/docs`).
Key endpoints: `POST /api/image/upload`, `GET /api/image/{id}/segment`,
`POST /api/image/{id}/resegment`, `POST /api/engine/load`,
`POST /api/transcribe` (SSE), `POST /api/compare/run`,
`POST /api/compare/ground-truth`, `POST /api/batch/export-zip`.

## Frontend development

The web UI is a SvelteKit SPA (Svelte 5 + TypeScript + Tailwind CSS 4) in
[`frontend/`](frontend/), built to static files and served by the backend:

```bash
cd frontend
npm install
npm run dev     # Vite dev server with hot reload, proxies /api to 127.0.0.1:8765
npm run build   # output in frontend/build/, picked up by `polyscriptor serve`
```

Run `polyscriptor serve` alongside `npm run dev` during frontend work, or just
serve the built output for normal use.

## Development

```bash
pip install -e ".[dev]"
python -m pytest polyscriptor/web/tests -q
```

### Project layout

```
polyscriptor/
├── core/          # engine base + registry, segmentation, PAGE XML, imaging, metrics
├── engines/       # the 14 HTR engine plugins (+ venv-isolated workers)
├── inference/     # page pipeline, pylaia-native CRNN, qwen3, commercial API clients
├── data/          # Transkribus/ALTO parsers, pylaia conversion, blla model
├── training/      # CRNN-CTC trainer, TrOCR trainer, KenLM trainer
├── cli/           # Click CLI (serve + batch/segment/train/parse commands)
└── web/           # FastAPI app, API-key auth, key manager, tests, legacy static UI
frontend/          # SvelteKit SPA (built into frontend/build, served at /)
```

## Credits

Polyscriptor is written and maintained by **Achim Rabus**.

The 2026 restructuring — packaging into a `polyscriptor` package with a Click CLI,
removal of the PyQt6 GUI, the SvelteKit web interface, the honest error channel,
and the CI setup — was done by **`johnlockejrr`** using [ZCode](https://zcode.ai)
with the Qwen38-Flash-Next model.

## License

MIT — see [LICENSE](LICENSE).
