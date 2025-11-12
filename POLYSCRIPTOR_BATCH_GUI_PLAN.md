# Polyscriptor Batch GUI - Implementation Plan

## Honest Assessment

**Does this make sense?** → **YES, BUT with caveats**

### Pros:
1. **Lower barrier to entry**: Non-technical users can run batch processing without CLI
2. **Visual feedback**: Progress bars, live statistics, image previews
3. **Configuration persistence**: Save/load processing presets
4. **Error prevention**: Input validation, model compatibility checks
5. **Leverages existing code**: batch_processing.py already has all the logic

### Cons:
1. **Maintenance overhead**: Another GUI to maintain alongside transcription_gui_plugin.py
2. **Redundancy**: Plugin GUI already supports single-image processing
3. **Limited value-add**: CLI already has dry-run, resume, verbose flags
4. **Target audience unclear**: Power users prefer CLI, beginners might prefer web interface

### Recommendation: **Minimal Qt6 GUI focused on essentials**

Build a **lightweight launcher** that generates and executes batch_processing.py commands, rather than duplicating functionality.

---

## Design Philosophy

**"CLI wrapper, not reimplementation"**

- GUI builds command-line arguments for batch_processing.py
- Executes via subprocess with live stdout/stderr capture
- Displays progress, logs, and final summary
- Saves configurations as .json presets

This approach:
- Avoids duplicating 1000+ lines of batch_processing.py logic
- Ensures CLI and GUI always behave identically
- Keeps GUI simple (~300-400 lines)
- Easy to maintain

---

## UI Layout (Single Window)

```
┌─────────────────────────────────────────────────────────────┐
│ Polyscriptor Batch - HTR Batch Processing                  │
├─────────────────────────────────────────────────────────────┤
│ ┌─ Input ─────────────────────────────────────────────────┐ │
│ │ Input Folder:  [HTR_Images/my_folder]      [Browse...]  │ │
│ │ Output Folder: [output]                    [Browse...]  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─ Engine Configuration ──────────────────────────────────┐ │
│ │ Engine:        [PyLaia ▼]                                │ │
│ │ Model Path:    [models/pylaia_church_slavonic...]        │ │
│ │                [Browse Local...] [HuggingFace Hub ID...] │ │
│ │ Device:        [cuda:0 ▼]                                │ │
│ │ Batch Size:    [32] (auto-optimized)                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─ Segmentation ──────────────────────────────────────────┐ │
│ │ Method:        [Kraken ▼] (HPP / None)                   │ │
│ │ ☑ Use PAGE XML (auto-detect from page/ folder)          │ │
│ │ Sensitivity:   [0.1] ──────────                          │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─ Output Options ────────────────────────────────────────┐ │
│ │ Formats:  ☑ TXT  ☐ CSV  ☐ PAGE XML                      │ │
│ │ ☑ Resume (skip processed)  ☐ Verbose logging            │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─ Presets ───────────────────────────────────────────────┐ │
│ │ [Church Slavonic (PyLaia) ▼] [Save] [Load] [Delete]     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ [Dry Run (Test First)]        [Start Batch Processing]      │
│                                                               │
│ ┌─ Progress ──────────────────────────────────────────────┐ │
│ │ [████████████████░░░░░░░░] 67% (200/300 images)          │ │
│ │ Current: 0145_manuscript_page.jpg                        │ │
│ │ Lines: 8,432 | Chars: 245,891 | Avg Conf: 87.3%          │ │
│ │ Time elapsed: 12m 34s | ETA: 6m 12s                      │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─ Log Output ────────────────────────────────────────────┐ │
│ │ Processing 0145_manuscript_page.jpg...                   │ │
│ │   Using PAGE XML: 0145_manuscript_page.xml               │ │
│ │   Processing 168 line(s)...                              │ │
│ │ ✓ 0145_manuscript_page.jpg (168 lines, 89.2% conf)       │ │
│ │ Processing 0146_manuscript_page.jpg...                   │ │
│ │   Segmented 142 lines (Kraken)                           │ │
│ │ [Auto-scroll ▼]                                          │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ [Stop Processing]  [View Output Folder]  [Close]            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Engine Selection
**Dropdown with engine-specific config:**
- PyLaia → Show model path browser, vocabulary file (optional)
- TrOCR → Show model_id field, num_beams spinner
- Qwen3-VL → Show HF model ID, warn about slowness, require confirmation checkbox
- Party → Show model path, language dropdown
- Kraken → Show model path

**Dynamic UI**: Show/hide engine-specific options based on selection

### 2. Segmentation Options
**Method dropdown:**
- Kraken (default) → Show sensitivity slider
- HPP → Show min_line_height, min_gap spinners
- None → Show info label "Assumes pre-segmented line images"

**PAGE XML checkbox:**
- Default: ON
- When disabled: gray out "Custom XML folder" field

### 3. Presets System
**JSON-based configuration storage:**
```json
{
  "name": "Church Slavonic (PyLaia)",
  "engine": "PyLaia",
  "model_path": "models/pylaia_church_slavonic_20251103_162857/best_model.pt",
  "segmentation_method": "kraken",
  "use_pagexml": true,
  "device": "cuda:0",
  "output_format": ["txt"]
}
```

**Presets location**: `~/.config/polyscriptor/batch_presets.json`

**Built-in presets:**
- Church Slavonic (PyLaia + Kraken + PAGE XML)
- Ukrainian (PyLaia + Kraken)
- Russian (TrOCR + HF model)
- Glagolitic (PyLaia + Kraken)

### 4. Live Progress Monitoring
**Subprocess stdout parsing:**
- Regex patterns to extract:
  - Current image name
  - Lines processed count
  - Average confidence
  - Progress percentage (X/Y images)

**Progress bar calculation:**
- Total images discovered at start
- Update after each "✓ image.jpg" completion line

**ETA estimation:**
- Track time per image
- Rolling average of last 10 images
- Multiply by remaining images

### 5. Dry Run Mode
**Button: "Dry Run (Test First)"**
- Executes with `--dry-run` flag
- Shows estimated time, first image test result
- User must click "Confirmed, Start" to proceed
- Prevents accidents (wrong folder, wrong model)

### 6. Error Handling
**Common errors with user-friendly messages:**
- Model file not found → "Browse to select a valid model file"
- CUDA out of memory → "Try reducing batch size or switching to CPU"
- No images found → "Check input folder path"
- Engine not available → "Install dependencies: pip install [package]"

---

## Implementation Strategy

### Phase 1: Core GUI (4-6 hours)
- [x] Basic window layout with Qt6
- [x] Engine dropdown + model path browser
- [x] Input/output folder browsers
- [x] Segmentation options
- [x] "Start" button → subprocess execution

### Phase 2: Progress Monitoring (2-3 hours)
- [x] QProcess for subprocess execution
- [x] Stdout/stderr capture and parsing
- [x] Progress bar updates
- [x] Live log display with auto-scroll

### Phase 3: Presets System (2 hours)
- [x] JSON save/load for configurations
- [x] Preset dropdown with built-in defaults
- [x] Add/delete custom presets

### Phase 4: Polish (2 hours)
- [x] Input validation (folder exists, model valid)
- [x] Engine-specific UI (show/hide fields)
- [x] Tooltips for all options
- [x] Error dialogs with helpful messages
- [x] "View Output Folder" button

**Total estimate: 10-13 hours** (2-3 coding sessions)

---

## Technical Architecture

### File: `polyscriptor_batch_gui.py`

**Key classes:**
1. **BatchConfigWidget**: Input/output/engine configuration panel
2. **BatchProgressWidget**: Progress bar, statistics, ETA display
3. **BatchLogWidget**: QTextEdit with auto-scroll, color-coded messages
4. **BatchProcessorWindow**: Main window orchestrating everything

**Subprocess execution:**
```python
class BatchProcessRunner(QObject):
    progress_update = pyqtSignal(dict)  # {image: str, lines: int, conf: float}
    log_message = pyqtSignal(str, str)  # (message, level: info/warning/error)
    finished = pyqtSignal(dict)  # {total_lines: int, avg_conf: float, errors: int}

    def start(self, args: Dict[str, Any]):
        cmd = self._build_command(args)
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self._parse_stdout)
        self.process.start("python", ["batch_processing.py"] + cmd)

    def _parse_stdout(self):
        output = self.process.readAllStandardOutput().data().decode()
        # Parse lines like "✓ image.jpg (168 lines, 89.2% conf)"
        # Emit progress_update signal
```

**Command builder:**
```python
def _build_command(self, config: Dict) -> List[str]:
    cmd = [
        "--input-folder", config["input_folder"],
        "--engine", config["engine"],
        "--device", config["device"],
    ]

    if config.get("model_path"):
        cmd += ["--model-path", config["model_path"]]
    elif config.get("model_id"):
        cmd += ["--model-id", config["model_id"]]

    if config["segmentation_method"] != "none":
        cmd += ["--segmentation-method", config["segmentation_method"]]

    if config.get("use_pagexml", True):
        cmd += ["--use-pagexml"]

    # ... etc

    return cmd
```

---

## Alternative: Web-based UI

**Consideration**: Could also build a Flask/FastAPI web interface

**Pros:**
- No Qt6 dependency
- Remote server access via browser
- Mobile-friendly
- Multiple users can submit jobs

**Cons:**
- Requires web server setup
- More complex deployment
- Overkill for single-user desktop use

**Verdict**: Qt6 GUI is simpler for desktop-only use case

---

## Decision: Should We Build This?

### My honest opinion:

**Build it if:**
1. You have non-technical collaborators who need batch processing
2. You frequently switch between different configurations
3. You want visual monitoring of long-running batches
4. The CLI feels cumbersome for your workflow

**Skip it if:**
1. You're comfortable with CLI (which you clearly are)
2. You use shell scripts for common configurations
3. Time is better spent on improving models or preprocessing
4. The plugin GUI already meets your needs

### Pragmatic middle ground:

**Build a MINIMAL version** (~4 hours):
- Just input/output folders, engine dropdown, model path
- "Start" button that opens a terminal running batch_processing.py
- No subprocess management, no progress parsing
- Essentially a "configuration form → command generator"

This gives 80% of the value with 20% of the effort.

---

## Your Call

What do you think? Should we:

**A)** Build the full GUI with progress monitoring (10-13 hours)
**B)** Build the minimal "command generator" version (4 hours)
**C)** Skip it and focus on other features

I'm leaning toward **B** - a simple launcher that makes it easy to configure and run batches, without reinventing the CLI's subprocess management.
