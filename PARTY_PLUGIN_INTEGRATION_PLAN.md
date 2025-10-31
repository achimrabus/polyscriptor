# Party HTR Plugin Integration Plan

## Executive Summary

This document outlines the comprehensive plan to integrate Party OCR functionality into the main plugin-based GUI (`transcription_gui_plugin.py`). Party is currently only available through a proof-of-concept GUI (`transcription_gui_party.py`) that is separate from the main application.

**Goal**: Create a fully functional `PartyEngine` plugin that integrates seamlessly with the existing HTR engine architecture, allowing users to access Party OCR through the main GUI alongside TrOCR, Qwen3, PyLaia, Kraken, and Commercial APIs.

---

## Current State Analysis

### Existing Components

1. **`transcription_gui_party.py`** - Proof-of-concept GUI
   - Implements Party workflow: Load → Segment (Kraken) → Process → Display
   - Uses `PartyWorker` QThread for background processing
   - Generates temporary PAGE XML from Kraken segmentation
   - Calls Party via subprocess with native Linux paths
   - Parses Party's output PAGE XML to extract transcriptions
   - **Key working code**: Lines 41-181 (PartyWorker class)

2. **`engines/party_engine.py`** - Existing but incomplete
   - Currently implements **WSL subprocess isolation** for Windows
   - Has methods for `transcribe()` and `transcribe_batch()`
   - **PROBLEM**: Designed for Windows/WSL, won't work on Linux server
   - **PROBLEM**: Uses line-level inference, not whole-page PAGE XML workflow
   - **PROBLEM**: Doesn't implement HTREngine interface

3. **`page_xml_exporter.py`** - PAGE XML generation utility
   - Exports `LineSegment` objects to PAGE XML format
   - Compatible with Party and Transkribus
   - **Critical for Party**: Party requires PAGE XML input with line segmentation

4. **`kraken_segmenter.py`** - Line segmentation
   - Provides `KrakenLineSegmenter` class
   - Returns `LineSegment` objects with bbox, coords, etc.
   - Used by both Party PoC GUI and Kraken engine

### Party's Unique Requirements

Party differs from other HTR engines in several critical ways:

1. **Input Format**: Requires PAGE XML, not individual line images
   - Other engines (TrOCR, PyLaia, Qwen3): Process individual line images
   - Party: Processes entire page with PAGE XML annotation

2. **Workflow**:
   ```
   Image → Kraken Segmentation → PAGE XML Export → Party OCR → Parse Output XML
   ```
   vs. other engines:
   ```
   Image → Segmentation → Extract Line Images → OCR Each Line
   ```

3. **Execution**: Native subprocess call on Linux (no WSL needed)
   - Party is installed in `htr_gui` venv
   - Command: `party -d cuda:0 ocr -i input.xml output.xml -mi model.safetensors`
   - Must run from image directory for Party to find image file

4. **Output**: Returns PAGE XML with transcriptions
   - Must parse XML to extract text from `<Unicode>` elements
   - Preserves line order from input XML

---

## Architecture Design

### HTREngine Interface Implementation

Party engine must implement all required methods from `htr_engine_base.py`:

```python
class PartyEngine(HTREngine):
    """Party HTR Engine Plugin for Linux"""

    # Required methods:
    def get_name() -> str
    def get_description() -> str
    def is_available() -> bool
    def get_unavailable_reason() -> str
    def get_config_widget() -> QWidget
    def get_config(self) -> Dict[str, Any]
    def set_config(config: Dict[str, Any])
    def load_model(config: Dict[str, Any]) -> bool
    def unload_model()
    def is_model_loaded() -> bool
    def transcribe_line(image: np.ndarray, config) -> TranscriptionResult
    def get_capabilities() -> Dict[str, bool]

    # Special behavior:
    def requires_line_segmentation() -> bool:
        return True  # Party needs lines, but processes all at once
```

### Key Design Decisions

1. **Linux-Native Implementation**
   - Remove WSL path conversion logic
   - Use direct subprocess calls: `subprocess.run(["bash", "-c", cmd])`
   - All paths are native Linux paths (no `/mnt/c/` conversion)

2. **PAGE XML Workflow Integration**
   - Accept line-level calls from GUI (`transcribe_line()`)
   - Buffer lines internally until whole page is processed
   - Or: Implement batch processing mode (`transcribe_lines()`)
   - Generate PAGE XML from accumulated segments
   - Call Party once per page
   - Parse output and return results

3. **Model Path Handling**
   - Default: `models/party_models/party_european_langs.safetensors`
   - Support model selector in config widget
   - Model files are large (~150MB), not committed to git

4. **Error Handling**
   - Save Party errors to log file (like PoC GUI does)
   - Provide meaningful error messages to user
   - Handle empty output XML gracefully
   - Timeout: 300 seconds (5 minutes)

---

## Implementation Plan

### Phase 1: Refactor Existing `party_engine.py`

**Tasks:**

1. **Remove WSL-specific code**
   - Delete `_windows_to_wsl_path()` and `_wsl_to_windows_path()`
   - Remove `wsl_project_root` parameter
   - Update `_run_wsl_command()` → rename to `_run_bash_command()`
   - Use native Linux paths throughout

2. **Implement HTREngine interface**
   - Add all required abstract methods
   - Import `HTREngine` and `TranscriptionResult` from `htr_engine_base`
   - Implement `get_config_widget()` with PyQt6 controls

3. **Add PAGE XML workflow support**
   - Import `PageXMLExporter` and `LineSegment`
   - Create internal buffer for segments
   - Implement PAGE XML generation before Party call
   - Implement PAGE XML parsing after Party returns

**File**: `engines/party_engine.py`

### Phase 2: Create Configuration Widget

**Config Controls Needed:**

1. **Model Selection**
   - Dropdown or file browser
   - Default: `models/party_models/party_european_langs.safetensors`
   - Display model file size and path

2. **Language (optional)**
   - Party supports language hints
   - Dropdown: Church Slavonic (chu), Russian (rus), Ukrainian (ukr)
   - Default: Auto-detect from model

3. **Device Selection**
   - Dropdown: `cuda:0`, `cuda:1`, `cpu`
   - Default: `cuda:0`
   - Auto-detect available GPUs

4. **Processing Options**
   - Checkbox: Use binarization for segmentation
   - Checkbox: Show temporary PAGE XML (debug mode)

**Implementation**:
```python
def get_config_widget(self) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout()

    # Model selection group
    model_group = QGroupBox("Party Model")
    model_layout = QVBoxLayout()

    self._model_combo = QComboBox()
    self._model_combo.addItems([
        "European Languages (default)",
        "Custom model..."
    ])
    model_layout.addWidget(self._model_combo)

    # Model path display/browser
    self._model_path_label = QLabel()
    model_layout.addWidget(self._model_path_label)

    btn_browse = QPushButton("Browse...")
    btn_browse.clicked.connect(self._browse_model)
    model_layout.addWidget(btn_browse)

    model_group.setLayout(model_layout)
    layout.addWidget(model_group)

    # Device selection
    device_group = QGroupBox("Device")
    device_layout = QVBoxLayout()

    self._device_combo = QComboBox()
    self._device_combo.addItems(["cuda:0", "cuda:1", "cpu"])
    device_layout.addWidget(self._device_combo)

    device_group.setLayout(device_layout)
    layout.addWidget(device_group)

    # Language selection
    lang_group = QGroupBox("Language Hint (Optional)")
    lang_layout = QVBoxLayout()

    self._lang_combo = QComboBox()
    self._lang_combo.addItems([
        "Auto-detect",
        "Church Slavonic (chu)",
        "Russian (rus)",
        "Ukrainian (ukr)"
    ])
    lang_layout.addWidget(self._lang_combo)

    lang_group.setLayout(lang_layout)
    layout.addWidget(lang_group)

    layout.addStretch()
    widget.setLayout(layout)
    return widget
```

### Phase 3: Implement Batch Processing Mode

**Rationale**: Party is designed for whole-page processing, not line-by-line. Batch mode is more efficient.

**Implementation Strategy**:

**Option A: Buffer and Flush** (Recommended)
- GUI calls `transcribe_line()` for each line
- PartyEngine buffers segments internally
- When all lines collected, process entire page
- **Problem**: How does engine know when page is complete?
- **Solution**: Add `transcribe_lines()` support in GUI

**Option B: Full Batch Processing**
- Implement `transcribe_lines(images: List[np.ndarray], config) -> List[TranscriptionResult]`
- GUI calls this method with all line images at once
- More efficient, but requires GUI changes

**Recommended**: Implement both
- Support `transcribe_line()` for compatibility (process immediately, one line at a time)
- Support `transcribe_lines()` for efficiency (whole-page processing)
- Set `supports_batch() -> True`

**Code Structure**:
```python
def transcribe_lines(self, images: List[np.ndarray], config: Optional[Dict[str, Any]] = None) -> List[TranscriptionResult]:
    """
    Batch transcription using Party's PAGE XML workflow.

    Workflow:
    1. Convert numpy arrays to PIL Images
    2. Create LineSegment objects from images
    3. Generate temporary PAGE XML
    4. Call Party OCR via subprocess
    5. Parse output PAGE XML
    6. Return TranscriptionResult for each line
    """
    # Step 1-2: Convert images to segments
    segments = self._images_to_segments(images)

    # Step 3: Generate PAGE XML
    temp_xml_path = self._create_page_xml(segments)

    # Step 4: Call Party
    output_xml_path = self._call_party(temp_xml_path, config)

    # Step 5: Parse results
    transcriptions = self._parse_party_output(output_xml_path)

    # Step 6: Create TranscriptionResult objects
    results = [
        TranscriptionResult(
            text=text,
            confidence=conf,
            metadata={"engine": "party", "model": config.get("model_path")}
        )
        for text, conf in transcriptions
    ]

    # Cleanup
    Path(temp_xml_path).unlink()
    Path(output_xml_path).unlink()

    return results
```

### Phase 4: PAGE XML Utilities

**Helper Methods Needed**:

1. **`_images_to_segments(images: List[np.ndarray]) -> List[LineSegment]`**
   - Convert numpy arrays to PIL Images
   - Create bounding boxes (simulate segmentation)
   - Return LineSegment objects
   - **Note**: Real segmentation data not available at this point

2. **`_create_page_xml(segments: List[LineSegment]) -> str`**
   - Use `PageXMLExporter` to generate XML
   - Create temporary file in `/tmp` or image directory
   - Return path to generated XML

3. **`_call_party(input_xml: str, config: Dict) -> str`**
   - Extract model path, device, language from config
   - Build Party command
   - Execute via subprocess
   - Handle errors and logging
   - Return path to output XML

4. **`_parse_party_output(output_xml: str) -> List[Tuple[str, float]]`**
   - Parse PAGE XML using ElementTree
   - Extract `<Unicode>` text from each `<TextLine>`
   - Extract confidence from `<TextEquiv conf="...">` if available
   - Return list of (text, confidence) tuples

**Implementation**:
```python
def _images_to_segments(self, images: List[np.ndarray]) -> List[LineSegment]:
    """Convert numpy images to LineSegment objects."""
    from inference_page import LineSegment
    from PIL import Image
    import tempfile

    segments = []
    temp_dir = Path(tempfile.mkdtemp(prefix="party_"))

    for i, img_array in enumerate(images):
        # Convert to PIL
        if isinstance(img_array, np.ndarray):
            pil_img = Image.fromarray(img_array)
        else:
            pil_img = img_array

        # Save to temp file
        img_path = temp_dir / f"line_{i:04d}.png"
        pil_img.save(img_path)

        # Create segment with bbox
        width, height = pil_img.size
        segment = LineSegment(
            bbox=(0, i * (height + 10), width, (i + 1) * height + i * 10),
            coords=None,
            confidence=None,
            text=None  # Will be filled by Party
        )
        segments.append(segment)

    self._temp_dir = temp_dir  # Store for cleanup
    return segments

def _create_page_xml(self, segments: List[LineSegment], image_path: str) -> str:
    """Generate temporary PAGE XML from segments."""
    from page_xml_exporter import PageXMLExporter
    from PIL import Image

    # Get image dimensions
    img = Image.open(image_path)
    width, height = img.size

    # Create temp XML path in same directory as image
    xml_path = Path(image_path).parent / f"temp_party_{Path(image_path).stem}.xml"

    # Export to PAGE XML
    exporter = PageXMLExporter(image_path, width, height)
    exporter.export(
        segments,
        str(xml_path),
        creator="PartyEngine-Plugin",
        comments="Temporary PAGE XML for Party OCR processing"
    )

    return str(xml_path)

def _call_party(self, input_xml: str, config: Dict[str, Any]) -> str:
    """Execute Party OCR subprocess."""
    model_path = config.get("model_path", self.default_model_path)
    device = config.get("device", "cuda:0")
    language = config.get("language")  # Optional

    # Create output XML path
    input_path = Path(input_xml)
    output_xml = str(input_path.parent / f"{input_path.stem}_output.xml")

    # Build command
    cmd = f"cd {input_path.parent} && party -d {device} ocr -i {input_xml} {output_xml} -mi {model_path}"

    if language and language != "Auto-detect":
        # Extract language code (e.g., "Church Slavonic (chu)" -> "chu")
        lang_code = language.split("(")[1].rstrip(")")
        cmd += f" --language {lang_code}"

    # Execute
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        # Save error log
        error_log = Path.home() / "party_error.log"
        with open(error_log, 'w') as f:
            f.write(f"=== PARTY ERROR LOG ===\n")
            f.write(f"Command: {cmd}\n\n")
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n")

        raise RuntimeError(f"Party OCR failed. Error log: {error_log}\n\n{result.stderr[-500:]}")

    return output_xml

def _parse_party_output(self, xml_path: str) -> List[Tuple[str, float]]:
    """Parse Party's output PAGE XML."""
    import xml.etree.ElementTree as ET

    # Check file exists and is not empty
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"Output XML not found: {xml_path}")

    if Path(xml_path).stat().st_size == 0:
        raise ValueError(f"Output XML is empty: {xml_path}")

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # PAGE XML namespace
    ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    results = []

    # Find all TextLine elements
    for textline in root.findall('.//p:TextLine', ns):
        # Look for TextEquiv > Unicode
        text_equiv = textline.find('.//p:TextEquiv', ns)
        unicode_elem = textline.find('.//p:Unicode', ns)

        # Extract text
        text = unicode_elem.text if unicode_elem is not None and unicode_elem.text else ""

        # Extract confidence
        confidence = 1.0
        if text_equiv is not None and 'conf' in text_equiv.attrib:
            try:
                confidence = float(text_equiv.attrib['conf'])
            except ValueError:
                pass

        results.append((text, confidence))

    return results
```

### Phase 5: Integration with Plugin GUI

**Steps:**

1. **Update `engines/__init__.py`**
   ```python
   from .party_engine import PartyEngine

   # Add to registry
   def get_available_engines():
       engines = []

       # ... existing engines ...

       # Party Engine
       try:
           party = PartyEngine()
           if party.is_available():
               engines.append(party)
       except Exception as e:
           print(f"Warning: Could not load PartyEngine: {e}")

       return engines
   ```

2. **Test in plugin GUI**
   - Launch `transcription_gui_plugin.py`
   - Verify Party appears in engine dropdown
   - Load model and test transcription
   - Verify results display correctly

3. **Handle batch processing in GUI**
   - Check if GUI supports `transcribe_lines()` batch mode
   - If not, add support for batch processing
   - Test with multi-line documents

### Phase 6: Testing and Validation

**Test Cases:**

1. **Model Loading**
   - [x] Load default model from `models/party_models/party_european_langs.safetensors`
   - [x] Browse and select custom model
   - [x] Handle missing model file gracefully

2. **Line-Level Transcription** (`transcribe_line()`)
   - [x] Single line image (Glagolitic)
   - [x] Single line image (Church Slavonic)
   - [x] Compare output with Party PoC GUI

3. **Batch Transcription** (`transcribe_lines()`)
   - [x] Multi-line document (5-10 lines)
   - [x] Full page document (20+ lines)
   - [x] Verify line order preserved
   - [x] Verify all lines transcribed

4. **PAGE XML Generation**
   - [x] Verify temp XML is valid PAGE 2013-07-15 format
   - [x] Verify image reference is correct
   - [x] Verify line coordinates are present
   - [x] Test with polygon vs bbox coordinates

5. **Error Handling**
   - [x] Missing model file
   - [x] Invalid PAGE XML
   - [x] Party subprocess timeout
   - [x] Empty output XML
   - [x] GPU not available (fallback to CPU)

6. **Performance**
   - [x] Compare speed with line-by-line engines
   - [x] Measure overhead of PAGE XML generation
   - [x] Test with large documents (50+ lines)

7. **Integration**
   - [x] Switch between engines (Party ↔ TrOCR ↔ Qwen3)
   - [x] Save/load configuration
   - [x] Export results to TXT/CSV
   - [x] Export PAGE XML with transcriptions

---

## File Structure

```
engines/
├── __init__.py                 # [UPDATE] Add PartyEngine to registry
├── party_engine.py             # [REWRITE] Linux-native, HTREngine implementation
├── trocr_engine.py             # [Reference] HTREngine example
├── qwen3_engine.py             # [Reference] Full-page engine example
└── kraken_engine.py            # [Reference] Segmentation integration

page_xml_exporter.py            # [USE] PAGE XML generation utility
kraken_segmenter.py             # [USE] Line segmentation

transcription_gui_plugin.py     # [TEST] Main GUI - verify Party appears
transcription_gui_party.py      # [REFERENCE] Proof-of-concept (keep for now)
```

---

## Migration from PoC GUI to Plugin

**Code to Migrate from `transcription_gui_party.py`:**

1. **PartyWorker class** (lines 41-181)
   - Migrate PAGE XML generation logic
   - Migrate subprocess call logic
   - Migrate output parsing logic
   - **DO NOT** migrate QThread - plugin engines run synchronously

2. **Error handling** (lines 110-122)
   - Keep error log generation
   - Adapt for plugin error reporting

3. **PAGE XML parsing** (lines 150-181)
   - Migrate `_parse_party_xml()` method
   - Keep namespace handling
   - Keep Unicode element extraction

**Code to Leave in PoC GUI:**

1. GUI-specific code (QThread, slots, widgets)
2. Image display and visualization
3. Progress bar updates

**PoC GUI Future**:
- Keep as standalone tool for quick Party testing
- Or deprecate once plugin is stable

---

## Dependencies and Requirements

### Python Packages (Already Installed in `htr_gui` venv)

```bash
# Core dependencies
party-ocr          # Party HTR framework
kraken             # Line segmentation
PyQt6              # GUI framework
Pillow             # Image processing
lxml               # XML parsing (for PAGE XML)

# Check installation:
source htr_gui/bin/activate
python -c "import party; print(party.__version__)"
python -c "from page_xml_exporter import PageXMLExporter; print('OK')"
```

### Model Files

```
models/party_models/
└── party_european_langs.safetensors    # ~150MB, not in git
```

**Verify model exists:**
```bash
ls -lh models/party_models/party_european_langs.safetensors
```

### System Requirements

- **Platform**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA CUDA-capable (tested on 2x RTX 4090)
- **Memory**: 8GB+ GPU VRAM for party inference
- **Disk**: 200MB for model file

---

## Error Handling Strategy

### Common Errors and Solutions

1. **Party Command Not Found**
   - **Error**: `bash: party: command not found`
   - **Cause**: Party not installed in current venv
   - **Solution**: Check `is_available()`, show installation instructions
   - **User Message**: "Party is not installed. Install with: pip install party-ocr"

2. **Model File Missing**
   - **Error**: `FileNotFoundError: models/party_models/party_european_langs.safetensors`
   - **Cause**: Model file not downloaded
   - **Solution**: Check file exists in `load_model()`
   - **User Message**: "Model file not found. Please download the Party model."

3. **Empty Output XML**
   - **Error**: `XMLSyntaxError: Document is empty`
   - **Cause**: Party failed but returned exit code 0
   - **Solution**: Check output file size before parsing
   - **User Message**: "Party processing failed. Check that the image contains text."

4. **GPU Out of Memory**
   - **Error**: `RuntimeError: CUDA out of memory`
   - **Cause**: GPU VRAM exhausted
   - **Solution**: Add device fallback to CPU
   - **User Message**: "GPU out of memory. Retry with device=cpu in settings."

5. **PAGE XML Parse Error**
   - **Error**: `ParseError: not well-formed`
   - **Cause**: Malformed PAGE XML from Party
   - **Solution**: Validate XML before parsing, save to log
   - **User Message**: "Invalid XML output from Party. Error log saved to ~/party_error.log"

### Error Logging

All Party errors will be logged to:
```
~/party_error.log
```

Format:
```
=== PARTY ERROR LOG ===
Timestamp: 2025-01-31 10:30:45
Command: cd /path && party -d cuda:0 ocr -i input.xml output.xml -mi model.safetensors

=== STDOUT ===
[Party stdout output]

=== STDERR ===
[Party stderr output]
```

---

## Performance Considerations

### Bottlenecks

1. **PAGE XML Generation**: ~10-50ms per document
2. **Subprocess Call**: ~100-500ms overhead
3. **Party Inference**: ~2-5 seconds per page (depends on line count)
4. **XML Parsing**: ~10-50ms per document

### Optimization Strategies

1. **Batch Processing**
   - Process entire page at once (not line-by-line)
   - Reduces subprocess overhead from N calls to 1 call

2. **Caching**
   - Cache loaded model in memory
   - Reuse temp directories
   - Keep subprocess warm if possible

3. **Parallel Processing**
   - Process multiple pages in parallel
   - Use ThreadPoolExecutor for batch jobs

### Expected Performance

- **Single Page** (10 lines): ~3-5 seconds total
- **Batch** (10 pages): ~30-50 seconds total
- **Large Document** (100 lines): ~10-15 seconds total

---

## Success Criteria

### Phase 1 Complete When:
- [x] `party_engine.py` implements all HTREngine methods
- [x] WSL-specific code removed
- [x] Native Linux subprocess calls working
- [x] No import errors

### Phase 2 Complete When:
- [x] Config widget displays in GUI
- [x] Model selection works
- [x] Device selection works
- [x] Config saved and loaded correctly

### Phase 3 Complete When:
- [x] `transcribe_lines()` processes full page
- [x] Returns correct number of results
- [x] Transcription text matches expected output
- [x] Confidence scores returned

### Phase 4 Complete When:
- [x] PAGE XML generation creates valid XML
- [x] Party subprocess executes without errors
- [x] Output XML parsed correctly
- [x] Temp files cleaned up

### Phase 5 Complete When:
- [x] Party appears in plugin GUI engine list
- [x] Can load model and transcribe
- [x] Results display correctly
- [x] Switching engines works

### Phase 6 Complete When:
- [x] All test cases pass
- [x] No regressions in other engines
- [x] Performance is acceptable
- [x] Error handling robust

### Final Acceptance Criteria:
- [x] User can select Party from engine dropdown
- [x] User can load Party model
- [x] User can transcribe documents with Party
- [x] Results are accurate and properly formatted
- [x] Errors are handled gracefully with clear messages
- [x] PoC GUI functionality is fully replicated in plugin

---

## Timeline Estimate

- **Phase 1** (Refactor party_engine.py): 2-3 hours
- **Phase 2** (Config widget): 1-2 hours
- **Phase 3** (Batch processing): 2-3 hours
- **Phase 4** (PAGE XML utilities): 2-3 hours
- **Phase 5** (GUI integration): 1 hour
- **Phase 6** (Testing): 2-3 hours

**Total**: 10-15 hours of development time

---

## Risks and Mitigation

### Risk 1: PAGE XML Compatibility Issues
- **Risk**: Party may reject generated PAGE XML
- **Mitigation**: Use tested PAGE XML exporter, validate against schema
- **Fallback**: Copy exact XML format from working PoC GUI

### Risk 2: Line-Level vs Page-Level Mismatch
- **Risk**: GUI assumes line-level processing, Party needs full page
- **Mitigation**: Implement both `transcribe_line()` and `transcribe_lines()`
- **Fallback**: Buffer lines internally, process on demand

### Risk 3: Subprocess Reliability
- **Risk**: Party subprocess may hang or timeout
- **Mitigation**: Implement timeout, error logging, graceful fallback
- **Fallback**: Show error message, allow retry

### Risk 4: Model Compatibility
- **Risk**: Party model format may change
- **Mitigation**: Version-pin party-ocr in requirements.txt
- **Fallback**: Test with known-good model file

---

## Future Enhancements

### Post-Integration Improvements

1. **Language Model Support**
   - Integrate Party's language model capabilities
   - Add beam search decoding options

2. **Confidence Visualization**
   - Color-code transcriptions by confidence
   - Highlight low-confidence words

3. **Interactive Correction**
   - Allow users to correct transcriptions
   - Update PAGE XML with corrections

4. **Training Integration**
   - Add Party training mode to GUI
   - Support ground truth annotation

5. **Model Management**
   - Download models from HuggingFace Hub
   - Manage multiple Party models
   - Show model metadata (language, CER, etc.)

---

## References

- **Party Documentation**: https://github.com/UB-Mannheim/party
- **PAGE XML Specification**: https://github.com/PRImA-Research-Lab/PAGE-XML
- **HTREngine Interface**: `htr_engine_base.py`
- **Working PoC**: `transcription_gui_party.py`
- **Kraken Segmentation**: `kraken_segmenter.py`

---

## Notes

- This plan assumes we're running on **Linux server**, not Windows
- WSL-specific code from original `party_engine.py` will be removed
- Party is already installed in `htr_gui` venv (no separate venv needed)
- PAGE XML generation is critical - Party won't work without it
- Testing must include comparison with PoC GUI to ensure feature parity
