# Party GUI Integration Plan

## Overview

Integrate Party (multilingual page-wise HTR model) into the GUI plugin system by hiding the PAGE XML requirement from the user.

**User Workflow**: Load Image → Segment Lines → Select Party Engine → Process → View Results

**Internal Process**: GUI automatically creates temp PAGE XML → Calls Party via WSL → Parses output → Displays in GUI

---

## Architecture

### Current Situation

**Two GUIs exist:**
1. `transcription_gui_qt.py` - Original, hardcoded engines
2. `transcription_gui_plugin.py` - Plugin-based architecture (preferred)

**Party exists but not integrated:**
- `engines/party_engine.py` - WSL subprocess wrapper
- NOT registered as a plugin
- Requires PAGE XML input (incompatible with line-level plugin interface)

### Proposed Solution

**Adapter Pattern**: Create a plugin wrapper that:
1. Accepts line segments (like other plugins)
2. Internally generates temp PAGE XML
3. Calls Party via WSL subprocess
4. Parses output back to line transcriptions
5. Returns results to GUI

### Key Insight

Party processes whole pages via PAGE XML, but we can:
- Generate PAGE XML programmatically from segmentation data
- Use temp files (or in-memory if possible)
- Parse Party's output back to individual line transcriptions
- Make it appear as a line-level engine to the GUI

---

## Implementation Phases

### Phase 1: Proof-of-Concept GUI ✓ (CURRENT)

**Goal**: Standalone GUI to validate the approach

**File**: `transcription_gui_party.py` (separate, simple GUI)

**Features**:
- Load image
- Segment with Kraken
- Generate temp PAGE XML
- Call Party via WSL
- Display transcriptions

**Why separate GUI?**
- Quick to develop and test
- No risk to existing GUIs
- Easy to debug Party integration
- Can be discarded after validation

**Deliverables**:
- Working PoC GUI
- Validated temp PAGE XML approach
- Confirmed Party subprocess calls work
- Output parsing logic

---

### Phase 2: Plugin Architecture Design

**Goal**: Design clean plugin interface for Party

**File**: `engines/party_plugin.py`

**Interface**:
```python
class PartyHTREngine(HTREngine):
    """Party plugin compatible with GUI plugin system."""

    def get_name(self) -> str:
        return "Party (Multilingual)"

    def get_description(self) -> str:
        return "Whole-page recognition with 90+ languages"

    def get_config_widget(self, parent) -> QWidget:
        """Return Qt widget for Party-specific settings."""
        # Model path selector
        # Language selection (optional)
        # Device selection (cuda/cpu)

    def transcribe(self, image: Image.Image) -> TranscriptionResult:
        """Transcribe single line via temp PAGE XML."""
        # Called by GUI for each line
        # But Party works better on whole pages
        # Solution: Batch processing

    def transcribe_batch(self, segments: List[LineSegment],
                        image_path: str) -> List[TranscriptionResult]:
        """Transcribe all lines at once (more efficient)."""
        # 1. Generate temp PAGE XML with all segments
        # 2. Call Party once for whole page
        # 3. Parse all transcriptions
        # 4. Return list of results
```

**Batch Processing Strategy**:
- Override default line-by-line processing
- Process entire page in one Party call
- More efficient, matches Party's design
- GUI plugin system needs to support batch mode

---

### Phase 3: Temp PAGE XML Handler

**Goal**: Reusable module for PAGE XML generation

**File**: `party_page_xml_adapter.py`

**Functions**:
```python
def create_temp_page_xml(segments: List[LineSegment],
                         image_path: str) -> str:
    """
    Create temporary PAGE XML file from segmentation data.

    Returns:
        Path to temporary XML file
    """

def parse_party_output(xml_path: str) -> List[str]:
    """
    Parse Party's output PAGE XML to extract transcriptions.

    Returns:
        List of transcription strings (one per line)
    """

def cleanup_temp_xml(xml_path: str):
    """Clean up temporary PAGE XML file."""
```

**Implementation Options**:

**Option A: Temp File (Recommended for PoC)**
- Use `tempfile.NamedTemporaryFile`
- Easier to debug (can inspect XML)
- Party CLI requires file path anyway
- Cleanup with try/finally

**Option B: In-Memory (Future Optimization)**
- Generate XML string
- Write to temp file only for Party call
- Faster, no disk I/O

---

### Phase 4: WSL Subprocess Integration

**Goal**: Reliable Party execution via WSL

**File**: `engines/party_engine.py` (update existing)

**Improvements**:
```python
class PartyWSLExecutor:
    """Execute Party commands via WSL subprocess."""

    def __init__(self, model_path: str, wsl_venv: str = "venv_party_wsl"):
        self.model_path = model_path
        self.wsl_venv = wsl_venv
        self.wsl_project_root = "/mnt/c/Users/Achim/Documents/TrOCR/dhlab-slavistik"

    def recognize_page(self, xml_path: str, device: str = "cuda") -> str:
        """
        Call Party OCR on PAGE XML file.

        Args:
            xml_path: Path to PAGE XML file (Windows path)
            device: 'cuda' or 'cpu'

        Returns:
            Party's stdout output
        """
        # Convert Windows path to WSL path
        wsl_xml_path = self._windows_to_wsl_path(xml_path)

        # Build command
        cmd = f"""cd {self.wsl_project_root} && \
source {self.wsl_venv}/bin/activate && \
party ocr {self.model_path} {wsl_xml_path} --device {device}"""

        # Execute via WSL
        result = subprocess.run(
            ["wsl", "bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Party failed: {result.stderr}")

        return result.stdout
```

**Error Handling**:
- WSL not available → show clear error message
- Party model not found → guide user to download
- Timeout → show progress, allow cancellation
- CUDA out of memory → suggest CPU fallback

---

### Phase 5: GUI Plugin Integration

**Goal**: Add Party to `transcription_gui_plugin.py`

**Changes to GUI**:

**1. Register Party Plugin** (`transcription_gui_plugin.py`):
```python
# Add after other plugin imports
try:
    from engines.party_plugin import PartyHTREngine
    engine_registry.register_engine(PartyHTREngine)
    PARTY_AVAILABLE = True
except ImportError:
    PARTY_AVAILABLE = False
```

**2. Support Batch Processing** (if needed):
```python
class TranscriptionWorker(QThread):
    """Worker thread for OCR processing."""

    def run(self):
        # Check if engine supports batch processing
        if hasattr(self.engine, 'transcribe_batch'):
            # Process all lines at once
            results = self.engine.transcribe_batch(
                self.line_segments,
                str(self.image_path)
            )
            self.transcriptions = [r.text for r in results]
        else:
            # Process line by line (existing code)
            for seg in self.line_segments:
                result = self.engine.transcribe(seg.image)
                self.transcriptions.append(result.text)
```

**3. Party Config Widget**:
- Model path file selector
- Device dropdown (CUDA/CPU)
- Show available GPU memory
- Option to save temp PAGE XML for debugging

---

## Technical Details

### Temp PAGE XML Structure

**Minimal PAGE XML for Party**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Metadata>
    <Creator>Party-GUI-Adapter</Creator>
    <Created>2025-10-30T16:00:00</Created>
  </Metadata>
  <Page imageFilename="page.jpg" imageWidth="3000" imageHeight="4000">
    <TextRegion id="region_1">
      <Coords points="100,200 2900,200 2900,3800 100,3800"/>
      <TextLine id="line_1">
        <Coords points="120,220 2880,220 2880,280 120,280"/>
        <Baseline points="120,275 2880,275"/>
      </TextLine>
      <TextLine id="line_2">
        <Coords points="120,300 2880,300 2880,360 120,360"/>
        <Baseline points="120,355 2880,355"/>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
```

**What Party Needs**:
- `imageFilename`: Path to the image file
- `TextLine` elements with `Coords` (bounding box or polygon)
- `Baseline` (optional, can be approximated from bbox)

**What Party Returns**:
- Updated XML with `<TextEquiv><Unicode>transcription</Unicode></TextEquiv>` inside each `TextLine`
- OR stdout with line-by-line transcriptions (depends on Party's output mode)

### Party Output Parsing

**Option A: Parse XML Output**
```python
def parse_party_xml(xml_path: str) -> List[str]:
    """Parse Party's output XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    transcriptions = []
    for textline in root.findall('.//p:TextLine', ns):
        unicode_elem = textline.find('.//p:Unicode', ns)
        if unicode_elem is not None and unicode_elem.text:
            transcriptions.append(unicode_elem.text)
        else:
            transcriptions.append("")

    return transcriptions
```

**Option B: Parse stdout**
```python
def parse_party_stdout(stdout: str) -> List[str]:
    """Parse Party's stdout output."""
    # Party might output: "line_1: transcription text"
    lines = []
    for line in stdout.strip().split('\n'):
        if ':' in line:
            _, text = line.split(':', 1)
            lines.append(text.strip())
    return lines
```

---

## Testing Strategy

### Unit Tests

**1. Test Temp PAGE XML Generation**:
- Generate XML from sample segments
- Validate XML structure
- Test with various image paths (absolute, relative)

**2. Test Party WSL Execution**:
- Mock subprocess calls
- Test error handling
- Test timeout handling

**3. Test Output Parsing**:
- Parse sample Party XML output
- Handle missing transcriptions
- Handle malformed XML

### Integration Tests

**1. Test PoC GUI**:
- Load sample image
- Segment with Kraken
- Call Party
- Verify transcriptions appear

**2. Test Plugin in GUI**:
- Select Party from dropdown
- Process multiple lines
- Verify results match expected

**3. Test Error Cases**:
- WSL not available
- Party model missing
- Invalid image
- Timeout on large pages

---

## Advantages of This Approach

✅ **User-Friendly**: No manual PAGE XML creation
✅ **Plugin Compatible**: Works with existing architecture
✅ **Efficient**: Process whole page in one Party call
✅ **Debuggable**: Can save temp XML for inspection
✅ **Flexible**: Supports both batch and line-by-line
✅ **Maintainable**: Clear separation of concerns

---

## Challenges & Mitigations

### Challenge 1: Party is Slow
**Mitigation**:
- Run in background thread (already exists in GUI)
- Show progress bar
- Allow cancellation
- Cache results

### Challenge 2: WSL Overhead
**Mitigation**:
- Batch process entire page at once
- Consider persistent WSL session
- Profile and optimize

### Challenge 3: Temp File Management
**Mitigation**:
- Use context managers
- Cleanup in finally blocks
- Option to keep temp files for debugging

### Challenge 4: Path Handling
**Mitigation**:
- Robust Windows → WSL path conversion
- Handle spaces and special characters
- Use absolute paths

### Challenge 5: Error Messages
**Mitigation**:
- Parse Party stderr
- Show user-friendly error messages
- Suggest solutions (install model, check WSL, etc.)

---

## File Structure

```
dhlab-slavistik/
├── PARTY_GUI_INTEGRATION_PLAN.md         # This file
│
├── transcription_gui_party.py             # Phase 1: PoC GUI
│
├── party_page_xml_adapter.py              # Phase 3: PAGE XML utilities
│
├── engines/
│   ├── party_engine.py                    # Phase 4: Updated WSL executor
│   └── party_plugin.py                    # Phase 2: HTR plugin interface
│
└── transcription_gui_plugin.py            # Phase 5: Updated to support Party
```

---

## Implementation Order

### Stage 1: Proof-of-Concept (TODAY) ✓
- [x] Create `PARTY_GUI_INTEGRATION_PLAN.md`
- [ ] Create `transcription_gui_party.py` (standalone PoC GUI)
- [ ] Test with sample Glagolitic/Ukrainian images
- [ ] Validate temp PAGE XML approach

### Stage 2: Adapter Module (NEXT)
- [ ] Create `party_page_xml_adapter.py`
- [ ] Implement temp XML generation
- [ ] Implement output parsing
- [ ] Unit tests

### Stage 3: Plugin Implementation
- [ ] Update `engines/party_engine.py`
- [ ] Create `engines/party_plugin.py`
- [ ] Add batch processing support
- [ ] Register plugin

### Stage 4: GUI Integration
- [ ] Update `transcription_gui_plugin.py`
- [ ] Add Party config widget
- [ ] Test end-to-end workflow
- [ ] Document usage

---

## Future Enhancements

### Short-term
- Add language selection (if Party supports it)
- Show confidence scores
- Export Party results to PAGE XML
- Batch process multiple images

### Long-term
- Direct Python API instead of subprocess (if Party adds it)
- Fine-tune Party models
- Compare Party vs other engines
- Ensemble predictions

---

## Notes

**Model Location**: `party_models/party_european_langs.safetensors` (819 MB)

**WSL Environment**: `venv_party_wsl` (separate from main venv to avoid conflicts)

**Party Repository**: `party_repo/` (cloned from GitHub with syntax fix)

**Known Issues**:
- Party requires PAGE XML (hence this adapter)
- WSL subprocess has some overhead
- Works best on whole pages, not individual lines

**Dependencies**:
- Working WSL installation
- Party installed in `venv_party_wsl`
- Party model downloaded
- `page_xml_exporter.py` (already exists)

---

## Success Criteria

✅ User can select "Party (Multilingual)" from engine dropdown
✅ Process entire page with one click
✅ Results appear in GUI just like other engines
✅ No manual PAGE XML creation required
✅ Error messages are clear and actionable
✅ Performance is acceptable (<30s for typical page)
