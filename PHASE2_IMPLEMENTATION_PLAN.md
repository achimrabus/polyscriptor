# Phase 2 Implementation Plan - Transcription GUI

## Overview

Phase 2 focuses on enhancing usability and professional features to make the GUI production-ready for archival work.

**Status**: Phase 1 Complete ✓ (including CPU/GPU selection + multi-image navigation)

---

## Phase 2 Features (Priority Order)

### ✅ Already Implemented in Phase 1+
1. Seamless zoom and pan (QGraphicsView)
2. Font selection and formatting
3. Drag & drop import (single + multiple files)
4. Multi-image navigation (Previous/Next buttons + keyboard shortcuts)
5. Background normalization toggle
6. Keyboard shortcuts (Ctrl+O, Ctrl+S, Ctrl+0, etc.)
7. CPU/GPU device selection

### 🎯 Phase 2 Tasks

#### 1. HuggingFace Model Integration (HIGH PRIORITY)
**Rationale**: Allow users to test community models without manual downloads

**Implementation**:
- Add "HuggingFace" tab in model selection
- Text input for model ID (e.g., `kazars24/trocr-base-handwritten-ru`)
- Search button to validate model on Hub
- Show model card info (description, language, CER if available)
- Download button with progress bar
- Cache management (list cached models, delete old models)

**Technical Details**:
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load from HuggingFace Hub
model_id = "kazars24/trocr-base-handwritten-ru"
processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)
```

**Files to Modify**:
- `transcription_gui_qt.py` - Add HuggingFace tab to model selection
- `inference_page.py` - Update TrOCRInference to support HF model IDs

**UI Mockup**:
```
┌─────────────────────────────────────────┐
│ Model Selection                         │
├─────────────────────────────────────────┤
│ [Local] [HuggingFace] ← Tabs            │
├─────────────────────────────────────────┤
│ Model ID: [kazars24/trocr-base-...  ] 🔍│
│                                          │
│ Model Card:                             │
│ ┌────────────────────────────────────┐ │
│ │ TrOCR Base Russian Handwritten     │ │
│ │ Language: Russian                  │ │
│ │ Training: Kazars dataset           │ │
│ │ CER: ~7%                           │ │
│ └────────────────────────────────────┘ │
│                                          │
│ [Download Model] or [Use Cached]        │
└─────────────────────────────────────────┘
```

---

#### 2. PAGE XML Import Support (HIGH PRIORITY)
**Rationale**: Reuse existing Transkribus segmentation, don't reinvent the wheel

**Implementation**:
- Auto-detect PAGE XML when loading image (look for matching .xml file)
- Parse PAGE XML to extract TextLine coordinates
- Use polygon coordinates for segmentation (instead of rectangles)
- Show "Using PAGE XML segmentation" indicator
- Manual import option (Browse for XML file)

**Technical Details**:
```python
from lxml import etree

def parse_page_xml(xml_path: Path) -> List[LineSegment]:
    """Parse PAGE XML and extract line segments."""
    tree = etree.parse(str(xml_path))

    # PAGE XML namespace
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    segments = []
    for text_line in tree.xpath('//pc:TextLine', namespaces=ns):
        # Get polygon coordinates
        coords = text_line.xpath('.//pc:Coords/@points', namespaces=ns)[0]
        points = parse_coords(coords)  # "x1,y1 x2,y2 ..." -> [(x1,y1), ...]

        # Get bounding box
        bbox = get_bbox_from_polygon(points)

        segments.append(LineSegment(bbox=bbox, polygon=points))

    return segments
```

**Files to Modify**:
- `transcription_gui_qt.py` - Add PAGE XML import button and auto-detect
- `inference_page.py` - Add `parse_page_xml()` helper function

**UI Changes**:
```
┌─────────────────────────────────────────┐
│ Line Segmentation                       │
├─────────────────────────────────────────┤
│ Source: [Auto-detect ▼]                 │
│         • Auto (Horizontal Projection)  │
│         • PAGE XML (if available)       │
│         • Manual                        │
│                                          │
│ [📄 Import PAGE XML...]                 │
│ ✓ Using: document_001.xml               │
│                                          │
│ [Detect Lines]  Lines: 24               │
└─────────────────────────────────────────┘
```

---

#### 3. CSV Export Format (MEDIUM PRIORITY)
**Rationale**: Structured data for analysis and archival databases

**Implementation**:
- Add CSV option to export dialog
- Include columns: filename, line_number, bbox_x1, bbox_y1, bbox_x2, bbox_y2, text
- UTF-8 encoding with BOM for Excel compatibility
- Quote escaping for text with commas

**Technical Details**:
```python
import csv

def export_to_csv(output_path: Path, image_name: str, segments: List[LineSegment]):
    """Export transcription to CSV format."""
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        # Header
        writer.writerow(['filename', 'line', 'x1', 'y1', 'x2', 'y2', 'text'])

        # Data rows
        for idx, segment in enumerate(segments, 1):
            x1, y1, x2, y2 = segment.bbox
            writer.writerow([image_name, idx, x1, y1, x2, y2, segment.text])
```

**Files to Modify**:
- `transcription_gui_qt.py` - Update export dialog with CSV option
- Add `export_to_csv()` method

**UI Changes**:
```
File > Save Transcription (Ctrl+S)
  → Format: [TXT ▼]
            • TXT (Plain Text)
            • CSV (Structured Data) ← NEW
            • JSON (Metadata) ← NEW
```

---

#### 4. Mini-Map Navigation Widget (MEDIUM PRIORITY)
**Rationale**: Easier navigation of large zoomed images

**Implementation**:
- Small thumbnail view in bottom-right corner
- Red rectangle showing current viewport
- Click to jump to area
- Toggle on/off with View menu

**Technical Details**:
```python
class MiniMapWidget(QWidget):
    """Mini-map overview for large images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 150)
        self.thumbnail = None
        self.viewport_rect = QRectF()

    def set_image(self, pixmap: QPixmap):
        """Set the thumbnail image."""
        self.thumbnail = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.update()

    def set_viewport(self, viewport_rect: QRectF):
        """Update current viewport rectangle."""
        self.viewport_rect = viewport_rect
        self.update()

    def paintEvent(self, event):
        """Draw thumbnail and viewport rectangle."""
        painter = QPainter(self)

        # Draw thumbnail
        if self.thumbnail:
            painter.drawPixmap(0, 0, self.thumbnail)

        # Draw viewport rectangle
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        painter.drawRect(self.viewport_rect)
```

**Files to Modify**:
- `transcription_gui_qt.py` - Add MiniMapWidget class
- Connect to ZoomableGraphicsView to track viewport changes

---

#### 5. Comparison Mode (MEDIUM PRIORITY)
**Rationale**: Compare outputs from different models to choose best transcription

**Implementation**:
- Split transcription panel vertically
- Load 2 models simultaneously
- Process with both models and show side-by-side
- Highlight differences
- Select better transcription for each line

**Technical Details**:
```python
class ComparisonPanel(QWidget):
    """Side-by-side model comparison."""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()

        # Model A
        self.text_a = QTextEdit()
        self.text_a.setReadOnly(True)
        layout.addWidget(QLabel("Model A:"))
        layout.addWidget(self.text_a)

        # Model B
        self.text_b = QTextEdit()
        self.text_b.setReadOnly(True)
        layout.addWidget(QLabel("Model B:"))
        layout.addWidget(self.text_b)

        self.setLayout(layout)

    def set_transcriptions(self, text_a: str, text_b: str):
        """Set transcriptions and highlight differences."""
        self.text_a.setPlainText(text_a)
        self.text_b.setPlainText(text_b)
        self._highlight_differences(text_a, text_b)

    def _highlight_differences(self, text_a: str, text_b: str):
        """Highlight character differences."""
        # Use difflib to find differences
        import difflib
        diff = difflib.SequenceMatcher(None, text_a, text_b)

        # Highlight mismatches in yellow
        # ... implementation ...
```

**UI Mockup**:
```
┌────────────────────────────────────────────────────┐
│ Transcription                        [Comparison] ☑│
├────────────────────────────────────────────────────┤
│ Model A: checkpoint-3000    │ Model B: HF/kazars24 │
├─────────────────────────────┼──────────────────────┤
│ Добрий день, мої дорожен-   │ Добрий день, мої     │
│ кі! Пишу тобі зі самого     │ дорогенькі! Пишу тобі│
│ ранку! Все гарно...         │ зі самого ранку!     │
│                             │ Все гарно...         │
├─────────────────────────────┼──────────────────────┤
│ [Use This] [Copy]           │ [Use This] [Copy]    │
└────────────────────────────────────────────────────┘
```

---

#### 6. Settings Persistence (HIGH PRIORITY)
**Rationale**: Remember user preferences between sessions

**Implementation**:
- Save settings to JSON file on close
- Load settings on startup
- Settings to save:
  - Last used model path
  - Device preference (CPU/GPU)
  - Normalize background default
  - Beam search / max length defaults
  - Window size and position
  - Last working directory
  - Font selection

**Technical Details**:
```python
import json
from pathlib import Path

class SettingsManager:
    """Manage application settings persistence."""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path.home() / '.trocr_gui_settings.json'
        self.settings = self.load()

    def load(self) -> dict:
        """Load settings from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return self.default_settings()

    def save(self, settings: dict):
        """Save settings to file."""
        with open(self.config_path, 'w') as f:
            json.dump(settings, f, indent=2)

    def default_settings(self) -> dict:
        """Return default settings."""
        return {
            'model_path': './models/ukrainian_model_normalized/checkpoint-3000',
            'device': 'cuda',
            'normalize_bg': True,
            'num_beams': 4,
            'max_length': 128,
            'window_geometry': None,
            'last_directory': str(Path.home()),
            'font_family': 'Arial',
            'font_size': 12
        }
```

**Files to Modify**:
- `transcription_gui_qt.py` - Add SettingsManager class
- Load settings in `__init__()`, save in `closeEvent()`

---

#### 7. JSON Export Format (LOW PRIORITY)
**Rationale**: Structured export with metadata for programmatic processing

**Implementation**:
```python
def export_to_json(output_path: Path, image_path: Path,
                  segments: List[LineSegment], metadata: dict):
    """Export transcription to JSON format."""
    data = {
        'image': {
            'filename': image_path.name,
            'path': str(image_path),
            'width': metadata.get('width'),
            'height': metadata.get('height')
        },
        'model': {
            'checkpoint': metadata.get('model_path'),
            'device': metadata.get('device'),
            'normalize_bg': metadata.get('normalize_bg'),
            'num_beams': metadata.get('num_beams')
        },
        'lines': [
            {
                'line_number': idx + 1,
                'bbox': {
                    'x1': seg.bbox[0],
                    'y1': seg.bbox[1],
                    'x2': seg.bbox[2],
                    'y2': seg.bbox[3]
                },
                'text': seg.text,
                'confidence': seg.confidence if hasattr(seg, 'confidence') else None
            }
            for idx, seg in enumerate(segments)
        ],
        'statistics': {
            'total_lines': len(segments),
            'total_characters': sum(len(s.text) for s in segments),
            'processing_time': metadata.get('processing_time')
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

---

## Implementation Order (Recommended)

### Week 1: Core Enhancements
1. **HuggingFace Model Integration** (2 days)
   - Most requested feature
   - Enables testing community models easily

2. **PAGE XML Import** (1 day)
   - High value for Transkribus users
   - Reuses existing segmentation work

3. **Settings Persistence** (1 day)
   - QoL improvement
   - Foundation for future features

### Week 2: Export & Navigation
4. **CSV Export** (0.5 day)
   - Quick win, highly useful

5. **JSON Export** (0.5 day)
   - Similar to CSV, easy to implement

6. **Mini-Map Widget** (1 day)
   - Nice-to-have navigation enhancement

7. **Comparison Mode** (2 days)
   - Advanced feature for power users

---

## Testing Plan

### Manual Testing Checklist
- [ ] HuggingFace model downloads correctly
- [ ] PAGE XML import detects .xml files automatically
- [ ] Settings persist across application restarts
- [ ] CSV export opens correctly in Excel
- [ ] JSON export validates with schema
- [ ] Mini-map updates on zoom/pan
- [ ] Comparison mode highlights differences

### Integration Testing
- [ ] Batch processing with HF models
- [ ] PAGE XML + multi-image navigation
- [ ] Export all formats for single image
- [ ] Settings reload after crash recovery

---

## Dependencies to Add

```bash
# Already have: PyQt6, torch, transformers, pillow, opencv-python

# May need to add:
pip install lxml           # PAGE XML parsing
pip install difflib        # Text comparison (built-in, no install needed)
```

---

## Documentation Updates

After Phase 2 implementation, update:
1. **GUI_TESTING_GUIDE.md** - Add Phase 2 feature checklist
2. **README.md** - Update feature list and screenshots
3. **TRANSCRIPTION_GUI_REQUIREMENTS.md** - Mark Phase 2 items as complete

---

## Success Criteria

Phase 2 is complete when:
- ✅ Can load models directly from HuggingFace Hub
- ✅ PAGE XML files are auto-detected and used for segmentation
- ✅ Settings are remembered between sessions
- ✅ Can export in 3 formats (TXT, CSV, JSON)
- ✅ Mini-map provides easy navigation
- ✅ Can compare 2 models side-by-side
- ✅ All features tested and documented

---

## Phase 3 Preview

After Phase 2 completion, Phase 3 will focus on:
1. PAGE XML Export (reverse of import)
2. Manual segmentation tools (rectangle/polygon drawing)
3. Confidence scoring visualization
4. DOCX/PDF export
5. Project/session management

---

**Current Status**: Planning complete, ready to begin implementation.

**Next Action**: Start with HuggingFace model integration (highest priority).
