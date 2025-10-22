# Transcription GUI Requirements

## Overview
Professional GUI application for transcribing handwritten Ukrainian documents using TrOCR models. Designed for researchers and archivists working with historical manuscripts.

---

## Core Features

### 1. Image Import & Management

#### 1.1 Drag & Drop
- **Single file**: Drop single image onto main window
- **Multiple files**: Drop multiple images for batch processing
- **Folder import**: Drop entire folder to import all images recursively
- **Supported formats**: PNG, JPG, JPEG, TIFF, PDF (convert to images)

#### 1.2 File Browser Import
- Traditional file dialog for single/multiple file selection
- Recent files list (last 10 opened)
- Session persistence (remember last working directory)

#### 1.3 Batch Management
- **Image list panel**: Show all loaded images with thumbnails
- **Batch queue**: Process multiple images sequentially or in parallel
- **Progress tracking**: Per-image and overall batch progress
- **Skip/retry**: Ability to skip failed images or retry with different settings

---

### 2. Model Management

#### 2.1 Model Selection
- **Local models**: Browse and select checkpoint directories
  - Auto-detect checkpoints in `./models/` directory
  - Show model metadata (training date, CER, dataset info)
  - Recent models list

- **Hugging Face models**:
  - Search and download models directly from Hugging Face Hub
  - Show model card information (language, architecture, metrics)
  - Cached models list (avoid re-downloading)
  - Model version management

#### 2.2 Model Settings
- **Base model override**: Specify base model for processor (default: auto-detect)
- **Background normalization toggle**: Match training preprocessing
  - Auto-detect from `dataset_info.json` if available
  - Manual override option
- **Device selection**: CPU / GPU 0 / GPU 1 / Auto
- **Batch size**: For batch processing (1-16)

---

### 3. Image Viewing & Manipulation

#### 3.1 Seamless Zoom
- **Mouse wheel**: Zoom in/out centered on cursor position
- **Zoom controls**: Buttons for Fit to Window / 100% / 200% / Custom
- **Zoom range**: 10% to 500%
- **Keyboard shortcuts**:
  - `Ctrl+0`: Fit to window
  - `Ctrl++`: Zoom in
  - `Ctrl+-`: Zoom out
  - `Ctrl+1`: 100% zoom

#### 3.2 Pan & Navigate
- **Click and drag**: Pan around zoomed image
- **Mini-map**: Small overview in corner showing current viewport
- **Reset view**: Double-click to reset zoom and center

#### 3.3 Image Enhancement (Optional)
- **Brightness/Contrast sliders**: Adjust for better visibility (doesn't affect OCR)
- **Grayscale toggle**: View in grayscale
- **Rotation**: 90° increments for misoriented scans
- **Reset all**: Return to original image

---

### 4. Line Segmentation

#### 4.1 Automatic Segmentation
- **Horizontal projection**: Default automatic line detection
- **Adjustable parameters**:
  - Min line height (pixels)
  - Min gap between lines (pixels)
  - Sensitivity threshold
- **Preview mode**: Show detected lines overlaid on image before processing

#### 4.2 PAGE XML Segmentation
- **Import PAGE XML**: Use existing Transkribus segmentation
- **Auto-detect**: Look for matching `.xml` file when image is loaded
- **Validation**: Verify polygon coordinates are within image bounds

#### 4.3 Manual Segmentation
- **Rectangle tool**: Click and drag to define line regions
- **Polygon tool**: Click points to define irregular line shapes
- **Edit mode**: Adjust existing line boundaries
- **Line reordering**: Drag to change line processing order
- **Delete lines**: Remove incorrect detections

---

### 5. Transcription & Editing

#### 5.1 OCR Processing
- **Process single line**: Click line to transcribe
- **Process all lines**: Batch transcribe all detected lines
- **Process selection**: Transcribe only selected lines
- **Real-time preview**: Show transcription as it completes

#### 5.2 Text Editor
- **Font selection**:
  - Unicode font picker (support Cyrillic)
  - Font size adjustment (8pt - 24pt)
  - Suggested fonts: Arial, Times New Roman, DejaVu Sans, Noto Sans
- **Text formatting**:
  - Line-by-line editing
  - Undo/redo (Ctrl+Z / Ctrl+Y)
  - Find and replace
- **Confidence indicators**: Highlight low-confidence characters (optional)

#### 5.3 Parallel View
- **Split pane**: Image on left, transcription on right
- **Synchronized scrolling**: Click line in image → highlights text and vice versa
- **Line linking**: Visual connection between image line and text line

---

### 6. Export Options

#### 6.1 Plain Text Export
- **TXT**: Simple line-by-line text file
- **CSV**: Image filename, line bbox, transcription text
- **JSON**: Structured format with metadata
- **Encoding options**: UTF-8 (default), UTF-16, Windows-1251

#### 6.2 PAGE XML Export ⭐ **Highly Requested**
- **Generate PAGE XML**: Create Transkribus-compatible PAGE XML
  - Image filename and dimensions
  - TextRegion coordinates (from segmentation)
  - TextLine coordinates with polygon points
  - Unicode text content
  - Confidence scores (if available)
- **Update existing PAGE XML**: Preserve layout, update text only
- **Schema version**: PAGE XML 2013-07-15 (Transkribus standard)
- **Validation**: Verify XML against schema before export

#### 6.3 Document Formats
- **DOCX**: Microsoft Word document with images and transcriptions
- **PDF**: Searchable PDF with text layer
- **HTML**: Web page with image + text side-by-side

#### 6.4 Batch Export
- Export multiple processed images at once
- Preserve folder structure
- Naming conventions: `{original_name}_transcription.{ext}`

---

### 7. Quality Control & Review

#### 7.1 Confidence Scoring
- Show character-level confidence from model (if available)
- Highlight uncertain words/characters in yellow/red
- Filter lines by confidence threshold

#### 7.2 Manual Review Mode
- **Review queue**: Show only lines needing review
- **Keyboard shortcuts**:
  - `Enter`: Accept and move to next
  - `Ctrl+E`: Edit current line
  - `Ctrl+D`: Mark for deletion
- **Skip reviewed**: Hide lines already approved

#### 7.3 Statistics
- Total lines processed
- Average confidence score
- Processing time per line/page
- Character/word count

---

### 8. Settings & Preferences

#### 8.1 Inference Settings
- **Beam search**: Number of beams (1-10)
- **Max length**: Maximum sequence length (64-256)
- **Temperature**: Sampling temperature (greedy=1.0)
- **Save settings**: Remember last used settings

#### 8.2 UI Preferences
- **Theme**: Light / Dark / System
- **Language**: English / Ukrainian / German (for UI, not OCR)
- **Auto-save**: Interval for auto-saving work (disabled / 5min / 10min)
- **Layout**: Save/restore window size and panel positions

#### 8.3 Advanced
- **Cache size**: Limit model cache (GB)
- **Temp directory**: Where to store temporary files
- **Logging level**: Debug / Info / Warning / Error

---

## Additional Suggestions

### 9. Keyboard Shortcuts Summary
| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open image |
| `Ctrl+S` | Save transcription |
| `Ctrl+Shift+S` | Save As... |
| `Ctrl+E` | Export |
| `Ctrl+P` | Process all lines |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+F` | Find in text |
| `Ctrl+0` | Fit to window |
| `Ctrl++` | Zoom in |
| `Ctrl+-` | Zoom out |
| `F5` | Refresh / Reload |
| `Esc` | Cancel operation |

### 10. Project/Session Management
- **Project files**: Save entire session (images, segmentation, transcriptions)
  - `.trocr_project` format (JSON-based)
  - Reopen and continue work later
- **Recent projects**: Quick access to last 5 projects
- **Templates**: Save segmentation settings as templates

### 11. Collaboration Features (Future)
- **Comments/Notes**: Add notes to specific lines
- **User tracking**: Track who transcribed which lines (multi-user)
- **Diff view**: Compare transcriptions from different models
- **Ground truth mode**: Load reference transcriptions for comparison

### 12. Integration & Automation
- **CLI mode**: Command-line interface for batch processing
- **Watch folder**: Monitor folder for new images and auto-process
- **Transkribus integration**: Direct import/export from Transkribus projects
- **API endpoint**: REST API for programmatic access (optional)

### 13. Help & Documentation
- **Interactive tutorial**: First-run walkthrough
- **Tooltips**: Hover help for all controls
- **Context help**: F1 key for help on current feature
- **Model documentation**: Show model card when hovering over model name

---

## Technical Requirements

### Dependencies
- **Python**: 3.9+
- **GUI Framework**: PyQt6 or PySide6 (recommended for professional feel)
  - Alternative: Tkinter (simpler but less powerful)
- **Image handling**: Pillow, OpenCV
- **ML Framework**: PyTorch, Transformers
- **XML handling**: lxml (for PAGE XML generation/parsing)
- **Document export**: python-docx, reportlab

### Performance Targets
- **Startup time**: < 5 seconds
- **Image loading**: < 1 second (up to 10MB images)
- **Zoom responsiveness**: 60 FPS smooth panning
- **OCR processing**: < 2 seconds per line (depends on hardware)

### Platform Support
- **Primary**: Windows 10/11
- **Secondary**: Linux (Ubuntu 20.04+), macOS (optional)

---

## Implementation Priority

### Phase 1 (MVP - Minimum Viable Product)
1. ✅ Basic image loading and display
2. ✅ Model selection (local checkpoints only)
3. ✅ Automatic line segmentation (horizontal projection)
4. ✅ OCR processing with progress display
5. ✅ Text editor with basic editing
6. ✅ Export to TXT/CSV

### Phase 2 (Enhanced Usability)
7. Seamless zoom and pan
8. Font selection and formatting
9. Drag & drop import
10. Batch processing queue
11. PAGE XML import (use existing segmentation)
12. Background normalization toggle
13. Keyboard shortcuts

### Phase 3 (Professional Features)
14. PAGE XML export ⭐
15. Manual segmentation tools
16. Hugging Face model integration
17. Confidence scoring and review mode
18. DOCX/PDF export
19. Project/session management
20. Settings persistence

### Phase 4 (Advanced Features)
21. Mini-map for navigation
22. Parallel view with synchronized scrolling
23. Statistics and quality metrics
24. Watch folder automation
25. Multi-language UI

---

## Design Mockup Suggestions

### Main Window Layout
```
+----------------------------------------------------------+
| File  Edit  View  Process  Export  Settings  Help       |
+----------------------------------------------------------+
| [Open] [Save] [Zoom] [Process All] [Export]   Model: ▼  |
+----------------------------------------------------------+
|                    |                                      |
|  Image List        |         Image Viewer                |
|  (thumbnails)      |         (with zoom/pan)             |
|                    |                                      |
|  □ image_001.jpg   |    [===============================] |
|  ☑ image_002.jpg   |    |                               | |
|  □ image_003.jpg   |    |      Page Image               | |
|                    |    |      with line overlays       | |
|  [Load More...]    |    |                               | |
|                    |    [===============================] |
|                    |                                      |
+--------------------+--------------------------------------+
|                             |                             |
|  Line Segmentation          |  Transcription Editor       |
|  [Detected Lines: 24]       |  [Font: Arial ▼] [Size: 12]|
|                             |                             |
|  Line 1 [✓] [Edit] [Del]    |  Добрий день, мої дорожен- |
|  Line 2 [✓] [Edit] [Del]    |  кі! Пишу тобі зі самого  |
|  Line 3 [○] [Edit] [Del]    |  ранку! Все гарно...      |
|  ...                        |                             |
|                             |  [Character count: 245]    |
+-----------------------------+-----------------------------+
| Status: Processing line 3/24 | GPU: 89% | Step: 1.2s     |
+----------------------------------------------------------+
```

---

## Open Questions for Discussion

1. **PAGE XML Priority**: Is PAGE XML export essential for MVP or can it wait for Phase 3?
2. **Manual Segmentation**: How important is the ability to manually adjust line boundaries?
3. **Confidence Scoring**: Should we show character-level confidence or just line-level?
4. **Font Handling**: Should transcription font match the handwriting style (e.g., cursive fonts)?
5. **Multi-page Documents**: Should we support PDF import with multi-page handling?
6. **Comparison Mode**: Would you like to compare transcriptions from different models side-by-side?

---

## Notes
- **Existing Code**: `inference_page_gui.py` provides basic foundation - can be extended
- **PAGE XML**: Python's `lxml` library supports full PAGE XML creation/parsing
- **Seamless Zoom**: PyQt's QGraphicsView provides built-in smooth zoom/pan
- **Drag & Drop**: Both PyQt and Tkinter support drag-drop events
- **Font Selection**: QFontDialog (PyQt) or tkinter.font provides font picker

---

## Success Criteria
- Transcribe a 100-page document in < 30 minutes (with user review)
- No crashes or data loss during long sessions
- Intuitive enough for non-technical archivists
- Seamless workflow from image → transcription → PAGE XML export
