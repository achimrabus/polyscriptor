# PyQt6 Transcription GUI - Testing Guide

## Phase 1 MVP Implementation Complete

The PyQt6 transcription GUI has been successfully implemented with all Phase 1 features.

## How to Launch

```bash
python transcription_gui_qt.py
```

## Phase 1 Features Checklist

### 1. Image Loading
- [x] **File Menu > Open Image** (Ctrl+O) - Traditional file dialog (single image)
- [x] **Load Images...** button - Multi-select file dialog (batch mode)
- [x] **Drag & Drop** - Drag single or multiple image files directly onto the window
- [x] Supported formats: PNG, JPG, JPEG, TIFF, TIF

### 2. Image Navigation (NEW!)
- [x] **< Previous / Next >** buttons - Navigate through loaded images
- [x] **Image counter** - Shows current position (e.g., "3 / 15")
- [x] **Keyboard shortcuts**:
  - Left Arrow / Page Up: Previous image
  - Right Arrow / Page Down: Next image
- [x] Automatic navigation UI updates

### 3. Model Selection
- [x] **Model Path** dropdown in settings panel
- [x] **Browse** button to select custom checkpoint
- [x] Pre-filled with example path: `./models/ukrainian_model_normalized/checkpoint-3000`

### 4. Device Selection (CPU/GPU) (NEW!)
- [x] **Radio buttons** for GPU/CPU selection
- [x] **Auto-detection** - GPU selected by default if CUDA available
- [x] **Status feedback** - Shows which device is active
- [x] **Training compatibility** - Use CPU for inference while GPU is training
- [x] Automatic model reload when device changes

### 3. OCR Settings
- [x] **Normalize Background** checkbox - Enable if model was trained with normalization
- [x] **Beam Search** spinbox (1-10) - Default: 4
- [x] **Max Length** spinbox (64-256) - Default: 128

### 4. Line Segmentation
- [x] **Segment Lines** button - Automatic line detection
- [x] Visual overlays with green bounding boxes
- [x] Line numbering displayed on each box

### 5. OCR Processing
- [x] **Process All Lines** button - Runs OCR in background thread
- [x] Progress bar with status updates
- [x] Non-blocking UI - window remains responsive during processing

### 6. Zoom & Pan
- [x] **Mouse wheel zoom** - Smooth scaling centered on cursor
- [x] **Click & drag pan** - Move around the image
- [x] **View menu shortcuts**:
  - Ctrl+0: Fit to window
  - Ctrl++: Zoom in
  - Ctrl+-: Zoom out

### 7. Text Editing
- [x] Multi-line text editor with detected/transcribed lines
- [x] **Font Selection** button (Ctrl+F) - Choose custom font
- [x] Character count display
- [x] Word count display

### 8. Export
- [x] **File > Save Transcription** (Ctrl+S) - Export to TXT file
- [x] Confirmation message after save

### 9. UI Layout
- [x] Split-pane design: Image viewer (left) | Transcription editor (right)
- [x] Settings panel with all configuration options
- [x] Menu bar with File/View/Edit menus
- [x] Toolbar with common actions
- [x] Status bar with progress indicator

## Testing Workflow

### Typical Usage Flow (Single Image):

1. **Launch GUI**:
   ```bash
   python transcription_gui_qt.py
   ```

2. **Load Image**:
   - Either: File > Open Image (Ctrl+O)
   - Or: Drag & drop image file onto window

3. **Configure Settings**:
   - Select model checkpoint path
   - Choose device (CPU/GPU) - **NEW!**
   - Enable "Normalize Background" if needed
   - Adjust beam search/max length if desired

4. **Segment Lines**:
   - Click "Detect Lines" button
   - Green boxes appear around detected text lines

5. **Process OCR**:
   - Click "Process All Lines" button
   - Watch progress bar as OCR runs
   - Transcribed text appears in editor

6. **Review & Edit**:
   - Scroll through transcription
   - Make manual corrections as needed
   - Use font selector to adjust readability

7. **Zoom/Navigate**:
   - Mouse wheel to zoom in/out
   - Click & drag to pan
   - Ctrl+0 to fit to window

8. **Export**:
   - File > Save Transcription (Ctrl+S)
   - Choose output location
   - TXT file saved with transcription

### Batch Processing Workflow (NEW!):

1. **Load Multiple Images**:
   - Click "Load Images..." button
   - Select multiple image files (Ctrl+Click or Shift+Click)
   - Or: Drag & drop multiple image files onto window

2. **Navigate Images**:
   - Click "< Previous" / "Next >" buttons
   - Or: Use Left/Right arrow keys
   - Or: Use Page Up/Page Down keys
   - Image counter shows position (e.g., "3 / 15")

3. **Process Each Image**:
   - For current image: Click "Detect Lines"
   - Then: Click "Process All Lines"
   - Review/edit transcription
   - Save transcription for current image

4. **Move to Next Image**:
   - Click "Next >" or press Right Arrow
   - Repeat steps 3-4 for all images

### Using CPU Mode (Recommended During Training):

1. **Why CPU Mode?**:
   - Your GPUs may be busy with training
   - CPU inference allows transcription without interrupting training
   - Slightly slower but doesn't compete for GPU memory

2. **How to Use**:
   - In settings panel, select "CPU" radio button
   - Status bar confirms: "Device set to: CPU"
   - Model automatically reloads on CPU
   - Process images normally

3. **Switching Back to GPU**:
   - After training completes, select "GPU" radio button
   - Much faster inference (especially for large batches)

## Known Limitations (Phase 1)

- No HuggingFace model loading yet (Phase 2)
- No PAGE XML import/export yet (Phase 2-3)
- No comparison mode yet (Phase 2)
- No manual segmentation tools yet (Phase 3)
- Only single image processing (no bulk mode yet)

## Next Steps (Phase 2)

After Phase 1 testing and user feedback:
1. HuggingFace model integration
2. Enhanced zoom controls (mini-map)
3. Comparison mode (side-by-side model outputs)
4. CSV export format
5. PAGE XML import support

## Troubleshooting

### GUI doesn't launch
- Ensure PyQt6 is installed: `pip install PyQt6`
- Check Python version (requires 3.7+)

### Model loading fails
- Verify checkpoint path exists
- Check that checkpoint contains `config.json` and `pytorch_model.bin`
- Ensure base model `kazars24/trocr-base-handwritten-ru` is cached

### OCR processing hangs
- Check GPU availability: `nvidia-smi`
- Monitor GPU memory usage
- Try reducing batch size (not exposed in Phase 1 GUI)

### Zoom is sluggish
- Expected with very large images (>5000px)
- QGraphicsView uses hardware acceleration where available
- Performance should be smooth for typical scan sizes (2000-3000px)

## Dependencies

All required dependencies:
```bash
pip install PyQt6 torch transformers pillow opencv-python numpy
```

Already included in project's `requirements.txt`.

## File Structure

```
transcription_gui_qt.py          # Main GUI application
inference_page.py                # OCR backend (TrOCRInference, LineSegmenter)
TRANSCRIPTION_GUI_REQUIREMENTS.md # Full requirements specification
GUI_TESTING_GUIDE.md             # This guide
```

## Feedback

After testing, please provide feedback on:
1. UI responsiveness and performance
2. Zoom/pan behavior
3. OCR accuracy with selected model
4. Any crashes or error messages
5. Feature requests for Phase 2

Report issues or suggestions to continue improving the GUI!
