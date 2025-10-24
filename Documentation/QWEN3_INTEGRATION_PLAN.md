# Qwen3 VLM Integration Plan for GUI

## Overview

Qwen3 VLM (Vision-Language Model) represents a paradigm shift from line-based OCR:

### TrOCR/PyLaia (Line-based)
```
Page → Line Segmentation → Line Recognition → Combine Lines
```

### Qwen3 VLM (Whole-page)
```
Page → Direct Transcription (no segmentation needed!)
```

## Key Advantages

1. **No Segmentation Needed** - Processes entire pages directly
2. **Layout Aware** - Understands document structure
3. **Context Understanding** - Uses visual context for better accuracy
4. **Flexible Prompting** - Can customize behavior via prompts
5. **Multi-GPU Support** - Distributes 8B model across GPUs automatically

## GUI Integration Architecture

### Current Flow (TrOCR)
```python
1. Load Image
2. Segment Lines (HPP/Kraken)
3. Process Each Line (TrOCR)
4. Display Results (with confidence)
```

### New Flow (Qwen3)
```python
1. Load Image
2. Process Entire Page (Qwen3) ← No step 2!
3. Display Results (full page text)
```

## Implementation Plan

### Step 1: Add Qwen3 Tab to Model Selection

```python
# In transcription_gui_qt.py

def _setup_ui(self):
    # ... existing code ...

    # Add Qwen3 VLM tab (after PyLaia tab)
    qwen3_tab = QWidget()
    qwen3_layout = QGridLayout(qwen3_tab)

    # Model selection dropdown
    qwen3_layout.addWidget(QLabel("Qwen3 Model:"), 0, 0)
    self.combo_qwen3_model = QComboBox()

    # Populate with available models
    from inference_qwen3 import QWEN3_MODELS
    for model_id, info in QWEN3_MODELS.items():
        display_name = f"{model_id} ({info['vram']})"
        self.combo_qwen3_model.addItem(display_name, model_id)

    self.combo_qwen3_model.setToolTip("Select Qwen3 VLM model variant")
    qwen3_layout.addWidget(self.combo_qwen3_model, 0, 1, 1, 2)

    # Custom adapter path (optional)
    qwen3_layout.addWidget(QLabel("Custom Adapter:"), 1, 0)
    self.txt_qwen3_adapter = QLineEdit()
    self.txt_qwen3_adapter.setPlaceholderText("Optional: your-username/qwen3-ukrainian-adapter")
    qwen3_layout.addWidget(self.txt_qwen3_adapter, 1, 1, 1, 2)

    # Prompt customization
    qwen3_layout.addWidget(QLabel("Prompt:"), 2, 0)
    self.txt_qwen3_prompt = QTextEdit()
    self.txt_qwen3_prompt.setPlainText("Transcribe the text shown in this image.")
    self.txt_qwen3_prompt.setMaximumHeight(80)
    qwen3_layout.addWidget(self.txt_qwen3_prompt, 2, 1, 1, 3)

    # Advanced settings
    advanced_group = QGroupBox("Advanced Settings")
    advanced_layout = QGridLayout()

    advanced_layout.addWidget(QLabel("Max Tokens:"), 0, 0)
    self.spin_qwen3_max_tokens = QSpinBox()
    self.spin_qwen3_max_tokens.setRange(512, 8192)
    self.spin_qwen3_max_tokens.setValue(2048)
    advanced_layout.addWidget(self.spin_qwen3_max_tokens, 0, 1)

    advanced_layout.addWidget(QLabel("Image Size:"), 0, 2)
    self.spin_qwen3_img_size = QSpinBox()
    self.spin_qwen3_img_size.setRange(512, 2048)
    self.spin_qwen3_img_size.setValue(1536)
    advanced_layout.addWidget(self.spin_qwen3_img_size, 0, 3)

    advanced_group.setLayout(advanced_layout)
    qwen3_layout.addWidget(advanced_group, 3, 0, 1, 4)

    self.model_tabs.addTab(qwen3_tab, "Qwen3 VLM")
```

### Step 2: Modify Processing Logic

```python
def _process_image(self):
    """Main processing method - handles both line-based and page-based models."""

    if not self.current_image_path or not self.current_pixmap:
        QMessageBox.warning(self, "Warning", "Please load an image first!")
        return

    # Determine model type
    current_tab = self.model_tabs.currentIndex()

    if current_tab == 3:  # Qwen3 VLM tab
        self._process_with_qwen3()
    else:
        # Existing line-based processing
        self._process_with_line_segmentation()

def _process_with_qwen3(self):
    """Process entire page with Qwen3 VLM (no segmentation)."""

    # Get model configuration
    model_id = self.combo_qwen3_model.currentData()
    custom_adapter = self.txt_qwen3_adapter.text().strip()
    prompt = self.txt_qwen3_prompt.toPlainText().strip()
    max_tokens = self.spin_qwen3_max_tokens.value()
    max_img_size = self.spin_qwen3_img_size.value()

    from inference_qwen3 import QWEN3_MODELS, Qwen3VLMInference

    # Get model info
    model_config = QWEN3_MODELS[model_id]

    # Override adapter if custom one provided
    adapter = custom_adapter if custom_adapter else model_config["adapter"]

    try:
        self.status_bar.showMessage(f"Loading Qwen3 VLM: {model_id}...")

        # Initialize Qwen3
        self.qwen3 = Qwen3VLMInference(
            base_model=model_config["base"],
            adapter_model=adapter,
            device="auto" if self.device == "cuda" else "cpu",
            max_image_size=max_img_size
        )

        # Load full page image
        from PIL import Image
        page_image = Image.open(self.current_image_path)

        self.status_bar.showMessage("Transcribing full page with Qwen3...")
        self.btn_segment.setEnabled(False)
        self.btn_process.setEnabled(False)

        # Transcribe entire page
        result = self.qwen3.transcribe_page(
            page_image,
            prompt=prompt,
            max_new_tokens=max_tokens
        )

        # Display result
        self.text_editor.setPlainText(result.text)

        # Update status
        self.status_bar.showMessage(
            f"Qwen3 transcription complete! Time: {result.processing_time:.2f}s"
        )

        # Show memory usage
        memory_usage = self.qwen3.get_memory_usage()
        if memory_usage:
            memory_str = ", ".join([f"{gpu}: {stats['utilization']}" for gpu, stats in memory_usage.items()])
            print(f"GPU Usage: {memory_str}")

        self.btn_segment.setEnabled(False)  # No segmentation for Qwen3
        self.btn_process.setEnabled(True)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Qwen3 processing failed:\n{str(e)}")
        self.btn_segment.setEnabled(True)
        self.btn_process.setEnabled(True)

def _process_with_line_segmentation(self):
    """Existing line-based processing (TrOCR/PyLaia)."""
    # ... existing code for TrOCR ...
    pass
```

### Step 3: Hide Segmentation Controls for Qwen3

```python
def _on_model_tab_changed(self, index):
    """Handle model tab changes."""
    is_qwen3 = (index == 3)  # Qwen3 tab

    # Hide/show segmentation section
    self.segmentation_group.setVisible(not is_qwen3)

    # Hide/show line count
    self.lbl_lines_count.setVisible(not is_qwen3)

    # Update button text
    if is_qwen3:
        self.btn_process.setText("Transcribe Page")
        self.btn_segment.setVisible(False)
    else:
        self.btn_process.setText("Process All Lines")
        self.btn_segment.setVisible(True)

# Connect in __init__
self.model_tabs.currentChanged.connect(self._on_model_tab_changed)
```

### Step 4: Update Export to Handle Qwen3 Results

```python
def _save_transcription(self):
    """Save transcription to file."""
    if not self.text_editor.toPlainText():
        QMessageBox.warning(self, "Warning", "No transcription to save!")
        return

    # Determine source
    is_qwen3 = (self.model_tabs.currentIndex() == 3)

    file_path, _ = QFileDialog.getSaveFileName(
        self, "Save Transcription", "",
        "Text Files (*.txt);;All Files (*)"
    )

    if file_path:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Add metadata header
                if is_qwen3:
                    model_id = self.combo_qwen3_model.currentData()
                    f.write(f"# Generated by Qwen3 VLM ({model_id})\n")
                    f.write(f"# Source: {self.current_image_path}\n\n")

                f.write(self.text_editor.toPlainText())

            self.status_bar.showMessage(f"Saved: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
```

## Workflow Comparison

### TrOCR Workflow
```
1. User: Load image
2. User: Click "Segment Lines" → Wait
3. GUI: Shows line boxes on image
4. User: Click "Process All Lines" → Wait
5. GUI: Shows transcription with confidence
```

### Qwen3 Workflow
```
1. User: Load image
2. User: Click "Transcribe Page" → Wait (longer)
3. GUI: Shows full page transcription
```

**Simpler but slower per page**

## Performance Expectations

### TrOCR (Line-based)
- Segmentation: 1-5 seconds
- Recognition: 0.1s per line × 30 lines = 3 seconds
- **Total: 4-8 seconds per page**

### Qwen3 VLM (8B model)
- No segmentation needed
- Recognition: 10-30 seconds per page
- **Total: 10-30 seconds per page**

**Trade-off**: Slower but handles complex layouts better

## Memory Requirements

| Model | VRAM | Multi-GPU |
|-------|------|-----------|
| TrOCR Base | 2 GB | No |
| TrOCR Large | 4 GB | No |
| Qwen3 2B | 4-6 GB | Optional |
| Qwen3 8B | 12-16 GB | **Recommended** |

**Your Setup**: 2× RTX 4090 (24GB each) = Perfect for Qwen3 8B!

## Prompt Engineering

### Basic Prompts
```python
prompts = {
    "basic": "Transcribe the text shown in this image.",

    "detailed": "Extract all handwritten text from this page, preserving the original language and formatting.",

    "line_by_line": "Transcribe all text line by line, maintaining the original order.",

    "with_structure": "Transcribe this document preserving its structure, including paragraphs and line breaks.",

    "ukrainian": "Транскрибуй весь текст з цього зображення українською мовою.",
}
```

### Advanced Prompts (for finetuned models)
```python
# For historical documents
"Transcribe this historical handwritten document, preserving archaic spelling and abbreviations."

# For specific scripts
"Transcribe the Cyrillic handwritten text, maintaining original orthography."

# With metadata
"Transcribe this page and indicate any uncertain or illegible sections with [?]."
```

## Adding New Finetuned Models

When new Qwen3 adapters become available:

```python
# In inference_qwen3.py, add to QWEN3_MODELS dict:

"qwen3-vl-8b-ukrainian-handwriting": {
    "base": "Qwen/Qwen3-VL-8B-Instruct",
    "adapter": "your-username/qwen3-ukrainian-handwriting",
    "description": "Finetuned for Ukrainian historical handwriting",
    "vram": "12-16 GB",
    "speed": "Medium"
},
```

GUI will automatically detect and add to dropdown!

## Testing Strategy

### Phase 1: Single Page Test
1. Load Qwen3 8B with Old Church Slavonic adapter
2. Test on 1 Ukrainian page
3. Compare with TrOCR results
4. Evaluate:
   - Accuracy (CER)
   - Layout preservation
   - Processing time

### Phase 2: Batch Test
1. Process 10 representative pages
2. Measure:
   - Average CER
   - Time per page
   - Memory usage
   - GPU utilization

### Phase 3: Prompt Optimization
1. Test different prompts
2. Find optimal prompt for Ukrainian handwriting
3. Save as preset in GUI

## Advantages Over TrOCR

1. ✅ **No segmentation errors** - Most OCR errors come from bad segmentation
2. ✅ **Layout awareness** - Understands page structure
3. ✅ **Context usage** - Uses surrounding text for better accuracy
4. ✅ **Flexible** - Can customize via prompts
5. ✅ **Modern architecture** - State-of-the-art VLM

## Disadvantages

1. ❌ **Slower** - 3-5x slower than line-based
2. ❌ **More VRAM** - Needs 12-16 GB
3. ❌ **No per-line confidence** - Only page-level output
4. ❌ **Less predictable** - VLMs can be inconsistent
5. ❌ **Harder to debug** - Can't see which part failed

## When to Use Qwen3 vs TrOCR

### Use Qwen3 VLM When:
- Complex page layouts
- Segmentation is failing
- Need layout preservation
- Have adequate VRAM (12+ GB)
- Quality > Speed

### Use TrOCR When:
- Simple line-by-line documents
- Need fast processing
- Limited VRAM (<8 GB)
- Need per-line confidence
- Speed > Layout

## Future Enhancements

### Short-term
1. Add preset prompts dropdown
2. Save/load custom prompts
3. Batch processing for Qwen3
4. Temperature/sampling controls

### Long-term
1. Hybrid mode: Qwen3 for layout detection + TrOCR for recognition
2. Confidence estimation for Qwen3 (using multiple runs)
3. Post-processing: Structure extraction from Qwen3 output
4. Fine-tune Qwen3 on your Ukrainian data

## Installation Requirements

```bash
# Additional packages for Qwen3
pip install transformers>=4.37.0
pip install accelerate  # For multi-GPU
pip install peft  # For LoRA adapters
pip install bitsandbytes  # Optional: 8-bit quantization

# Verify installation
python -c "from transformers import Qwen3VLForConditionalGeneration; print('Qwen3 OK')"
```

## Summary

Qwen3 VLM integration adds a powerful alternative to line-based OCR:
- **No segmentation needed** - Biggest advantage
- **Better for complex layouts**
- **Requires more VRAM but you have it**
- **Slower but potentially more accurate**
- **Easy to integrate** - Just add one more tab to GUI

The implementation is straightforward and follows the same patterns as TrOCR integration.
