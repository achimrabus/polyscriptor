# GUI Improvement Plan - Context-Aware Interface

## Overview
Create a smoother, more intuitive GUI that dynamically shows/hides controls based on the active model tab (TrOCR vs Qwen3 VLM), with better defaults and improved usability.

---

## 1. Context-Aware UI Elements

### Problem
Currently, line segmentation controls are visible even when using Qwen3 VLM, which doesn't need segmentation. This creates confusion and clutters the interface.

### Solution: Dynamic Control Visibility

#### Controls to Hide in Qwen3 Mode:
- **Line Segmentation Group** (entire `seg_group`)
  - Method selection dropdown
  - Kraken model selection
  - "Detect Lines" button
  - HPP threshold/min-height controls
  - Line count label

#### Controls to Show in Qwen3 Mode:
- **Qwen3-specific settings** (already in tab)
  - Model source (preset/custom)
  - Prompt customization
  - Max tokens, image size
  - Confidence estimation checkbox

#### Implementation Strategy:
```python
def _on_model_tab_changed(self, index):
    """Handle model tab changes - show/hide controls based on context."""
    is_qwen3 = QWEN3_AVAILABLE and (index == self.model_tabs.count() - 1)

    # Hide/show segmentation group
    self.seg_group.setVisible(not is_qwen3)

    # Update button text
    if is_qwen3:
        self.btn_process.setText("Transcribe Page")
    else:
        self.btn_process.setText("Process All Lines")
```

**Files to modify:**
- `transcription_gui_qt.py` lines 338-420 (segmentation group)
- `transcription_gui_qt.py` lines 1078-1100 (`_on_model_tab_changed`)

---

## 2. Default Settings Changes

### Current Defaults vs Proposed

| Setting | Current | Proposed | Rationale |
|---------|---------|----------|-----------|
| Device | CPU | **GPU (cuda)** | GPU is 5-20x faster, most users have CUDA |
| Font Size | 10pt | **12pt** | Better readability, modern standard |
| Confidence Display | ON | ON | Keep - users want to see this |
| Segmentation | Kraken | Kraken | Keep - more robust than HPP |

### Implementation:
```python
# In __init__ (line ~228):
# Change from:
self.device = "cpu"

# To:
self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# In text editor setup (find QTextEdit creation):
# Add font size configuration:
default_font = QFont()
default_font.setPointSize(12)  # Increase from default ~10
self.text_editor.setFont(default_font)
```

**Files to modify:**
- `transcription_gui_qt.py` line 229 (device default)
- `transcription_gui_qt.py` around line 450-470 (text editor font)

---

## 3. Qwen3 VLM Speed Optimization Guide

### Factors Affecting Qwen3 Speed

#### 1. **Token Generation (PRIMARY BOTTLENECK)**
- **Impact**: 🔴 CRITICAL
- **`max_new_tokens` parameter**: Each token requires one forward pass
- **Speed per token**: ~50-200ms depending on model size and hardware

**Token Count Analysis:**
```
Average page transcriptions:
- Short page (10 lines):    ~200-500 tokens   → 10-25 seconds
- Medium page (25 lines):   ~500-1500 tokens  → 25-75 seconds
- Long page (50+ lines):    ~1500-4000 tokens → 75-200 seconds
```

**Recommendation:**
- Start with `max_new_tokens=1024` for testing
- Increase to `2048` for production
- Only use `4096+` for very dense historical manuscripts

#### 2. **Image Size**
- **Impact**: 🟡 MODERATE
- **`max_image_size` parameter**: Larger images = more vision tokens to process

**Image Size Trade-offs:**
```
512x512:   Fast (2-3s processing) but may miss fine details
1024x1024: Balanced (5-8s) - good for most manuscripts
1536x1536: Slow (10-15s) but preserves fine details (CURRENT DEFAULT)
2048x2048: Very slow (20-30s) - only for high-res requirements
```

**Recommendation:**
- Default: `1536` (current) for historical manuscripts
- Reduce to `1024` if speed is critical
- Increase to `2048` only for very high-DPI scans

#### 3. **Confidence Estimation**
- **Impact**: 🟢 MINOR (~5-10% overhead)
- Extracting token probabilities adds minimal processing time
- Worthwhile for quality assessment

#### 4. **GPU Memory & Batch Size**
- **Impact**: 🟡 MODERATE
- Qwen3-VL-8B requires ~16-20GB VRAM in float16
- Multi-GPU distribution helps but doesn't speed up single-page inference
- Batch processing not applicable (one page at a time in GUI)

#### 5. **Model Precision**
- **Current**: `torch.float16` (half precision)
- **Alternative**: `torch.bfloat16` on newer GPUs (A100, 4090)
- **Impact**: Minimal speed difference, but bfloat16 can be more stable

#### 6. **Beam Search**
- **Current**: `num_beams=1` (greedy decoding)
- **Alternative**: `num_beams=4-5` for better quality
- **Impact**: 🔴 CRITICAL - each beam multiplies processing time
- **Recommendation**: Keep at 1 for speed, increase to 3-4 only if quality issues

### Speed Optimization Matrix

| Configuration | Speed | Quality | Use Case |
|---------------|-------|---------|----------|
| **Fast** | ⚡⚡⚡ | ⭐⭐ | Quick preview |
| `max_new_tokens=512` | | | |
| `max_image_size=1024` | | | |
| `num_beams=1` | | | |
| | | | |
| **Balanced (DEFAULT)** | ⚡⚡ | ⭐⭐⭐ | Production |
| `max_new_tokens=2048` | | | |
| `max_image_size=1536` | | | |
| `num_beams=1` | | | |
| | | | |
| **Quality** | ⚡ | ⭐⭐⭐⭐ | Critical documents |
| `max_new_tokens=4096` | | | |
| `max_image_size=2048` | | | |
| `num_beams=3` | | | |

### Practical Speed Expectations

**Hardware: RTX 3090 (24GB VRAM)**
- Fast config: ~10-20 seconds/page
- Balanced config: ~30-60 seconds/page
- Quality config: ~90-180 seconds/page

**Hardware: RTX 4090 (24GB VRAM)**
- Fast config: ~7-15 seconds/page
- Balanced config: ~20-40 seconds/page
- Quality config: ~60-120 seconds/page

**Hardware: A100 (40GB VRAM)**
- Fast config: ~5-10 seconds/page
- Balanced config: ~15-30 seconds/page
- Quality config: ~45-90 seconds/page

---

## 4. UI/UX Improvements

### 4.1 Add Speed Profile Presets (Optional Enhancement)

Add preset buttons in Qwen3 tab for quick configuration:

```python
# In Qwen3 tab advanced settings:
preset_layout = QHBoxLayout()
preset_layout.addWidget(QLabel("Speed Preset:"))

btn_fast = QPushButton("⚡ Fast")
btn_fast.clicked.connect(lambda: self._set_qwen3_preset("fast"))
preset_layout.addWidget(btn_fast)

btn_balanced = QPushButton("⚖️ Balanced")
btn_balanced.clicked.connect(lambda: self._set_qwen3_preset("balanced"))
preset_layout.addWidget(btn_balanced)

btn_quality = QPushButton("⭐ Quality")
btn_quality.clicked.connect(lambda: self._set_qwen3_preset("quality"))
preset_layout.addWidget(btn_quality)

# Add to advanced_layout
advanced_layout.addLayout(preset_layout, 2, 0, 1, 4)
```

```python
def _set_qwen3_preset(self, preset: str):
    """Apply Qwen3 speed/quality preset."""
    presets = {
        "fast": {
            "max_tokens": 512,
            "img_size": 1024,
            "confidence": False
        },
        "balanced": {
            "max_tokens": 2048,
            "img_size": 1536,
            "confidence": False
        },
        "quality": {
            "max_tokens": 4096,
            "img_size": 2048,
            "confidence": True
        }
    }

    config = presets[preset]
    self.spin_qwen3_max_tokens.setValue(config["max_tokens"])
    self.spin_qwen3_img_size.setValue(config["img_size"])
    self.chk_qwen3_confidence.setChecked(config["confidence"])
```

### 4.2 Real-time Token Estimate

Show estimated processing time based on current settings:

```python
# Add label in Qwen3 tab:
self.lbl_qwen3_estimate = QLabel("Est. time: ~30-60 sec/page")
self.lbl_qwen3_estimate.setStyleSheet("color: gray; font-style: italic;")

# Connect to spinbox changes:
self.spin_qwen3_max_tokens.valueChanged.connect(self._update_qwen3_estimate)
self.spin_qwen3_img_size.valueChanged.connect(self._update_qwen3_estimate)

def _update_qwen3_estimate(self):
    """Update estimated processing time."""
    tokens = self.spin_qwen3_max_tokens.value()
    img_size = self.spin_qwen3_img_size.value()

    # Rough estimate: 50ms per token + image processing overhead
    base_time = tokens * 0.05  # 50ms per token
    img_overhead = (img_size / 1536) * 10  # Image processing scales with size

    total_time = base_time + img_overhead

    if total_time < 30:
        estimate = f"Est. time: ~{int(total_time)}-{int(total_time*1.5)} sec/page"
    else:
        estimate = f"Est. time: ~{int(total_time/60)}-{int(total_time*1.5/60)} min/page"

    self.lbl_qwen3_estimate.setText(estimate)
```

---

## 5. Implementation Checklist

### Phase 1: Core Improvements (HIGH PRIORITY)
- [ ] Change default device to GPU (line 229)
- [ ] Increase default font size to 12pt (text editor setup)
- [ ] Store segmentation group as `self.seg_group` for visibility control
- [ ] Update `_on_model_tab_changed` to hide segmentation in Qwen3 mode
- [ ] Test tab switching shows/hides correct controls

### Phase 2: Optional Enhancements (MEDIUM PRIORITY)
- [ ] Add speed preset buttons to Qwen3 tab
- [ ] Add estimated processing time label
- [ ] Add tooltips explaining token/image size impact
- [ ] Add GPU memory usage display during Qwen3 processing

### Phase 3: Polish (LOW PRIORITY)
- [ ] Add keyboard shortcuts for preset switching
- [ ] Add visual indicator when GPU is being used
- [ ] Add warning if max_tokens is very high (>4096)
- [ ] Add batch processing support for multiple pages

---

## 6. Testing Plan

### Test Cases
1. **TrOCR mode**: Verify segmentation controls visible
2. **Qwen3 mode**: Verify segmentation controls hidden
3. **Tab switching**: Verify smooth transition between modes
4. **GPU detection**: Verify GPU is default when available
5. **Font size**: Verify text editor uses 12pt font
6. **Speed presets**: Verify presets update spinboxes correctly
7. **Token estimate**: Verify estimate updates with settings

### Performance Benchmarks
Run with different token settings and measure:
- 512 tokens: Time to transcribe sample page
- 1024 tokens: Time to transcribe sample page
- 2048 tokens: Time to transcribe sample page
- 4096 tokens: Time to transcribe sample page

---

## 7. Code Files to Modify

| File | Lines | Changes |
|------|-------|---------|
| `transcription_gui_qt.py` | 229 | Change device default to GPU |
| `transcription_gui_qt.py` | 338-420 | Store `seg_group` as instance variable |
| `transcription_gui_qt.py` | 450-470 | Increase text editor font to 12pt |
| `transcription_gui_qt.py` | 1078-1100 | Enhance `_on_model_tab_changed` for visibility |
| `transcription_gui_qt.py` | 525-555 | Add speed presets (optional) |
| `transcription_gui_qt.py` | New method | Add `_set_qwen3_preset()` |
| `transcription_gui_qt.py` | New method | Add `_update_qwen3_estimate()` |

---

## Summary

**Core Goals:**
1. ✅ **Context-aware UI**: Hide segmentation controls in Qwen3 mode
2. ✅ **Better defaults**: GPU, larger font
3. ✅ **Speed optimization**: Document token count impact
4. ✅ **User guidance**: Presets and estimates (optional)

**Expected Impact:**
- Cleaner, less cluttered interface
- Faster default experience (GPU)
- Better readability (12pt font)
- Clearer understanding of speed/quality trade-offs
- More professional, polished appearance
