# HTR Dataset Preprocessing Checklist

## ⚠️ MANDATORY CHECKS - DO NOT SKIP

This checklist prevents common data quality bugs that can ruin training. Follow every step before starting training.

---

## 1. ⚠️ CRITICAL: EXIF Rotation Check

**Why**: 32% of Prosta Mova training data had vertical text due to ignored EXIF tags. This bug cost weeks of training time.

### Check for EXIF-rotated images:

```bash
# Count images with rotation tags (EXIF 6 or 8)
find <dataset_dir> -name "*.jpg" -exec identify -format "%f %[EXIF:Orientation]\n" {} \; | \
    grep -E " (6|8)$" | wc -l

# If count > 0, YOU MUST use ImageOps.exif_transpose()
```

### Verify transkribus_parser.py has EXIF handling:

```python
# Lines 212-215 MUST contain:
page_image = Image.open(image_path)
page_image = ImageOps.exif_transpose(page_image)  # ← CRITICAL LINE
page_image = page_image.convert('RGB')
```

### EXIF Tag Reference:
- `1` = Normal (no rotation)
- `3` = Rotate 180°
- `6` = Rotate 90° CW (270° CCW)
- `8` = Rotate 270° CW (90° CCW) ← **Common in smartphone photos**

**Impact of missing this**:
- Text appears vertical instead of horizontal
- Model cannot learn from rotated data
- Training CER stays above 15-20% even after many epochs
- Validation CER similarly poor

---

## 2. Visual Inspection of Random Samples

**Why**: Automated checks can miss subtle issues. Human eyes catch orientation bugs instantly.

### Create inspection notebook:

```python
import random
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
train_df = pd.read_csv('data/<dataset>/train.csv')
val_df = pd.read_csv('data/<dataset>/val.csv')

# Sample random images
train_sample = train_df.sample(30)
val_sample = val_df.sample(15)

# Display
fig, axes = plt.subplots(15, 2, figsize=(20, 50))
for idx, (_, row) in enumerate(train_sample.iterrows()[:15]):
    img = Image.open(row['image_path'])
    axes[idx, 0].imshow(img, cmap='gray')
    axes[idx, 0].set_title(f"Train: {row['text'][:50]}")
    axes[idx, 0].axis('off')
```

### What to look for:
- ✅ Text baseline is horizontal
- ✅ Characters are upright (not 90° rotated)
- ✅ No blank/corrupted images
- ✅ Reasonable aspect ratios (width >> height for lines)
- ✅ Text is readable (not too blurry or tiny)

**Red flags**:
- ❌ Vertical text (letters sideways)
- ❌ Images with extreme widths (>10,000px) or heights
- ❌ Blank white/black images
- ❌ Text cut off at top/bottom

---

## 3. Dataset Statistics Validation

**Why**: Abnormal statistics often indicate preprocessing bugs.

### Check average line heights:

```bash
cat data/<dataset>/dataset_info.json | grep avg_line_height
```

### Expected ranges:
- **Excellent**: 40-50px (Church Slavonic: 42.8px → 3.17% CER)
- **Good**: 50-70px (Prosta Mova V4: 64.0px → 7.5% CER)
- **Acceptable**: 70-90px (may limit performance)
- **⚠️ Problematic**: >90px (likely loose segmentation or rotation bug)

### Check line count:
- **Minimum for good performance**: 10,000+ lines
- **Recommended**: 50,000+ lines (Prosta Mova V4: 58,843)
- **Excellent**: 300,000+ lines (Church Slavonic: 309,959)

### Check vocabulary size:
```bash
wc -l data/<dataset>/syms.txt
```
- **Typical**: 100-300 symbols
- **⚠️ Too small**: <50 symbols (may be missing characters)
- **⚠️ Too large**: >500 symbols (may include noise/unicode variants)

---

## 4. Verify Preprocessing Consistency

**Why**: Inference must match training preprocessing exactly.

### Check dataset_info.json:

```json
{
  "preserve_aspect_ratio": true,    // ← Should be true for line images
  "target_height": 128,              // ← Standard for PyLaia/TrOCR
  "background_normalized": false,    // ← Note for inference
  "use_polygon_mask": true,          // ← Polygon vs bounding box
  "total_lines": 58843,
  "avg_line_height": 64.0
}
```

### Document preprocessing for inference:
- If `background_normalized: true` → inference MUST use `--normalize-background`
- If `target_height: 128` → inference should resize to 128px
- If `preserve_aspect_ratio: true` → inference should preserve aspect ratio

---

## 5. Train CER Early Warning System

**Why**: High train CER after 10 epochs indicates data quality issues, not model problems.

### Monitor training logs:

```
Epoch 1:  Train CER: 45%, Val CER: 48%
Epoch 5:  Train CER: 18%, Val CER: 22%
Epoch 10: Train CER: 8%,  Val CER: 10%  ✅ Healthy
```

### Red flags:
- **Train CER stuck above 20% after 10 epochs** → Check for:
  - EXIF rotation bug (text vertical)
  - Vocabulary mismatch (symbols missing)
  - Corrupted images
- **Train CER < 5% but Val CER > 15%** → Overfitting:
  - Need more training data
  - Increase augmentation
  - Reduce model capacity

### Healthy training pattern:
- Epoch 1-5: Train CER drops rapidly (45% → 10%)
- Epoch 5-20: Train and Val CER converge (gap < 3%)
- Epoch 20+: Both plateau together (early stopping triggers)

---

## 6. Vocabulary Format Validation

**Why**: Wrong parsing of KALDI format caused 100% CER in Glagolitic training.

### Check vocabulary file format:

**List format** (one symbol per line):
```
<SPACE>
о
а
и
```

**KALDI format** (symbol + index):
```
<space> 1
о 27
а 28
и 29
```

### Verify parser handles both:

```python
# train_pylaia.py lines 64-96 should auto-detect format
with open(symbols_file, 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    if ' ' in first_line and first_line.split()[-1].isdigit():
        # KALDI format: split on space, take first part
        symbols = [line.split()[0].rstrip() for line in f]
    else:
        # List format: one per line
        symbols = [line.rstrip('\n\r') for line in f]
```

### Check space token handling:
- `<SPACE>` or `<space>` should map to actual space `' '`
- Do NOT use `.strip()` (removes TAB characters)
- Use `.rstrip('\n\r')` instead

---

## 7. GPU Memory and Performance Check

**Why**: OOM errors or slow training indicate configuration issues.

### Before training starts:

```bash
# Check GPU availability
nvidia-smi

# Check PyTorch sees GPUs
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### Expected batch sizes (for NVIDIA L40S 48GB):
- **PyLaia CRNN**: 32-64 (58K lines takes ~12 min/epoch)
- **TrOCR**: 16-32
- **Qwen3-VL**: 2-4 (very memory hungry)

### If OOM occurs:
- Reduce batch_size (32 → 16 → 8)
- Disable image caching (`cache_images: false`)
- Reduce num_workers (12 → 4)

---

## 8. Sanity Test Export

**Why**: Test on small subset before committing to full export.

### Test on single page:

```bash
# Create test output with just 1 page
python transkribus_parser.py \
    --input_dir <dataset_dir> \
    --output_dir data/<dataset>_test \
    --train_ratio 1.0 \
    --preserve-aspect-ratio \
    --target-height 128 \
    --use-polygon-mask \
    --num-workers 1
```

### Verify test output:
- ✅ Line images look horizontal
- ✅ Reasonable file sizes (100-500 KB per line)
- ✅ Vocabulary file created
- ✅ No error messages in log

**Only proceed with full export after test passes.**

---

## 9. Version Control Dataset Metadata

**Why**: Track which preprocessing was used for reproducibility.

### Save to dataset directory:

```bash
# Create metadata file
cat > data/<dataset>/PREPROCESSING_LOG.md << EOF
# Dataset: <name>
# Date: $(date +%Y-%m-%d)
# Preprocessed by: $(whoami)

## Source
- Input: <path_to_transkribus_export>
- Pages: <count>

## Preprocessing Command
\`\`\`bash
python transkribus_parser.py \\
    --input_dir <dir> \\
    --output_dir <dir> \\
    --preserve-aspect-ratio \\
    --target-height 128 \\
    --use-polygon-mask \\
    --num-workers 12
\`\`\`

## Statistics
- Training lines: <count>
- Validation lines: <count>
- Avg line height: <px>
- Vocabulary size: <count>

## Quality Checks
- [x] EXIF rotation handled
- [x] Visual inspection passed (30 train + 15 val samples)
- [x] No vertical text found
- [x] Average line height in acceptable range
- [x] Vocabulary format validated
EOF
```

---

## 10. Final Checklist Before Training

**ALL items must be checked:**

- [ ] **EXIF rotation**: Verified `ImageOps.exif_transpose()` in code
- [ ] **EXIF tags checked**: Counted rotated source images
- [ ] **Visual inspection**: Reviewed 30+ random samples, all horizontal
- [ ] **Line heights**: Within expected range (40-90px)
- [ ] **Vocabulary**: Format validated, space token correct
- [ ] **Preprocessing metadata**: Saved to dataset directory
- [ ] **Test export**: Sanity test passed on 1 page
- [ ] **Full export**: Completed without errors
- [ ] **Statistics**: Training/val split reasonable (80/20 or 90/10)
- [ ] **GPU check**: CUDA available, sufficient memory

**If ANY item unchecked → DO NOT START TRAINING**

---

## Historical Bug Reference

### Prosta Mova V2/V3 EXIF Rotation Bug (2025-11-21)

**Symptom**: 19% CER despite 58K training lines and proven architecture

**Root cause**: 32% of training data had vertical text due to ignored EXIF tags

**Detection**: Visual inspection showed "orthogonal line snippets"

**Fix**: Added `ImageOps.exif_transpose()` at line 212 of transkribus_parser.py

**Result**: V4 achieved 7.5% CER (61% improvement)

**Lesson**: ALWAYS check EXIF orientation tags before extracting line images

---

## Quick Reference: Common Data Quality Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **Text appears vertical** | EXIF rotation not handled | Add `ImageOps.exif_transpose()` |
| **Train CER stuck at 20%+** | Data quality issue | Check for rotation, vocabulary bugs |
| **Extreme image widths** | Wrong orientation + polygon coords | Fix EXIF handling |
| **Blank/corrupted images** | PAGE XML coord mismatch | Check EXIF orientation |
| **Train CER < 5%, Val CER > 15%** | Overfitting | More data or augmentation |
| **Model outputs gibberish** | Vocabulary mismatch | Check idx2char mapping |
| **Spaces missing in output** | `<SPACE>` not mapped to `' '` | Fix idx2char space handling |

---

## See Also

- [INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md) - Prosta Mova EXIF bug analysis
- [TRAIN_CER_LOGGING_EXPLANATION.md](TRAIN_CER_LOGGING_EXPLANATION.md) - How to use train CER for debugging
- [PYLAIA_FIXES_SUMMARY_20251106.md](PYLAIA_FIXES_SUMMARY_20251106.md) - Vocabulary parsing bugs
- [CLAUDE.md](CLAUDE.md) - Full project documentation
