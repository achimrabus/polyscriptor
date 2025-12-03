# Investigation Summary - Prosta Mova V3 Polygon Extraction Issues

## ⚠️ CRITICAL BUG ALERT - EXIF ROTATION

**ALWAYS CHECK THIS FIRST** when investigating poor HTR performance or rotated/vertical text in line images.

## Problem Discovered

User reported seeing "orthogonal line snippets" in the V3 validation dataset - line images where the text itself was rotated 90 degrees (vertical letters instead of horizontal).

## Root Cause: EXIF Orientation Tags

**Issue**: 32% of training images and 11% of validation images have EXIF orientation tags indicating they're stored rotated. The `transkribus_parser.py` script was not respecting these EXIF tags, causing:

1. **Rotated text**: Letters appearing vertical instead of horizontal
2. **Wrong coordinates**: PAGE XML coordinates assume correct orientation, but images were loaded in wrong orientation
3. **Corrupted extractions**: Some images had extreme widths (88,832px!) or were completely blank

## Impact Assessment

### Training Set (sample of 100)
- **68 images**: Normal orientation (EXIF=1)
- **32 images**: Rotated (EXIF=6 or EXIF=8) → **32% affected**

### Validation Set (sample of 47)
- **42 images**: Normal orientation (EXIF=1)
- **5 images**: Rotated (EXIF=6 or EXIF=8) → **11% affected**

### EXIF Orientation Values
- `1` = Normal
- `6` = Rotate 90° CW (270° CCW)
- `8` = Rotate 270° CW (90° CCW)

## The Fix

### Code Changes to transkribus_parser.py

**Line 22**: Added `ImageOps` import
```python
from PIL import Image, ImageDraw, ImageOps
```

**Lines 212-215**: Apply EXIF rotation before processing
```python
page_image = Image.open(image_path)
# CRITICAL: Apply EXIF orientation (handles rotated images)
page_image = ImageOps.exif_transpose(page_image)
page_image = page_image.convert('RGB')
```

### Verification Results

✓ Tested on page 0506 (previously had rotated text)
✓ **10/10 images correctly oriented** - no more vertical text
✓ Normal image sizes (1890-2081px × 128px)
✓ No more blank/corrupted images

## Required Actions

### 1. Re-export BOTH Training and Validation Sets

**Training** (will take ~5 minutes with 12 workers):
```bash
source htr_gui/bin/activate && python transkribus_parser.py \
    --input_dir /home/achimrabus/htr_gui/Prosta_Mova/Prosta_Mova_train \
    --output_dir data/pylaia_prosta_mova_v4_train \
    --train_ratio 1.0 \
    --preserve-aspect-ratio \
    --target-height 128 \
    --use-polygon-mask \
    --num-workers 12
```

**Validation** (will take ~1 minute with 12 workers):
```bash
source htr_gui/bin/activate && python transkribus_parser.py \
    --input_dir /home/achimrabus/htr_gui/Prosta_Mova/Prosta_Mova_val \
    --output_dir data/pylaia_prosta_mova_v4_val \
    --train_ratio 1.0 \
    --preserve-aspect-ratio \
    --target-height 128 \
    --use-polygon-mask \
    --num-workers 12
```

### 2. Verify Dataset Quality

After export, check for rotation issues:
```bash
source htr_gui/bin/activate && python3 << 'EOF'
from PIL import Image
import numpy as np
import glob

def check_rotation(img_path):
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    gy, gx = np.gradient(img_array.astype(float))
    ratio = np.abs(gx).sum() / (np.abs(gy).sum() + 1)
    return ratio < 0.8

# Check random sample
import random
train_images = list(glob.glob('data/pylaia_prosta_mova_v4_train/line_images/*.png'))
val_images = list(glob.glob('data/pylaia_prosta_mova_v4_val/line_images/*.png'))

sample = random.sample(train_images + val_images, 100)
rotated = sum(1 for img in sample if check_rotation(img))

print(f"Rotation check: {rotated}/100 images still rotated")
print(f"Expected: 0/100")
print(f"Status: {'✓ PASS' if rotated == 0 else '❌ FAIL'}")
EOF
```

### 3. Compare V3 vs V4 Dataset Statistics

After export, check if the fix improved data quality:
```bash
# Compare line heights
cat data/pylaia_prosta_mova_v3_train/dataset_info.json | grep avg_line_height
cat data/pylaia_prosta_mova_v4_train/dataset_info.json | grep avg_line_height

cat data/pylaia_prosta_mova_v3_val/dataset_info.json | grep avg_line_height
cat data/pylaia_prosta_mova_v4_val/dataset_info.json | grep avg_line_height
```

### 4. Train Prosta Mova V4 Model

Once V4 dataset is verified:
```bash
python start_pylaia_prosta_mova_v4_training.py
```

## Expected Outcomes

### Data Quality Improvements
- ✅ All text correctly oriented (no vertical letters)
- ✅ Proper coordinate extraction (no 88k pixel widths)
- ✅ No blank/corrupted images
- ✅ Consistent line heights

### Potential CER Improvements
- **V2/V3**: 19% CER (with 20-32% corrupted data)
- **V4 Target**: < 15% CER (with clean data)
- **Ultimate Goal**: < 10% CER (matching Church Slavonic quality)

## Files Generated

### Documentation
- `POLYGON_EXTRACTION_BUG_ANALYSIS.md` - Detailed technical analysis
- `INVESTIGATION_SUMMARY.md` - This file (executive summary)

### Test Outputs
- `test_fixed_0506.png` - Visual verification sample
- `rotated_*.png` (6 files) - Examples of the bug (for reference)
- `data/pylaia_prosta_mova_v4_test/` - Test export with fix applied

### Code Changes
- `transkribus_parser.py` - EXIF rotation fix applied (lines 22, 212-215)

## ⚠️ PREVENTION CHECKLIST FOR FUTURE DATASETS

To avoid this bug in the future, **ALWAYS**:

1. **Check EXIF tags** before data extraction:
   ```bash
   # Count images with rotation tags
   find <dataset_dir> -name "*.jpg" -exec identify -format "%f %[EXIF:Orientation]\n" {} \; | \
       grep -E " (6|8)$" | wc -l
   ```

2. **Verify transkribus_parser.py has EXIF handling**:
   ```python
   # Line 212-215 should contain:
   page_image = Image.open(image_path)
   page_image = ImageOps.exif_transpose(page_image)  # CRITICAL!
   page_image = page_image.convert('RGB')
   ```

3. **Visual inspection of random samples**:
   - Sample 30 training + 15 validation images
   - Look for vertical/rotated text
   - Check if text baseline is horizontal

4. **Check for train CER anomalies**:
   - If train CER stays above 15-20% after 10 epochs → likely data quality issue
   - If train CER high AND val CER high → check for rotation bug

5. **Compare average line heights** to known-good datasets:
   - Church Slavonic: 42.8px (excellent)
   - Prosta Mova V4: 64.0px (good)
   - If > 80px → likely segmentation or rotation issue

## Next Steps for User

1. **Review** this summary and `POLYGON_EXTRACTION_BUG_ANALYSIS.md`
2. **Inspect** `test_fixed_0506.png` to visually verify fix
3. **Run** training and validation re-exports (commands above)
4. **Verify** dataset quality (rotation check script above)
5. **Train** V4 model with clean data
6. **Compare** V4 CER vs V2/V3 CER

## Secondary Investigation: Church Slavonic Comparison

A parallel investigation found that Church Slavonic has tighter line segmentation (42.8px avg height vs Prosta Mova's 67.2px). This 36% height difference likely contributes to the CER gap (3.17% vs 19%).

Solution created: `tighten_page_xml.py` script to automatically tighten loose PAGE XML polygons (see `SEGMENTATION_ANALYSIS_REPORT.md`).

However, **fixing the rotation bug should be done first** before attempting to tighten polygons, as rotation issues could interfere with polygon analysis.
