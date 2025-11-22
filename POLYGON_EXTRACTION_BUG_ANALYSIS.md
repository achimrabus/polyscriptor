# Polygon Extraction Bug Analysis - Rotated Pages

## Executive Summary

**ROOT CAUSE FOUND**: ~20% of Prosta Mova validation images have **vertically rotated text** (letters are vertical instead of horizontal) due to EXIF orientation tags not being respected during polygon extraction.

## The Problem

User reported seeing "orthogonal line snippets" - line images where:
- The image dimensions are correct (horizontal, e.g., 1939×128px)
- But the **text content itself is rotated 90 degrees** (vertical letters)
- Text is unreadable because it's sideways

## Investigation Results

### 1. Affected Images

**Sample of 30 validation images:**
- 6 images (20%) have vertically rotated text
- All from the same source pages: 0266, 0442, 0506
- Edge gradient analysis shows horizontal dominance (ratio < 0.8)

**Examples:**
```
❌ 0026_bibliasiriechkni01luik_orig_0266_region_1566558943691_1159l21.png
   Edge ratio: 0.54 (vertical text)

❌ 0028_bibliasiriechkni01luik_orig_0506_region_1567116598630_152l48.png
   Edge ratio: 0.00 (blank/corrupted)
   Size: 88832×128px (!) - massively wrong width
```

### 2. Root Cause: EXIF Orientation Not Handled

**Source Images:**
```
0028_bibliasiriechkni01luik_orig_0506.jpg: 4368×2912px
  EXIF Orientation: 8 (Rotate 270° CW / 90° CCW)

0026_bibliasiriechkni01luik_orig_0266.jpg: 4368×2912px
  EXIF Orientation: 8 (Rotate 270° CW / 90° CCW)

0027_bibliasiriechkni01luik_orig_0442.jpg: 4368×2912px
  EXIF Orientation: 8 (Rotate 270° CW / 90° CCW)
```

**EXIF Orientation Tag Values:**
- 1 = Normal
- 3 = Rotate 180°
- 6 = Rotate 90° CW (270° CCW)
- **8 = Rotate 270° CW (90° CCW)** ← Our problem

**What Happens:**
1. Transkribus creates PAGE XML with coordinates assuming **portrait orientation** (2912×4368)
2. Source JPG is physically stored in **landscape orientation** (4368×2912) with EXIF tag 8
3. `transkribus_parser.py` uses `Image.open()` which **does NOT auto-rotate**
4. PAGE XML coordinates (e.g., x=1169, y=806) are applied to **wrong orientation**
5. Result: Extracting horizontal slice from vertically-oriented text = rotated snippet

### 3. Why V2 (Bounding Box) Worked Better

V2 may have had different handling or these specific pages weren't included in V2 training/validation split.

### 4. The 88,832px Width Mystery

One image (`152l48.png`) has width of **88,832 pixels** - this is catastrophically wrong. This occurs when:
- Polygon coordinates span nearly the entire wrong dimension
- With incorrect orientation, a vertical polygon becomes horizontal
- A 2000px tall region becomes a 2000px+ wide region after incorrect extraction

## The Bug in transkribus_parser.py

**Location:** Line 212 in `extract_lines_from_page()`

```python
page_image = Image.open(image_path).convert('RGB')
```

**Problem:** `Image.open()` loads the raw pixel data without applying EXIF transformations.

**Required Fix:** Use `ImageOps.exif_transpose()` to auto-rotate based on EXIF:

```python
from PIL import Image, ImageOps

page_image = Image.open(image_path)
page_image = ImageOps.exif_transpose(page_image)  # Apply EXIF rotation
page_image = page_image.convert('RGB')
```

## Impact Assessment

### Validation Set
- **20% of images affected** (6/30 sample)
- Specific pages: 0266, 0442, 0506
- Makes these lines completely untrainable (text is sideways)

### Training Set
Need to check if training set also has rotated pages:
```bash
find /home/achimrabus/htr_gui/Prosta_Mova/Prosta_Mova_train -name "*.jpg" -exec identify -format "%f %[EXIF:Orientation]\n" {} \; | grep -v "^$" | sort | uniq -c
```

### Model Performance Impact
- V3 validation avg height: 80.7px (vs 67.2px training)
- But height isn't the only issue - **20% of validation data is garbage**
- This explains why V3 didn't improve over V2

## Solution

### Fix transkribus_parser.py

```python
# Line 212 - BEFORE:
page_image = Image.open(image_path).convert('RGB')

# Line 212 - AFTER:
page_image = Image.open(image_path)
page_image = ImageOps.exif_transpose(page_image)  # Handle EXIF rotation
page_image = page_image.convert('RGB')
```

### Re-export V4 Dataset

After fixing the code:
```bash
# Training set
python transkribus_parser.py \
    --input_dir /home/achimrabus/htr_gui/Prosta_Mova/Prosta_Mova_train \
    --output_dir data/pylaia_prosta_mova_v4_train \
    --train_ratio 1.0 \
    --preserve-aspect-ratio \
    --target-height 128 \
    --use-polygon-mask \
    --num-workers 12

# Validation set
python transkribus_parser.py \
    --input_dir /home/achimrabus/htr_gui/Prosta_Mova/Prosta_Mova_val \
    --output_dir data/pylaia_prosta_mova_v4_val \
    --train_ratio 1.0 \
    --preserve-aspect-ratio \
    --target-height 128 \
    --use-polygon-mask \
    --num-workers 12
```

### Verify Fix

```python
# Check that no more rotated text exists
python3 << 'EOF'
from PIL import Image
import numpy as np
import glob

def check_rotation(img_path):
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    gy, gx = np.gradient(img_array.astype(float))
    ratio = np.abs(gx).sum() / (np.abs(gy).sum() + 1)
    return ratio < 0.8  # True if rotated

rotated_count = 0
total_count = 0

for img_path in glob.glob('data/pylaia_prosta_mova_v4_val/line_images/*.png'):
    total_count += 1
    if check_rotation(img_path):
        rotated_count += 1
        print(f"Still rotated: {img_path}")

print(f"\nRotation bug rate: {rotated_count}/{total_count} = {100*rotated_count/total_count:.1f}%")
print("Target: 0%")
EOF
```

## Expected Outcome

After fix:
- ✅ All text oriented correctly (horizontal)
- ✅ No more 88,000px wide images
- ✅ Validation set usable for training
- ✅ Potential CER improvement (eliminating 20% garbage data)

## Files for User Inspection

Rotated image samples saved to project root:
```
rotated_0026_bibliasiriechkni01luik_orig_0266_region_1566558943691_1159l21.png
rotated_0027_bibliasiriechkni01luik_orig_0442_region_1567088727934_413l3.png
rotated_0028_bibliasiriechkni01luik_orig_0506_region_1567116598630_152l48.png
rotated_0027_bibliasiriechkni01luik_orig_0442_region_1567088727934_413l1.png
rotated_0026_bibliasiriechkni01luik_orig_0266_region_1566558943691_1159l45.png
rotated_0026_bibliasiriechkni01luik_orig_0266_region_1566558980199_1172l4.png
```

User can open these to see the vertical text problem.
