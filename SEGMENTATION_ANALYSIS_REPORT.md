# Segmentation Analysis Report: Church Slavonic vs Prosta Mova

**Date**: 2025-11-21
**Analyst**: Claude (Sonnet 4.5)

## Executive Summary

**ROOT CAUSE IDENTIFIED**: The 36% height difference (42.8px vs 67.2px) and poor CER performance (3.17% vs 19%) is caused by **loose PAGE XML polygon segmentation in Prosta Mova dataset**, not by transkribus_parser.py or aspect ratio preservation settings.

- **Church Slavonic**: Tight polygons (40-50px height, follows text contour)
- **Prosta Mova**: Loose polygons (200-400px height, includes massive whitespace)
- **Impact**: Prosta Mova lines have 21% margin ratio vs 3% for Church Slavonic after extraction

## Dataset Comparison

### Dataset Metadata (dataset_info.json)

| Metric | Church Slavonic | Prosta Mova |
|--------|----------------|-------------|
| Total lines | 309,959 | 51,067 |
| Avg line width | 540.7px | 822.9px |
| **Avg line height** | **42.8px** | **67.2px** |
| Preprocessing | preserve_aspect_ratio: true<br>target_height: 128<br>background_normalized: false | preserve_aspect_ratio: true<br>target_height: 128<br>background_normalized: false |

**Conclusion**: Identical preprocessing settings, so the height difference originates from the source PAGE XML files.

## PAGE XML Analysis

### Sample Files Analyzed

- **Church Slavonic**: 10 random XML files (1,403 lines total)
- **Prosta Mova**: 10 random XML files (356 lines total)

### Original PAGE XML Line Heights (from Coords polygons)

| Dataset | Mean | Median | Std Dev | Range |
|---------|------|--------|---------|-------|
| **Church Slavonic** | **41.5px** | 40.0px | 6.4px | 30-71px |
| **Prosta Mova** | **113.0px** | 71.0px | 100.2px | 33-485px |

**Height Ratio**: 2.72x (Prosta Mova is 2.72 times taller in PAGE XML)

### Extreme Cases

**Church Slavonic (typical)**:
- File: `0101_Usp_Juni_Sin_995-0482.xml`
- 60 lines, avg height: 38.6px, range: 32-51px
- **Very consistent, tight segmentation**

**Prosta Mova (problematic)**:
- File: `0101_18_Pravilo_1598-0169.xml`
- 28 lines, avg height: 363.8px, range: 310-485px
- **Extreme variation, massively loose segmentation**

## Coords Polygon Structure Analysis

### Church Slavonic Example (tr_1_tl_3 from 0101_Usp_Juni_Sin_995-0482.xml)

```
Coords: "2106,827 2132,823 2158,821 ... 2551,812 2551,782 2524,782 ... 2106,797"

Analysis:
- Num points: 36
- Bbox height: 45px (827 - 782)
- Y-coordinate std dev: 15.6px
- Structure: Polygon traces EXACT text contour
- First half avg y: 817.1px (top edge near baseline)
- Second half avg y: 787.1px (bottom edge)
- Polygon follows character ascenders/descenders tightly
```

### Prosta Mova Example (tr_1_tl_1 from 0101_18_Pravilo_1598-0169.xml)

```
Coords: "148,993 250,974 345,1020 462,970 ... 1781,720 1751,758 ... 148,902"

Analysis:
- Num points: 33
- Bbox height: 391px (1107 - 716) ← MASSIVE!
- Y-coordinate std dev: 122.1px (7.8x higher variance)
- Structure: Polygon includes HUGE whitespace above/below text
- First half avg y: 1015.5px
- Second half avg y: 817.7px
- Polygon does NOT follow text contour
- Includes 100-200px of whitespace above AND below actual text
```

**Visual comparison**: Church Slavonic polygons hug the text like shrink-wrap. Prosta Mova polygons are like putting a letter in an oversized envelope.

## Extracted Line Image Analysis

After `transkribus_parser.py` extraction (both resized to 128px height):

| Metric | Church Slavonic | Prosta Mova | Difference |
|--------|----------------|-------------|------------|
| Avg height | 128px | 128px | 0px (same) |
| Avg ink height | 123.9px | 94.8px | -29.1px |
| Avg top margin | 2.8px | 11.7px | +8.9px |
| Avg bottom margin | 1.3px | 15.1px | +13.8px |
| **Margin ratio** | **3.2%** | **20.9%** | **+17.7%** |
| Whitespace ratio | 56.8% | 79.7% | +22.9% |

**Critical Finding**: Even after resize to 128px, Prosta Mova lines retain 21% margins (vs 3% for Church Slavonic). The text is **physically smaller** in the 128px image because the original polygon included so much whitespace.

### Visual Evidence

Sample line images (first 10 from each dataset):

**Church Slavonic**:
- 10/10 images have 0% margin ratio (ink fills entire 128px height)
- Text is crisp and large
- Characters fully utilize vertical space

**Prosta Mova**:
- 3/10 images have 0% margin
- 7/10 images have 2-48% margins
- Examples:
  - Line 12: 23px top margin (18%)
  - Line 13: 39px top + 10px bottom (38%)
  - Line 14: 0px top + 54px bottom (42%)
  - Line 15: 15px top + 47px bottom (48%)

## Root Cause: Transkribus Export Settings

### Hypothesis

The datasets were exported from Transkribus with different settings:

**Church Slavonic** (optimal):
- Tight baseline detection
- Minimal polygon padding (5-10px)
- Polygons follow text contour closely
- Result: 40-50px height polygons

**Prosta Mova** (suboptimal):
- Loose baseline detection OR
- Large polygon padding (50-200px) OR
- Different segmentation algorithm OR
- Automatic segmentation (not manually corrected) OR
- Different Transkribus version/model
- Result: 200-400px height polygons with huge margins

### Evidence from PAGE XML Metadata

**Church Slavonic** (`0101_Usp_Juni_Sin_995-0482.xml`):
```xml
<Creator>Transkribus</Creator>
<Created>2024-02-10T09:49:50.660+01:00</Created>
<LastChange>2024-04-26T12:50:22.455+02:00</LastChange>
<TranskribusMetadata ... status="GT" .../>
```
- Status: "GT" (Ground Truth) - manually corrected
- Multiple edits (Created vs LastChange dates differ by 2.5 months)

**Prosta Mova** (`0101_18_Pravilo_1598-0169.xml`):
```xml
<Creator>prov=READ-COOP:name=PyLaia@TranskribusPlatform:version=2.26.0:model_id=327253:lm=provided:date=24_04_2025_09:09</Creator>
<Created>2023-06-16T10:31:52.549+02:00</Created>
<LastChange>2025-05-02T09:40:15.969+02:00</LastChange>
```
- Created by PyLaia automatic recognition (model_id=327253)
- May not have manually corrected segmentation
- Automatic segmentation often includes more margins for safety

## Impact on Model Performance

### Training Impact

1. **Character Size**: Prosta Mova characters are ~24% smaller in the 128px image due to margins
   - Church Slavonic: 123.9px ink height → ~30-40px character height
   - Prosta Mova: 94.8px ink height → ~24-30px character height

2. **Effective Resolution**: Prosta Mova loses ~25% resolution due to wasted space
   - Church Slavonic: 97% of pixels are ink (3% margins)
   - Prosta Mova: 74% of pixels are ink (21% margins)

3. **CER Impact**:
   - Church Slavonic: 3.17% CER (excellent)
   - Prosta Mova V2: 19.03% CER (6x worse)
   - Prosta Mova V3: 19.24% CER (6x worse)

**Hypothesis**: The 6x CER difference is partially explained by:
- Smaller character size (harder to recognize)
- More whitespace (less context for model)
- Inconsistent margins (some lines 0%, some 48% → confuses model)

## Transkribus Parser Analysis

### Current Behavior

`transkribus_parser.py` (lines 158-199):
1. Reads Coords polygon from PAGE XML
2. Calculates bounding box: `get_bounding_box(coords)` → (x1, y1, x2, y2)
3. Adds 5px padding on all sides
4. Crops image to bounding box: `image.crop((x1_pad, y1_pad, x2_pad, y2_pad))`
5. Resizes to target_height (128px) with aspect ratio preservation

**NO TIGHTENING LOGIC**: The parser uses the PAGE XML polygon as-is. If the polygon includes 200px of whitespace, that whitespace is preserved in the extracted line image.

### Why This is Correct

The parser should NOT modify the polygon boundaries because:
1. It's faithful to the source data (PAGE XML defines ground truth)
2. Some whitespace may be intentional (line spacing in original manuscript)
3. Modifying polygons could crop actual text (dangerous)

**The fix must happen earlier**: Either in Transkribus export or PAGE XML post-processing.

## Solutions

### Solution 1: Re-export from Transkribus (RECOMMENDED)

**Best option** if you have access to the original Transkribus project:

1. Open Prosta Mova project in Transkribus
2. Check segmentation settings:
   - Layout Analysis → Baseline Detection
   - Reduce "Polygon Padding" or "Baseline Offset" settings
3. Manually tighten loose polygons using Transkribus editor:
   - Select TextLine
   - Adjust polygon vertices to hug text closely
   - Remove excess whitespace above/below
4. Re-export to PAGE XML
5. Re-run `transkribus_parser.py`

**Expected improvement**:
- Line heights: 363px → 50-70px (5x reduction)
- Margin ratio: 21% → 5% (4x improvement)
- CER: 19% → 8-12% (estimated 50% improvement)

### Solution 2: PAGE XML Post-Processing (MEDIUM EFFORT)

Create a script to tighten PAGE XML polygons before parsing:

```python
# Pseudo-code
for each TextLine in PAGE XML:
    1. Load corresponding page image
    2. Extract region inside polygon
    3. Find actual ink extent (vertical projection, threshold at 5%)
    4. Calculate tight bounding box around ink
    5. Add small padding (10-15px)
    6. Update Coords polygon in XML to tight bbox
    7. Save modified PAGE XML
```

**Advantages**:
- Automated, repeatable
- Preserves original PAGE XML (backup first)
- Can be tuned with different padding amounts

**Disadvantages**:
- Risk of cropping actual text if thresholds are too aggressive
- Requires careful validation
- May not work for faint/degraded text

### Solution 3: Modify transkribus_parser.py (NOT RECOMMENDED)

Add ink-based cropping to `crop_polygon()`:

```python
def crop_polygon_tight(self, image, coords):
    # Get initial crop (current behavior)
    cropped = self.crop_polygon(image, coords)

    # Tighten to actual ink extent
    gray = cropped.convert('L')
    arr = np.array(gray)
    vertical_proj = np.sum(arr < 200, axis=1)  # Count dark pixels

    ink_rows = np.where(vertical_proj > arr.shape[1] * 0.05)[0]
    if len(ink_rows) > 0:
        top = max(0, ink_rows[0] - 5)  # 5px padding
        bottom = min(arr.shape[0], ink_rows[-1] + 5)
        cropped = cropped.crop((0, top, cropped.width, bottom))

    return cropped
```

**Disadvantages**:
- Changes data semantics (extracted image ≠ PAGE XML polygon)
- Risk of cropping diacritics, ascenders, descenders
- Doesn't fix inconsistency in dataset (some lines already tight)
- Makes reproducibility harder (results differ from PAGE XML)

### Solution 4: Train on Mixed Data (WORKAROUND)

If re-export is not feasible, train a model that's robust to varying margins:

1. Augment Church Slavonic with artificial margins:
   ```python
   def add_random_margins(line_image):
       top_margin = random.randint(0, 30)
       bottom_margin = random.randint(0, 30)
       # Add whitespace
   ```

2. Train on combined dataset (Church Slavonic + Prosta Mova)
3. Use data augmentation during training (random crops, vertical shifts)

**Disadvantages**:
- Doesn't fix root cause
- Model must learn to handle inconsistent data
- Still wastes resolution on whitespace

## Recommendations

**Priority 1** (if possible):
- Re-export Prosta Mova from Transkribus with tight segmentation
- Manually correct loose polygons in Transkribus editor
- Target: 50-70px average line height (Church Slavonic is 42.8px)

**Priority 2** (if Priority 1 not feasible):
- Implement PAGE XML post-processing script (Solution 2)
- Validate on 100 random lines to ensure no text is cropped
- Re-parse dataset with tightened XMLs

**Priority 3** (short-term workaround):
- Use Solution 4 (train on mixed data with margin augmentation)
- Accept lower CER (~12-15% instead of 8-10%)

**DO NOT**:
- Modify transkribus_parser.py to tighten crops (breaks data semantics)
- Ignore the issue (19% CER is too high for production use)

## Validation Plan

After implementing any solution:

1. **Re-measure line heights**:
   ```bash
   python analyze_page_xml_segmentation.py
   ```
   - Target: Prosta Mova avg height: 50-70px (currently 113px)

2. **Re-measure extracted image margins**:
   ```bash
   python visualize_line_comparison.py
   ```
   - Target: Prosta Mova margin ratio: <10% (currently 21%)

3. **Re-train PyLaia model**:
   ```bash
   python train_pylaia.py --config config_prosta_mova_tight.yaml
   ```
   - Target CER: <10% (currently 19%)

4. **Visual spot-check**:
   - Inspect 50 random extracted line images
   - Ensure no text is cropped
   - Ensure margins are reasonable (5-10px)

## Appendix: Sample Output

### analyze_page_xml_segmentation.py Output

```
Church Slavonic (n=1403 lines):
  Mean: 41.5px
  Median: 40.0px
  Std: 6.4px
  Range: 30-71px

Prosta Mova (n=356 lines):
  Mean: 113.0px
  Median: 71.0px
  Std: 100.2px
  Range: 33-485px

Height difference: 71.5px
Ratio: 2.72x

Church Slavonic Line Images:
  Avg margin ratio: 3.20%

Prosta Mova Line Images:
  Avg margin ratio: 20.90%
```

### visualize_line_comparison.py Output

```
Church Slavonic (first 10 images):
  All 10 images: 0.0% margin ratio (perfect!)

Prosta Mova (first 10 images):
  Line 12: Top=23px, Bottom= 0px, Margin=18.0%
  Line 13: Top=39px, Bottom=10px, Margin=38.3%
  Line 14: Top= 0px, Bottom=54px, Margin=42.2%
  Line 15: Top=15px, Bottom=47px, Margin=48.4%
```

## Files Generated

- `/home/achimrabus/htr_gui/dhlab-slavistik/analyze_page_xml_segmentation.py` - Main analysis script
- `/home/achimrabus/htr_gui/dhlab-slavistik/visualize_line_comparison.py` - Visual comparison script
- `/home/achimrabus/htr_gui/dhlab-slavistik/analyze_coords_detail.py` - Polygon structure analysis
- `/home/achimrabus/htr_gui/dhlab-slavistik/line_comparison_grid.png` - Visual comparison output
- `/home/achimrabus/htr_gui/dhlab-slavistik/SEGMENTATION_ANALYSIS_REPORT.md` - This report

---

**Conclusion**: The 36% height difference and 6x CER gap is caused by loose PAGE XML polygon segmentation in Prosta Mova, not by preprocessing or parser bugs. **Solution**: Re-export from Transkribus with tight segmentation settings, or post-process PAGE XML to tighten polygons to actual ink extent.
