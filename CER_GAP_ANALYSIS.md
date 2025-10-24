# CER Gap Analysis: TrOCR (23%) vs PyLaia (6%) Baseline

## Problem Statement

Current TrOCR model achieves **23.07% CER** on Ukrainian handwriting, but PyLaia baseline achieves **6% CER** on the same dataset - a gap of **17 percentage points** (3.8x worse).

## Training Results Summary

**Final Results:**
- Training samples: 17,602 lines
- Validation samples: 748 lines
- Final CER: 23.07% (epoch 19.1)
- Training loss: 0.169 (low, indicating model has learned)
- Validation loss: 0.926 (high, indicating generalization gap)

**CER Progression:**
- Epoch 1.36: 34.05%
- Epoch 5.46: 24.44%
- Epoch 10.91: 23.82%
- Epoch 19.1: 23.07%

**Observation:** CER plateaued around epoch 5-6 and barely improved after that (24.4% → 23%). This suggests a fundamental limitation, not just needing more training.

---

## Hypothesis Analysis (Step-by-Step)

### ❌ Hypothesis 1: Background Normalization

**Testing:** We implemented background normalization (CLAHE + LAB color space).

**Result:** CER improved from 34% → 23% but still 17 points above PyLaia baseline.

**Conclusion:** Background normalization helped (32% relative improvement) but is NOT the root cause.

---

### 🔍 Hypothesis 2: Segmentation Quality (CRITICAL)

**PyLaia approach:** Uses **polygon masks** from Transkribus PAGE XML
- Precise line boundaries following text contours
- Excludes background noise between lines
- Tight crop around actual text

**Our TrOCR approach:** Uses **rectangular bounding boxes**
- Horizontal min/max coordinates
- Includes whitespace above/below text
- May include parts of adjacent lines
- More background noise

**Example from data:**
```
Letters_Shwedowa_11_2022-0018_r1l12.png,від самого ранку! Все гарно
```

**Critical Difference:**
- PyLaia: Polygon mask extracts ONLY the curved/slanted text line
- TrOCR: Rectangle captures the line + potentially whitespace/noise above/below

**Evidence from config:**
- `transkribus_parser.py` has `use_polygon_mask` parameter but it's set to `False`
- We're using simple rectangular crops

**Impact Assessment:** 🔴 **HIGH PRIORITY**
- Segmentation quality directly affects what the model "sees"
- Extra background = harder to learn character boundaries
- PyLaia's precise polygon masks = cleaner input

---

### 🔍 Hypothesis 3: Sequence Length Mismatch

**TrOCR config:** `max_length: 128`
**Efendiev config:** `max_length: 64`

**Ukrainian lines are MUCH longer:**
- Avg line width: 2,262px (from config comment)
- Efendiev lines: ~657px (mentioned in config)
- **3.4x longer!**

**Sample from data showing long lines:**
```
"V. Яко.-Демченко, N N 1, 59, 75. Конощенко, III, №96 - Роздольський Людкeвич, NN 57-58."
```

**Problem:**
- TrOCR model has `max_position_embeddings: 512` (from model config)
- With longer lines, we need more tokens to represent them
- If actual transcriptions exceed 128 tokens, they get truncated
- Model never learns to handle full-length Ukrainian lines

**Impact Assessment:** 🟡 **MEDIUM-HIGH PRIORITY**
- Check actual token distribution in training data
- May need to increase max_length to 192 or 256
- Could explain why model plateaus - it simply can't generate longer sequences

---

### 🔍 Hypothesis 4: Effective Batch Size

**TrOCR (Ukrainian):**
- Batch size: 6 per GPU
- Gradient accumulation: 4
- GPUs: 2
- **Effective batch size: 6 × 4 × 2 = 48**

**TrOCR (Efendiev - achieved 6-7% CER):**
- Batch size: 32 per GPU
- Gradient accumulation: 2
- GPUs: 2
- **Effective batch size: 32 × 2 × 2 = 128**

**Difference:** Efendiev had **2.67x larger effective batch size!**

**Why this matters:**
- Larger batches → more stable gradients → better convergence
- Transformer models especially benefit from large batch sizes
- We reduced batch size to avoid OOM, but this may hurt performance

**Current GPU utilization:**
```
Training samples: 17,602
Batch processing: ~1.5-1.6s per step
```

**Impact Assessment:** 🟡 **MEDIUM PRIORITY**
- Could try batch_size=8 or 12 with gradient checkpointing
- Or increase gradient_accumulation_steps to 6-8

---

### 🔍 Hypothesis 5: Data Quality / Ground Truth

**Critical questions:**
1. Is the PyLaia 6% CER measured on the SAME validation set we're using?
2. Could there be ground truth inconsistencies?
3. Are we using the same Transkribus export that PyLaia used?

**Evidence needed:**
- Compare a few validation samples manually
- Check if our rectangular crops match PyLaia's polygon masks
- Verify ground truth transcriptions are identical

**Impact Assessment:** 🔴 **HIGH PRIORITY - NEEDS VERIFICATION**
- If we're measuring against different data, comparison is invalid
- If ground truth differs, CER metrics aren't comparable

---

### 🔍 Hypothesis 6: Model Architecture Differences

**PyLaia:**
- CNN + LSTM architecture
- Designed specifically for line-level OCR
- Works directly with 1D sequence (line height × width)
- CTC loss function

**TrOCR:**
- Vision Transformer (ViT) encoder + Transformer decoder
- Designed for general image-to-text
- Processes full 2D image (384×384 patches)
- Autoregressive generation with cross-entropy loss

**Key difference:**
- PyLaia: Input is the ACTUAL line image (could be 5000px × 100px)
- TrOCR: Input is RESIZED to 384×384 (fixed aspect ratio)

**Potential issue:**
- Long Ukrainian lines (2262px avg) get heavily downsampled
- Text becomes tiny/unreadable at 384px width
- Fine details (diacritics, similar letters) lost in downsampling

**Impact Assessment:** 🔴 **CRITICAL - ARCHITECTURAL LIMITATION**
- TrOCR may not be suitable for very long lines
- Resolution loss could be fundamental bottleneck
- Would need to test with line splitting or different encoder

---

### 🔍 Hypothesis 7: Preprocessing Differences

**What we know PyLaia does:**
- Polygon mask extraction (precise line boundaries)
- Likely: Height normalization to consistent pixel height
- Likely: No width resizing (preserves aspect ratio)

**What TrOCR does:**
- Rectangular cropping
- Resize to 384×384 (destroys aspect ratio for long lines!)
- Background normalization (we added this)

**Critical issue:** TrOCR's `TrOCRProcessor` does:
```python
images = images.resize((384, 384))  # SQUARE!
```

For a 2262px × 80px line:
- Aspect ratio: 28:1
- Resized to: 384×384 (1:1)
- Result: Massively distorted OR huge black padding

**Impact Assessment:** 🔴 **CRITICAL - PREPROCESSING MISMATCH**
- This could be THE main issue
- Need to check how processor handles long lines
- May need custom preprocessing

---

## Root Cause Ranking (Most Likely → Least Likely)

### 🥇 #1: Image Resizing / Resolution Loss (CRITICAL)
**Evidence:**
- Ukrainian lines are 2262px wide on average
- TrOCR resizes to 384×384
- 6x downsampling destroys detail
- PyLaia likely preserves resolution

**Test:** Check TrOCRProcessor resize behavior on actual training images

### 🥈 #2: Rectangular vs Polygon Segmentation (HIGH)
**Evidence:**
- We use rectangular bounding boxes
- PyLaia uses precise polygon masks
- Extra background noise affects training

**Test:** Re-run with `use_polygon_mask=True` in transkribus_parser

### 🥉 #3: Sequence Length Limit (MEDIUM-HIGH)
**Evidence:**
- max_length=128 may be too short for Ukrainian
- Long lines get truncated
- Model can't learn full sequences

**Test:** Analyze token length distribution, try max_length=192

### 4️⃣ #4: Effective Batch Size (MEDIUM)
**Evidence:**
- Batch size 48 vs Efendiev's 128
- Smaller batches = noisier gradients
- Transformer models need large batches

**Test:** Increase gradient_accumulation_steps or enable gradient checkpointing

### 5️⃣ #5: Validation Set Mismatch (NEEDS VERIFICATION)
**Evidence:**
- Unknown if we're measuring on same data as PyLaia
- Could invalidate comparison

**Test:** Verify validation set matches PyLaia baseline

---

## Recommended Action Plan

### Phase 1: Immediate Investigation (1-2 hours)

1. **Check TrOCR image preprocessing:**
   ```python
   from transformers import TrOCRProcessor
   processor = TrOCRProcessor.from_pretrained("kazars24/trocr-base-handwritten-ru")

   # Load a sample Ukrainian line
   from PIL import Image
   img = Image.open("data/ukrainian_train_normalized/line_images/sample.png")
   print(f"Original size: {img.size}")

   # See what processor does
   pixel_values = processor(images=img, return_tensors="pt").pixel_values
   print(f"Processed shape: {pixel_values.shape}")
   ```

2. **Analyze token length distribution:**
   ```python
   # Check if 128 is enough
   from transformers import TrOCRProcessor
   processor = TrOCRProcessor.from_pretrained("kazars24/trocr-base-handwritten-ru")

   import pandas as pd
   df = pd.read_csv("data/ukrainian_train_normalized/train.csv", header=None)

   token_lengths = []
   for text in df[1]:
       tokens = processor.tokenizer(text, return_tensors="pt").input_ids
       token_lengths.append(tokens.shape[1])

   print(f"Max tokens: {max(token_lengths)}")
   print(f"95th percentile: {np.percentile(token_lengths, 95)}")
   print(f"Mean: {np.mean(token_lengths)}")
   ```

3. **Verify validation set:**
   - Check validation samples manually
   - Compare against PyLaia's validation metrics

### Phase 2: Quick Fixes (if Phase 1 confirms issues)

**If resizing is the problem:**
- Option A: Split long lines into multiple segments
- Option B: Use a different base model with larger input size
- Option C: Custom preprocessing to preserve aspect ratio

**If polygon masks help:**
```python
# Re-run preprocessing with polygon masks
python transkribus_parser.py \
    --input_dir "C:\Users\Achim\Documents\TrOCR\Ukrainian_Data\training_set" \
    --output_dir "data/ukrainian_train_polygon" \
    --normalize-background \
    --use-polygon-mask  # ENABLE THIS
```

**If sequence length is the issue:**
```yaml
# Update config_ukrainian_normalized.yaml
max_length: 192  # or 256
generation_max_length: 192
```

### Phase 3: Systematic Experiments

Run ablation study:
1. Baseline (current): Rectangular + resize + max_len=128 → **23% CER**
2. Test A: Polygon + resize + max_len=128 → **? CER**
3. Test B: Rectangular + resize + max_len=192 → **? CER**
4. Test C: Polygon + resize + max_len=192 → **? CER**
5. Test D: Polygon + custom_resize + max_len=192 → **? CER** (best hope)

---

## Expected Improvements

**Conservative estimate:**
- Fix preprocessing (polygon + proper resize): **18-20% CER** (-3-5 points)
- Increase max_length: **16-18% CER** (-2-3 points)
- Increase batch size: **15-17% CER** (-1-2 points)

**Optimistic estimate (if all fixes work):**
- Combined fixes: **10-12% CER** (-11-13 points)
- Still above PyLaia's 6%, but much closer

**Realistic target:**
- Get to **12-15% CER** with TrOCR (acceptable given architectural differences)
- If we need 6%, may need to switch to PyLaia or hybrid approach

---

## Next Steps

**Immediate:**
1. Run Phase 1 investigations (image preprocessing + token lengths)
2. Report findings
3. Decide on Phase 2 fixes based on evidence

**Would you like me to run the Phase 1 investigation scripts now?**
