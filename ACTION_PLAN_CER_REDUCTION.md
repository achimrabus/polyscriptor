# Action Plan: Reducing CER from 23% to ~6% Baseline

## Executive Summary

**Current Status**: TrOCR achieves 23% CER vs PyLaia baseline of 6%
**Gap**: 17 percentage points (283% higher error rate)
**Root Cause Identified**: Image preprocessing (10.6x resolution loss)
**Tokenizer Status**: ✓ VERIFIED WORKING (byte-level BPE handles Ukrainian perfectly)

---

## Diagnostic Results

### ✓ What We've Confirmed WORKS

1. **Tokenizer Coverage** (Step 1 from TROCR_CER_REDUCTION_STEPS.md)
   - Ukrainian chars (і, ї, є, ґ) properly handled
   - Byte-level BPE tokenization: 0% character loss
   - No UNK tokens generated
   - **CONCLUSION**: Tokenizer is NOT the problem

2. **Background Normalization** (Partial Step 2)
   - CLAHE + LAB color space conversion implemented
   - Improved CER from 34% → 23% (32% relative improvement)
   - **CONCLUSION**: Helpful but not sufficient

3. **Training Setup** (Step 5)
   - Adafactor optimizer ✓
   - Beam search (num_beams=4) ✓
   - Data augmentation enabled ✓
   - **CONCLUSION**: Training hyperparameters are reasonable

### ✗ What We've Confirmed is BROKEN

1. **Image Resolution Loss** (Step 2 - CRITICAL)
   - Ukrainian lines: Average 4077×357px (aspect ratio 11.6:1)
   - TrOCR resizes to: 384×384 (square, brutal resize)
   - **Width downsampling**: 10.6x (4077→384)
   - **Impact**: Characters reduced from ~80px to ~7px width
   - **Result**: Fine details (diacritics, similar letters) completely destroyed

2. **Rectangular vs Polygon Segmentation** (Step 2)
   - Currently using rectangular bounding boxes
   - 773 PAGE XML files have polygon coordinates available
   - Extra background noise may affect quality
   - **CONCLUSION**: Secondary issue, should test

3. **Text Normalization** (Step 3) - NOT TESTED
   - Unknown if using NFC normalization
   - Unknown if apostrophe is standardized
   - Unknown if PyLaia uses different normalization
   - **CONCLUSION**: Need to verify

---

## Prioritized Action Plan

### PHASE 1: Image Preprocessing Fix (HIGH IMPACT)

**Goal**: Preserve character details by avoiding brutal 10.6x downsampling

#### Option A: Custom Preprocessing with Aspect Ratio Preservation (RECOMMENDED)

**Implementation**:
1. Resize lines to target height (96-128px as per TROCR_CER_REDUCTION_STEPS.md)
2. Keep aspect ratio intact
3. Pad to square (384×384) with white/gray background
4. This preserves character width and prevents detail loss

**Expected Character Width**:
- Current: 4077×357 → 384×384 (chars: 80px → 7px) ❌
- With fix: 4077×357 → 128px height → 1467×128 → pad to 384×384 (chars: 80px → 28px) ✓
- **Improvement**: 4x better character resolution

**Code Changes Required**:
- Modify [transkribus_parser.py](transkribus_parser.py) preprocessing
- Create custom image processor for training
- Update inference to match training preprocessing

#### Option B: Polygon Mask Preprocessing (COMPLEMENTARY)

**Implementation**:
1. Use existing `--use-polygon-mask` flag in transkribus_parser.py
2. Extract lines using polygon coordinates instead of rectangles
3. Reduces background noise, tighter cropping

**Expected Impact**: 2-5% CER improvement (based on typical HTR improvements)

#### Option C: Test Alternative Base Models (IF A+B fail)

**Research**:
- Check if other ViT models support larger input sizes (e.g., 384×768)
- Evaluate Donut (preserves aspect ratio natively)
- Consider returning to PyLaia if TrOCR fundamentally limited

### PHASE 2: Text Normalization (MEDIUM IMPACT)

**Goal**: Ensure fair comparison with PyLaia baseline

#### Tasks:
1. **NFC Normalization**
   - Apply `unicodedata.normalize('NFC', text)` to all training/eval data
   - Ensures consistent character representation

2. **Apostrophe Standardization**
   - Audit apostrophe variants in data (', ', ʼ, etc.)
   - Standardize to single code point (recommend U+2019)

3. **PyLaia Evaluation Parity**
   - Check PyLaia's evaluation script
   - Match normalization, space handling, case sensitivity
   - Ensure apples-to-apples comparison

4. **Evaluation Breakdown**
   - Report CER with/without punctuation
   - Identify if punctuation errors dominate

### PHASE 3: Error Analysis (DIAGNOSTIC)

**Goal**: Understand remaining error sources after fixes

#### Tasks:
1. **Confusion Matrix**
   - Top 20 character substitutions
   - Check for expected pairs: і↔и, є↔е, ї↔йі, ґ↔г, '↔'
   - Position-wise errors (start/end of line)

2. **Length Bias Analysis**
   - Check if predictions are too short (max_length issue)
   - Check if predictions are too long (length_penalty issue)

3. **Per-Document CER**
   - Identify if specific documents have high error rates
   - May indicate segmentation/scan quality issues

### PHASE 4: Advanced Optimization (IF NEEDED)

**Only pursue if Phases 1-3 don't reach ~6% CER**

1. **Two-Stage Training**
   - Stage 1: Freeze ViT encoder, train decoder only (5 epochs)
   - Stage 2: Unfreeze encoder, train full model (lower LR)
   - Rationale: Let decoder adapt to Ukrainian before fine-tuning vision

2. **Language Model Rescoring**
   - Train character 5-gram KenLM on training transcriptions
   - Generate num_beams=20, rescore with LM
   - Expected gain: 1-3% CER for diacritic disambiguation

3. **Controlled Ablation**
   - Train on clean 5k-line subset to isolate pipeline vs capacity issues

---

## Implementation Priority

### Immediate (Today):
1. ✓ Diagnose tokenizer (COMPLETED - verified working)
2. Implement custom preprocessing with aspect ratio preservation
3. Test on small subset (100 lines) to verify improvement

### Short-Term (This Week):
4. Run full preprocessing with custom image handler
5. Retrain model with preserved-aspect-ratio images
6. Implement polygon mask preprocessing as secondary test
7. Add text normalization (NFC + apostrophe)

### Medium-Term (Next Week):
8. Build confusion matrix and error analysis
9. Match PyLaia evaluation methodology
10. Two-stage training if needed

---

## Expected Outcomes

### Conservative Estimate:
- **Phase 1 (preprocessing)**: 23% → 12-15% CER (8-11 point improvement)
- **Phase 2 (normalization)**: 12-15% → 8-10% CER (4-5 point improvement)
- **Phase 3 (error-driven fixes)**: 8-10% → 6-8% CER (2-4 point improvement)

### Optimistic Estimate:
- **Phase 1**: 23% → 10% CER (if resolution loss is primary cause)
- **Phase 2**: 10% → 7% CER (with proper normalization)
- **Result**: Matching PyLaia baseline

---

## Key Insights from TROCR_CER_REDUCTION_STEPS.md

The document was HIGHLY accurate in diagnosis:

> "Height target 96–128 px for line HTR with ViT; keep aspect ⇒ **pad to square at the very end** for the encoder (e.g., 384×384) **rather than brutal resize**."

This is EXACTLY our problem - TrOCR does "brutal resize" to 384×384.

The document's "fix order" aligns with our findings:
1. ~~Tokenizer~~ (already working with byte-level BPE)
2. **Fix image preprocessing** ← OUR #1 PRIORITY
3. **Text normalization** ← OUR #2 PRIORITY
4. Error analysis ← OUR #3 PRIORITY
5. Advanced optimization ← If needed

---

## Next Steps

**Recommended**: Implement Option A (custom preprocessing with aspect ratio preservation) first, as it has highest expected impact and directly addresses the confirmed root cause.

**Question for User**:
- Proceed with implementing custom preprocessing?
- Test on small subset first or go straight to full retrain?
- Any preference on target line height (96px, 128px, or 150px)?
