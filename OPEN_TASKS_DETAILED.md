# Open Tasks - Detailed Elaboration

**Date**: 2025-11-06
**Status**: Active development roadmap

---

## 🔴 HIGH PRIORITY - Immediate Action Required

### Task 1: Fix PyLaia Ukrainian Model (CRITICAL)

**Issue**: PyLaia Ukrainian model outputs garbage instead of Ukrainian Cyrillic text

**Root Cause Analysis**:
- **Primary**: Checkpoint missing `idx2char` mapping (same bug as Church Slavonic)
- **Secondary**: Training script used `.strip()` which removed TAB from KALDI vocabulary
- **Status**: Inference code already fixed (line 139), but checkpoint needs patching

**Affected Models**:
1. **Ukrainian (10.80% CER)**: `models/pylaia_ukrainian_pagexml_20251101_182736/best_model.pt`
2. **Ukrainian (13.53% CER)**: `models/pylaia_ukrainian_retrain_20251102_213431/best_model.pt`

**Expected Symptoms**:
```
# Garbage output (current):
^! 56d( 4!`-(1[0 ?/(5!`j!ž

# Expected output (after fix):
український текст з пробілами
```

**Solution Steps**:

1. **Verify the problem exists**:
   - Run `python3 transcription_gui_plugin.py`
   - Load PyLaia engine with Ukrainian model
   - Transcribe a Ukrainian manuscript line
   - Check if output is garbage or proper Ukrainian

2. **Check checkpoint for idx2char**:
   - Load checkpoint with PyTorch
   - Verify `'idx2char'` key exists
   - If missing, needs patching

3. **Create and run fix script**:
   - Copy `fix_church_slavonic_checkpoint.py`
   - Modify paths for Ukrainian model
   - Run fix script
   - Verify idx2char added

4. **Test fixed model**:
   - Restart GUI
   - Load fixed Ukrainian model
   - Should now produce proper Ukrainian text

**Time Estimate**: 15-30 minutes
**Priority**: CRITICAL (model currently unusable)
**Difficulty**: Easy (copy existing fix script)

**Success Criteria**:
- ✅ Ukrainian model outputs proper Cyrillic text
- ✅ Spaces work correctly (not `"<SPACE>"` strings)
- ✅ CER matches training CER (~10-13%)

**Documentation to Update**:
- [FIX_UKRAINIAN_PYLAIA_TODO.md](FIX_UKRAINIAN_PYLAIA_TODO.md) - Mark as complete
- [PYLAIA_IDX2CHAR_BUG_FIX.md](PYLAIA_IDX2CHAR_BUG_FIX.md) - Add Ukrainian to "Fixed Models" list

---

### Task 2: Fix PyLaia Training Script Bug (IMPORTANT)

**Issue**: Training script uses `.strip()` which removes TAB character from vocabulary

**File**: `train_pylaia.py` line 67
**Impact**: All future PyLaia models will have idx2char bugs

**Root Cause**:
```python
# BROKEN (current):
symbols_raw = [line.strip() for line in f if line.strip()]
# Problem: .strip() removes ALL whitespace including TAB

# FIXED (needed):
symbols_raw = [line.rstrip('\n\r') for line in f if line.rstrip('\n\r')]
# Only removes newlines, preserves TAB and other whitespace
```

**Why TAB Matters**:
KALDI vocabulary format includes TAB character at index 131. When `.strip()` removes it, vocabulary parsing breaks and the model outputs garbage.

**Solution**:

1. **Locate the bug**: Find line 67 in `train_pylaia.py`
2. **Change code**: Replace `.strip()` with `.rstrip('\n\r')`
3. **Test**: Parse a KALDI vocabulary file and verify TAB preserved
4. **Verify**: Train small model and check idx2char[131] == '\t'

**Time Estimate**: 5 minutes (code change) + 1 hour (verification training)
**Priority**: HIGH (prevents future bugs)
**Difficulty**: Trivial (1-line change)

**Success Criteria**:
- ✅ Code changed to use `.rstrip('\n\r')`
- ✅ Verified with test vocabulary parsing
- ✅ New models include correct idx2char
- ✅ TAB character (index 131) maps correctly

**Related Bugs**:
- ✅ Fixed in `inference_pylaia_native.py` line 139 (already done)
- 🔴 Still broken in `train_pylaia.py` line 67 (needs fix)

---

## 🟡 MEDIUM PRIORITY - Planned Enhancements

### Task 3: Churro Fine-Tuning for Ukrainian Manuscripts

**Goal**: Improve Churro HTR accuracy on Ukrainian manuscripts through domain-specific fine-tuning

**Current Status**:
- Churro works zero-shot (no training)
- Accuracy unknown (no systematic evaluation yet)
- Potential for improvement through fine-tuning

**Why Fine-Tune?**:
1. **Domain Adaptation**: Churro pretrained on generic images, not historical manuscripts
2. **Script Specificity**: Ukrainian Cyrillic has unique characters (Ґ, Є, Ї, І)
3. **Writing Style**: Medieval handwriting differs from modern printed text
4. **Context**: Learn manuscript-specific abbreviations and conventions

**Available Data**:
- **Ukrainian**: 21,944 lines (400-800 pages) ✅
- **Church Slavonic**: 24,364 lines (500-1000 pages) ✅
- **Glagolitic**: 23,203 lines (400-800 pages) ✅
- **Total**: ~69,500 lines / 1200-2600 pages ✅ **SUFFICIENT!**

**Hardware Requirements**:
- **Current**: 2x NVIDIA L40S (46GB VRAM each = 92GB total) ✅ **EXCELLENT!**
- **Needed**: 12GB VRAM minimum (with QLoRA)
- **Batch Size**: 8-16 (vs 4-8 for Qwen3-8B)
- **Training Time**: 2-6 hours (depending on epochs)

**Fine-Tuning Method: QLoRA**

**What is QLoRA?**:
- **LoRA** (Low-Rank Adaptation): Trains small adapter layers, freezes base model
- **QLoRA**: Base model in 4-bit, adapters in full precision
- **Memory Savings**: 3B model in ~12GB (vs ~24GB full precision)
- **Quality**: Minimal accuracy loss, massive memory gain

**Training Approaches**:

**Option A: Single Universal Adapter** (Recommended First)
- Train on all scripts combined (Church Slavonic + Glagolitic + Ukrainian)
- Benefits from 69K combined lines
- Single model handles all scripts
- Faster training, easier GUI integration

**Option B: Script-Specific Adapters**
- Train separate adapters for each script
- Highly specialized per script
- Can swap adapters in GUI
- 3x training time

**Implementation Steps**:

1. Prepare data in Churro format (PAGE XML)
2. Install dependencies: `pip install trl peft qwen-vl-utils datasets`
3. Create training script (adapt from `finetune_qwen_ukrainian.py`)
4. Run training (2-4 hours on 2x L40S)
5. Evaluate CER improvement
6. Integrate adapter loading into GUI

**Time Estimate**:
- Data preparation: 1-2 hours
- Training: 2-6 hours
- Evaluation: 1 hour
- GUI integration: 1 hour
- **Total**: 5-10 hours

**Priority**: MEDIUM (optimization, not critical)
**Difficulty**: Moderate (requires LoRA/PEFT knowledge)

**Success Criteria**:
- ✅ CER reduced by ≥5% vs zero-shot Churro
- ✅ Handles Ukrainian-specific characters correctly
- ✅ Adapter loads in GUI without issues
- ✅ Inference speed unchanged

**Documentation**:
- [CHURRO_FINETUNING_PLAN.md](CHURRO_FINETUNING_PLAN.md) - Full training guide

---

### Task 4: Church Slavonic Preprocessing Optimization

**Goal**: Find optimal preprocessing pipeline for Church Slavonic manuscripts

**Current Status**:
- Using default preprocessing (CLAHE normalization, aspect ratio preservation)
- No systematic evaluation of preprocessing impact
- Church Slavonic CER: 3.51% (PyLaia) - **already good!**

**Why Optimize?**:
Church Slavonic manuscripts have unique characteristics:
- Titlos (overlines for abbreviations): ◌҃
- Yer vowels: ь, ъ
- Historical letters: ѣ, ѳ, ѵ
- Varied ink colors and fading
- Preprocessing can improve OCR accuracy by 5-20%

**Preprocessing Variables to Test**:

1. **Background Normalization**: CLAHE, global histogram, LAB color, none
2. **Binarization**: Otsu, adaptive, Sauvola, none
3. **Aspect Ratio**: Preserve, stretch, pad
4. **Target Height**: 96px, 128px, 150px, 192px
5. **Denoising**: Gaussian blur, bilateral filter, non-local means, none

**Experimental Setup**:
- Test combinations on validation set (200 lines)
- Measure CER (primary) and inference time (secondary)
- Grid search across variables
- Statistical analysis of results

**Implementation Steps**:

1. Create preprocessing variant functions
2. Run automated grid search
3. Analyze results for best combination
4. Update training pipeline with optimal method
5. Optionally retrain model with better preprocessing

**Time Estimate**:
- Setup: 2 hours
- Experiments: 4-8 hours (automated)
- Analysis: 1 hour
- Implementation: 2 hours
- **Total**: 9-13 hours

**Priority**: MEDIUM (CER already good at 3.51%)
**Difficulty**: Moderate (requires CV knowledge)

**Success Criteria**:
- ✅ Systematic evaluation of ≥10 preprocessing variants
- ✅ CER improvement ≥0.5% over baseline
- ✅ No significant slowdown (<20% time increase)
- ✅ Best method documented and deployed

**Documentation**:
- [CHURCH_SLAVONIC_PREPROCESSING_PLAN.md](CHURCH_SLAVONIC_PREPROCESSING_PLAN.md)

---

## 🟢 LOW PRIORITY - Future Enhancements

### Task 5: Comparison Feature V2

**Goal**: Advanced comparison features for manuscript analysis

**Current Status**:
- V1 complete: Engine vs engine, ground truth, CER/WER, CSV export ✅
- Line-by-line navigation ✅
- Color-coded diff ✅

**V2 Enhancements**:

1. **Batch Processing**: Compare entire manuscripts (100s of pages) with parallel GPU processing
2. **HTML Export**: Interactive diff viewer with embedded CSS/JS
3. **Error Pattern Analysis**: Common substitutions, confusion matrix, statistical reports
4. **Multi-Engine Voting**: Consensus transcription from 3+ engines
5. **LLM Post-Correction**: Use GPT-4 to fix obvious errors

**Time Estimate**: 20-40 hours (full V2)
**Priority**: LOW (V1 sufficient for now)

---

## 📦 ARCHIVABLE - Completed Plans

These implementation plans can be moved to `docs/archive/` as they're already implemented:

1. **Churro Integration** → ✅ Complete (`engines/churro_engine.py`)
2. **Party Integration** → ✅ Complete (`engines/party_engine.py`)
3. **Qwen3 Custom Prompts** → ✅ Complete (`qwen3_prompts.py`)
4. **Metadata Display** → ✅ Complete (statistics panel)
5. **Comparison Feature** → ✅ Complete (`comparison_widget.py`)

**Archive Command**:
```bash
mkdir -p docs/archive
mv CHURRO_INTEGRATION_PLAN.md docs/archive/
mv PARTY_*_PLAN.md docs/archive/
mv QWEN3_PROMPT_MODIFICATION_PLAN.md docs/archive/
mv METADATA_DISPLAY_PLAN.md docs/archive/
```

---

## 📊 Task Summary

| Task | Priority | Time | Difficulty | Impact |
|------|----------|------|------------|--------|
| Fix Ukrainian PyLaia | 🔴 HIGH | 30min | Easy | CRITICAL |
| Fix Training Script | 🔴 HIGH | 5min | Trivial | HIGH |
| Churro Fine-tuning | 🟡 MEDIUM | 5-10h | Moderate | MEDIUM |
| CS Preprocessing | 🟡 MEDIUM | 9-13h | Moderate | LOW |
| Comparison V2 | 🟢 LOW | 20-40h | Hard | LOW |

**Recommended Order**:
1. **Fix Ukrainian PyLaia** (30 min) - DO NOW
2. **Fix Training Script** (5 min) - DO NOW
3. Archive completed plans (5 min)
4. Churro Fine-tuning (when ready for experiments)
5. CS Preprocessing (when seeking further improvements)
6. Comparison V2 (when V1 limitations become blocking)

---

**Last Updated**: 2025-11-06
