# Features and Bugfixes to Address Next

**Date**: 2025-11-06
**Current Status**: Comparison feature completed, logo updated

---

## ✅ Recently Completed

### 1. Transcription Comparison Feature (COMPLETE)
- **Status**: ✅ Implemented and deployed
- **Commits**: `37011bf` (comparison), `e9a3480` (logo)
- **Documentation**: 
  - [COMPARISON_FEATURE_IMPLEMENTATION_SUMMARY.md](COMPARISON_FEATURE_IMPLEMENTATION_SUMMARY.md)
  - [COMPARISON_BUGFIX_20251105.md](COMPARISON_BUGFIX_20251105.md)
- **Files**: `comparison_widget.py`, `transcription_metrics.py`
- **All 6 bugs fixed**: Page-based models, engine iteration, LineSegment, TrOCR format, dynamic linking

### 2. Application Rename (COMPLETE)
- **Status**: ✅ Polyscript → Polyscriptor
- **Commits**: `8c58d5e` (rename), `47e6f87` (gitignore)
- **Files**: GUI, logos, all documentation updated

### 3. Professional Logo (COMPLETE)
- **Status**: ✅ Logo resized and integrated
- **Commit**: `e9a3480`
- **Files**: `assets/logo.png` (400×112px), `assets/icon.png` (256×256px)

---

## 📋 Open Features and Plans

### High Priority (Active Development)

#### 1. PyLaia Ukrainian Retraining
- **File**: [FIX_UKRAINIAN_PYLAIA_TODO.md](FIX_UKRAINIAN_PYLAIA_TODO.md)
- **Issue**: PyLaia Ukrainian model outputs garbage (idx2char bug in checkpoint)
- **Status**: Bug identified, fix script exists
- **Action**: Run `fix_ukrainian_checkpoint.py` and retrain if needed
- **Priority**: HIGH (model currently unusable)

#### 2. PyLaia idx2char Bug Fix
- **File**: [PYLAIA_IDX2CHAR_BUG_FIX.md](PYLAIA_IDX2CHAR_BUG_FIX.md)
- **Issue**: Training script uses `.strip()` which removes TAB from KALDI vocabulary
- **Status**: ✅ Fixed in inference, needs fix in training script
- **Action**: Fix `train_pylaia.py:67` to use `.rstrip('\n\r')`
- **Priority**: MEDIUM (affects future training)

### Medium Priority (Planning Stage)

#### 3. Churro GUI Implementation
- **Files**: 
  - [CHURRO_GUI_IMPLEMENTATION_PLAN.md](CHURRO_GUI_IMPLEMENTATION_PLAN.md)
  - [CHURRO_INTEGRATION_PLAN.md](CHURRO_INTEGRATION_PLAN.md)
- **Goal**: Integrate Churro HTR engine into GUI
- **Status**: ✅ COMPLETE - Churro engine already implemented
- **Note**: Can be archived/removed

#### 4. Churro Fine-tuning
- **File**: [CHURRO_FINETUNING_PLAN.md](CHURRO_FINETUNING_PLAN.md)
- **Goal**: Fine-tune Churro on Ukrainian manuscript dataset
- **Status**: Planning stage
- **Dependencies**: Ukrainian dataset prepared
- **Priority**: MEDIUM (improve accuracy for Ukrainian)

#### 5. Church Slavonic Preprocessing
- **File**: [CHURCH_SLAVONIC_PREPROCESSING_PLAN.md](CHURCH_SLAVONIC_PREPROCESSING_PLAN.md)
- **Goal**: Optimize preprocessing for Church Slavonic manuscripts
- **Status**: Planning stage
- **Priority**: MEDIUM

#### 6. Party Optimization
- **Files**:
  - [PARTY_OPTIMIZATION_PLAN.md](PARTY_OPTIMIZATION_PLAN.md)
  - [PARTY_GUI_INTEGRATION_PLAN.md](PARTY_GUI_INTEGRATION_PLAN.md)
  - [PARTY_PLUGIN_INTEGRATION_PLAN.md](PARTY_PLUGIN_INTEGRATION_PLAN.md)
- **Goal**: Optimize Party OCR performance and integration
- **Status**: ✅ COMPLETE - Party engine integrated with optimizations
- **Note**: Can be archived/removed

#### 7. PyLaia Training Improvements
- **File**: [PYLAIA_TRAINING_PLAN.md](PYLAIA_TRAINING_PLAN.md)
- **Goal**: Improve PyLaia training pipeline
- **Status**: Planning stage
- **Priority**: LOW (training already works)

### Low Priority (Future Enhancements)

#### 8. GUI Improvements
- **Files**:
  - [GUI_IMPROVEMENT_PLAN.md](GUI_IMPROVEMENT_PLAN.md)
  - [GUI_IMPROVEMENTS_MASTER_PLAN.md](GUI_IMPROVEMENTS_MASTER_PLAN.md)
- **Goals**: Various UX/UI enhancements
- **Status**: Collection of ideas
- **Priority**: LOW (GUI already functional)

#### 9. Qwen3 Prompt Modification
- **File**: [QWEN3_PROMPT_MODIFICATION_PLAN.md](QWEN3_PROMPT_MODIFICATION_PLAN.md)
- **Goal**: Implement custom prompts for Qwen3-VL
- **Status**: ✅ COMPLETE - Custom prompts already implemented
- **Note**: Can be archived/removed

#### 10. Metadata Display
- **File**: [METADATA_DISPLAY_PLAN.md](METADATA_DISPLAY_PLAN.md)
- **Goal**: Enhanced metadata display in GUI
- **Status**: ✅ COMPLETE - Statistics panel shows metadata
- **Note**: Can be archived/removed

#### 11. CER Reduction Action Plan
- **File**: [ACTION_PLAN_CER_REDUCTION.md](ACTION_PLAN_CER_REDUCTION.md)
- **Goal**: Systematic approach to reduce character error rate
- **Status**: Strategic document
- **Priority**: ONGOING (always relevant)

---

## 🔧 Recommended Next Steps

### Immediate (This Week)

1. **Fix PyLaia Ukrainian Model** ([FIX_UKRAINIAN_PYLAIA_TODO.md](FIX_UKRAINIAN_PYLAIA_TODO.md))
   - Run `fix_ukrainian_checkpoint.py`
   - Test inference with fixed checkpoint
   - Retrain if needed (use fixed `train_pylaia.py`)

2. **Fix PyLaia Training Script** ([PYLAIA_IDX2CHAR_BUG_FIX.md](PYLAIA_IDX2CHAR_BUG_FIX.md))
   - Change `train_pylaia.py:67` from `.strip()` to `.rstrip('\n\r')`
   - Prevents future models from having idx2char bugs

3. **Archive Completed Plans**
   - Move CHURRO_* plans to `docs/archive/` (already implemented)
   - Move PARTY_* plans to `docs/archive/` (already implemented)
   - Move QWEN3/METADATA plans to `docs/archive/` (already implemented)

### Short-Term (Next 2 Weeks)

4. **Churro Fine-tuning for Ukrainian** ([CHURRO_FINETUNING_PLAN.md](CHURRO_FINETUNING_PLAN.md))
   - Prepare Ukrainian training data (PAGE XML format)
   - Run fine-tuning script
   - Evaluate CER improvement

5. **Church Slavonic Preprocessing** ([CHURCH_SLAVONIC_PREPROCESSING_PLAN.md](CHURCH_SLAVONIC_PREPROCESSING_PLAN.md))
   - Test different preprocessing strategies
   - Measure impact on CER

### Long-Term (Next Month)

6. **Comparison Feature V2** (from COMPARISON_FEATURE_IMPLEMENTATION_SUMMARY.md)
   - Batch comparison script (process entire manuscripts)
   - HTML export with interactive diff viewer
   - Error pattern analysis

7. **GUI Enhancements** ([GUI_IMPROVEMENTS_MASTER_PLAN.md](GUI_IMPROVEMENTS_MASTER_PLAN.md))
   - Implement ideas from master plan
   - User feedback integration

---

## 📊 Plan Status Summary

| Plan | Status | Priority | Action |
|------|--------|----------|--------|
| Comparison Feature | ✅ COMPLETE | - | Archive |
| Rename to Polyscriptor | ✅ COMPLETE | - | Archive |
| Professional Logo | ✅ COMPLETE | - | Archive |
| PyLaia Ukrainian Fix | 🔴 OPEN | HIGH | **Fix Now** |
| PyLaia Training Bug | 🔴 OPEN | MEDIUM | **Fix Now** |
| Churro Integration | ✅ COMPLETE | - | Archive |
| Party Integration | ✅ COMPLETE | - | Archive |
| Qwen3 Prompts | ✅ COMPLETE | - | Archive |
| Metadata Display | ✅ COMPLETE | - | Archive |
| Churro Fine-tuning | 🟡 PLANNED | MEDIUM | Schedule |
| Church Slavonic Prep | 🟡 PLANNED | MEDIUM | Schedule |
| GUI Improvements | 🟡 IDEAS | LOW | Backlog |
| CER Reduction | 🟢 ONGOING | ONGOING | Monitor |

---

**Legend**:
- ✅ COMPLETE - Implemented and deployed
- 🔴 OPEN - Needs immediate attention
- 🟡 PLANNED - Ready for implementation
- 🟢 ONGOING - Continuous process

---

**Next Review**: After fixing PyLaia Ukrainian model
