# Features and Bugfixes to Address Next

**Date**: 2025-11-07
**Current Status**: PyLaia fixes complete, GUI responsiveness plan created

---

## ✅ Recently Completed

### 1. PyLaia Ukrainian Model Fix (COMPLETE - 2025-11-06)
- **Status**: ✅ Fixed and documented
- **Commits**: `8b422f2` (initial fix), `9a9ecae` (CTC blank documentation)
- **Issue**: Missing idx2char in checkpoint + CTC blank token bug
- **Solution**: Created fix script with proper CTC blank at index 0
- **Documentation**:
  - [PYLAIA_FIXES_SUMMARY_20251106.md](PYLAIA_FIXES_SUMMARY_20251106.md)
  - [PYLAIA_IDX2CHAR_BUG_FIX.md](PYLAIA_IDX2CHAR_BUG_FIX.md)
  - [FIX_UKRAINIAN_PYLAIA_TODO.md](FIX_UKRAINIAN_PYLAIA_TODO.md)
- **Key Learning**: PyLaia models ALWAYS require CTC blank token at index 0

### 2. PyLaia Training Script Fix (COMPLETE - 2025-11-06)
- **Status**: ✅ Fixed to prevent future bugs
- **Commit**: `8b422f2`
- **Issue**: `.strip()` removes TAB character from KALDI vocabulary
- **Solution**: Changed to `.rstrip('\n\r')` in train_pylaia.py:67
- **Impact**: All future PyLaia models will have correct vocabulary parsing

### 3. Transcription Comparison Feature (COMPLETE - 2025-11-05)
- **Status**: ✅ Implemented and deployed
- **Commits**: `37011bf` (comparison), `e9a3480` (logo)
- **Documentation**:
  - [COMPARISON_FEATURE_IMPLEMENTATION_SUMMARY.md](COMPARISON_FEATURE_IMPLEMENTATION_SUMMARY.md)
  - [COMPARISON_BUGFIX_20251105.md](COMPARISON_BUGFIX_20251105.md)
- **Files**: `comparison_widget.py`, `transcription_metrics.py`
- **All 6 bugs fixed**: Page-based models, engine iteration, LineSegment, TrOCR format, dynamic linking

### 4. Application Rename (COMPLETE - 2025-11-05)
- **Status**: ✅ Polyscript → Polyscriptor
- **Commits**: `8c58d5e` (rename), `47e6f87` (gitignore)
- **Files**: GUI, logos, all documentation updated

### 5. Professional Logo (COMPLETE - 2025-11-05)
- **Status**: ✅ Logo resized and integrated
- **Commit**: `e9a3480`
- **Files**: `assets/logo.png` (400×112px), `assets/icon.png` (256×256px)

---

## 📋 Open Features and Plans

### High Priority (Active Planning)

#### 1. GUI Responsiveness & Layout Improvements (NEW - 2025-11-07)
- **File**: [GUI_RESPONSIVENESS_MASTER_PLAN.md](GUI_RESPONSIVENESS_MASTER_PLAN.md)
- **Issues**:
  - At FHD resolution, transcription window too small (~400-500px)
  - Image panel takes excessive space (~800px)
  - No dynamic resizing between panels
  - User cannot adjust window proportions
- **Solutions**:
  - Add top-level splitter (Image ↔ Controls+Results)
  - Enhance results splitter (Text ↔ Stats/Comparison)
  - Save/restore window geometry and splitter positions
  - Optional view presets (Default, Image Focus, Transcription Focus)
- **Priority**: HIGH (critical UX issue)
- **Estimated Time**: 3.5-5.5 hours

#### 2. Comparison Feature V2 - Page-Based Support (NEW - 2025-11-07)
- **File**: [GUI_RESPONSIVENESS_MASTER_PLAN.md](GUI_RESPONSIVENESS_MASTER_PLAN.md) (Priority 2)
- **Issue**: Cannot compare line-based model (PyLaia) with page-based model (Qwen3)
- **Solution**:
  - Parse page output → split into lines → align with segmentation
  - Support fuzzy alignment when line counts mismatch
  - Visual indicators for mixed comparison modes
  - Alignment quality metrics
- **Priority**: MEDIUM (feature enhancement)
- **Estimated Time**: 4.5 hours

#### 3. Confidence Score Fixes (NEW - 2025-11-07)
- **File**: [GUI_RESPONSIVENESS_MASTER_PLAN.md](GUI_RESPONSIVENESS_MASTER_PLAN.md) (Priority 3)
- **Issues**:
  - Some engines don't return confidence (None)
  - Statistics panel shows errors
  - Character-level confidence not displayed
- **Solutions**:
  - Standardize confidence handling in TranscriptionResult
  - Update statistics panel to handle None gracefully
  - Optional character-level confidence visualization
- **Priority**: MEDIUM (quality-of-life improvement)
- **Estimated Time**: 2.5-4 hours

### Medium Priority (Planning Stage)

#### 4. NuMarkdown-8B-Thinking Integration (NEW - 2025-11-07)
- **File**: [GUI_RESPONSIVENESS_MASTER_PLAN.md](GUI_RESPONSIVENESS_MASTER_PLAN.md) (Priority 4)
- **What**: "First reasoning OCR VLM" - converts documents to Markdown with thinking steps
- **Base**: Qwen 2.5-VL-7B (8B params)
- **Use Cases**: Complex layouts, tables, multi-column manuscripts
- **Not For**: Simple line transcription (use PyLaia/TrOCR instead)
- **Implementation**: New engine plugin (similar to Qwen3)
- **Fine-tuning**: Technically possible but NOT recommended (overkill for HTR)
- **Priority**: LOW-MEDIUM (interesting but limited use case)
- **Estimated Time**: 7 hours

#### 5. Churro Fine-tuning
- **File**: [CHURRO_FINETUNING_PLAN.md](CHURRO_FINETUNING_PLAN.md)
- **Goal**: Fine-tune Churro on Ukrainian manuscript dataset
- **Status**: Planning stage
- **Dependencies**: Ukrainian dataset prepared
- **Priority**: MEDIUM (improve accuracy for Ukrainian)

#### 6. Church Slavonic Preprocessing
- **File**: [CHURCH_SLAVONIC_PREPROCESSING_PLAN.md](CHURCH_SLAVONIC_PREPROCESSING_PLAN.md)
- **Goal**: Optimize preprocessing for Church Slavonic manuscripts
- **Status**: Planning stage
- **Priority**: MEDIUM

### Low Priority (Future Enhancements / Archive Candidates)

#### 7. Churro GUI Implementation
- **Files**:
  - [CHURRO_GUI_IMPLEMENTATION_PLAN.md](CHURRO_GUI_IMPLEMENTATION_PLAN.md)
  - [CHURRO_INTEGRATION_PLAN.md](CHURRO_INTEGRATION_PLAN.md)
- **Goal**: Integrate Churro HTR engine into GUI
- **Status**: ✅ COMPLETE - Churro engine already implemented
- **Action**: Archive plans

#### 8. Party Optimization
- **Files**:
  - [PARTY_OPTIMIZATION_PLAN.md](PARTY_OPTIMIZATION_PLAN.md)
  - [PARTY_GUI_INTEGRATION_PLAN.md](PARTY_GUI_INTEGRATION_PLAN.md)
  - [PARTY_PLUGIN_INTEGRATION_PLAN.md](PARTY_PLUGIN_INTEGRATION_PLAN.md)
- **Goal**: Optimize Party OCR performance and integration
- **Status**: ✅ COMPLETE - Party engine integrated with optimizations
- **Action**: Archive plans

#### 9. Qwen3 Prompt Modification
- **File**: [QWEN3_PROMPT_MODIFICATION_PLAN.md](QWEN3_PROMPT_MODIFICATION_PLAN.md)
- **Goal**: Implement custom prompts for Qwen3-VL
- **Status**: ✅ COMPLETE - Custom prompts already implemented
- **Action**: Archive plan

#### 10. Metadata Display
- **File**: [METADATA_DISPLAY_PLAN.md](METADATA_DISPLAY_PLAN.md)
- **Goal**: Enhanced metadata display in GUI
- **Status**: ✅ COMPLETE - Statistics panel shows metadata
- **Action**: Archive plan

#### 11. PyLaia Training Improvements
- **File**: [PYLAIA_TRAINING_PLAN.md](PYLAIA_TRAINING_PLAN.md)
- **Goal**: Improve PyLaia training pipeline
- **Status**: Planning stage
- **Priority**: LOW (training already works)

#### 12. GUI Improvements (General)
- **Files**:
  - [GUI_IMPROVEMENT_PLAN.md](GUI_IMPROVEMENT_PLAN.md)
  - [GUI_IMPROVEMENTS_MASTER_PLAN.md](GUI_IMPROVEMENTS_MASTER_PLAN.md)
- **Goals**: Various UX/UI enhancements
- **Status**: Collection of ideas
- **Priority**: LOW (superseded by GUI_RESPONSIVENESS_MASTER_PLAN.md)

#### 13. CER Reduction Action Plan
- **File**: [ACTION_PLAN_CER_REDUCTION.md](ACTION_PLAN_CER_REDUCTION.md)
- **Goal**: Systematic approach to reduce character error rate
- **Status**: Strategic document
- **Priority**: ONGOING (always relevant)

---

## 🔧 Recommended Next Steps

### Week 1: GUI Responsiveness (HIGH PRIORITY)

1. **Implement Top-Level Splitter** (1-2 hours)
   - Replace QHBoxLayout with QSplitter
   - Add image ↔ controls+results split
   - Set initial sizes (40/60 at FHD)
   - Add minimum/maximum constraints

2. **Enhance Results Splitter** (30 minutes)
   - Remove fixed maximum width on stats panel
   - Add collapsible behavior
   - Improve sizing ratios

3. **Settings Persistence** (1 hour)
   - Save window geometry
   - Save splitter positions
   - Restore on startup
   - Add closeEvent handler

4. **View Presets (Optional)** (1 hour)
   - Add View menu
   - Implement layout presets
   - Add keyboard shortcuts

5. **Testing** (1 hour)
   - Test at multiple resolutions
   - Test splitter edge cases
   - Test settings persistence

### Week 2: Comparison V2 & Confidence (MEDIUM PRIORITY)

6. **Page-to-Line Parsing** (1 hour)
   - Implement parse_page_to_lines()
   - Handle newline splitting

7. **Line Alignment** (2 hours)
   - Implement align_lines_to_segments()
   - Fuzzy alignment for mismatched counts
   - Merge excess lines strategy

8. **UI Indicators** (30 minutes)
   - Add comparison mode label
   - Color coding for mixed modes
   - Alignment quality display

9. **Confidence Standardization** (1 hour)
   - Add helper methods to TranscriptionResult
   - Update all engines

10. **Statistics Panel Updates** (1 hour)
    - Handle None confidence gracefully
    - Add "N/A" display
    - Optional char-level visualization

11. **Testing** (1 hour)
    - Test mixed comparisons
    - Test confidence edge cases

### Week 3: Optional Enhancements

12. **NuMarkdown Integration** (7 hours)
    - If user wants reasoning OCR for complex layouts

13. **Archive Completed Plans**
    - Move completed plans to docs/archive/

---

## 📊 Plan Status Summary

| Plan | Status | Priority | Estimated Time | Action |
|------|--------|----------|----------------|--------|
| PyLaia Ukrainian Fix | ✅ COMPLETE | - | - | Done |
| PyLaia Training Bug | ✅ COMPLETE | - | - | Done |
| Comparison Feature V1 | ✅ COMPLETE | - | - | Archive |
| Rename to Polyscriptor | ✅ COMPLETE | - | - | Archive |
| Professional Logo | ✅ COMPLETE | - | - | Archive |
| **GUI Responsiveness** | 🔴 **PLANNED** | **HIGH** | **3.5-5.5h** | **Week 1** |
| **Comparison V2** | 🟡 **PLANNED** | **MEDIUM** | **4.5h** | **Week 2** |
| **Confidence Fixes** | 🟡 **PLANNED** | **MEDIUM** | **2.5-4h** | **Week 2** |
| NuMarkdown Integration | 🟡 PLANNED | LOW-MED | 7h | Week 3 (Optional) |
| Churro Integration | ✅ COMPLETE | - | - | Archive |
| Party Integration | ✅ COMPLETE | - | - | Archive |
| Qwen3 Prompts | ✅ COMPLETE | - | - | Archive |
| Metadata Display | ✅ COMPLETE | - | - | Archive |
| Churro Fine-tuning | 🟡 PLANNED | MEDIUM | TBD | Schedule Later |
| Church Slavonic Prep | 🟡 PLANNED | MEDIUM | TBD | Schedule Later |
| GUI Improvements (Old) | 🟡 IDEAS | LOW | - | Superseded |
| CER Reduction | 🟢 ONGOING | ONGOING | - | Monitor |

---

**Legend**:
- ✅ COMPLETE - Implemented and deployed
- 🔴 PLANNED - High priority, ready for implementation
- 🟡 PLANNED - Medium/low priority, ready when needed
- 🟢 ONGOING - Continuous process

---

**Next Review**: After Week 1 implementation (GUI Responsiveness)

**Questions for User**:
1. Preferred default layout ratio at FHD? (Currently proposing 40% image / 60% controls+text)
2. Should view presets be included, or is manual splitter enough?
3. What should happen if page model returns different line count than segmentation?
4. Should char-level confidence highlighting be enabled by default or opt-in?
5. Is NuMarkdown integration needed now, or can it wait?
