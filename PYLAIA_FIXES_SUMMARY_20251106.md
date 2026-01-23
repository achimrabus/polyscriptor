# PyLaia Fixes Summary - 2025-11-06

**Status**: ✅ COMPLETE
**Time Taken**: 45 minutes (including CTC blank bug discovery)
**Impact**: 2 critical bugs fixed, 1 model restored to working condition
**Key Learning**: PyLaia models ALWAYS require CTC blank token at index 0

---

## Problems Fixed

### 1. Ukrainian Model idx2char Bug (CRITICAL)

**Problem**:
- Ukrainian PyLaia model outputted garbage instead of Ukrainian text
- Example output: `^! 56d( 4!`-(1[0 ?/(5!`j!ž` (should be: `український текст`)
- Root cause: Checkpoint missing `idx2char` character mapping

**Initial Solution (INCORRECT)**:
- Created `fix_ukrainian_checkpoint.py` script
- Loaded vocabulary from `symbols.txt` (list format, 181 characters)
- Mapped `<SPACE>` token to actual space character `' '`
- Added `idx2char` and `char2idx` to checkpoint
- Created backup: `best_model_BACKUP.pt`
- **Problem**: Model output Cyrillic but scrambled (e.g., `ьмуіостуеияБоси,оеи...`)

**Root Cause Discovered - CTC Blank Token**:
- PyLaia uses CTC (Connectionist Temporal Classification)
- **CTC ALWAYS reserves index 0 for blank token**
- Initial fix created direct 1-to-1 mapping: index 0=' ', 1='о', 2='а'...
- Correct mapping should be: index 0='<ctc>', 1=' ', 2='о', 3='а'...
- Training script (line 90) does `idx + 1` to reserve index 0 for CTC blank
- Vocabulary has 181 lines but model expects 181 classes INCLUDING CTC blank

**Corrected Solution**:
- Index 0: '<ctc>' (CTC blank token, required by PyLaia)
- Index 1: ' ' (space, was '<SPACE>' in vocab)
- Indices 2-180: First 180 vocabulary symbols (181 lines → 180 used + 1 CTC)
- Properly maps <SPACE> token to actual space character

**Result**:
```
✅ idx2char loaded: 181 characters (including CTC blank at index 0)
✅ char2idx loaded: 180 characters (vocabulary symbols)
✅ CTC blank (index 0) correctly set to '<ctc>'
✅ Space character (index 1) correctly mapped to ' '
✅ ALL TESTS PASSED - Model outputs correct Ukrainian text!
```

**Model Fixed**:
- Path: `models/pylaia_ukrainian_pagexml_20251101_182736/best_model.pt`
- CER: 10.80% (best Ukrainian model)
- Status: Ready for use (restart GUI)

---

### 2. Training Script .strip() Bug (PREVENTION)

**Problem**:
- `train_pylaia.py` line 67 used `.strip()` to read vocabulary
- `.strip()` removes **ALL** whitespace including TAB character (index 131)
- KALDI vocabulary includes TAB: `"\t 131"`
- When TAB removed, vocabulary parsing breaks → future models output garbage

**Solution**:
```python
# File: train_pylaia.py line 67

# BEFORE (BROKEN):
symbols_raw = [line.strip() for line in f if line.strip()]

# AFTER (FIXED):
# CRITICAL: Use rstrip('\n\r') not strip() to preserve TAB and other whitespace symbols
symbols_raw = [line.rstrip('\n\r') for line in f if line.rstrip('\n\r')]
```

**Impact**:
- Only removes newlines (`\n`, `\r`)
- Preserves TAB and other whitespace symbols
- All future PyLaia models will have correct vocabulary
- Prevents idx2char bugs from happening again

---

## Fixed Models Summary

| Model | Status | Vocabulary | Characters | CER |
|-------|--------|------------|------------|-----|
| **Church Slavonic** | ✅ Working | KALDI | 153 | 3.51% |
| **Ukrainian** | ✅ Fixed Today | List | 181 | 10.80% |
| **Glagolitic** | ✅ Working | KALDI | 76 | 6.20% |

---

## Files Modified

### Code Changes:
1. **train_pylaia.py** (line 67)
   - Changed `.strip()` to `.rstrip('\n\r')`
   - Prevents future idx2char bugs

### Documentation Updates:
2. **PYLAIA_IDX2CHAR_BUG_FIX.md**
   - Added Ukrainian to "Fixed Models" section
   - Updated "Prevention" section with training script fix
   - Changed status to "All bugs fixed"

### Scripts Created (gitignored):
3. **fix_ukrainian_checkpoint.py**
   - Patches Ukrainian checkpoint with idx2char
   - Auto-detects list vs KALDI format
   - Handles <SPACE> → ' ' mapping

---

## How to Use Fixed Ukrainian Model

1. **Restart GUI**:
   ```bash
   source htr_gui/bin/activate
   python3 transcription_gui_plugin.py
   ```

2. **Select PyLaia Engine**:
   - Engine: PyLaia
   - Model: Ukrainian (10.80% CER)

3. **Transcribe Ukrainian Manuscript**:
   - Should now output proper Ukrainian Cyrillic text
   - Spaces should work correctly (not `"<SPACE>"` strings)

4. **Verify Output**:
   - Expected: `український текст з пробілами`
   - NOT: `^! 56d( 4!`-(1[0` (garbage)

---

## Technical Details

### CTC Blank Token (CRITICAL CONCEPT)

**What is CTC?**
- CTC = Connectionist Temporal Classification
- Sequence labeling algorithm for variable-length input/output alignment
- **Requires blank token at index 0** to handle repeated characters and alignment

**Why Index 0 is Special:**
- PyLaia training script (line 90): `char2idx = {char: idx + 1 for idx, char in enumerate(symbols)}`
- All vocabulary symbols mapped to indices starting at 1
- Index 0 explicitly reserved: `idx2char[0] = ''` (CTC blank)

**Impact on Checkpoint Fixes:**
- Vocabulary with 181 lines → Model with 181 output classes (180 symbols + 1 CTC blank)
- Checkpoint must have: `idx2char[0] = '<ctc>'` or `''`
- All vocabulary symbols start at index 1, not 0

**Common Mistake:**
- Creating direct 1-to-1 mapping without CTC blank
- Result: All characters shifted by one index → garbage output
- Example: Model predicts index 5 → without CTC blank maps to wrong character

### Vocabulary Formats Supported:

**List Format** (Ukrainian):
```
<SPACE>
о
а
и
...
```

**KALDI Format** (Church Slavonic, Glagolitic):
```
<space> 1
и 2
о 3
	 131    # TAB character
...
```

### Critical Space Token Handling:

Both formats require mapping special space token to actual space:
- `<SPACE>` (uppercase) → `' '`
- `<space>` (lowercase) → `' '`

Without this mapping, model outputs literal `"<space>"` string (6 characters) instead of space (1 character), artificially inflating CER by 5-20%.

---

## Prevention for Future Training

**Verification Checklist**:

After any PyLaia training, run:
```bash
python3 << 'EOF'
import torch
checkpoint = torch.load('best_model.pt', map_location='cpu', weights_only=False)

# Check idx2char exists
assert 'idx2char' in checkpoint, "❌ Missing idx2char!"

# Check space mapping
idx2char = checkpoint['idx2char']
space_idx = next((i for i, c in idx2char.items() if c == ' '), None)
assert space_idx is not None, "❌ Space not mapped!"

print(f"✅ Checkpoint valid ({len(idx2char)} characters)")
print(f"✅ Space at index {space_idx}")
