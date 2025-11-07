# PyLaia Fixes Summary - 2025-11-06

**Status**: ✅ COMPLETE
**Time Taken**: 20 minutes
**Impact**: 2 critical bugs fixed, 1 model restored to working condition

---

## Problems Fixed

### 1. Ukrainian Model idx2char Bug (CRITICAL)

**Problem**:
- Ukrainian PyLaia model outputted garbage instead of Ukrainian text
- Example output: `^! 56d( 4!`-(1[0 ?/(5!`j!ž` (should be: `український текст`)
- Root cause: Checkpoint missing `idx2char` character mapping

**Solution**:
- Created `fix_ukrainian_checkpoint.py` script
- Loaded vocabulary from `symbols.txt` (list format, 181 characters)
- Mapped `<SPACE>` token to actual space character `' '`
- Added `idx2char` and `char2idx` to checkpoint
- Created backup: `best_model_BACKUP.pt`

**Result**:
```
✅ idx2char loaded: 181 characters
✅ char2idx loaded: 181 characters  
✅ Space character (index 0) correctly mapped to ' '
✅ ALL TESTS PASSED!
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
