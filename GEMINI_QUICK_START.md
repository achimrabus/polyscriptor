# Quick Start: Gemini 3 Enhancements

## Setup (One-Time)

1. **Activate virtualenv**:
   ```bash
   source htr_gui/bin/activate
   ```

2. **Ensure API key in environment** (optional; GUI also supports manual entry):
   ```bash
   echo "GOOGLE_API_KEY=your_key_here" >> .env
   ```

## Using the Enhanced GUI

### Basic Transcription (Default Settings)

1. Launch GUI: `python transcription_gui_party.py` (or your preferred GUI entry point)
2. Select **Commercial APIs** engine
3. Choose provider: **Gemini**
4. Select model: **gemini-3-pro-preview** (restriction prompt auto-injected)
5. Thinking mode: **Auto (Low for preview)** ← Recommended
6. Click **Transcribe**

**Expected console output**:
```
⚡ Fast-direct mode enabled: prompting for immediate output
🧠 Using LOW thinking mode (direct decoding)
🔓 Using relaxed safety settings for preview model: gemini-3-pro-preview
   Increasing max_output_tokens to 4096 for preview model
[tokens] prompt=1137 candidates=18 total=1180
✅ Early streamed output (331 chars) [early-exit]
ℹ️ Continuation attempt 1 produced no new text; stopping.
```

### Advanced: Fine-Tuning for Complex Manuscripts

**Expand "Gemini Advanced" section**:

| Setting | Value | When to Adjust |
|---------|-------|----------------|
| **Early exit** | ✓ Checked | Uncheck if model producing truncated output; forces full stream collection. |
| **Auto continuation** | ✓ Checked | Check to enable multi-pass retrieval of missed text. |
| **Max passes** | 2 | Increase to 3-4 for very long manuscripts; decrease to 1 for short lines. |
| **Min new chars** | 50 | Lower to 25 if continuation stops prematurely; raise to 100 to reduce false positives. |
| **Low-mode tokens** | 6144 | Raise to 7168 if LOW mode hits token limit often; lower to 5120 for faster responses. |
| **Fallback %** | 0.6 | Lower to 0.5 for aggressive early fallback; raise to 0.7 to give model more reasoning time. |

### Monitoring Token Usage

**Real-time console**:
- Watch for `[tokens]` lines showing `prompt`, `candidates`, `total` counts.
- If `candidates=0` and `total` grows rapidly, internal reasoning is consuming budget.

**Post-run analysis**:
1. Open `gemini_runs.csv` in spreadsheet or terminal:
   ```bash
   cat gemini_runs.csv | column -t -s,
   ```
2. Look for:
   - **High `internal_tok`** with **low `emitted_chars`** → increase LOW-mode tokens or lower fallback %.
   - **`outcome=fallback_success`** rows → fallback is working as designed.
   - Multiple **`stream_early_exit`** with consistent chars → system stable.

## Troubleshooting

### Problem: "Streaming produced no early text"
**Solution**: Model consumed tokens internally. System will auto-fallback. If persistent:
- Increase **Low-mode tokens** to 7168 or 8192.
- Switch to **High (More reasoning)** thinking mode.
   (Model switching to flash/pro not recommended for Church Slavonic fidelity.)

### Problem: Continuation adding duplicate text
**Solution**: Increase **Min new chars** threshold to 75 or 100.

### Problem: Transcription stops mid-sentence
**Solution**:
- Uncheck **Early exit** to collect full stream.
- Enable **Auto continuation** with **Max passes = 3**.
- Raise **Max output tokens** to 6144 or 8192.

### Problem: Early reasoning fallback triggering too often
**Solution**: Raise **Fallback %** from 0.6 to 0.7 or 0.8 to give model more thinking time.

## Recommended Workflows

### Restriction Prompt Behavior
Preview model calls automatically prepend a concise instruction limiting internal reasoning and enforcing direct transcription output.

### 2. Church Slavonic Manuscripts (Default)
```
Model: gemini-3-pro-preview
Thinking: Low (Fast)
Temperature: 1.0
Max tokens: (empty, defaults to 4096)
Low-mode tokens: 6144
Early exit: ✓
Auto continuation: ✓ (passes=2)
Min new chars: 50
Fallback %: 0.6
```
**Use when**: Transcribing Cyrillic manuscripts; balance speed and accuracy.

### 3. Maximum Accuracy (Fallback Path)
```
Model: gemini-3-pro-preview
Thinking: High (More reasoning)
Temperature: 1.0
Max tokens: 8192
Early exit: ✗
Auto continuation: ✓ (passes=3)
Min new chars: 50
Fallback %: 0.7
```
**Use when**: Complex, heavily abbreviated manuscripts; willing to trade speed for completeness. Restriction prompt still applied.

## Testing Your Setup

Run validation script:
```bash
python validate_gemini_enhancements.py
```

**Expected output**:
```
✓ PASS     Module imports
✓ PASS     Parameter signatures
✓ PASS     CSV logging
✓ PASS     GUI controls
✓ All validation checks PASSED
```

If any test fails, check:
- Virtualenv activated (`source htr_gui/bin/activate`)
- All dependencies installed (`pip install -r requirements.txt`)
- Branch is `gemini-3-adjustments` (`git branch --show-current`)

## CSV Analysis Examples

### Find runs with high internal token waste:
```bash
awk -F, '$8 > 1500 {print $2, $3, $8, $9}' gemini_runs.csv
```
Output: `model thinking_mode internal_tok emitted_chars`

### Calculate average internal tokens by thinking mode:
```bash
awk -F, 'NR>1 && $8 != "" {sum[$3] += $8; count[$3]++} END {for (mode in sum) print mode, sum[mode]/count[mode]}' gemini_runs.csv
```

### Count fallback activations:
```bash
grep -c "fallback_success" gemini_runs.csv
```

## Getting Help

- **Documentation**: `GEMINI_3_ENHANCEMENTS.md` (full technical spec)
- **Console logs**: All diagnostic messages prefixed with emoji (⚡, 🧠, ⚠️, ✅, etc.)
- **CSV logs**: `gemini_runs.csv` tracks every transcription attempt with token breakdown

## Next Steps

After verifying the branch works:
1. Merge to `batch-processing-improvements`: `git checkout batch-processing-improvements && git merge gemini-3-adjustments`
2. Push to remote: `git push origin batch-processing-improvements`
3. Open PR to merge into `main` branch on GitHub.
